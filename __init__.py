"""Mage-Flow ARA loader for ComfyUI.

ARA (accuracy recovery adapter) is a low-rank correction trained to sit *next to* a quantized linear:
    y = Q(W) x + (alpha / rank) * up(down(x))
The rank is read per layer from lora_down, so one file may mix ranks (the W4A8 release uses rank 32 in
blocks 0-9 and rank 64 in blocks 10-11).
ComfyUI's regular LoRA loaders merge a LoRA into the weight and requantize it, which rounds an ARA
correction (smaller than one INT4 step) away. This node keeps the quantized weight untouched and adds the
correction as a full-precision residual at runtime, exactly as during ARA training.
"""
import logging

import torch
import torch.nn.functional as F

import comfy.utils
import folder_paths


class _ARAResidualForward:
    def __init__(self, module, down, up, scale):
        # Bind the class forward, not `module.forward`: when another patched copy of this model is still loaded,
        # `module.forward` can be that copy's ARA wrapper, and adapters would stack across ARA loads.
        self.base_forward = type(module).forward.__get__(module)
        self.down_cpu, self.up_cpu, self.scale = down, up, scale
        self._cache = None

    def _weights(self, device, dtype):
        # Kept on the GPU in the activation dtype (bf16): no per-call fp32 copies of the (large) input.
        # Tensor-core matmuls accumulate in fp32; the correction is ~0.5% of the layer output, so bf16
        # rounding of it is far below the output's own bf16 rounding.
        if self._cache is None or self._cache[0] != (device, dtype):
            self._cache = ((device, dtype), self.down_cpu.to(device, dtype), (self.up_cpu * self.scale).to(device, dtype))
        return self._cache[1], self._cache[2]

    def __call__(self, x, *args, **kwargs):
        out = self.base_forward(x, *args, **kwargs)
        down, up = self._weights(x.device, out.dtype)
        return out + F.linear(F.linear(x.to(out.dtype), down), up)


class MageFlowARALoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "ara_name": (folder_paths.get_filename_list("loras"),),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "loaders/mage-flow"
    DESCRIPTION = "Adds a Mage-Flow ARA correction as a runtime residual next to the quantized weights (no merge, no requantization)."

    def load(self, model, ara_name, strength):
        path = folder_paths.get_full_path_or_raise("loras", ara_name)
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        layers = sorted({k[: -len(".lora_down.weight")] for k in sd if k.endswith(".lora_down.weight")})
        if not layers:
            raise ValueError(f"{ara_name} contains no lora_down/lora_up pairs")
        patched = model.clone()
        diffusion = patched.get_model_object("diffusion_model")
        count = 0
        for layer in layers:
            name = layer[len("diffusion_model."):] if layer.startswith("diffusion_model.") else layer
            try:
                module = diffusion.get_submodule(name)
            except AttributeError:
                logging.warning("Mage ARA: layer %s not found in model, skipped", name)
                continue
            down, up = sd[f"{layer}.lora_down.weight"], sd[f"{layer}.lora_up.weight"]
            alpha = float(sd[f"{layer}.alpha"].item()) if f"{layer}.alpha" in sd else float(down.shape[0])
            scale = strength * alpha / down.shape[0]
            patched.add_object_patch(f"diffusion_model.{name}.forward", _ARAResidualForward(module, down, up, scale))
            count += 1
        logging.info("Mage ARA: attached %d residual branches from %s (strength %.2f)", count, ara_name, strength)
        return (patched,)


NODE_CLASS_MAPPINGS = {"MageFlowARALoader": MageFlowARALoader}
NODE_DISPLAY_NAME_MAPPINGS = {"MageFlowARALoader": "Load Mage-Flow ARA (residual)"}
