# ComfyUI-MageFlow-ARA

One node, **Load Mage-Flow ARA (residual)**, that attaches a Mage-Flow *accuracy recovery adapter* (ARA) to a quantized Mage-Flow model in ComfyUI.

Made for [SearchingMan/Mage-Flow-W4A8-ARA](https://huggingface.co/SearchingMan/Mage-Flow-W4A8-ARA): a native ComfyUI W4A8 checkpoint plus an ARA computed and fine-tuned against exactly that checkpoint's quantized weights (rank 32 in blocks 0–9, rank 64 in blocks 10–11).

## Why a separate node

An ARA is trained to run *next to* a quantized linear layer:

```
y = Q(W)·x + up(down(x))
```

Its correction is smaller than one INT4 step. ComfyUI's regular **LoraLoader** merges a LoRA into the weight and requantizes it, so most of that correction rounds away. This node leaves the quantized weight untouched and adds the correction at runtime, the same way it was computed during training (bf16 tensor-core matmuls, about 15% extra generation time). The base model still runs on ComfyUI's native quantized kernels.

The ARA file is a standard LoRA file (`diffusion_model.<layer>.lora_down/lora_up/alpha`), so it also loads in a regular LoraLoader, just with lower quality.

The rank is read separately for each layer from its `lora_down` shape and scaled by that layer's `alpha / rank`, so one file can mix ranks. The release file does: rank 32 in blocks 0–9 and rank 64 in blocks 10–11.

## Install

Clone it into ComfyUI's `custom_nodes` folder and restart ComfyUI:

```
cd ComfyUI/custom_nodes
git clone https://github.com/kgonia/ComfyUI-MageFlow-ARA
```

No extra Python dependencies.

## Use

```
Load Diffusion Model (mage_flow_w4a8.safetensors)
  → Load Mage-Flow ARA (residual)   ara_name: mage_flow_w4a8_ara.safetensors, strength 1.0
  → KSampler
```

Put the ARA file in `ComfyUI/models/loras/`. `strength` scales the correction. Keep 1.0 unless you are experimenting.

An example workflow is available in [kgonia/comfy-workflows](https://github.com/kgonia/comfy-workflows) and in the Hugging Face repo.

## Requirements

- A ComfyUI version with native Mage-Flow support and the `asym_w4a8_int8` quantization format (comfy-kitchen ≥ 0.2.37).
- The node only adds the residual branch. Quantized execution itself is ComfyUI's.

## License

MIT
