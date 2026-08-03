Day0 Trainer: Krea 2 / Ideogram 4 / MiniMax H3 LoRA Trainer

by Yewcake

Universal direct Diffusers + PEFT LoRA/LoKr trainer with a web UI. Ships with Krea 2 (Raw/Turbo), Ideogram 4, and MiniMax H3 (video, experimental); new models plug in via `models.json`.

![Day0 Trainer sample output 1](https://raw.githubusercontent.com/Yewcake/day0-trainer/main/docs/images/sample-1.jpg)
![Day0 Trainer sample output 2](https://raw.githubusercontent.com/Yewcake/day0-trainer/main/docs/images/sample-2.jpg)

Repo & source: https://github.com/Yewcake/day0-trainer (fully public, inspect before you trust it)

## Supported models

- **Krea 2** (Raw / Turbo): LoRA + LoKr
- **Ideogram 4**: LoRA
- **MiniMax H3** (video): LoRA, experimental

## Before you deploy

Set your own `UI_PASSWORD` environment variable when creating the pod. If left blank, the web UI is open to anyone who has your pod's URL.

## First boot can be slow, this is normal

The image is ~10GB (CUDA + PyTorch + Diffusers). Public/anonymous image pulls are throttled harder by the registry than authenticated ones, so depending on which host you land on, the first boot can take anywhere from 2 minutes to ~30-40 minutes. Check the pod's Logs tab: as long as you see layers actively "Downloading" or "Extracting" (not stuck with no movement), it's working, just slow. It only happens once per host; restarting the same pod later is fast.

## After deploying

1. Open the pod's **Dashboard** (HTTP service on port 8888).
2. Enter the password you set.
3. Upload a dataset (drag a folder, .zip/.rar/.7z, or loose images/clips + .txt captions).
4. Caption with the built-in Gemini captioner (needs your own Gemini API key, set in Settings) or bring your own captions.
5. Pick a model, start a LoRA or LoKr training run. Live loss chart, sample images, and checkpoint downloads (ComfyUI-ready `.safetensors`) are all in the UI.

## Notes

- One job at a time per pod (one GPU).
- The Docker image only holds the heavy environment (CUDA/PyTorch/Diffusers). Trainer + UI code is pulled fresh from the public GitHub repo at boot, and can be updated live from the "Update trainer" button without a pod restart.
- Recommended container disk: 30GB+ (holds the pulled image itself).
- Recommended volume: 150GB+, mounted at `/workspace` (model cache, datasets, checkpoints all live here).
