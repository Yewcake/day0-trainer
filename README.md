# Day0 Trainer — Made by Yewcake

Universal direct Diffusers + PEFT trainer with a web UI. Ships with Krea 2 (Raw/Turbo, LoRA + LoKr); new models plug in via `models.json` + a trainer adapter. Runs as a RunPod template.

## Features

- Model dropdown driven by `models.json` — add architectures without touching the UI
- LoRA/LoKr with factor + full-rank options, ComfyUI-native export
- Live loss chart (EMA smoothed), per-checkpoint sample gallery, checkpoint downloads
- Dataset manager: drag and drop folders, .zip/.rar/.7z archives, or loose images
- Built-in Gemini captioner: fetched model list, editable instruction prompt, trigger-word injection, batch progress, per-image caption editing
- Gemini checkpoint analysis: sends the loss curve + one sample per checkpoint, returns best-candidate verdict
- Settings page for Hugging Face token (model downloads) and Gemini API key
- Light + dark mode, soft purple theme


## How updates work

The Docker image contains only the heavy environment (CUDA, PyTorch, Diffusers, PEFT). The trainer and UI code in this repo is pulled fresh from GitHub every time a pod boots, and can also be updated live from the UI ("Update trainer" button, which does a git pull and restarts).

- Change UI or trainer code → push to `main` → restart pod or click Update. No rebuild.
- Change dependencies (Dockerfile) → push → GitHub Action rebuilds and pushes the image to GHCR automatically.

## Deploying your own pod

This repo and its GHCR image are both public, so no credentials are needed to pull either.

1. Deploy a pod from the **day0-trainer** RunPod template (or build your own from this repo — see below).
2. When deploying, set your own **`UI_PASSWORD`** — the template does not ship with a fixed one, so leaving it unset means the UI is open to anyone with the pod's URL.
3. Open the HTTP service on port 8888, enter your password, upload a dataset, start training.

## One-time setup (building your own template)

1. Fork or push this repo to GitHub.
2. Let the GitHub Action build the image once (Actions tab → "Build and push trainer image" → Run workflow). It lands at `ghcr.io/<you>/<repo>:latest`.
3. Create the RunPod template:
   - **Container image**: `ghcr.io/<you>/<repo>:latest`
   - **Expose HTTP port**: `8888`
   - **Volume**: mount path `/workspace`, 150 GB+ recommended (model cache, datasets, checkpoints all persist here)
   - **Environment variables**:
     - `TRAINER_REPO_URL` — this repo's clone URL (use an `https://<token>@github.com/...` URL only if your fork is private)
     - `TRAINER_REPO_BRANCH` — optional, default `main`
     - `HF_TOKEN` — Hugging Face token for pulling Krea 2 weights
     - `UI_PASSWORD` — access password for the web UI (leave unset to force each deployer to set their own if you publish the template)

## Layout

```
Dockerfile                  environment image (rebuild rarely)
docker/start.sh             pod boot: git pull code, launch UI
app/main.py                 FastAPI backend (jobs, metrics, samples, downloads)
app/static/index.html       web UI (loss chart, sample gallery, checkpoints)
trainer/train_krea2_lora_direct.py   the actual trainer (LoRA + LoKr)
```

## Job output

Each run lives in `/workspace/jobs/<id>/`:

```
config.json           form snapshot
log.txt               full trainer output
run/metrics.jsonl     per-step loss/lr — the UI chart reads this
run/samples/step-*/   sample images per checkpoint
run/checkpoints/*/krea2_comfy_native_lora.safetensors   ComfyUI-ready file
```

The exported `.safetensors` loads in ComfyUI's standard LoRA loader for both network types. Use strength ~1.35 for LoRA runs, 1.0 for LoKr runs.

## Notes

- One job at a time (one GPU per pod). The API refuses a second concurrent job.
- Datasets are zips of images with matching `.txt` caption files; uploaded via the UI to `/workspace/datasets/<name>/`.
- The CLI launcher `Train_Krea2_Direct_Diffusers.sh` still works over SSH for headless runs; the UI and CLI share the same trainer script.
