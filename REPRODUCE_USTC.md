# USTC Reproduction Notes

This branch is for reproducing Gated-StreamV2V on the USTC undergraduate computing platform.

## Working Model

Use three places for three different jobs:

- Local Mac: edit code with VS Code.
- GitHub: keep version history and transfer code.
- USTC server: run GPU jobs with Slurm.

Typical loop:

```bash
# Local Mac
git status
git add .
git commit -m "describe the change"
git push

# USTC server
cd /home/scc/pb22020481/projects/Gated-StreamV2V
git pull
```

## Paths

Local Mac clone:

```text
/Users/andylee/My files/AI/杰毕设/Gated-StreamV2V
```

USTC server clone:

```text
/home/scc/pb22020481/projects/Gated-StreamV2V
```

Project conda environment to create later:

```text
/home/scc/pb22020481/conda-envs/gated-streamv2v
```

## What Goes Into Git

Good to commit:

- Python source code
- Slurm scripts
- Shell scripts
- Small config files
- Notes and reproduction logs

Do not commit:

- Model weights
- Large videos
- Generated outputs
- Large logs
- API keys or tokens

## Environment

Create or refresh the USTC conda environment with:

```bash
cd /home/scc/pb22020481/projects/Gated-StreamV2V
bash scripts/ustc/setup_env.sh
```

The current USTC-tested environment uses:

- Python 3.10
- PyTorch 2.7.0+cu128
- torchvision 0.22.0+cu128
- xformers 0.0.30
- diffusers 0.27.0
- transformers 4.37.2
- huggingface-hub 0.25.2

Why cu128: the Students partition currently assigns NVIDIA GeForce RTX 5090 nodes. Older PyTorch cu121 wheels can see the GPU, but fail with `no kernel image is available for execution on the device`.

## Model Downloads

The USTC jobs use:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/scc/pb22020481/.cache/huggingface
```

Reason: `huggingface.co` DNS resolution failed on the platform, while `hf-mirror.com` worked.

## First Target

Run a single video-to-video smoke test from:

```text
vid2vid/main.py
```

using a small demo video from:

```text
vid2vid/demo_selfie/tennis.mp4
```

Submit it with:

```bash
cd /home/scc/pb22020481/projects/Gated-StreamV2V
sbatch scripts/ustc/run_single_video.sbatch
```

Successful run on 2026-06-14:

```text
job: 16452
node: anode05
gpu: NVIDIA GeForce RTX 5090
output: /home/scc/pb22020481/projects/Gated-StreamV2V/vid2vid/outputs_ustc/single_similarity/tennis_similarity_ustc.mp4
size: 2983956 bytes
frames: 70
fps: 25.0
width: 912
height: 512
```
