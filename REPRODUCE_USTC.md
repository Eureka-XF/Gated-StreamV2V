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

## First Target

Run a single video-to-video smoke test from:

```text
vid2vid/main.py
```

using a small demo video from:

```text
vid2vid/source_video
```
