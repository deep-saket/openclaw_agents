# Pipecat NVIDIA Voice Test

This is a minimal **NVIDIA-only** Pipecat voice test bot.

## Prerequisites

1. `NVIDIA_API_KEY` set in environment.
2. Pipecat NVIDIA dependencies installed in your runtime env:

```bash
pip install "pipecat-ai[nvidia,webrtc]"
```

## Run

```bash
source /Users/saketm10/miniconda3/etc/profile.d/conda.sh
conda activate ecs
cd /Users/saketm10/Projects/openclaw_agents
set -a; source .env; set +a
python tests/pipcat/test.py -t webrtc
```

The Pipecat runner will print a local URL. Open it in browser, click connect, allow mic, and speak.

Expected behavior: bot says hello and repeats: `I heard: <your speech>`.
