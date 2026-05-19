"""NVIDIA-only Pipecat voice smoke test (WebRTC).

Usage:
  source /Users/saketm10/miniconda3/etc/profile.d/conda.sh
  conda activate ecs
  set -a; source .env; set +a
  python tests/pipcat/test.py -t webrtc

Requirements:
  - NVIDIA_API_KEY in environment
  - pip install "pipecat-ai[nvidia,webrtc]"
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure repository root is importable when running as `python tests/pipcat/test.py`.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.interfaces import build_runner_bot, run_pipecat_main


@dataclass(slots=True)
class TestConfig:
    vad_enabled: bool = True
    audio_in_sample_rate: int = 16000
    audio_out_sample_rate: int = 24000


def _load_config() -> TestConfig:
    return TestConfig(
        vad_enabled=os.getenv("PIPECAT_TEST_VAD_ENABLED", "true").strip().lower() != "false",
        audio_in_sample_rate=int(os.getenv("PIPECAT_TEST_AUDIO_IN", "16000")),
        audio_out_sample_rate=int(os.getenv("PIPECAT_TEST_AUDIO_OUT", "24000")),
    )


async def _run_bot(transport: Any, runner_args: Any) -> None:
    from pipecat.frames.frames import TranscriptionFrame, TTSSpeakFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.services.nvidia.stt import NvidiaSTTService
    from pipecat.services.nvidia.tts import NvidiaTTSService

    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise ValueError("NVIDIA_API_KEY is required for this test bot.")

    stt = NvidiaSTTService(api_key=api_key)
    tts = NvidiaTTSService(api_key=api_key)

    class EchoProcessor(FrameProcessor):
        async def process_frame(self, frame: Any, direction: FrameDirection) -> None:
            await super().process_frame(frame, direction)

            if isinstance(frame, TranscriptionFrame):
                finalized = bool(getattr(frame, "finalized", True))
                if not finalized:
                    return
                text = str(getattr(frame, "text", "")).strip()
                if not text:
                    return

                response = f"I heard: {text}"
                await self.push_frame(TTSSpeakFrame(response), FrameDirection.DOWNSTREAM)
                return

            await self.push_frame(frame, direction)

    cfg = _load_config()
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            EchoProcessor(),
            tts,
            transport.output(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=cfg.audio_in_sample_rate,
            audio_out_sample_rate=cfg.audio_out_sample_rate,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    try:
        @transport.event_handler("on_client_connected")
        async def _on_client_connected(transport_obj: Any, client: Any) -> None:
            del transport_obj, client
            await task.queue_frame(TTSSpeakFrame("Hello. NVIDIA voice test is ready. Please speak now."))
    except Exception:
        pass

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


def _build_bot() -> Any:
    from pipecat.transports.base_transport import TransportParams

    vad_analyzer = None
    cfg = _load_config()
    if cfg.vad_enabled:
        try:
            from pipecat.audio.vad.silero import SileroVADAnalyzer

            vad_analyzer = SileroVADAnalyzer()
        except Exception:
            vad_analyzer = None

    # Keep this test lightweight: only expose webrtc transport params.
    transport_params = {
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=vad_analyzer,
        )
    }
    return build_runner_bot(run_bot=_run_bot, transport_params=transport_params)


bot = _build_bot()


if __name__ == "__main__":
    run_pipecat_main()
