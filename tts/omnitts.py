"""
vLLM Omni TTS via OpenAI-compatible speech API.

POST /v1/audio/speech  with JSON body, returns raw PCM audio.

Supported models (vLLM Omni):
  - Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice  (24 kHz)
  - Qwen/Qwen3-TTS-12Hz-1.7B-Base          (24 kHz)
  - Qwen/Qwen3-TTS-12Hz-0.6B-Base   (24 kHz)
  - fishaudio/s2-pro                        (44.1 kHz)
  - FunAudioLLM/Fun-CosyVoice3-0.5B-2512   (24 kHz)
  - mistralai/Voxtral-4B-TTS-2603           (16 kHz)
  - openbmb/VoxCPM2                         (16 kHz)
  - OpenMOSS-Team/MOSS-TTS-Nano             (24 kHz)

Usage:
    python app.py --tts omnitts --TTS_SERVER http://127.0.0.1:8091 \
                  --REF_FILE vivian "
"""

import time
import io
import soundfile as sf
import numpy as np
import resampy
import requests
from typing import Iterator

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register

# Default source sample rate for the TTS model output.
# Qwen3-TTS / CosyVoice3 / MOSS-TTS-Nano: 24000
# Fish Speech S2 Pro:                      44100
# Voxtral / VoxCPM2:                       16000
DEFAULT_SRC_SR = 24000


@register("tts", "omnitts")
class OmniTTS(BaseTTS):
    """TTS client that talks to a vLLM Omni server via OpenAI-compatible API."""

    def __init__(self, opt, parent):
        super().__init__(opt, parent)

        # ── required ──────────────────────────────────────────
        self.server_url = opt.TTS_SERVER.rstrip("/")       # e.g. http://127.0.0.1:8091
        self.voice = opt.REF_FILE or "default"               # speaker name
        #self.ref_text = opt.REF_TEXT or ""                  # reference transcript (for Base mode)

        # ── optional, configurable via opt ────────────────────
        # self.model = getattr(opt, "omni_tts_model",
        #                      "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
        self.language = getattr(opt, "omni_tts_language", "Auto")
        self.speed = float(getattr(opt, "omni_tts_speed", 1.0))
        self.response_format = getattr(opt, "omni_tts_format", "wav")
        self.task_type = getattr(opt, "omni_tts_task_type", "CustomVoice")
        self.instructions = getattr(opt, "omni_tts_instructions", "")
        self.src_sr = int(getattr(opt, "omni_tts_src_sr", DEFAULT_SRC_SR))

        logger.info(
            f"OmniTTS init: server={self.server_url}, "
            f"voice={self.voice}, src_sr={self.src_sr}"
        )

    # ── main entry ────────────────────────────────────────────

    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg

        # per-message overrides (from datainfo / avatar config)
        tts_cfg = textevent.get("tts", {})
        voice = tts_cfg.get("ref_file", self.voice)
        # ref_text = tts_cfg.get("ref_text", self.ref_text)
        language = tts_cfg.get("language", self.language)
        speed = float(tts_cfg.get("speed", self.speed))
        instructions = tts_cfg.get("instructions", self.instructions)
        # model = tts_cfg.get("model", self.model)
        task_type = tts_cfg.get("task_type", self.task_type)

        self.stream_tts(
            self._synthesize(
                text=text,
                voice=voice,
                language=language,
                speed=speed,
                instructions=instructions,
                task_type=task_type,
            ),
            msg,
        )

    # ── API call ──────────────────────────────────────────────

    def _synthesize(
        self,
        text: str,
        voice: str,
        language: str,
        speed: float,
        instructions: str,
        task_type: str,
    ) -> Iterator[bytes]:
        """
        Call POST /v1/audio/speech and yield raw PCM chunks.
        """
        url = f"{self.server_url}/v1/audio/speech"

        body = {
            "input": text,
            "voice": voice,
            "response_format": self.response_format,
            "speed": speed,
            "language": language,
            "task_type": task_type,
            "stream": True,
            "stream_format": "audio",
        }
        if instructions:
            body["instructions"] = instructions
        # if ref_text:
        #     body["ref_text"] = ref_text

        start = time.perf_counter()
        logger.info(f"OmniTTS POST {url} voice={voice} text={text[:60]}...")

        try:
            res = requests.post(
                url,
                json=body,
                stream=True,
                headers={"Content-Type": "application/json"},
                timeout=120,
            )

            if res.status_code != 200:
                logger.error("OmniTTS server error: %s", res.text)
                return

            first = True
            for chunk in res.iter_content(chunk_size=None):
                logger.info('chunk len:%d',len(chunk))
                if first:
                    end = time.perf_counter()
                    logger.info(f"OmniTTS first chunk latency: {end - start:.2f}s")
                    first = False
                if chunk and self.state == State.RUNNING:
                    yield chunk

        except requests.exceptions.Timeout:
            logger.error("OmniTTS request timeout")
        except Exception:
            logger.exception("OmniTTS synthesize error")

    # ========================== 流式音频处理 ==========================
    
    def stream_tts(self, audio_stream, msg: tuple[str, dict]):
        """
        将音频字节流按 20ms chunk 推送给父级 avatar。
        自动检测 WAV/PCM 格式：首个 chunk 若为 WAV 则用 soundfile 解码，
        否则按 int16 PCM 处理。若输出采样率与项目标准 16kHz 不同则重采样。
        """
        text, textevent = msg
        first = True
        last_stream = np.array([], dtype=np.float32)
        wav_sr = None  # WAV 文件中的实际采样率

        for chunk in audio_stream:
            if chunk is None or len(chunk) == 0:
                continue

            if first and len(chunk) >= 4 and chunk[:4] == b'RIFF':
                # WAV 格式（含文件头），用 soundfile 解码
                with io.BytesIO(chunk) as buf:
                    stream, wav_sr = sf.read(buf, dtype='float32', always_2d=False)
                if stream.ndim > 1:
                    stream = stream[:, 0]
                logger.info(f"OmniTTS: WAV decoded, sr={wav_sr}, "
                            f"frames={len(stream)}")
            else:
                # 裸 PCM: int16 → float32
                stream = (np.frombuffer(chunk, dtype=np.int16)
                            .astype(np.float32) / 32768.0)

            # 确定源采样率并重采样到 16kHz
            actual_sr = wav_sr or self.src_sr
            if actual_sr != self.sample_rate and stream.shape[0] > 0:
                stream = resampy.resample(
                    x=stream, sr_orig=actual_sr, sr_new=self.sample_rate
                )

            # 拼接上次剩余
            if last_stream.shape[0] > 0:
                stream = np.concatenate([last_stream, stream])

            # 按 self.chunk (320 samples = 20ms @16kHz) 分块推送
            total = stream.shape[0]
            idx = 0
            while total - idx >= self.chunk and self.state == State.RUNNING:
                eventpoint = {}
                if first:
                    eventpoint = {"status": "start", "text": text}
                    first = False
                eventpoint.update(**textevent)
                self.parent.put_audio_frame(
                    stream[idx:idx + self.chunk], eventpoint
                )
                idx += self.chunk

            # 不足一 chunk 的留到下次
            last_stream = stream[idx:]

        # ── send end-of-stream marker ──
        eventpoint = {"status": "end", "text": text}
        eventpoint.update(**textevent)
        self.parent.put_audio_frame(
            np.zeros(self.chunk, dtype=np.float32), eventpoint
        )
