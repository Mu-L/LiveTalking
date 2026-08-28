import os
import time
import math
import numpy as np
import azure.cognitiveservices.speech as speechsdk

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register

@register("tts", "azuretts")
class AzureTTS(BaseTTS):
    """Azure 语音合成 (认知服务) — 基于 http 的流式合成。

    音频输出格式固定为 16kHz 16bit mono PCM，通过 synthesizing 回调逐块推送。

    需要设置环境变量 AZURE_SPEECH_KEY 与 AZURE_TTS_ENDPOINT
    REF_FILE 用作音色 (voice name)，例如 zh-CN-XiaoxiaoMultilingualNeural。

    """

    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        self.audio_buffer = b''
        self.voice = opt.REF_FILE or "zh-CN-XiaoxiaoMultilingualNeural"   # https://learn.microsoft.com/zh-cn/azure/ai-services/speech-service/language-support?tabs=tts
        speech_key = os.getenv("AZURE_SPEECH_KEY")
        speech_endpoint = os.getenv("AZURE_TTS_ENDPOINT")

        if not speech_key or not speech_endpoint:
            logger.warning("AzureTTS: AZURE_SPEECH_KEY / AZURE_TTS_ENDPOINT 未设置，请检查环境变量")

        #speech_endpoint = f"wss://{tts_region}.tts.speech.microsoft.com/cognitiveservices/websocket/v2"
        self.speech_config = speechsdk.SpeechConfig(subscription=speech_key, endpoint=speech_endpoint)
        self.speech_config.speech_synthesis_voice_name = self.voice
        self.speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm)

        # 获取内存中流形式的结果
        self.speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=None)
        self.speech_synthesizer.synthesizing.connect(self._on_synthesizing)
        self.speech_synthesizer.synthesis_completed.connect(self._on_completed)
        self.speech_synthesizer.synthesis_canceled.connect(self._on_canceled)

        # 当前消息的流式上下文，供回调附加 eventpoint
        self._cur_text = ""
        self._cur_textevent = {}
        self._first = True       # 是否尚未发送 start 事件
        self._ended = False      # 是否已发送 end 事件 (防止重复)
        self._t_start = None     # 首块延迟计时

        logger.info(f"AzureTTS init: voice={self.voice}")

    def txt_to_audio(self, msg: tuple[str, dict]):
        msg_text, textevent = msg

        # 每条消息可覆盖音色
        tts_cfg = textevent.get("tts", {})
        voice = tts_cfg.get("ref_file", self.voice)

        # 重置本次消息的流式状态
        self.audio_buffer = b''
        self._cur_text = msg_text
        self._cur_textevent = textevent
        self._first = True
        self._ended = False
        self._t_start = time.perf_counter()

        logger.info(f"AzureTTS synthesize: voice={voice} text={msg_text[:60]}...")

        try:
            result = self.speech_synthesizer.speak_text(msg_text) 

            # # 延迟指标
            # fb_latency = int(result.properties.get_property(
            #     speechsdk.PropertyId.SpeechServiceResponse_SynthesisFirstByteLatencyMs
            # ))
            # fin_latency = int(result.properties.get_property(
            #     speechsdk.PropertyId.SpeechServiceResponse_SynthesisFinishLatencyMs
            # ))
            # logger.info(f"azure音频生成: 首字节延迟 {fb_latency} ms, 完成延迟 {fin_latency} ms, result_id={result.result_id}")
        except Exception:
            logger.exception("AzureTTS synthesize error")
            self._finish()

        # 兜底: 若完成回调未触发 (异常等), 这里补发结束标记
        self._finish()

    # ── 流式回调 ──────────────────────────────────────────────

    def _on_synthesizing(self, evt: speechsdk.SpeechSynthesisEventArgs):
        """每个音频 chunk 到达时触发 (reason == SynthesizingAudio)。"""
        if evt.result.reason == speechsdk.ResultReason.Canceled:
            self._finish(error="canceled")
            return
        if self.state != State.RUNNING:
            self.audio_buffer = b''
            return

        data = evt.result.audio_data
        if not data:
            return

        self.audio_buffer += data
        chunk_bytes = self.chunk * 2  # 320 samples * 2 bytes (int16)
        while len(self.audio_buffer) >= chunk_bytes:
            raw = self.audio_buffer[:chunk_bytes]
            self.audio_buffer = self.audio_buffer[chunk_bytes:]

            frame = (np.frombuffer(raw, dtype=np.int16)
                       .astype(np.float32) / 32767.0)

            eventpoint = {}
            if self._first:
                eventpoint = {"status": "start", "text": self._cur_text}
                self._first = False
                logger.info(f"azure 首块延迟: {time.perf_counter() - self._t_start:.2f}s")
            eventpoint.update(**self._cur_textevent)
            self.parent.put_audio_frame(frame, eventpoint)

    def _on_completed(self, evt: speechsdk.SpeechSynthesisEventArgs):
        """合成完成 (reason == SynthesizingAudioCompleted)。"""
        self._finish()

    def _on_canceled(self, evt: speechsdk.SpeechSynthesisEventArgs):
        details = evt.result.cancellation_details
        error = f"canceled: {details.reason}"
        if details.reason == speechsdk.CancellationReason.Error:
            error = f"canceled: {details.reason}, {details.error_details}"
        self._finish(error=error)

    # ── 结束处理 (幂等) ───────────────────────────────────────

    def _finish(self, error: str = None):
        if self._ended:
            return
        self._ended = True

        if error:
            logger.error(f"AzureTTS 合成结束异常: {error}")

        # 发送结束标记 (与 doubao/omnitts 模式一致)
        if self.state == State.RUNNING:
            eventpoint = {"status": "end", "text": self._cur_text}
            eventpoint.update(**self._cur_textevent)
            self.parent.put_audio_frame(
                np.zeros(self.chunk, dtype=np.float32), eventpoint
            )

    def stop_tts(self):
        try:
            self.speech_synthesizer.stop_speaking()
        except Exception:
            pass
        logger.info("AzureTTS stopped")
