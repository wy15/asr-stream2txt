from __future__ import annotations

import datetime as dt
import unittest
from pathlib import Path

import numpy as np

from asr_stream2txt.cli import (
    DeviceInfo,
    SpeechSegmenter,
    build_audio_writer_args,
    choose_recommended_device,
    date_dir_name,
    format_transcript_line,
    resolve_audio_extension,
    sanitize_text,
    time_file_stem,
    transcript_file_name,
)


class CliHelpersTest(unittest.TestCase):
    def test_choose_recommended_device_avoids_virtual_audio(self) -> None:
        device = choose_recommended_device(
            [
                DeviceInfo(index=4, name="Microsoft Teams Audio", default_samplerate=48000, max_input_channels=2),
                DeviceInfo(index=2, name="qiAirPods3", default_samplerate=24000, max_input_channels=1),
            ]
        )
        self.assertIsNotNone(device)
        self.assertEqual(device.index, 2)

    def test_path_and_format_helpers(self) -> None:
        now = dt.datetime(2026, 3, 18, 9, 1, 5)
        self.assertEqual(transcript_file_name(now), "2026-03-18.txt")
        self.assertEqual(date_dir_name(now), "2026-03-18")
        self.assertEqual(time_file_stem(now), "090105")
        self.assertEqual(resolve_audio_extension("opus"), "opus")
        self.assertEqual(format_transcript_line(now, now, "测试"), "[09:01:05 - 09:01:05] 测试")
        self.assertEqual(sanitize_text(" 你好 \n 世界 "), "你好 世界")

    def test_build_audio_writer_args_supports_segmented_opus(self) -> None:
        args = build_audio_writer_args(
            audio_format="opus",
            output_path=Path("/tmp/audio/090105-%03d.opus"),
            segment_minutes=30,
        )
        self.assertIn("segment", args)
        self.assertIn("1800", args)
        self.assertIn("/tmp/audio/090105-%03d.opus", args)

    def test_speech_segmenter_flushes_after_silence(self) -> None:
        segmenter = SpeechSegmenter(
            sample_rate=16_000,
            rms_threshold=0.01,
            silence_seconds=0.6,
            min_speech_seconds=0.3,
            max_segment_seconds=8.0,
            pre_roll_seconds=0.2,
        )

        start = dt.datetime(2026, 3, 18, 9, 0, 0)
        speech = np.full(1600, 6000, dtype=np.int16)
        silence = np.zeros(1600, dtype=np.int16)

        out = []
        out.extend(segmenter.push(speech, start + dt.timedelta(seconds=0.1)))
        out.extend(segmenter.push(speech, start + dt.timedelta(seconds=0.2)))
        out.extend(segmenter.push(silence, start + dt.timedelta(seconds=0.5)))
        out.extend(segmenter.push(silence, start + dt.timedelta(seconds=0.9)))

        self.assertEqual(len(out), 1)
        self.assertGreater(out[0].audio.size, 0)
        self.assertEqual(out[0].start_at.date(), start.date())

    def test_speech_segmenter_flushes_on_shutdown(self) -> None:
        segmenter = SpeechSegmenter(
            sample_rate=16_000,
            rms_threshold=0.01,
            silence_seconds=0.8,
            min_speech_seconds=0.3,
            max_segment_seconds=8.0,
            pre_roll_seconds=0.2,
        )
        start = dt.datetime(2026, 3, 18, 9, 0, 0)
        speech = np.full(3200, 5000, dtype=np.int16)
        segmenter.push(speech, start + dt.timedelta(seconds=0.2))
        out = segmenter.flush(start + dt.timedelta(seconds=0.5))
        self.assertEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()
