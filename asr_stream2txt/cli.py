from __future__ import annotations

import argparse
import datetime as dt
import inspect
import queue
import shutil
import signal
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import sounddevice as sd
from mlx_audio.stt.utils import load_model

SAMPLE_RATE = 16_000
DEFAULT_MODEL = "mlx-community/Qwen3-ASR-0.6B-4bit"
DEFAULT_OUTPUT_DIR = Path("data/transcripts")
DEFAULT_AUDIO_DIR = Path("data/audio")
QUEUE_POLL_SECONDS = 0.2


@dataclass(slots=True)
class DeviceInfo:
    index: int
    name: str
    default_samplerate: float
    max_input_channels: int


@dataclass(slots=True)
class RuntimeConfig:
    device_index: int | None
    model: str
    output_dir: Path
    language: str | None
    prompt: str
    save_audio: bool
    audio_dir: Path
    audio_format: str
    audio_segment_minutes: float
    rms_threshold: float
    silence_seconds: float
    min_speech_seconds: float
    max_segment_seconds: float
    pre_roll_seconds: float


@dataclass(slots=True)
class Utterance:
    audio: np.ndarray
    start_at: dt.datetime
    end_at: dt.datetime


class SpeechSegmenter:
    def __init__(
        self,
        *,
        sample_rate: int,
        rms_threshold: float,
        silence_seconds: float,
        min_speech_seconds: float,
        max_segment_seconds: float,
        pre_roll_seconds: float,
    ) -> None:
        self.sample_rate = sample_rate
        self.rms_threshold = rms_threshold
        self.silence_seconds = silence_seconds
        self.min_speech_seconds = min_speech_seconds
        self.max_segment_seconds = max_segment_seconds
        self.pre_roll_seconds = pre_roll_seconds

        self._pre_roll: deque[tuple[np.ndarray, float]] = deque()
        self._speech_chunks: list[np.ndarray] = []
        self._speech_duration = 0.0
        self._pre_roll_duration = 0.0
        self._speech_active = False
        self._segment_start_at: dt.datetime | None = None
        self._last_speech_at: dt.datetime | None = None

    def push(self, pcm: np.ndarray, now: dt.datetime) -> list[Utterance]:
        if pcm.size == 0:
            return []

        duration = pcm.size / self.sample_rate
        rms = float(np.sqrt(np.mean(np.square(pcm.astype(np.float32) / 32768.0))))
        utterances: list[Utterance] = []

        self._append_pre_roll(pcm, duration)

        if not self._speech_active:
            if rms >= self.rms_threshold:
                self._speech_active = True
                self._speech_chunks = [chunk.copy() for chunk, _ in self._pre_roll]
                self._speech_chunks.append(pcm.copy())
                self._speech_duration = self._pre_roll_duration + duration
                self._segment_start_at = now - dt.timedelta(
                    seconds=self._speech_duration
                )
                self._last_speech_at = now
            return utterances

        self._speech_chunks.append(pcm.copy())
        self._speech_duration += duration

        if rms >= self.rms_threshold:
            self._last_speech_at = now

        silence_elapsed = (
            (now - self._last_speech_at).total_seconds()
            if self._last_speech_at
            else 0.0
        )

        if self._speech_duration >= self.max_segment_seconds or (
            silence_elapsed >= self.silence_seconds
            and self._speech_duration >= self.min_speech_seconds
        ):
            utterance = self._finalize(now)
            if utterance is not None:
                utterances.append(utterance)

        return utterances

    def flush(self, now: dt.datetime) -> list[Utterance]:
        utterance = self._finalize(now)
        return [utterance] if utterance is not None else []

    def _append_pre_roll(self, pcm: np.ndarray, duration: float) -> None:
        self._pre_roll.append((pcm.copy(), duration))
        self._pre_roll_duration += duration
        while self._pre_roll_duration > self.pre_roll_seconds and self._pre_roll:
            _, removed_duration = self._pre_roll.popleft()
            self._pre_roll_duration -= removed_duration

    def _finalize(self, now: dt.datetime) -> Utterance | None:
        if (
            not self._speech_active
            or not self._speech_chunks
            or self._segment_start_at is None
        ):
            self._reset()
            return None

        audio = np.concatenate(self._speech_chunks)
        utterance = Utterance(audio=audio, start_at=self._segment_start_at, end_at=now)
        self._reset()
        return utterance

    def _reset(self) -> None:
        self._speech_chunks = []
        self._speech_duration = 0.0
        self._speech_active = False
        self._segment_start_at = None
        self._last_speech_at = None


class AudioWriter:
    def __init__(
        self,
        process: subprocess.Popen[bytes],
        output_hint: str,
        *,
        output_dir: Path,
        base_stem: str,
        extension: str,
        segmented: bool,
    ) -> None:
        self.process = process
        self.output_hint = output_hint
        self.output_dir = output_dir
        self.base_stem = base_stem
        self.extension = extension
        self.segmented = segmented

    @classmethod
    def create(
        cls,
        *,
        audio_dir: Path,
        audio_format: str,
        segment_minutes: float,
        started_at: dt.datetime,
    ) -> "AudioWriter":
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                "ffmpeg is required for --save-audio but was not found in PATH."
            )

        audio_dir = audio_dir / date_dir_name(started_at)
        audio_dir.mkdir(parents=True, exist_ok=True)
        base_stem = time_file_stem(started_at)
        extension = resolve_audio_extension(audio_format)

        if segment_minutes > 0:
            output_path = audio_dir / f"{base_stem}-%03d.{extension}"
            args = build_audio_writer_args(
                audio_format=audio_format,
                output_path=output_path,
                segment_minutes=segment_minutes,
            )
            output_hint = str(output_path)
            segmented = True
        else:
            output_path = audio_dir / f"{base_stem}.{extension}"
            args = build_audio_writer_args(
                audio_format=audio_format,
                output_path=output_path,
                segment_minutes=0,
            )
            output_hint = str(output_path)
            segmented = False

        process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if process.stdin is None or process.stderr is None:
            raise RuntimeError("Failed to start ffmpeg audio writer.")
        return cls(
            process=process,
            output_hint=output_hint,
            output_dir=audio_dir,
            base_stem=base_stem,
            extension=extension,
            segmented=segmented,
        )

    def write(self, pcm_bytes: bytes) -> None:
        if self.process.stdin is None:
            raise RuntimeError("Audio writer stdin is not available.")
        self.process.stdin.write(pcm_bytes)
        self.process.stdin.flush()

    def close(self, *, interrupted: bool = False) -> None:
        stderr = b""
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.stderr is not None:
            stderr = self.process.stderr.read()
        code = self.process.wait()
        if interrupted and code in (255, -2, 130):
            return
        if code != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg audio writer failed with code {code}: {detail}")


class TranscriptionWorker(threading.Thread):
    def __init__(
        self,
        *,
        model_name: str,
        output_dir: Path,
        language: str | None,
        system_prompt: str | None,
        task_queue: "queue.Queue[Utterance | None]",
        stdout,
        ready_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.model_name = model_name
        self.output_dir = output_dir
        self.language = language
        self.system_prompt = system_prompt
        self.task_queue = task_queue
        self.stdout = stdout
        self.ready_event = ready_event
        self.error: Exception | None = None
        self._model = None
        self.written_segments = 0

    def run(self) -> None:
        try:
            self._model = load_model(self.model_name)
            validate_loaded_model(self.model_name, self._model)
            self.ready_event.set()
            while True:
                task = self.task_queue.get()
                try:
                    if task is None:
                        return
                    self._transcribe(task)
                finally:
                    self.task_queue.task_done()
        except Exception as exc:  # noqa: BLE001
            self.error = exc
            self.ready_event.set()
            while True:
                try:
                    self.task_queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    self.task_queue.task_done()

    def _transcribe(self, utterance: Utterance) -> None:
        if self._model is None:
            raise RuntimeError("MLX model failed to load.")
        result = self._model.generate(
            utterance.audio.astype(np.float32) / 32768.0,
            language=self.language,
            verbose=False,
            system_prompt=self.system_prompt or None,
        )
        text = sanitize_text(getattr(result, "text", ""))
        if not text:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = self.output_dir / transcript_file_name(utterance.start_at)
        line = format_transcript_line(utterance.start_at, utterance.end_at, text)
        with transcript_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")
        self.stdout.write(f"{line}\n")
        self.stdout.flush()
        self.written_segments += 1


def transcript_file_name(moment: dt.datetime) -> str:
    return f"{moment.year:04d}-{moment.month:02d}-{moment.day:02d}.txt"


def date_dir_name(moment: dt.datetime) -> str:
    return f"{moment.year:04d}-{moment.month:02d}-{moment.day:02d}"


def time_file_stem(moment: dt.datetime) -> str:
    return f"{moment.hour:02d}{moment.minute:02d}{moment.second:02d}"


def format_clock(moment: dt.datetime) -> str:
    return f"{moment.hour:02d}:{moment.minute:02d}:{moment.second:02d}"


def format_transcript_line(
    start_at: dt.datetime, end_at: dt.datetime, text: str
) -> str:
    return f"[{format_clock(start_at)} - {format_clock(end_at)}] {text}"


def sanitize_text(text: str) -> str:
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


def resolve_audio_extension(audio_format: str) -> str:
    if audio_format == "opus":
        return "opus"
    if audio_format == "flac":
        return "flac"
    raise ValueError(f"Unsupported audio format: {audio_format}")


def build_audio_writer_args(
    *, audio_format: str, output_path: Path, segment_minutes: float
) -> list[str]:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    args = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        "-i",
        "pipe:0",
    ]

    if segment_minutes > 0:
        args.extend(
            [
                "-f",
                "segment",
                "-segment_time",
                str(int(round(segment_minutes * 60))),
                "-reset_timestamps",
                "1",
                "-segment_start_number",
                "0",
                "-segment_format",
                audio_format,
            ]
        )

    if audio_format == "opus":
        args.extend(
            [
                "-c:a",
                "libopus",
                "-b:a",
                "24k",
                "-vbr",
                "on",
                "-compression_level",
                "10",
                str(output_path),
            ]
        )
        return args

    if audio_format == "flac":
        args.extend(["-c:a", "flac", str(output_path)])
        return args

    raise ValueError(f"Unsupported audio format: {audio_format}")


def validate_loaded_model(model_name: str, model) -> None:  # noqa: ANN001
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type == "qwen3_forced_aligner" or "forcedaligner" in model_name.lower():
        raise RuntimeError(
            "The selected model is a forced aligner, not a speech-to-text model. "
            "Use 'mlx-community/Qwen3-ASR-0.6B-4bit' instead."
        )

    signature = inspect.signature(model.generate)
    text_param = signature.parameters.get("text")
    if text_param is not None and text_param.default is inspect._empty:
        raise RuntimeError(
            "The selected model requires existing transcript text for alignment and cannot do speech-to-text. "
            "Use 'mlx-community/Qwen3-ASR-0.6B-4bit' instead."
        )


def list_input_devices() -> list[DeviceInfo]:
    devices: list[DeviceInfo] = []
    for index, item in enumerate(sd.query_devices()):
        if item["max_input_channels"] <= 0:
            continue
        devices.append(
            DeviceInfo(
                index=index,
                name=str(item["name"]),
                default_samplerate=float(item["default_samplerate"]),
                max_input_channels=int(item["max_input_channels"]),
            )
        )
    return devices


def choose_recommended_device(devices: Iterable[DeviceInfo]) -> DeviceInfo | None:
    ranked = []
    for device in devices:
        name = device.name.lower()
        score = 0
        if any(token in name for token in ("macbook", "built-in", "internal")):
            score += 40
        if any(token in name for token in ("microphone", "mic", "麦克风")):
            score += 30
        if any(
            token in name for token in ("airpods", "headset", "headphones", "earpods")
        ):
            score += 15
        if any(
            token in name
            for token in (
                "teams",
                "zoom",
                "loopback",
                "blackhole",
                "obs",
                "soundflower",
            )
        ):
            score -= 50
        ranked.append((score, -device.index, device))
    if not ranked:
        return None
    ranked.sort(reverse=True)
    return ranked[0][2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="asr-stream2txt")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser(
        "setup", help="Download/cache the default MLX model"
    )
    setup_parser.add_argument("--model", default=DEFAULT_MODEL)

    subparsers.add_parser("devices", help="List input devices from sounddevice")

    start_parser = subparsers.add_parser(
        "start", help="Start realtime microphone transcription"
    )
    start_parser.add_argument("--device-index", type=int, default=None)
    start_parser.add_argument("--model", default=DEFAULT_MODEL)
    start_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    start_parser.add_argument("--language", default="Chinese")
    start_parser.add_argument("--prompt", default="")
    start_parser.add_argument("--save-audio", action="store_true")
    start_parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    start_parser.add_argument(
        "--audio-format", choices=("opus", "flac"), default="opus"
    )
    start_parser.add_argument("--audio-segment-minutes", type=float, default=0.0)
    start_parser.add_argument("--rms-threshold", type=float, default=0.012)
    start_parser.add_argument("--silence-seconds", type=float, default=0.8)
    start_parser.add_argument("--min-speech-seconds", type=float, default=0.6)
    start_parser.add_argument("--max-segment-seconds", type=float, default=8.0)
    start_parser.add_argument("--pre-roll-seconds", type=float, default=0.3)
    return parser.parse_args(argv)


def run_setup(model_name: str) -> int:
    print(f"Caching model: {model_name}")
    model = load_model(model_name)
    validate_loaded_model(model_name, model)
    print("Model is ready.")
    return 0


def run_devices() -> int:
    devices = list_input_devices()
    recommended = choose_recommended_device(devices)
    if not devices:
        print("No input devices found.", file=sys.stderr)
        return 1
    for device in devices:
        suffix = (
            " (recommended)"
            if recommended and device.index == recommended.index
            else ""
        )
        print(f"[{device.index}] {device.name}{suffix}")
    return 0


def build_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    language = None if str(args.language).lower() == "auto" else args.language
    return RuntimeConfig(
        device_index=args.device_index,
        model=args.model,
        output_dir=args.output_dir.resolve(),
        language=language,
        prompt=args.prompt,
        save_audio=bool(args.save_audio),
        audio_dir=args.audio_dir.resolve(),
        audio_format=args.audio_format,
        audio_segment_minutes=float(args.audio_segment_minutes),
        rms_threshold=float(args.rms_threshold),
        silence_seconds=float(args.silence_seconds),
        min_speech_seconds=float(args.min_speech_seconds),
        max_segment_seconds=float(args.max_segment_seconds),
        pre_roll_seconds=float(args.pre_roll_seconds),
    )


def run_start(config: RuntimeConfig) -> int:
    devices = list_input_devices()
    if not devices:
        raise RuntimeError("No input devices found. Check microphone permissions.")

    selected = (
        choose_recommended_device(devices)
        if config.device_index is None
        else next(
            (device for device in devices if device.index == config.device_index), None
        )
    )
    if selected is None:
        raise RuntimeError(f"Device index {config.device_index} not found.")

    print(f"Using audio device {selected.index}: {selected.name}")
    print(f"Using model {config.model}")
    print(f"Writing transcripts to {config.output_dir}")

    started_at = dt.datetime.now()
    audio_writer = None
    if config.save_audio:
        audio_writer = AudioWriter.create(
            audio_dir=config.audio_dir,
            audio_format=config.audio_format,
            segment_minutes=config.audio_segment_minutes,
            started_at=started_at,
        )
        prefix = (
            "Recording segmented audio to"
            if config.audio_segment_minutes > 0
            else "Recording audio to"
        )
        print(f"{prefix} {audio_writer.output_hint}")

    capture_queue: "queue.Queue[bytes | None]" = queue.Queue()
    transcribe_queue: "queue.Queue[Utterance | None]" = queue.Queue()
    stop_event = threading.Event()
    worker = TranscriptionWorker(
        model_name=config.model,
        output_dir=config.output_dir,
        language=config.language,
        system_prompt=config.prompt or None,
        task_queue=transcribe_queue,
        stdout=sys.stdout,
        ready_event=threading.Event(),
    )
    worker.start()
    print(f"Loading model {config.model} ...")
    worker.ready_event.wait()
    if worker.error:
        raise worker.error
    print("Model ready.")

    segmenter = SpeechSegmenter(
        sample_rate=SAMPLE_RATE,
        rms_threshold=config.rms_threshold,
        silence_seconds=config.silence_seconds,
        min_speech_seconds=config.min_speech_seconds,
        max_segment_seconds=config.max_segment_seconds,
        pre_roll_seconds=config.pre_roll_seconds,
    )

    def callback(indata, frames, time_info, status):  # noqa: ANN001
        del frames, time_info
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        if stop_event.is_set():
            return
        capture_queue.put(bytes(indata))

    def process_chunk(chunk: bytes) -> None:
        if audio_writer is not None:
            audio_writer.write(chunk)
        pcm = np.frombuffer(chunk, dtype=np.int16).copy()
        now = dt.datetime.now()
        for utterance in segmenter.push(pcm, now):
            transcribe_queue.put(utterance)

    def request_stop(signum, frame):  # noqa: ANN001
        del frame
        stop_event.set()
        print(f"Stopping on signal {signum}")

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=int(SAMPLE_RATE * 0.03),
            device=selected.index,
            dtype="int16",
            channels=1,
            callback=callback,
        ):
            print("Listening...")
            while True:
                try:
                    chunk = capture_queue.get(timeout=QUEUE_POLL_SECONDS)
                except queue.Empty:
                    if worker.error:
                        raise worker.error
                    if stop_event.is_set():
                        break
                    continue

                if chunk is None:
                    continue

                process_chunk(chunk)

                if worker.error:
                    raise worker.error
                if stop_event.is_set() and capture_queue.empty():
                    break
    finally:
        stop_event.set()
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        while True:
            try:
                chunk = capture_queue.get_nowait()
            except queue.Empty:
                break
            else:
                if chunk is not None:
                    process_chunk(chunk)
        now = dt.datetime.now()
        for utterance in segmenter.flush(now):
            transcribe_queue.put(utterance)
        transcribe_queue.put(None)
        transcribe_queue.join()
        worker.join(timeout=1)
        if audio_writer is not None:
            audio_writer.close(interrupted=stop_event.is_set())
        if worker.error:
            raise worker.error

    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if args.command == "setup":
            return run_setup(args.model)
        if args.command == "devices":
            return run_devices()
        if args.command == "start":
            return run_start(build_runtime_config(args))
        raise RuntimeError(f"Unsupported command: {args.command}")
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
