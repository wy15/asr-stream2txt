import { spawn } from "node:child_process";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { StringDecoder } from "node:string_decoder";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, "..");

export const PINNED_QWEN_ASR_REF = "b00b789b17051aea61e9717458171100662318a4";
export const DEFAULT_MODEL_DIR = path.join(REPO_ROOT, "models", "qwen3-asr-0.6b");
export const DEFAULT_OUTPUT_DIR = path.join(REPO_ROOT, "data", "transcripts");
export const DEFAULT_VENDOR_DIR = path.join(REPO_ROOT, "vendor", "qwen-asr");
export const DEFAULT_FLUSH_DELAY_MS = 2200;

export function parseAvfoundationAudioDevices(output) {
  const lines = output.split(/\r?\n/);
  const devices = [];
  let inAudioSection = false;

  for (const line of lines) {
    if (line.includes("AVFoundation audio devices:")) {
      inAudioSection = true;
      continue;
    }
    if (line.includes("AVFoundation video devices:")) {
      inAudioSection = false;
      continue;
    }
    if (!inAudioSection) continue;

    const match = line.match(/\[(\d+)\]\s+(.+)$/);
    if (!match) continue;
    devices.push({ index: Number(match[1]), name: match[2].trim() });
  }

  return devices;
}

export function chooseRecommendedDevice(devices) {
  if (!devices.length) return null;

  const score = (device) => {
    const name = device.name.toLowerCase();
    let value = 0;
    if (/(macbook|built-in|internal)/.test(name)) value += 30;
    if (/(microphone|mic|麦克风)/.test(name)) value += 20;
    if (/blackhole|teams|zoom|loopback/.test(name)) value -= 10;
    return value;
  };

  return devices
    .map((device) => ({ device, score: score(device) }))
    .sort((a, b) => b.score - a.score || a.device.index - b.device.index)[0]
    .device;
}

export function normalizeLanguage(language) {
  if (!language || language.toLowerCase() === "auto") return null;
  return language;
}

export function sanitizeSegmentText(text) {
  return text.replace(/\r/g, "").replace(/\n+/g, " ").replace(/\s+/g, " ").trim();
}

export function formatClock(date) {
  return [
    String(date.getHours()).padStart(2, "0"),
    String(date.getMinutes()).padStart(2, "0"),
    String(date.getSeconds()).padStart(2, "0"),
  ].join(":");
}

export function transcriptFileName(date) {
  return [
    date.getFullYear(),
    String(date.getMonth() + 1).padStart(2, "0"),
    String(date.getDate()).padStart(2, "0"),
  ].join("-") + ".txt";
}

export function formatTranscriptLine(segment) {
  return `[${formatClock(segment.startAt)} - ${formatClock(segment.endAt)}] ${segment.text}`;
}

export class TranscriptBuffer {
  constructor({ flushDelayMs = DEFAULT_FLUSH_DELAY_MS, onFlush, now = () => new Date() }) {
    this.flushDelayMs = flushDelayMs;
    this.onFlush = onFlush;
    this.now = now;
    this.currentText = "";
    this.startedAt = null;
    this.timer = null;
  }

  noteChunk(text, at = this.now()) {
    const stripped = text.replace(/[\r\n]/g, "");
    if (!stripped) return;
    if (!this.currentText && !stripped.trim()) return;

    if (!this.currentText) this.startedAt = at;
    this.currentText += stripped;
    this.#scheduleTimer();
  }

  async flush(at = this.now()) {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }

    const text = sanitizeSegmentText(this.currentText);
    if (!text) {
      this.currentText = "";
      this.startedAt = null;
      return null;
    }

    const segment = {
      startAt: this.startedAt ?? at,
      endAt: at,
      text,
    };

    this.currentText = "";
    this.startedAt = null;

    if (this.onFlush) await this.onFlush(segment);
    return segment;
  }

  async close(at = this.now()) {
    return this.flush(at);
  }

  #scheduleTimer() {
    if (this.timer) clearTimeout(this.timer);
    this.timer = setTimeout(() => {
      this.flush().catch(() => {});
    }, this.flushDelayMs);
    this.timer.unref?.();
  }
}

function parseStartOptions(args) {
  const options = {
    deviceIndex: 0,
    modelDir: DEFAULT_MODEL_DIR,
    outputDir: DEFAULT_OUTPUT_DIR,
    language: "Chinese",
    prompt: "",
  };

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    const next = args[index + 1];

    switch (arg) {
      case "--device-index":
        if (next === undefined) throw new Error("--device-index requires a value");
        options.deviceIndex = Number(next);
        if (!Number.isInteger(options.deviceIndex) || options.deviceIndex < 0) {
          throw new Error("--device-index must be a non-negative integer");
        }
        index += 1;
        break;
      case "--model-dir":
        if (next === undefined) throw new Error("--model-dir requires a value");
        options.modelDir = path.resolve(next);
        index += 1;
        break;
      case "--output-dir":
        if (next === undefined) throw new Error("--output-dir requires a value");
        options.outputDir = path.resolve(next);
        index += 1;
        break;
      case "--language":
        if (next === undefined) throw new Error("--language requires a value");
        options.language = next;
        index += 1;
        break;
      case "--prompt":
        if (next === undefined) throw new Error("--prompt requires a value");
        options.prompt = next;
        index += 1;
        break;
      default:
        throw new Error(`Unknown option: ${arg}`);
    }
  }

  return options;
}

function usage() {
  return `asr-stream2txt

Usage:
  asr-stream2txt setup
  asr-stream2txt devices
  asr-stream2txt start [options]

Commands:
  setup                     Clone pinned qwen-asr, build it, download Qwen3-ASR-0.6B
  devices                   List AVFoundation audio input devices
  start                     Start live microphone transcription

start options:
  --device-index <n>        AVFoundation audio device index (default: 0)
  --model-dir <path>        qwen-asr model directory (default: ${DEFAULT_MODEL_DIR})
  --output-dir <path>       Transcript output directory (default: ${DEFAULT_OUTPUT_DIR})
  --language <lang|auto>    Force language, or auto to omit --language (default: Chinese)
  --prompt <text>           Optional qwen-asr prompt

Env overrides:
  ASR_STREAM2TXT_FFMPEG_BIN  Override ffmpeg path
  ASR_STREAM2TXT_QWEN_BIN    Override qwen_asr path
`;
}

function spawnProcess(command, args, options = {}) {
  return spawn(command, args, {
    stdio: ["pipe", "pipe", "pipe"],
    ...options,
  });
}

function assertPipedStreams(child, command) {
  if (!child.stdin || !child.stdout || !child.stderr) {
    throw new Error(`Failed to start ${command}. Check that the binary exists and is executable.`);
  }
}

function collectStream(stream) {
  return new Promise((resolve, reject) => {
    const decoder = new StringDecoder("utf8");
    let text = "";

    stream.on("data", (chunk) => {
      text += decoder.write(chunk);
    });
    stream.on("end", () => {
      text += decoder.end();
      resolve(text);
    });
    stream.on("error", reject);
  });
}

function waitForClose(child) {
  return new Promise((resolve) => {
    child.on("close", (code, signal) => resolve({ code, signal }));
  });
}

export async function listAudioDevices({ ffmpegBin = "ffmpeg" } = {}) {
  const child = spawnProcess(ffmpegBin, [
    "-f",
    "avfoundation",
    "-list_devices",
    "true",
    "-i",
    "",
  ]);
  assertPipedStreams(child, ffmpegBin);

  const stderrPromise = collectStream(child.stderr);
  const stdoutPromise = collectStream(child.stdout);
  const closePromise = waitForClose(child);
  const errorPromise = new Promise((_, reject) => {
    child.on("error", reject);
  });

  const [stderr, stdout] = await Promise.race([
    Promise.all([stderrPromise, stdoutPromise]),
    errorPromise,
  ]);
  await Promise.race([closePromise, errorPromise]);

  const combined = `${stdout}\n${stderr}`;
  const devices = parseAvfoundationAudioDevices(combined);
  const recommended = chooseRecommendedDevice(devices);

  return {
    devices,
    recommended,
    rawOutput: combined,
  };
}

async function ensureFileExists(targetPath, message) {
  try {
    await fs.access(targetPath);
  } catch {
    throw new Error(message);
  }
}

async function writeTranscriptSegment(outputDir, segment, stdout) {
  const filePath = path.join(outputDir, transcriptFileName(segment.startAt));
  const line = `${formatTranscriptLine(segment)}\n`;
  await fs.mkdir(outputDir, { recursive: true });
  await fs.appendFile(filePath, line, "utf8");
  stdout.write(line);
}

function pipeWithPrefix(stream, target, prefix) {
  const decoder = new StringDecoder("utf8");
  let buffered = "";

  stream.on("data", (chunk) => {
    buffered += decoder.write(chunk);
    const lines = buffered.split(/\r?\n/);
    buffered = lines.pop() ?? "";
    for (const line of lines) {
      if (line) target.write(`${prefix}${line}\n`);
    }
  });

  stream.on("end", () => {
    buffered += decoder.end();
    if (buffered) target.write(`${prefix}${buffered}\n`);
  });
}

function buildQwenArgs({ modelDir, language, prompt }) {
  const args = ["-d", modelDir, "--stdin", "--stream"];
  const normalizedLanguage = normalizeLanguage(language);
  if (normalizedLanguage) args.push("--language", normalizedLanguage);
  if (prompt) args.push("--prompt", prompt);
  return args;
}

function buildFfmpegArgs(deviceIndex) {
  return [
    "-nostdin",
    "-hide_banner",
    "-loglevel",
    "error",
    "-f",
    "avfoundation",
    "-i",
    `:${deviceIndex}`,
    "-f",
    "s16le",
    "-ar",
    "16000",
    "-ac",
    "1",
    "pipe:1",
  ];
}

export async function runSetup({ cwd = REPO_ROOT, stdout = process.stdout, stderr = process.stderr } = {}) {
  const scriptPath = path.join(REPO_ROOT, "scripts", "setup-qwen-asr.sh");
  await ensureFileExists(scriptPath, `Missing setup script at ${scriptPath}`);

  await new Promise((resolve, reject) => {
    const child = spawn("bash", [scriptPath], {
      cwd,
      stdio: ["inherit", "inherit", "inherit"],
    });

    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`setup failed with exit code ${code}`));
    });
  });

  stdout.write(`Pinned qwen-asr ref: ${PINNED_QWEN_ASR_REF}\n`);
  stderr.write("");
}

export async function runStart(
  options,
  {
    ffmpegBin = process.env.ASR_STREAM2TXT_FFMPEG_BIN || "ffmpeg",
    qwenBin = process.env.ASR_STREAM2TXT_QWEN_BIN || path.join(DEFAULT_VENDOR_DIR, "qwen_asr"),
    stdout = process.stdout,
    stderr = process.stderr,
    on = process.on.bind(process),
    off = process.off.bind(process),
  } = {},
) {
  await ensureFileExists(qwenBin, `Missing qwen_asr binary at ${qwenBin}. Run 'asr-stream2txt setup' first.`);
  await ensureFileExists(options.modelDir, `Missing model directory at ${options.modelDir}. Run 'asr-stream2txt setup' first.`);

  const deviceInfo = await listAudioDevices({ ffmpegBin });
  if (!deviceInfo.devices.length) {
    throw new Error(
      "No AVFoundation audio input devices found. Grant microphone access to your terminal in System Settings and retry.",
    );
  }
  if (!deviceInfo.devices.some((device) => device.index === options.deviceIndex)) {
    throw new Error(`Audio device index ${options.deviceIndex} not found. Run 'asr-stream2txt devices' to inspect available inputs.`);
  }

  await fs.mkdir(options.outputDir, { recursive: true });

  const ffmpegArgs = buildFfmpegArgs(options.deviceIndex);
  const qwenArgs = buildQwenArgs(options);
  stdout.write(`Using audio device ${options.deviceIndex}\n`);
  stdout.write(`Writing transcripts to ${options.outputDir}\n`);

  const ffmpeg = spawnProcess(ffmpegBin, ffmpegArgs);
  const qwen = spawnProcess(qwenBin, qwenArgs);
  assertPipedStreams(ffmpeg, ffmpegBin);
  assertPipedStreams(qwen, qwenBin);

  ffmpeg.stdout.pipe(qwen.stdin);
  pipeWithPrefix(ffmpeg.stderr, stderr, "[ffmpeg] ");
  pipeWithPrefix(qwen.stderr, stderr, "[qwen_asr] ");

  let ffmpegBytes = 0;
  let shuttingDown = false;
  let failure = null;

  ffmpeg.stdout.on("data", (chunk) => {
    ffmpegBytes += chunk.length;
  });
  qwen.stdin.on("error", (error) => {
    if (error.code !== "EPIPE") failure = error;
  });

  const buffer = new TranscriptBuffer({
    onFlush: async (segment) => {
      await writeTranscriptSegment(options.outputDir, segment, stdout);
    },
  });

  const decoder = new StringDecoder("utf8");
  qwen.stdout.on("data", (chunk) => {
    const text = decoder.write(chunk);
    buffer.noteChunk(text, new Date());
  });

  qwen.stdout.on("end", () => {
    const text = decoder.end();
    if (text) buffer.noteChunk(text, new Date());
  });

  const startupTimer = setTimeout(() => {
    if (ffmpegBytes === 0 && !failure) {
      failure = new Error("No microphone audio received within 5 seconds. Check the input device and microphone permissions.");
      ffmpeg.kill("SIGTERM");
      qwen.kill("SIGTERM");
    }
  }, 5000);
  startupTimer.unref?.();

  const shutdown = (signal) => {
    if (shuttingDown) return;
    shuttingDown = true;
    ffmpeg.kill("SIGTERM");
    qwen.kill("SIGTERM");
    buffer.flush(new Date()).catch(() => {});
    if (signal) stdout.write(`Stopping on ${signal}\n`);
  };

  const onSigint = () => shutdown("SIGINT");
  const onSigterm = () => shutdown("SIGTERM");
  on("SIGINT", onSigint);
  on("SIGTERM", onSigterm);

  ffmpeg.on("error", (error) => {
    failure = error;
    qwen.kill("SIGTERM");
  });
  qwen.on("error", (error) => {
    failure = error;
    ffmpeg.kill("SIGTERM");
  });

  const [ffmpegResult, qwenResult] = await Promise.all([waitForClose(ffmpeg), waitForClose(qwen)]);
  clearTimeout(startupTimer);
  off("SIGINT", onSigint);
  off("SIGTERM", onSigterm);

  await buffer.close(new Date());

  if (failure) throw failure;
  if (!shuttingDown && ffmpegResult.code !== 0) {
    throw new Error(`ffmpeg exited with code ${ffmpegResult.code ?? "unknown"} while capturing audio.`);
  }
  if (!shuttingDown && qwenResult.code !== 0) {
    throw new Error(`qwen_asr exited with code ${qwenResult.code ?? "unknown"} while transcribing audio.`);
  }
}

export async function main(argv, io = {}) {
  const {
    stdout = process.stdout,
    stderr = process.stderr,
    exit = (code) => {
      process.exitCode = code;
    },
  } = io;

  const [command, ...rest] = argv;

  try {
    switch (command) {
      case "setup":
        await runSetup(io);
        break;
      case "devices": {
        const ffmpegBin = io.env?.ASR_STREAM2TXT_FFMPEG_BIN || process.env.ASR_STREAM2TXT_FFMPEG_BIN || "ffmpeg";
        const info = await listAudioDevices({ ffmpegBin });
        if (!info.devices.length) {
          throw new Error("No audio devices found. Grant microphone access to your terminal and retry.");
        }
        for (const device of info.devices) {
          const suffix = info.recommended?.index === device.index ? " (recommended)" : "";
          stdout.write(`[${device.index}] ${device.name}${suffix}\n`);
        }
        break;
      }
      case "start": {
        const options = parseStartOptions(rest);
        await runStart(options, io);
        break;
      }
      case "-h":
      case "--help":
      case undefined:
        stdout.write(`${usage()}\n`);
        break;
      default:
        throw new Error(`Unknown command: ${command}`);
    }
  } catch (error) {
    stderr.write(`${error.message}\n`);
    if (command !== "-h" && command !== "--help" && command !== undefined) {
      stderr.write(`\n${usage()}\n`);
    }
    exit(1);
  }
}
