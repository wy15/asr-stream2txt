import test from "node:test";
import assert from "node:assert/strict";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";

import {
  TranscriptBuffer,
  audioDateDirName,
  audioTimeFileStem,
  buildFfmpegArgs,
  chooseRecommendedDevice,
  formatTranscriptLine,
  listAudioDevices,
  parseAvfoundationAudioDevices,
  resolveAudioOutputPaths,
  resolveAudioExtension,
  sanitizeSegmentText,
  transcriptFileName,
} from "../src/asr-stream2txt.mjs";

test("parseAvfoundationAudioDevices extracts audio devices", () => {
  const sample = `
[AVFoundation indev @ 0x1] AVFoundation video devices:
[AVFoundation indev @ 0x1] [0] Capture screen 0
[AVFoundation indev @ 0x1] AVFoundation audio devices:
[AVFoundation indev @ 0x1] [0] MacBook Air Microphone
[AVFoundation indev @ 0x1] [1] BlackHole 2ch
`;

  const devices = parseAvfoundationAudioDevices(sample);
  assert.deepEqual(devices, [
    { index: 0, name: "MacBook Air Microphone" },
    { index: 1, name: "BlackHole 2ch" },
  ]);
});

test("TranscriptBuffer flushes text after inactivity", async () => {
  const flushed = [];
  const buffer = new TranscriptBuffer({
    flushDelayMs: 20,
    onFlush: async (segment) => flushed.push(segment),
  });

  buffer.noteChunk("你好");
  await new Promise((resolve) => setTimeout(resolve, 40));

  assert.equal(flushed.length, 1);
  assert.equal(flushed[0].text, "你好");
});

test("TranscriptBuffer flushes remaining text on close", async () => {
  const flushed = [];
  const now = new Date("2026-03-18T09:00:00");
  const buffer = new TranscriptBuffer({
    onFlush: async (segment) => flushed.push(segment),
    now: () => now,
  });

  buffer.noteChunk("  你好\n世界  ", now);
  await buffer.close(now);

  assert.equal(flushed.length, 1);
  assert.equal(flushed[0].text, "你好世界");
});

test("chooseRecommendedDevice avoids virtual audio devices", () => {
  const recommended = chooseRecommendedDevice([
    { index: 0, name: "Microsoft Teams Audio" },
    { index: 1, name: "qiAirPods3" },
  ]);

  assert.deepEqual(recommended, { index: 1, name: "qiAirPods3" });
});

test("sanitizeSegmentText and transcriptFileName normalize output", () => {
  assert.equal(sanitizeSegmentText("  你好 \n 世界  "), "你好 世界");
  assert.equal(transcriptFileName(new Date("2026-03-18T00:00:00")), "2026-03-18.txt");
  assert.equal(audioDateDirName(new Date("2026-03-18T00:00:00")), "2026-03-18");
  assert.equal(audioTimeFileStem(new Date("2026-03-18T09:01:05")), "090105");
  assert.equal(resolveAudioExtension("opus"), "opus");
  assert.equal(resolveAudioExtension("flac"), "flac");
  assert.equal(
    formatTranscriptLine({
      startAt: new Date("2026-03-18T09:01:02"),
      endAt: new Date("2026-03-18T09:01:05"),
      text: "测试",
    }),
    "[09:01:02 - 09:01:05] 测试",
  );
});

test("resolveAudioOutputPaths builds segmented audio pattern", () => {
  const resolved = resolveAudioOutputPaths({
    audioDir: "/tmp/audio",
    audioFormat: "opus",
    startedAt: new Date("2026-03-18T09:01:05"),
    segmentMinutes: 30,
  });

  assert.equal(resolved.audioDir, "/tmp/audio/2026-03-18");
  assert.equal(resolved.audioPath, "/tmp/audio/2026-03-18/090105-%03d.opus");
  assert.equal(resolved.segmented, true);
});

test("buildFfmpegArgs adds compressed audio output when requested", () => {
  const args = buildFfmpegArgs(1, {
    audioFormat: "opus",
    audioPath: "/tmp/test.opus",
  });

  assert.equal(args[args.indexOf("-i") + 1], ":1");
  assert.equal(args[args.indexOf("-map") + 1], "0:a:0");
  assert.equal(args[args.indexOf("-c:a") + 1], "libopus");
  assert.equal(args[args.indexOf("-b:a") + 1], "24k");
  assert.equal(args[args.indexOf("-compression_level") + 1], "10");
  assert.equal(args.at(-1), "/tmp/test.opus");
});

test("buildFfmpegArgs supports segmented opus recording", () => {
  const args = buildFfmpegArgs(1, {
    audioFormat: "opus",
    audioPath: "/tmp/test-%03d.opus",
    audioSegmentMinutes: 30,
  });

  assert.ok(args.includes("segment"));
  assert.equal(args[args.indexOf("-segment_time") + 1], "1800");
  assert.equal(args[args.indexOf("-segment_format") + 1], "opus");
  assert.equal(args.at(-1), "/tmp/test-%03d.opus");
});

test("listAudioDevices parses ffmpeg stderr output", async () => {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "asr-devices-"));
  const fakeFfmpeg = path.join(tempDir, "ffmpeg");
  await fs.writeFile(
    fakeFfmpeg,
    `#!/usr/bin/env bash
if [[ "$*" == *"-list_devices true"* ]]; then
  cat >&2 <<'EOF'
[AVFoundation indev @ 0x1] AVFoundation audio devices:
[AVFoundation indev @ 0x1] [0] MacBook Air Microphone
EOF
  exit 1
fi
`,
    { mode: 0o755 },
  );

  const info = await listAudioDevices({ ffmpegBin: fakeFfmpeg });
  assert.equal(info.devices.length, 1);
  assert.equal(info.recommended?.index, 0);
});

test("CLI start writes transcript file from fake ffmpeg and qwen_asr", async () => {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "asr-start-"));
  const fakeFfmpeg = path.join(tempDir, "ffmpeg");
  const fakeQwen = path.join(tempDir, "qwen_asr");
  const modelDir = path.join(tempDir, "model");
  const outputDir = path.join(tempDir, "out");
  const cliPath = path.resolve("bin/asr-stream2txt.mjs");

  await fs.mkdir(modelDir, { recursive: true });

  await fs.writeFile(
    fakeFfmpeg,
    `#!/usr/bin/env bash
if [[ "$*" == *"-list_devices true"* ]]; then
  cat >&2 <<'EOF'
[AVFoundation indev @ 0x1] AVFoundation audio devices:
[AVFoundation indev @ 0x1] [0] MacBook Air Microphone
EOF
  exit 1
fi
printf '\\x01\\x00%.0s' {1..8000}
`,
    { mode: 0o755 },
  );

  await fs.writeFile(
    fakeQwen,
    `#!/usr/bin/env bash
cat >/dev/null
printf '你好'
sleep 0.1
printf '世界'
`,
    { mode: 0o755 },
  );

  const child = spawn(process.execPath, [cliPath, "start", "--model-dir", modelDir, "--output-dir", outputDir], {
    cwd: path.resolve("."),
    env: {
      ...process.env,
      ASR_STREAM2TXT_FFMPEG_BIN: fakeFfmpeg,
      ASR_STREAM2TXT_QWEN_BIN: fakeQwen,
    },
    stdio: ["ignore", "pipe", "pipe"],
  });

  let stdout = "";
  let stderr = "";
  child.stdout.on("data", (chunk) => {
    stdout += chunk.toString("utf8");
  });
  child.stderr.on("data", (chunk) => {
    stderr += chunk.toString("utf8");
  });

  const exitCode = await new Promise((resolve, reject) => {
    child.on("error", reject);
    child.on("close", resolve);
  });

  assert.equal(exitCode, 0, stderr);
  assert.match(stdout, /Using audio device 0: MacBook Air Microphone/);
  assert.match(stdout, /你好世界/);

  const files = await fs.readdir(outputDir);
  assert.equal(files.length, 1);
  const transcript = await fs.readFile(path.join(outputDir, files[0]), "utf8");
  assert.match(transcript, /\[\d{2}:\d{2}:\d{2} - \d{2}:\d{2}:\d{2}\] 你好世界/);
});

test("CLI start can save opus audio while transcribing", async () => {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "asr-audio-"));
  const fakeFfmpeg = path.join(tempDir, "ffmpeg");
  const fakeQwen = path.join(tempDir, "qwen_asr");
  const modelDir = path.join(tempDir, "model");
  const outputDir = path.join(tempDir, "out");
  const audioDir = path.join(tempDir, "audio");
  const cliPath = path.resolve("bin/asr-stream2txt.mjs");

  await fs.mkdir(modelDir, { recursive: true });

  await fs.writeFile(
    fakeFfmpeg,
    `#!/usr/bin/env bash
if [[ "$*" == *"-list_devices true"* ]]; then
  cat >&2 <<'EOF'
[AVFoundation indev @ 0x1] AVFoundation audio devices:
[AVFoundation indev @ 0x1] [0] MacBook Air Microphone
EOF
  exit 1
fi
last_arg="\${!#}"
if [[ "$last_arg" == *.opus ]]; then
  mkdir -p "$(dirname "$last_arg")"
  printf 'fake opus bytes' > "$last_arg"
fi
printf '\\x01\\x00%.0s' {1..4000}
`,
    { mode: 0o755 },
  );

  await fs.writeFile(
    fakeQwen,
    `#!/usr/bin/env bash
cat >/dev/null
printf '录音测试'
`,
    { mode: 0o755 },
  );

  const child = spawn(
    process.execPath,
    [
      cliPath,
      "start",
      "--model-dir",
      modelDir,
      "--output-dir",
      outputDir,
      "--save-audio",
      "--audio-dir",
      audioDir,
      "--audio-format",
      "opus",
    ],
    {
      cwd: path.resolve("."),
      env: {
        ...process.env,
        ASR_STREAM2TXT_FFMPEG_BIN: fakeFfmpeg,
        ASR_STREAM2TXT_QWEN_BIN: fakeQwen,
      },
      stdio: ["ignore", "pipe", "pipe"],
    },
  );

  let stdout = "";
  let stderr = "";
  child.stdout.on("data", (chunk) => {
    stdout += chunk.toString("utf8");
  });
  child.stderr.on("data", (chunk) => {
    stderr += chunk.toString("utf8");
  });

  const exitCode = await new Promise((resolve, reject) => {
    child.on("error", reject);
    child.on("close", resolve);
  });

  assert.equal(exitCode, 0, stderr);
  assert.match(stdout, /Recording audio to .+\.opus/);

  const dateDirs = await fs.readdir(audioDir);
  assert.equal(dateDirs.length, 1);
  const audioFiles = await fs.readdir(path.join(audioDir, dateDirs[0]));
  assert.equal(audioFiles.length, 1);
  assert.match(audioFiles[0], /^\d{6}\.opus$/);
  const audioBytes = await fs.readFile(path.join(audioDir, dateDirs[0], audioFiles[0]), "utf8");
  assert.equal(audioBytes, "fake opus bytes");
});

test("CLI start can segment saved audio by duration", async () => {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "asr-segmented-audio-"));
  const fakeFfmpeg = path.join(tempDir, "ffmpeg");
  const fakeQwen = path.join(tempDir, "qwen_asr");
  const modelDir = path.join(tempDir, "model");
  const outputDir = path.join(tempDir, "out");
  const audioDir = path.join(tempDir, "audio");
  const cliPath = path.resolve("bin/asr-stream2txt.mjs");

  await fs.mkdir(modelDir, { recursive: true });

  await fs.writeFile(
    fakeFfmpeg,
    `#!/usr/bin/env bash
if [[ "$*" == *"-list_devices true"* ]]; then
  cat >&2 <<'EOF'
[AVFoundation indev @ 0x1] AVFoundation audio devices:
[AVFoundation indev @ 0x1] [0] MacBook Air Microphone
EOF
  exit 1
fi
last_arg="\${!#}"
if [[ "$last_arg" == *"%03d.opus" ]]; then
  output_file="\${last_arg//%03d/000}"
  mkdir -p "$(dirname "$output_file")"
  printf 'fake segmented opus bytes' > "$output_file"
fi
printf '\\x01\\x00%.0s' {1..4000}
`,
    { mode: 0o755 },
  );

  await fs.writeFile(
    fakeQwen,
    `#!/usr/bin/env bash
cat >/dev/null
printf '分片测试'
`,
    { mode: 0o755 },
  );

  const child = spawn(
    process.execPath,
    [
      cliPath,
      "start",
      "--model-dir",
      modelDir,
      "--output-dir",
      outputDir,
      "--save-audio",
      "--audio-dir",
      audioDir,
      "--audio-format",
      "opus",
      "--audio-segment-minutes",
      "30",
    ],
    {
      cwd: path.resolve("."),
      env: {
        ...process.env,
        ASR_STREAM2TXT_FFMPEG_BIN: fakeFfmpeg,
        ASR_STREAM2TXT_QWEN_BIN: fakeQwen,
      },
      stdio: ["ignore", "pipe", "pipe"],
    },
  );

  let stdout = "";
  let stderr = "";
  child.stdout.on("data", (chunk) => {
    stdout += chunk.toString("utf8");
  });
  child.stderr.on("data", (chunk) => {
    stderr += chunk.toString("utf8");
  });

  const exitCode = await new Promise((resolve, reject) => {
    child.on("error", reject);
    child.on("close", resolve);
  });

  assert.equal(exitCode, 0, stderr);
  assert.match(stdout, /Recording segmented audio to .+-%03d\.opus/);

  const dateDirs = await fs.readdir(audioDir);
  assert.equal(dateDirs.length, 1);
  const audioFiles = await fs.readdir(path.join(audioDir, dateDirs[0]));
  assert.equal(audioFiles.length, 1);
  assert.match(audioFiles[0], /^\d{6}-000\.opus$/);
  const audioBytes = await fs.readFile(path.join(audioDir, dateDirs[0], audioFiles[0]), "utf8");
  assert.equal(audioBytes, "fake segmented opus bytes");
});
