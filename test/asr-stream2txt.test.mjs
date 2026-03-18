import test from "node:test";
import assert from "node:assert/strict";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";

import {
  TranscriptBuffer,
  chooseRecommendedDevice,
  formatTranscriptLine,
  listAudioDevices,
  parseAvfoundationAudioDevices,
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
  assert.equal(
    formatTranscriptLine({
      startAt: new Date("2026-03-18T09:01:02"),
      endAt: new Date("2026-03-18T09:01:05"),
      text: "测试",
    }),
    "[09:01:02 - 09:01:05] 测试",
  );
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
