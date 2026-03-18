# asr-stream2txt

本地 macOS CLI 工具，基于 [`antirez/qwen-asr`](https://github.com/antirez/qwen-asr) 做中文优先的实时麦克风转写，并把文本持续保存到本地文件。

## Requirements

- macOS arm64
- `ffmpeg`
- `git`
- `make`
- Node.js 20+

## Quick Start

```bash
npm install
./bin/asr-stream2txt.mjs setup
./bin/asr-stream2txt.mjs devices
./bin/asr-stream2txt.mjs start
```

首次运行前，给你的终端应用打开麦克风权限：

- System Settings
- Privacy & Security
- Microphone

## Commands

### `setup`

拉取固定版本的 `qwen-asr`，编译 `qwen_asr`，并下载 `Qwen3-ASR-0.6B` 到本地：

```bash
./bin/asr-stream2txt.mjs setup
```

默认目录：

- `vendor/qwen-asr`
- `models/qwen3-asr-0.6b`

### `devices`

列出 `AVFoundation` 音频输入设备：

```bash
./bin/asr-stream2txt.mjs devices
```

输出会标记一个推荐设备。`start` 默认会自动选择推荐设备，也可以显式传 `--device-index` 覆盖。

### `start`

开始实时转写并把结果写到 `data/transcripts/YYYY-MM-DD.txt`：

```bash
./bin/asr-stream2txt.mjs start
```

常用参数：

```bash
./bin/asr-stream2txt.mjs start \
  --device-index 0 \
  --model-dir ./models/qwen3-asr-0.6b \
  --output-dir ./data/transcripts \
  --language Chinese
```

如果想让模型自动识别语言：

```bash
./bin/asr-stream2txt.mjs start --language auto
```

如果你要给模型轻微纠偏术语：

```bash
./bin/asr-stream2txt.mjs start \
  --prompt "Preserve spelling: CUDA, PostgreSQL, Redis"
```

## Output Format

日志按天落盘，每行一段：

```text
[09:15:02 - 09:15:07] 你好，今天我们继续看这个需求。
```

分段规则：

- 收到首个稳定文本时开始计时
- 连续 2200ms 没有新文本时 flush
- 退出时强制 flush 最后一段

## Troubleshooting

`devices` 没有看到内置麦克风：

- 确认终端已获得麦克风权限
- 重新打开终端后再试
- 先执行 `ffmpeg -f avfoundation -list_devices true -i ""`

`start` 提示 5 秒内没有音频：

- 检查 `--device-index`
- 确认麦克风没有被别的应用独占
- 先用 `devices` 看当前 index

## Development

```bash
npm test
```
