# asr-stream2txt

这个分支是 `mlx-audio` 版本，核心引擎改成了 `mlx-community/Qwen3-ASR-0.6B-4bit`，面向 macOS 本机实时麦克风转写。

能力范围：

- 实时监听麦克风
- 中文优先转写，默认 `--language Chinese`
- 文本按天追加保存到本地
- 可选同时保存压缩录音
- 可选按时长自动切片录音

## Requirements

- macOS Apple Silicon
- Python 3.12+
- `uv`
- `ffmpeg`

## Setup

先创建并同步虚拟环境：

```bash
uv venv .venv
uv sync
```

首次下载模型缓存：

```bash
./bin/asr-stream2txt setup
```

默认模型：

- `mlx-community/Qwen3-ASR-0.6B-4bit`

## Commands

列出输入设备：

```bash
./bin/asr-stream2txt devices
```

当前这台机器上，Python 版枚举到的输入设备索引和 `ffmpeg` 版不同，实际以 `devices` 输出为准。

开始实时转写：

```bash
./bin/asr-stream2txt start
```

显式指定设备：

```bash
./bin/asr-stream2txt start --device-index 2
```

同时保存压缩录音：

```bash
./bin/asr-stream2txt start --save-audio
```

每 30 分钟切一个录音文件：

```bash
./bin/asr-stream2txt start \
  --save-audio \
  --audio-format opus \
  --audio-segment-minutes 30
```

如果你要关闭语言强制，让模型自己判断：

```bash
./bin/asr-stream2txt start --language auto
```

如果你要给模型一些术语提示：

```bash
./bin/asr-stream2txt start \
  --prompt "保留术语拼写：CUDA, PostgreSQL, Redis"
```

## Output

文本输出默认在：

```text
data/transcripts/YYYY-MM-DD.txt
```

每行格式：

```text
[09:15:02 - 09:15:07] 你好，今天我们继续看这个需求。
```

录音输出默认在：

```text
data/audio/YYYY-MM-DD/HHmmss.opus
```

如果启用自动切片：

```text
data/audio/YYYY-MM-DD/HHmmss-000.opus
data/audio/YYYY-MM-DD/HHmmss-001.opus
...
```

## Notes

- 这个分支不再依赖 `antirez/qwen-asr` 做推理核心。
- 录音压缩仍然通过 `ffmpeg` 完成，因为它在 `Opus/FLAC` 输出和切片上更稳。
- 转写分段目前是“本地实时分段 + utterance 级转写”，不是 token 级终端流式打印。
- 输入设备索引来自 `sounddevice`，不要和旧版 `ffmpeg avfoundation` 索引混用。

## Development

跑 Python 测试：

```bash
./.venv/bin/python -m unittest discover -s tests -v
```
