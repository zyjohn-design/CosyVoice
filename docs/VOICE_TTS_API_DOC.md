# CosyVoice TTS API v1 接口文档

> 版本: v1.0.0  
> 基础 URL: `http://{host}:{port}`  
> 默认端口: 8080  
> 模型: CosyVoice2-0.5B (zero_shot)

---

## 接口总览

| 接口 | 方法 | 说明 |
|------|------|------|
| `/tts` | GET/POST | 统一 TTS 入口 (兼容原 app.py 参数) |
| `/tts/zero_shot` | POST | 上传参考音频实时克隆声音 |
| `/tts/stream` | POST | 流式低延迟合成 |
| `/speakers` | GET | 列出所有说话人 |
| `/speakers/upload` | POST | 上传音频注册新说话人 |
| `/speakers/{name}` | DELETE | 删除说话人 |
| `/clear_wavs` | POST | 清理生成的临时音频 |
| `/health` | GET | 服务健康检查 |

---

## 1. GET/POST `/tts` — 统一 TTS 入口

### 说明

语音合成统一入口。支持 GET query string 和 POST (JSON/form) 两种调用方式。所有推理走 CosyVoice2 `inference_zero_shot`。

### 请求参数

| 参数 | 类型 | 必须 | 默认值 | 说明 |
|------|------|------|--------|------|
| `text` | string | ✅ | - | 要合成的文本 |
| `voice` | string | ❌ | `"2222"` | 说话人名称。匹配 speakers/ 下已注册的说话人, 不存在则回退到第一个已注册说话人。可以传　　pt后缀的名称，后端为了适配自动去掉。去掉后缀的名字，必须与speaker一致 |
| `custom_voice` | int | ❌ | `0` | >0 时覆盖 voice 参数 |
| `speed` | int | ❌ | `5` | 语速 1~9, 5=正常速度。映射关系: speed÷5=CosyVoice倍率 (1→0.2x慢, 5→1.0x正常, 9→1.8x快) |
| `wav` | int | ❌ | `0` | 1=直接返回 WAV 音频文件, 0=返回 JSON |
| `is_stream` | int | ❌ | `0` | 1=流式返回 chunked PCM 音频流 |
| `text_seed` | int | ❌ | `42` | 随机种子, 固定韵律和语调。相同 seed + 相同文本 = 相同输出 |

**兼容参数** (保留接口兼容, 当前模型未直接使用):

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt` | string | `""` | 文本润色提示词 |
| `temperature` | float | `0.3` | 采样温度 |
| `top_p` | float | `0.7` | Top-P 采样 |
| `top_k` | int | `20` | Top-K 采样 |
| `skip_refine` | int | `0` | 跳过文本润色 |
| `refine_max_new_token` | int | `384` | 润色最大 token |
| `infer_max_new_token` | int | `2048` | 推理最大 token |
| `uv_break` | string | `""` | 分段连接符 |

### 返回

#### `wav=0` (默认) — JSON

```json
{
    "code": 0,
    "msg": "ok",
    "audio_files": [
        {
            "filename": "/workspace/CosyVoice/static/wavs/114530_xxx.wav",
            "url": "http://localhost:8080/static/wavs/114530_xxx.wav",
            "relative_url": "/static/wavs/114530_xxx.wav",
            "inference_time": 2.38,
            "audio_duration": 1.08
        }
    ],
    "filename": "/workspace/CosyVoice/static/wavs/114530_xxx.wav",
    "url": "http://localhost:8080/static/wavs/114530_xxx.wav",
    "relative_url": "/static/wavs/114530_xxx.wav"
}
```

| 字段 | 说明 |
|------|------|
| `code` | 0=成功, 1=失败 |
| `msg` | 状态信息 |
| `audio_files[0].filename` | 服务器上的绝对路径 |
| `audio_files[0].url` | 可直接访问的音频 URL |
| `audio_files[0].inference_time` | 推理耗时 (秒) |
| `audio_files[0].audio_duration` | 音频时长 (秒) |
| `filename` / `url` / `relative_url` | 顶层字段, 兼容 pyVideoTrans |

#### `wav=1` — WAV 音频文件

直接返回 `audio/x-wav` 格式的音频文件流。

#### `is_stream=1` — 流式 PCM

返回 `audio/pcm` 格式的 chunked transfer encoding 流。

| Header | 值 | 说明 |
|--------|----|------|
| `X-Sample-Rate` | `24000` | 采样率 |
| `X-Channels` | `1` | 单声道 |
| `X-Bit-Depth` | `16` | 16位 PCM |

#### 错误返回

```json
{
    "code": 1,
    "msg": "text params lost"
}
```

### 调用示例

```bash
# GET — 最简调用, 返回 JSON
curl "http://localhost:8080/tts?text=你好世界&voice=1111-男声"

# GET — 直接返回 WAV
curl "http://localhost:8080/tts?text=你好世界&voice=1111-男声&wav=1" -o output.wav

# GET — 流式返回 PCM
curl "http://localhost:8080/tts?text=你好世界&voice=1111-男声&is_stream=1" -o stream.pcm
ffmpeg -f s16le -ar 24000 -ac 1 -i stream.pcm stream.wav

# POST JSON
curl -X POST http://localhost:8080/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"你好世界","voice":"1111-男声","wav":1}' \
  -o output.wav

# POST form
curl -X POST http://localhost:8080/tts \
  -d "text=你好世界&voice=1111-男声&speed=7&wav=1" \
  -o output.wav
```

**Python 调用示例:**

```python
import requests

# 非流式
resp = requests.post("http://localhost:8080/tts", data={
    "text": "你好世界",
    "voice": "1111-男声",
    "wav": 1,
})
with open("output.wav", "wb") as f:
    f.write(resp.content)

# 流式播放
resp = requests.post("http://localhost:8080/tts", data={
    "text": "你好世界",
    "voice": "1111-男声",
    "is_stream": 1,
}, stream=True)

import pyaudio
sr = int(resp.headers.get("X-Sample-Rate", 24000))
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=sr, output=True)
for chunk in resp.iter_content(chunk_size=4096):
    stream.write(chunk)
stream.close()
```

---

## 2. POST `/tts/zero_shot` — 实时声音克隆

### 说明

上传参考音频实时克隆声音。不需要预注册说话人。可选保存为持久化说话人。

### 请求参数 (multipart/form-data)

| 参数 | 类型 | 必须 | 默认值 | 说明 |
|------|------|------|--------|------|
| `text` | string | ✅ | - | 要合成的文本 |
| `prompt_text` | string | ✅ | - | 参考音频中说的话 (**必须准确**, 影响克隆质量) |
| `prompt_audio` | file | ✅ | - | 参考音频文件 (wav, >=16kHz, 3~10秒) |
| `speed` | float | ❌ | `1.0` | 语速倍率 |
| `stream` | bool | ❌ | `false` | 是否流式 |
| `seed` | int | ❌ | `0` | 随机种子, 0=随机 |
| `save_speaker` | string | ❌ | `null` | 保存为说话人的名称, 为空则不保存 |

### 返回

`audio/wav` 格式音频流。

| Response Header | 说明 |
|-----------------|------|
| `X-Inference-Time` | 推理耗时 (秒) |
| `X-Audio-Duration` | 音频时长 (秒) |
| `X-Saved-Speaker` | 保存的说话人文件名 (仅 save_speaker 非空时) |

### 调用示例

```bash
# 克隆声音
curl -X POST http://localhost:8080/tts/zero_shot \
  -F "text=今天天气真好" \
  -F "prompt_text=希望你以后能够做的比我还好呦" \
  -F "prompt_audio=@reference.wav" \
  -o cloned.wav

# 克隆并保存说话人 (后续可用 /tts?voice=我的声音)
curl -X POST http://localhost:8080/tts/zero_shot \
  -F "text=今天天气真好" \
  -F "prompt_text=希望你以后能够做的比我还好呦" \
  -F "prompt_audio=@reference.wav" \
  -F "save_speaker=我的声音" \
  -o cloned.wav
```

---

## 3. POST `/tts/stream` — 流式合成

### 说明

流式低延迟合成, 边生成边返回 raw PCM 音频块。支持已注册说话人和实时上传克隆两种模式。

### 请求参数 (multipart/form-data)

| 参数 | 类型 | 必须 | 默认值 | 说明 |
|------|------|------|--------|------|
| `text` | string | ✅ | - | 要合成的文本 |
| `voice` | string | ❌ | `""` | 已注册说话人名称 |
| `prompt_text` | string | ❌ | `null` | (实时克隆) 参考音频文本 |
| `prompt_audio` | file | ❌ | `null` | (实时克隆) 参考音频文件 |
| `speed` | float | ❌ | `1.0` | 语速倍率 |
| `seed` | int | ❌ | `0` | 随机种子 |

### 返回

`audio/pcm` — chunked transfer encoding, raw PCM 流 (16bit, mono, 24000Hz)。

| Header | 值 |
|--------|-----|
| `X-Sample-Rate` | `24000` |
| `X-Channels` | `1` |
| `X-Bit-Depth` | `16` |

### 调用示例

```bash
# 已注册说话人
curl -X POST http://localhost:8080/tts/stream \
  -F "text=这是一段比较长的文本用来测试流式合成" \
  -F "voice=1111-男声" \
  -o stream.pcm

# PCM 转 WAV
ffmpeg -f s16le -ar 24000 -ac 1 -i stream.pcm output.wav

# 实时克隆 + 流式
curl -X POST http://localhost:8080/tts/stream \
  -F "text=这是一段比较长的文本" \
  -F "prompt_text=参考音频的文字内容" \
  -F "prompt_audio=@reference.wav" \
  -o stream.pcm
```

---

## 4. GET `/speakers` — 列出说话人

### 返回

```json
{
    "code": 0,
    "msg": "ok",
    "speakers": ["1111-男声.wav", "10003-女声.wav"],
    "registered": ["1111-男声", "10003-女声"]
}
```

| 字段 | 说明 |
|------|------|
| `speakers` | speakers/ 目录下的音频文件列表 |
| `registered` | 已成功注册到模型的说话人 ID 列表 |

### 调用示例

```bash
curl http://localhost:8080/speakers
```

---

## 5. POST `/speakers/upload` — 注册说话人

### 请求参数 (multipart/form-data)

| 参数 | 类型 | 必须 | 默认值 | 说明 |
|------|------|------|--------|------|
| `name` | string | ✅ | - | 说话人名称 (将作为 spk_id) |
| `audio` | file | ✅ | - | 参考音频 (wav, >=16kHz, 3~10秒) |
| `prompt_text` | string | ❌ | `""` | 音频中说的话 (**强烈建议提供**) |

### 返回

```json
{"code": 0, "msg": "ok", "speaker": "张三.wav"}
```

### 调用示例

```bash
curl -X POST http://localhost:8080/speakers/upload \
  -F "name=张三" \
  -F "audio=@reference.wav" \
  -F "prompt_text=希望你以后能够做的比我还好呦"
```

---

## 6. DELETE `/speakers/{name}` — 删除说话人

### 路径参数

| 参数 | 说明 |
|------|------|
| `name` | 说话人名称, 如 `1111-男声` |

### 返回

```json
{"code": 0, "msg": "ok", "deleted": ["1111-男声.wav", "1111-男声.txt"]}
```

### 调用示例

```bash
curl -X DELETE http://localhost:8080/speakers/1111-男声
```

> **注意**: 删除文件后模型内存中的 embedding 无法热移除, 需重启服务才能完全清理。

---

## 7. POST `/clear_wavs` — 清理临时音频

### 说明

清理 `static/wavs/` 下的所有生成文件, 释放磁盘空间。不影响已注册的说话人。

### 返回

```json
{"code": 0, "msg": "Cleared 42 files"}
```

### 调用示例

```bash
curl -X POST http://localhost:8080/clear_wavs
```

---

## 8. GET `/health` — 健康检查

### 返回

```json
{
    "code": 0,
    "msg": "ok",
    "status": "healthy",
    "model_dir": "pretrained_models/CosyVoice2-0.5B",
    "sample_rate": 24000,
    "max_concurrency": 4,
    "registered_speakers": ["1111-男声", "10003-女声"],
    "gpu": {
        "gpu_name": "NVIDIA L20",
        "gpu_memory_total_mb": 49152,
        "gpu_memory_used_mb": 3200
    }
}
```

> `gpu` 字段仅在 CUDA 可用时返回。macOS / CPU 环境下为空对象 `{}`。

### 调用示例

```bash
curl http://localhost:8080/health
```

---

## 错误码

| code | 说明 |
|------|------|
| `0` | 成功 |
| `1` | 失败 (具体原因见 msg 字段) |

常见错误:

| msg | 原因 |
|-----|------|
| `text params lost` | 缺少 text 参数 |
| `没有已注册的说话人` | speakers/ 目录为空 |
| `说话人 'xxx' 不存在` | DELETE 时找不到对应文件 |
| `推理未产生音频` | 模型推理异常 |
