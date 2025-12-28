<div align="center">
    <img src="/icon.ico"/>
    <h1>ReazonSpeech</h1>
</div>

This repository provides access to the main user tooling of ReazonSpeech
project.

- <https://research.reazon.jp/projects/ReazonSpeech/>

## nemo版部署

### 零、配置环境

如果你的设备未安装python3.10，请照此[教程](https://pvt9.com/_posts/pythoninstall)安装

如果你在国内，请自行搜索“pip 使用国内镜像源”相关教程

[windows安装cuda和cudnn](https://www.cnblogs.com/RiverRiver/p/18103991)


### 一、下载程序

```bash
git clone https://github.com/wjz2001/ReazonSpeech
cd ReazonSpeech
```

### 二、虚拟环境

1.创建虚拟环境
```bash
python -m venv venv
```

2.激活虚拟环境
  - **Windows (CMD/PowerShell)**: `.\venv\Scripts\activate`

  - **macOS / Linux (Bash/Zsh)**: `source venv/bin/activate`

### 三、安装

1.  在根目录下新建 models 文件夹
2.  [下载模型
    model_quantized.onnx](https://huggingface.co/onnx-community/pyannote-segmentation-3.0/tree/main/onnx/)
3.  [下载模型
    reazonspeech-nemo-v2.nemo](https://huggingface.co/reazon-research/reazonspeech-nemo-v2/tree/main/)
4.  把以上两个模型放入 models 文件夹
5.  在**已激活虚拟环境**的终端中，运行`python install_for_nemoasr.py`

### 注意

1. 本模型可仅在 CPU 上运行，如果有 GPU 且支持 cuda 的话会更快
   - 如果要在有 cuda 的 GPU 上运行，建议在运行前检查是否安装了对应 cuda 版本的 torch，一般情况下安装脚本会自动处理好
2. 确保设备上有能全局使用的ffmpeg，否则无法转换音频/视频为可语音识别的文件
   - **Windows**：
    1.  从 [FFmpeg gyan下载](https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full.7z)  解压后得到`ffmpeg.exe`；
    2.  将下载的 **`ffmpeg.exe 和 ffprobe.exe` 文件直接放置在本项目根目录** (与 `asr.py` 文件在同一级)，程序会自动检测并使用它

   - **macOS（使用 Homebrew）**：
    ```bash
    brew install ffmpeg
    ```
   - **Linux（Debian/Ubuntu）**：
    ```bash
    sudo apt update && sudo apt install ffmpeg
    ```

## 用法

```bash
reazonspeech 文件路径 --zcr --auto_zcr --refine-tail -segment2srt
```

### 其他参数

| 参数 | 作用 | 默认值 |
|-----------|:-------------:|:---------:|
| `--debug` | 处理结束后不删除临时文件，并自动打开临时分块目录，显示更多日志 | `无` |
| `--batch-size` | 数字越大批量推理的速度越快，不填则自动根据显存估算（只使用 CPU 则默认为 1） | `1` |
| `--beam` | 设置集束搜索宽度，范围为 4 到 256 之间的整数，更大的值可能更准确但更慢 | `4` |
| `--no-remove-punc` | 禁止自动剔除句末标点，保留原始识别结果 | `无` |

### 音频滤镜参数

| 参数 | 作用 | 默认值 |
|-----------|:-------------:|:---------:|
| `--audio-filter` | 添加ffmpeg音频滤镜参数 | `highpass=f=60,lowpass=f=8000` |
| `--limiter-filter` | 添加ffmpeg音频滤镜参数，并自动在滤镜链末尾附加 alimiter 限制器 | `无` |

- `--audio-filter` 和 `--limiter-filter` 不能共存

- 不写 `--audio-filter` 或不写 `--limiter-filter`
  **不启用任何滤镜**，推荐在录音干净时使用

- 写 `--audio-filter` 但不带参数  
  启用默认滤镜：`highpass=f=60,lowpass=f=8000`

  - highpass=f=60：去掉 60Hz 超低频

  - lowpass=f=8000：压除 8 kHz 以上的高频噪声，同时保留较多辅音高频

- 写 `--audio-filter "[滤镜链参数]"` 或写 `--limiter-filter "[滤镜链参数]"`
  把滤镜链参数原样传给 `ffmpeg -af` 例如：

  `--audio-filter "highpass=f=60,lowpass=f=8000"`

  `--limiter-filter "highpass=f=60,lowpass=f=8000"`

  - `--limiter-filter`会自动在滤镜链末尾附加`alimiter=limit=0.98:level=disabled:attack=5:release=50:latency=1`

  - [ffmpeg音频滤镜列表](https://ffmpeg.org/ffmpeg-filters.html#Audio-Filters)

### VAD参数

| 参数 | 作用 | 默认值 |
|-----------|:-------------:|:---------:|
| `--no-chunk` | 禁止使用VAD | `无` |
| `--vad_threshold` | VAD判断为语音的置信度阈值（0.05-1） | `0.4` |
| `--vad_end_threshold` | VAD判断为语音结束后静音的置信度阈值（0.05-1） | `vad_threshold的值减去0.15` |
| `--min_speech_duration_ms` | 移除短于此时长（毫秒）的语音块 | `100` |
| `--min_silence_duration_ms` | 短于此时长（毫秒）的语音块不被视为间隔 | `200` |
| `--keep_silence` | 在语音块前后扩展的时长（毫秒） | `300` |

### 过零率检测参数（必须先开启VAD）

| 参数 | 作用 | 默认值 |
|-----------|:-------------:|:---------:|
| `--zcr` | 开启过零率检测，防止切断清辅音 | `无` |
| `--zcr_threshold` | 手动设置 ZCR 阈值 | `0.15` |
| `--auto_zcr` | 开启自适应 ZCR 阈值计算，zcr_threshold作为兜底 | `无` |

### 段尾精修参数（必须先开启VAD）

| 参数 | 作用 | 默认值 |
|-----------|:-------------:|:---------:|
| `--refine-tail` | 使用段尾精修 | `无` |
| `--tail_percentile` | 自适应阈值（0-100），值越大越容易将高概率语音区域判为静音 | `20` |
| `--tail_offset` | 在自适应阈值的基础上增加的固定偏移量，值越大越容易将高概率区域语音区域判为静音 | `0.05` |
| `--tail_energy_percentile` | 自适应能量阈值（0-100），通常取 20~40，低于此值则判定为静音 | `30` |
| `--tail_energy_offset` | 在自适应能量阈值基础上增加的固定偏移量，一般为 0，值越大判定标准越宽松 | `0` |
| `--tail_lookahead_ms` | 滞回检查向前看的时长（毫秒），用于确认静音的稳定性，不会马上又回到语音 | `80` |
| `--tail_safety_margin_ms` | 在找到的切点后增加的安全边距（毫秒） | `30` |
| `--tail_min_keep_ms` | 强制保留在段尾的最小时长（毫秒） | `30` |
| `--tail_zcr_high_ratio` | 疑似静音窗口内高于 ZCR 阈值的帧超过此比例时（0.1-0.5），才会判定为清音 | `0.3` |

### 输出参数

| 参数 | 作用 |
|-----------|-------------|
| `无` | 直接把完整的识别文本打印至控制台 |
| `-text` | 仅输出完整的识别文本并保存为 .txt 文件 |
| `-segment` | 输出带时间戳的文本片段（Segment）并保存为 .segments.txt 文件 |
| `-segment2srt` | 输出带时间戳的文本片段 (Segment)并转换为 .srt 字幕文件 |
| `-segment2vtt` | 输出带时间戳的文本片段（Segment）并转换为 .vtt 字幕文件 |
| `-segment2tsv` | 输出带时间戳的文本片段（Segment）并转换为由制表符分隔的 .tsv 文件 |
| `-subword` | 输出带时间戳的所有子词 (Subword) 并保存为 .subwords.txt 文件 |
| `-subword2srt` | 输出带时间戳的所有子词 (Subword) 并转换为 .subwords.srt 字幕文件 |
| `-subword2json` | 输出带时间戳的所有子词（Subword）并转换为 .subwords.json 文件 |
| `-kass` | 生成逐字计时的卡拉OK式 .ass 字幕文件 |

## API 服务部署

### 启动服务

在终端直接运行命令（不带任何参数）：

```bash
reazonspeech
```

启动成功后，服务默认监听端口 **8888**（如被占用会自动寻找空闲端口）
- API 地址：`http://127.0.0.1:8888/v1/audio/transcriptions`

### 接口参数

| 参数名 | 必填 | 说明 |
| :--- | :---: | :--- |
| **file** | 是 | 音频文件或视频文件 |
| **response_format** | 否 | **输出格式（必须二选一）：**<br>1. **OpenAI 标准格式**：`text`（默认），`json`，`srt`，`verbose_json`，`vtt`<br>2. **ReazonSpeech 专用格式**：如 `kass`，`segment2tsv` 等（即上面的输出参数去除开头短横线，多个参数用逗号分隔） |
| **prompt** | 否 | **配置除输出参数和debug参数外所有参数**：<br>在此处传入除输出参数和 debug 参数外所有的配置参数 |
| **timestamp_granularities** | 否 | 仅当 `response_format` 为 `verbose_json` 时有效：<br>可选值：`segment`（段级时间戳），`word`（单词级时间戳） |

- 如果你的应用不支持输入或自定义 prompt 提示词，那么可以在根目录下新建文件 `reazonspeechprompt.txt`，在其中填写 prompt 参数，示例如下：

```
--beam 5 --no-chunk
```

```
{
  "beam": 5,
  "no_chunk": true
}
```

### 调用示例

#### cURL 示例

- json风格参数必须包裹在大括号内，在 curl 中建议用单引号包裹，防止 shell 转义问题

```bash
curl -X POST "http://127.0.0.1:8888/v1/audio/transcriptions" \
  -F "file=@test.wav" \
  -F 'prompt={"no-chunk": true, "beam-size": 5}' \
  -F "response_format=verbose_json" \
  -F "timestamp_granularities=["segment"]"
```

- 命令行风格参数之间用空格分隔，不要加逗号

```bash
curl -X POST "http://127.0.0.1:8888/v1/audio/transcriptions" \
  -F "file=@test.wav" \
  -F "prompt=--audio-filter --beam-size 2" \
  -F "response_format=text,segment2srt"
```

#### Python 示例

- 确保已安装：`pip install requests`

- json风格参数必须包裹在大括号内

```python
import requests
import json

url = "http://127.0.0.1:8888/v1/audio/transcriptions"
file_path = "test.wav"

# 构造 JSON 格式的 prompt 字符串
prompt_config = {
    "no-chunk": True,
    "beam-size": 5
}

payload = {
    # 将字典转为 JSON 字符串
    "prompt": json.dumps(prompt_config),
    "response_format": "verbose_json",
    "timestamp_granularities": ["segment"]  # 可选: segment 或 word
}

with open(file_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, data=payload, files=files)

# 打印结果
print(response.status_code)
print(json.dumps(response.json(), indent=2, ensure_ascii=False))
```

- 命令行风格参数之间用空格分隔，不要加逗号

```python
import requests

url = "http://127.0.0.1:8888/v1/audio/transcriptions"
file_path = "test.wav"

payload = {
    # 直接写命令行参数风格的字符串，空格分隔
    "prompt": "--audio-filter --beam-size 2",
    
    # 自定义多选，逗号分隔，不要带前面的横线
    "response_format": "text,segment2srt" 
}

with open(file_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, data=payload, files=files)

# 打印结果
print(response.status_code)
# 返回的是 asr.py 的原始字典数据，不是 OpenAI 格式
data = response.json()
print("Text 内容:", data.get("text"))
print("SRT 内容:\n", data.get("segment2srt"))
```

## Packages

[reazonspeech.evaluation](pkg/evaluation)

-   Provides a set of tools to evaluate ReazonSpeech models and other
    speech recognition models.

[reazonspeech.nemo.asr](pkg/nemo-asr)

-   Implements a fast, accurate speech recognition based on
    FastConformer-RNNT.
-   The total number of parameters is 619M. Requires [Nvidia
    Nemo](https://github.com/NVIDIA/NeMo).

[reazonspeech.k2.asr](pkg/k2-asr)

-   Next-gen Kaldi model that is very fast and accurate.
-   The total number of parameters is 159M. Requires
    [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).
-   Also contains a bilingual (ja-en) model, which is highly accurate at
    language detection in bilingual settings of Japanese and English.
-   For development: \"ja-en-mls-5k\" model trained on 5k hours of
    ReazonSpeech and MLS English data each

[reazonspeech.espnet.asr](pkg/espnet-asr)

-   Speech recognition with a Conformer-Transducer model.
-   The total number of parameters is 120M. Requires
    [ESPnet](https://github.com/espnet/espnet).

[reazonspeech.espnet.oneseg](pkg/espnet-oneseg)

-   Provides a set of tools to analyze Japanese \"one-segment\" TV
    stream.
-   Use this package to create Japanese audio corpus.

## LICENSE

    Copyright 2022-2025 Reazon Holdings, inc.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.