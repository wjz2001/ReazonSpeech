<div align="center">
    <img src="/icon.ico"/>
    <h1>ReazonSpeech</h1>
</div>
<br/>

This repository provides access to the main user tooling of ReazonSpeech
project.

- <https://research.reazon.jp/projects/ReazonSpeech/>

## nemo版部署

```bash
git clone https://github.com/wjz2001/ReazonSpeech
cd ReazonSpeech
python -m venv venv
.\venv\Scripts\activate # Linux或MacOS：source venv/bin/activate
python install_for_nemoasr.py
```

1.  在根目录下新建 models 文件夹
2.  [下载模型
    model_quantized.onnx](https://huggingface.co/onnx-community/pyannote-segmentation-3.0/tree/main/onnx/)
3.  [下载模型
    reazonspeech-nemo-v2.nemo](https://huggingface.co/reazon-research/reazonspeech-nemo-v2/tree/main/)
4.  把以上两个模型放入 models 文件夹

### 注意

1. 本模型可仅在 CPU 上运行，如果有 GPU 且支持 cuda 的话会更快
   - 如果要在有 cuda 的 GPU 上运行，建议在运行前检查是否安装了对应 cuda 版本的 torch，一般情况下安装脚本会自动处理好
2. 确保设备上有能全局使用的ffmpeg，否则无法转换音频/视频为可语音识别的文件

## 用法

```bash
python asr.py 文件路径
```

### VAD参数

| 参数 | 作用 | 默认值 |
|-----------|:-------------:|:---------:|
| `--no-chunk` | 禁止使用VAD | `无` |
| `--vad_threshold` | VAD判断为语音的置信度阈值（0-1） | `0.2` |
| `--min_speech_duration_ms` | 移除短于此时长（毫秒）的语音块 | `100` |
| `--keep_silence` | 在语音块前后扩展的时长（毫秒） | `30` |

### 输出参数

| 参数 | 作用 |
|-----------|:-------------:|
| `无` | 直接把完整的识别文本打印至控制台 |
| `-text` | 仅输出完整的识别文本并保存为 .txt 文件 |
| `-segment` | 输出带时间戳的文本片段 (Segment)并保存为 .segments
| `-segment2srt` | 输出带时间戳的文本片段 (Segment)并转换为 .srt 字幕文件 |
| `-subword` | 输出带时间戳的所有子词 (Subword) 并保存为 .subwords.txt 文件 |
| `-subword2srt` | 输出带时间戳的所有子词 (Subword) 并转换为 .subwords.srt 字幕
| `-kass` | 生成逐字计时的卡拉OK式 .ass 字幕文件 |

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
