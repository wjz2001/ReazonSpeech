============
ReazonSpeech
============

This repository provides access to the main user tooling of ReazonSpeech project.

* https://research.reazon.jp/projects/ReazonSpeech/

Install
=======

.. code:: console

   git clone https://github.com/wjz2001/ReazonSpeech
   cd ReazonSpeech
   python -m venv venv
   .\venv\Scripts\activate # Linux或MacOS：source venv/bin/activate
   python install_for_nemoasr.py

#. 在根目录下新建 models 文件夹
#. `下载模型 model_quantized.onnx <https://huggingface.co/onnx-community/pyannote-segmentation-3.0/tree/main/onnx/>`_
#. `下载模型 reazonspeech-nemo-v2.nemo <https://huggingface.co/reazon-research/reazonspeech-nemo-v2/tree/main/>`_
#. 把以上两个模型放入 models 文件夹

.. note::
   #. 注意torch版本是否适合已安装的cuda版本，一般情况下不会有问题
   #. 确保设备上有可用的ffmpeg


用法
====
.. code:: console

   python asr.py 文件路径

VAD参数
====
.. code:: console

   --no-chunk：禁用分块

   --vad_threshold：VAD判断为语音的置信度阈值 (0-1)
   --min_speech_duration_s：移除短于此时长(秒)的语音块
   --keep_silence：在语音块前后扩展的时长（毫秒）

输出参数
====
.. code:: console

   不填写任何参数：直接把完整的识别文本打印至控制台

   -text：仅输出完整的识别文本并保存为 .txt 文件

   -segment：输出带时间戳的文本片段 (Segment)并保存为 .segments.txt 文件
   -segment2srt：输出带时间戳的文本片段 (Segment)并转换为 .srt 字幕文件

   -subword：输出所有的子词 (Subword) 及其时间戳并保存为 .subwords.txt 文件
   -subword2srt：输出所有的子词 (Subword) 及其时间戳并转换为 .subwords.srt 字幕文件

   -kass：生成逐字计时的卡拉OK式 ASS 字幕文件 (.ass)

Packages
========

`reazonspeech.evaluation <pkg/evaluation>`_

* Provides a set of tools to evaluate ReazonSpeech models and other speech recognition models.


`reazonspeech.nemo.asr <pkg/nemo-asr>`_

* Implements a fast, accurate speech recognition based on FastConformer-RNNT.
* The total number of parameters is 619M. Requires `Nvidia Nemo <https://github.com/NVIDIA/NeMo>`_.

`reazonspeech.k2.asr <pkg/k2-asr>`_

* Next-gen Kaldi model that is very fast and accurate.
* The total number of parameters is 159M. Requires `sherpa-onnx <https://github.com/k2-fsa/sherpa-onnx>`_.
* Also contains a bilingual (ja-en) model, which is highly accurate at language detection in bilingual settings of Japanese and English.
* For development: "ja-en-mls-5k" model trained on 5k hours of ReazonSpeech and MLS English data each

`reazonspeech.espnet.asr <pkg/espnet-asr>`_

* Speech recognition with a Conformer-Transducer model.
* The total number of parameters is 120M. Requires `ESPnet <https://github.com/espnet/espnet>`_.

`reazonspeech.espnet.oneseg <pkg/espnet-oneseg>`_

* Provides a set of tools to analyze Japanese "one-segment" TV stream.
* Use this package to create Japanese audio corpus.

LICENSE
=======

::

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
