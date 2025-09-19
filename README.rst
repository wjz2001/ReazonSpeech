============
ReazonSpeech
============

This repository provides access to the main user tooling of ReazonSpeech project.

* https://research.reazon.jp/projects/ReazonSpeech/

用法
====
.. code:: console

   python asr.py 文件路径

推荐使用nemo版模型，注意先安装好对应cuda版本的torch

参数
====
.. code:: console

   --no-chunk 禁用分块

   --min_silence_len 最小静音时长（毫秒）
   --silence_thresh 静音音量阈值（dBFS）
   --keep_silence 分割块前后保留的静音时长（毫秒）

   -text 仅输出完整的识别文本并保存到 .txt 文件

   -segment 输出带时间戳的文本片段 (Segment)并保存到 .segments.txt 文件
   -segment2srt 将文本片段 (Segment) 转换为 SRT 字幕文件并保存

   -subword 输出所有的子词 (Subword) 及其时间戳并保存到保存到 .subwords.txt 文件=
   -subword2srt 将子词 (Subword) 转换为 SRT 字幕文件并保存

Install
=======

.. code:: console

   $ git clone https://github.com/reazon-research/ReazonSpeech
   $ pip install ReazonSpeech/pkg/nemo-asr  # or k2-asr, espnet-asr or espnet-oneseg

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
