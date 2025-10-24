import argparse
import os
import tempfile
import shutil
import torch
import torchaudio
import json
import time
import numpy as np
from pathlib import Path
from pydub import AudioSegment
from omegaconf import open_dict
from reazonspeech.nemo.asr import load_model
from reazonspeech.nemo.asr.decode import find_end_of_segment, decode_hypothesis, PAD_SECONDS, SECONDS_PER_STEP
from reazonspeech.nemo.asr.interface import Segment, Subword
from reazonspeech.nemo.asr.writer import (
    SRTWriter,
    ASSWriter,
    TextWriter,
    TSVWriter,
    VTTWriter,
)

# ONNX VAD 依赖
try:
    import onnxruntime
except ImportError:
    onnxruntime = None


# --- ONNX VAD 辅助函数 ---
def get_speech_timestamps_onnx(
    wav_tensor, onnx_session, threshold=0.5, sampling_rate=16000
):
    """使用 ONNX 模型和后处理来获取语音时间戳"""
    # ONNX 模型需要特定的输入形状
    if wav_tensor.dim() == 2:
        wav_tensor = wav_tensor.unsqueeze(0)

    # 运行 ONNX 模型
    ort_inputs = {"input_values": wav_tensor.numpy()}
    ort_outs = onnx_session.run(None, ort_inputs)
    logits = torch.from_numpy(ort_outs[0])[0]  # 获取 logits: [num_frames, num_classes]

    # Pyannote-segmentation-3.0 的输出中，索引2是 "speech"
    speech_probs = torch.sigmoid(logits[:, 2])
    speech_frames = speech_probs > threshold

    frame_duration_s = (wav_tensor.shape[2] / sampling_rate) / logits.shape[0]
    speech_timestamps = []
    is_speech = False
    start_time = 0

    for i, frame in enumerate(speech_frames):
        if frame and not is_speech:
            is_speech = True
            start_time = i * frame_duration_s
        elif not frame and is_speech:
            is_speech = False
            end_time = i * frame_duration_s  # 结束点是当前帧的开始
            speech_timestamps.append({"start": start_time, "end": end_time})
    if is_speech:
        end_time = len(speech_frames) * frame_duration_s
        speech_timestamps.append({"start": start_time, "end": end_time})

    return speech_timestamps


def format_duration(seconds):
    """将秒数格式化为 'X时Y分Z.ZZ秒' 的形式"""
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    parts = []
    if hours > 0:
        parts.append(f"{int(hours)} 小时")
    if minutes > 0:
        parts.append(f"{int(minutes)} 分")

    # 总是显示秒，并保留两位小数
    parts.append(f"{secs:.2f} 秒")

    return "".join(parts)


def create_precise_segments_from_subwords(
    all_subwords, vad_chunk_end_times_s=[], no_chunk=False
):
    if not all_subwords:
        return [], [], []

    # 计算子词平均持续时长和子词结束时间
    average_duration = SECONDS_PER_STEP

    if len(all_subwords) > 1:
        durations = [
            all_subwords[i + 1].seconds - all_subwords[i].seconds
            for i in range(len(all_subwords) - 1)
        ]
    else:
        durations = []

    positive_durations = [d for d in durations if d > 0]
    if positive_durations:
        average_duration = sum(positive_durations) / len(positive_durations)

    # --- 动态计算停顿阈值 ---
    pause_threshold = float('inf') # 默认阈值为无穷大，即默认禁用此功能
    # 只有当有足够多的数据点（例如超过20个子词间隔）时，才计算并启用阈值
    MIN_SAMPLES_FOR_THRESHOLD = 20 
    if len(durations) > MIN_SAMPLES_FOR_THRESHOLD:
        # 移除异常值，让统计更准确
        valid_durations = [d for d in durations if d > 0]
        if len(valid_durations) > MIN_SAMPLES_FOR_THRESHOLD:
            pause_threshold = np.percentile(valid_durations, 95) 

    subword_end_seconds = []
    vad_cursor = 0
    
    for i, sub in enumerate(all_subwords):
        # 统一计算潜在的结束时间
        potential_end_time = (
            all_subwords[i + 1].seconds
            if i < len(all_subwords) - 1
            else sub.seconds + average_duration
        )
    
        # 统一计算 VAD 边界（如果不使用 VAD，则默认为无穷大）
        vad_boundary_end_s = float("inf")
        if not no_chunk and vad_chunk_end_times_s:
            while (
                vad_cursor < len(vad_chunk_end_times_s)
                and sub.seconds > vad_chunk_end_times_s[vad_cursor]
            ):
                vad_cursor += 1
    
            if vad_cursor < len(vad_chunk_end_times_s):
                vad_boundary_end_s = vad_chunk_end_times_s[vad_cursor]
    
        end_time = min(potential_end_time, vad_boundary_end_s)
    
        # 最后修正：确保结束时间总是在开始时间之后
        if end_time <= sub.seconds:
            end_time = sub.seconds + 0.1
    
        subword_end_seconds.append(end_time)

    # 使用 VAD 优先、find_end_of_segment 其次、基于语速/停顿边界补充的逻辑生成片段
    new_segments = []
    segment_to_subword_map = []
    start = 0
    vad_cursor = 0
    while start < len(all_subwords):
        end_idx = -1  # 初始化结束索引

        # 检查 VAD 边界 (最高优先级)
        if not no_chunk and vad_chunk_end_times_s and start < len(all_subwords) - 1:
            # 定位当前 VAD 游标
            while (
                vad_cursor < len(vad_chunk_end_times_s)
                and all_subwords[start].seconds > vad_chunk_end_times_s[vad_cursor]
            ):
                vad_cursor += 1

            if vad_cursor < len(vad_chunk_end_times_s):
                current_vad_boundary = vad_chunk_end_times_s[vad_cursor]
                # 如果下一个词的时间戳越界，且当前词没有越界，就在当前词切分。
                if (
                    start + 1 < len(all_subwords)
                    and all_subwords[start + 1].seconds > current_vad_boundary
                    and all_subwords[start].seconds < current_vad_boundary
                ):
                    end_idx = start

        # 规则 2: 如果 VAD 未强制切分，则使用原始逻辑
        if end_idx == -1:
            end_idx = find_end_of_segment(all_subwords, start)

        # 规则 3: 基于语速/停顿的边界 (最低, 作为补充)
        # 只有当前片段依然很长 (例如超过 N 个子词)，并且没有被前两种规则切分时，才考虑此规则
        pause_split_indices = []
        current_segment_len = end_idx - start + 1
        
        if current_segment_len > MIN_SAMPLES_FOR_THRESHOLD:
            # 在 find_end_of_segment 划定的长片段内部，寻找所有显著的停顿点
            for i in range(start, end_idx):
                gap = all_subwords[i + 1].seconds - all_subwords[i].seconds
                if gap > pause_threshold:
                    pause_split_indices.append(i) # 记录所有停顿点的索引
        
        # 将片段的起始点和最终的结束点加入，形成完整的切分区间
        all_split_points = [start -1] + pause_split_indices + [end_idx]
        
        # 遍历所有切分点，生成多个片段
        for i in range(len(all_split_points) - 1):
            segment_start_idx = all_split_points[i] + 1
            segment_end_idx = all_split_points[i+1]

            # 提取当前片段的子词和索引
            current_subwords = all_subwords[segment_start_idx : segment_end_idx + 1]
            current_indices = list(range(segment_start_idx, segment_end_idx + 1))

            # 创建 Segment 对象
            start_s = current_subwords[0].seconds
            end_s = subword_end_seconds[segment_end_idx]
            text = "".join(s.token for s in current_subwords).replace(" ", " ").strip()
    
            if text:
                new_segment = Segment(start_seconds=start_s, end_seconds=end_s, text=text)
                new_segments.append(new_segment)
                segment_to_subword_map.append(
                    (new_segment, current_subwords, current_indices)
                )

        start = end_idx + 1

    return new_segments, segment_to_subword_map, subword_end_seconds


def main():
    # --- 设置命令行参数解析 ---
    parser = argparse.ArgumentParser(
        description="使用 ReazonSpeech 模型识别语音，并按指定格式输出结果。基于静音的智能分块方式识别长音频，以保证准确率并解决显存问题"
    )

    # 添加一个必须的位置参数：音频/视频文件路径
    parser.add_argument(
        "input_file",
        help="需要识别语音的音频/视频文件路径",
    )

    # --- --no-chunk 参数 ---
    parser.add_argument(
        "--no-chunk",
        action="store_true",
        help="禁用智能分块功能，一次性处理整个音频文件",
    )

    def beam_size_type(value):
        """自定义 argparse 类型函数，用于验证 beam_size 的范围"""
        try:
            ivalue = int(value)
        except ValueError:
            # 如果无法转换为整数，则引发错误
            raise argparse.ArgumentTypeError(f"beam_size 必须为 4 到 64 之间的整数，您提供的 {value} 不正确")
        
        if not (4 <= ivalue <= 64):
            # 如果不在指定的范围内，则引发错误
            raise argparse.ArgumentTypeError(f"beam_size 必须为 4 到 64 之间的整数，您提供的 {ivalue} 不正确")
        
        # 如果验证通过，返回整数值
        return ivalue

    parser.add_argument(
            "--beam",
            type=beam_size_type,
            default=4,
            help="设置集束搜索（Beam Search）宽度，范围为 4 到 64 之间的整数，默认值是 4 ，更大的值可能更准确但更慢",
        )

    # --- VAD核心参数 ---
    parser.add_argument(
        "--vad_threshold",
        type=float,
        default=0.2,
        help="【VAD】判断为语音的置信度阈值（0-1）",
    )
    parser.add_argument(
        "--min_speech_duration_ms",
        type=float,
        default=100,
        help="【过滤器】移除短于此时长（毫秒）的语音块",
    )
    parser.add_argument(
        "--keep_silence",
        type=int,
        default=300,
        help="在语音块前后扩展时长（毫秒）",
    )

    # 输出格式参数
    parser.add_argument(
        "-text",
        action="store_true",
        help="仅输出完整的识别文本并保存为 .txt 文件",
    )
    parser.add_argument(
        "-segment",
        action="store_true",
        help="输出带时间戳的文本片段（Segment）并保存为 .segments.txt 文件",
    )
    parser.add_argument(
        "-segment2srt",
        action="store_true",
        help="输出带时间戳的文本片段（Segment）并转换为 .srt 字幕文件",
    )
    parser.add_argument(
        "-segment2vtt",
        action="store_true",
        help="输出带时间戳的文本片段（Segment）并转换为 .vtt 字幕文件",
    )
    parser.add_argument(
        "-segment2tsv",
        action="store_true",
        help="输出带时间戳的文本片段（Segment）并转换为由制表符分隔的 .tsv 文件",
    )
    parser.add_argument(
        "-subword",
        action="store_true",
        help="输出带时间戳的所有子词（Subword）并保存为 .subwords.txt 文件",
    )
    parser.add_argument(
        "-subword2srt",
        action="store_true",
        help="输出带时间戳的所有子词（Subword）并转换为 .subwords.srt 字幕文件",
    )
    parser.add_argument(
        "-subword2json",
        action="store_true",
        help="输出带时间戳的所有子词（Subword）并转换为 .subwords.json 文件",
    )
    parser.add_argument(
        "-kass",
        action="store_true",
        help="生成逐字计时的卡拉OK式 .ass 字幕文件",
    )

    args = parser.parse_args()

    if not args.no_chunk:
        if onnxruntime is None:
            print(
                "【错误】智能分块功能需要 onnxruntime，请运行 'pip install onnxruntime'"
            )
            return

        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        local_onnx_model_path = script_dir / "models" / "model_quantized.onnx"

        if not local_onnx_model_path.exists():
            print(
                f"【错误】Pyannote-segmentation-3.0 模型未在 '{local_onnx_model_path}' 中找到"
            )
            print("请下载 model_quantized.onnx 并放入 models 文件夹")
            return

    # --- 准备路径和临时文件 ---
    input_path = args.input_file
    output_dir = os.path.dirname(os.path.abspath(input_path))
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    # 创建一个临时的 WAV 文件
    # delete=False 确保在 with 块外使用它，最后手动删除
    temp_wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav_path = temp_wav_file.name
    temp_wav_file.close()  # 关闭文件句柄，以便 ffmpeg 可以写入

    temp_chunk_dir = tempfile.mkdtemp()

    # --- 执行核心的语音识别流程 ---
    try:
        # --- ffmpeg 预处理：将输入文件转换为标准 WAV ---
        print(f"正在转换输入文件 '{input_path}' 为临时 WAV 文件……")
        audio = AudioSegment.from_file(input_path)
        # 转换为单声道，16kHz采样率，这是ASR模型的标准格式
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(temp_wav_path, format="wav")
        print("转换完成")

        # --- 加载模型 (只需一次) ---
        print("正在加载模型……")
        asr_model_load_start = time.time()  # <--- 计时开始
        model = load_model()
        asr_model_load_end = time.time()  # <--- 计时结束
        print(f"模型加载完成")

        # 保存原始的解码配置
        original_decoding_cfg = model.cfg.decoding
        
        # 创建一个新的配置副本并修改 beam_size
        new_decoding_cfg = original_decoding_cfg.copy()
        with open_dict(new_decoding_cfg): # 使用 open_dict 使配置可修改
            if 'beam' not in new_decoding_cfg:
                 new_decoding_cfg.beam = {} # 如果不存在 beam 节，先创建
            new_decoding_cfg.beam.beam_size = args.beam
        
        # 在所有识别任务开始前，应用这个新的解码策略
        print(f"正在应用新的解码策略（beam_size={args.beam}）……")
        model.change_decoding_strategy(new_decoding_cfg)

        # --- 逐块识别并校正时间戳 ---
        all_segments = []
        all_subwords = []
        vad_chunk_end_times_s = []  # 用于存储VAD块的结束时间

        # --- 计时开始：核心识别流程 ---
        recognition_start_time = time.time()

        # 使用 pydub 创建静音段用于填充
        # PAD_SECONDS 是以秒为单位，pydub 需要毫秒
        silence_padding = AudioSegment.silent(duration=PAD_SECONDS * 1000, frame_rate=16000)

        if args.no_chunk:
            # --- 不分块的逻辑 ---
            print("未使用VAD，一次性处理整个文件……")
            final_audio = silence_padding + audio + silence_padding
            final_audio.export(temp_wav_path, format="wav")
            hyp, _ = model.transcribe(
                [temp_wav_path],
                return_hypotheses=True,
                verbose=True,
            )
            if hyp and hyp[0]:
                ret = decode_hypothesis(model, hyp[0])
                all_subwords = ret.subwords

        else:
            print(
                "正在从本地路径加载 Pyannote-segmentation-3.0 模型（将在 CPU 上运行）……"
            )
            vad_model_load_start = time.time()  # <--- 计时开始
            onnx_session = onnxruntime.InferenceSession(
                str(local_onnx_model_path), providers=["CPUExecutionProvider"]
            )
            vad_model_load_end = time.time()  # <--- 计时结束
            print(f"Pyannote-segmentation-3.0 模型加载完成")
            print("正在使用 Pyannote-segmentation-3.0 侦测语音活动……")

            wav_tensor, sr = torchaudio.load(temp_wav_path)
            speech_timestamps_seconds = get_speech_timestamps_onnx(
                wav_tensor, onnx_session, args.vad_threshold
            )

            nonsilent_ranges_ms = [
                [ts["start"] * 1000, ts["end"] * 1000]
                for ts in speech_timestamps_seconds
            ]

            if not nonsilent_ranges_ms:
                print("【警告】未侦测到语音活动")
                return
            else:
                original_chunk_count = len(nonsilent_ranges_ms)
                min_speech_duration_ms = args.min_speech_duration_ms
                filtered_ranges = [
                    r
                    for r in nonsilent_ranges_ms
                    if (r[1] - r[0]) >= min_speech_duration_ms
                ]
                print(
                    f"VAD 侦测到 {original_chunk_count} 个语音块，已过滤不超过 {min_speech_duration_ms}ms 的部分，保留并处理 {len(filtered_ranges)} 个语音块"
                )

            # 记录上一个处理块的实际结束时间
            last_processed_end_ms = 0

            wav_audio = AudioSegment.from_wav(temp_wav_path)
            for i, time_range in enumerate(filtered_ranges):
                start_ms, end_ms = time_range
                # 记录这个VAD块在原始音频中的精确结束时间
                chunk_end_time_s = end_ms / 1000.0
                vad_chunk_end_times_s.append(chunk_end_time_s)
                # 计算带 padding 的起止时间
                start_ms = max(0, start_ms - args.keep_silence)
                end_ms = min(len(wav_audio), end_ms + args.keep_silence)
                # 如果当前块的开始时间早于上一个块的结束时间，则将当前块的开始时间强制设置为上一个块的结束时间，避免重叠
                if start_ms < last_processed_end_ms:
                    start_ms = last_processed_end_ms
                # 如果修正后出现 start >= end 的情况（说明这个块完全被前一个块的 padding 覆盖了）
                if start_ms >= end_ms:
                    continue         
                # 更新 last_processed_end_ms，为下一次循环做准备
                last_processed_end_ms = end_ms

                # 使用 pydub 切分音频块，将静音段添加到块的前后，完成填充
                chunk = silence_padding + wav_audio[start_ms:end_ms] + silence_padding

                time_offset_s = start_ms / 1000.0
                chunk_path = os.path.join(temp_chunk_dir, f"chunk_{i}.wav")
                print(
                    f"正在处理语音块 {i+1}/{len(filtered_ranges)} （该块起止时间：{SRTWriter._format_time(start_ms / 1000.0)} --> {SRTWriter._format_time(end_ms / 1000.0)}，持续时间：{(end_ms - start_ms) / 1000.0:.2f} 秒）……"
                )
                chunk.export(chunk_path, format="wav")
                
                hyp, _ = model.transcribe(
                    [chunk_path],
                    return_hypotheses=True,
                    verbose=False,
                )
                if hyp and hyp[0]:
                    ret = decode_hypothesis(model, hyp[0])
                    if ret.subwords:
                        for sub in ret.subwords:
                            all_subwords.append(
                                Subword(
                                    seconds=sub.seconds + time_offset_s,
                                    token_id=sub.token_id,
                                    token=sub.token,
                                )
                            )

        # 恢复原始解码策略，以防后续有其他操作
        print("正在恢复原始解码策略……")
        model.change_decoding_strategy(original_decoding_cfg)

        # --- 计时结束：核心识别流程 ---
        recognition_end_time = time.time()

        # 如果整个过程下来没有任何识别结果，提前告知用户并退出，避免生成空文件
        if not all_subwords:
            print("=" * 70)
            print("【信息】未识别到任何有效的语音内容，程序结束")
            return

        print("=" * 70)
        print("正在根据子词和VAD边界生成精确文本片段……")

        all_segments, segment_to_subword_map, subword_end_seconds = (
            create_precise_segments_from_subwords(
                all_subwords, vad_chunk_end_times_s, args.no_chunk
            )
        )

        if not all_segments:
            print("【错误】未能生成任何文本片段，程序结束")
            return

        print("文本片段生成完成")

        # --- 根据参数生成输出文件 ---
        print("=" * 70)
        print("识别完成，正在生成输出文件……")

        # 检查用户是否指定了任何一种文件输出格式
        file_output_requested = any(
            [
                args.text,
                args.segment,
                args.segment2srt,
                args.segment2vtt,
                args.segment2tsv,
                args.subword,
                args.subword2srt,
                args.subword2json,
                args.kass,
            ]
        )

        # 只有在用户完全没有指定任何输出参数时，才在控制台打印
        if not file_output_requested:
            full_text = " ".join([seg.text for seg in all_segments])
            print("\n识别结果（完整文本）：")
            print(full_text)
            print("=" * 70)
            print("提示：未指定输出参数，结果将打印至控制台")
            print("请使用 -text，-segment2srt，-kass 等参数将结果保存为文件")

        if args.text:
            output_path = os.path.join(output_dir, f"{base_name}.txt")
            full_text = " ".join([seg.text for seg in all_segments])
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            print(f"完整的识别文本已保存为：{output_path}")

        if args.segment:
            output_path = os.path.join(output_dir, f"{base_name}.segments.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                writer = TextWriter(f)
                writer.write_header()  # 虽然为空，但保持接口一致性
                for seg in all_segments:
                    writer.write(seg)
            print(f"带时间戳的文本片段已保存为：{output_path}")

        if args.segment2srt:
            output_path = os.path.join(output_dir, f"{base_name}.srt")
            with open(output_path, "w", encoding="utf-8") as f:
                writer = SRTWriter(f)
                writer.write_header()
                for seg in all_segments:
                    writer.write(seg)
            print(f"文本片段 SRT 字幕文件已保存为：{output_path}")

        if args.segment2vtt:
            output_path = os.path.join(output_dir, f"{base_name}.vtt")
            with open(output_path, "w", encoding="utf-8") as f:
                writer = VTTWriter(f)
                writer.write_header()
                for seg in all_segments:
                    writer.write(seg)
            print(f"文本片段 WebVTT 字幕文件已保存为：{output_path}")

        if args.segment2tsv:
            output_path = os.path.join(output_dir, f"{base_name}.tsv")
            with open(output_path, "w", encoding="utf-8") as f:
                writer = TSVWriter(f)
                writer.write_header()
                for seg in all_segments:
                    writer.write(seg)
            print(f"文本片段 TSV 文件已保存为：{output_path}")

        if args.subword:
            output_path = os.path.join(output_dir, f"{base_name}.subwords.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                for sub in all_subwords:
                    f.write(
                        f"[{SRTWriter._format_time(sub.seconds)}] {sub.token.replace(' ', '')}\n"
                    )
            print(f"带时间戳的所有子词信息已保存为：{output_path}")

        if args.subword2srt:
            output_path = os.path.join(output_dir, f"{base_name}.subwords.srt")
            with open(output_path, "w", encoding="utf-8") as f:
                if subword_end_seconds:
                    for i, sub in enumerate(all_subwords):
                        start_time_str = SRTWriter._format_time(sub.seconds)
                        end_time_str = SRTWriter._format_time(subword_end_seconds[i])
                        text = sub.token.replace(" ", " ").strip()

                        f.write(f"{i+1}\n")
                        f.write(f"{start_time_str} --> {end_time_str}\n")
                        f.write(f"{text}\n\n")

            print(f"所有子词信息的 SRT 文件已保存为：{output_path}")

        if args.subword2json:
            output_path = os.path.join(output_dir, f"{base_name}.subwords.json")
            subwords_for_json = []
            for sub in all_subwords:
                subwords_for_json.append(
                    {"token": sub.token.replace(" ", " "), "timestamp": sub.seconds}
                )

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(subwords_for_json, f, ensure_ascii=False, indent=4)

            print(f"所有子词信息的 JSON 文件已保存为：{output_path}")

        if args.kass:
            output_path = os.path.join(output_dir, f"{base_name}.ass")
            with open(output_path, "w", encoding="utf-8-sig") as f:

                # 使用 writer 生成标准文件头
                writer = ASSWriter(f)
                writer.write_header()

                dialogue_lines = []

                # 【直接遍历预先计算好的映射关系
                for seg, segment_subwords, segment_indices in segment_to_subword_map:

                    if not segment_subwords:
                        continue

                    karaoke_text = ""
                    for i, sub in enumerate(segment_subwords):
                        global_index = segment_indices[i]
                        duration_s = subword_end_seconds[global_index] - sub.seconds
                        duration_cs = max(1, round(duration_s * 100))
                        clean_token = sub.token.replace(" ", " ")
                        karaoke_text += f"{{\\k{duration_cs}}}{clean_token}"

                    # 格式化 Dialogue 行，使用 writer 的时间格式化函数
                    start_time = ASSWriter._format_time(seg.start_seconds)
                    end_time = ASSWriter._format_time(seg.end_seconds)
                    dialogue_lines.append(
                        f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{karaoke_text}"
                    )

                # 写入文件
                f.write("\n".join(dialogue_lines))

            print(f"卡拉OK式 ASS 字幕已保存为：{output_path}")

    finally:
        print("=" * 70)
        print(
            f"ReazonSpeech模型加载耗时：{format_duration(asr_model_load_end - asr_model_load_start)}"
        )
        # 安全地获取 VAD 加载的起止时间，如果不存在则默认为 0
        vad_model_load_start = locals().get("vad_model_load_start", 0)
        vad_model_load_end = locals().get("vad_model_load_end", 0)
        # 只有当VAD加载时间大于0时才打印，否则不显示
        if vad_model_load_end - vad_model_load_start > 0:
            print(
                f"Pyannote-segmentation-3.0 模型加载耗时：{format_duration(vad_model_load_end - vad_model_load_start)}"
            )
            print(
                f"语音识别核心流程耗时：{format_duration(recognition_end_time - recognition_start_time - (vad_model_load_end - vad_model_load_start))}"
            )
        else:
            print(
                f"语音识别核心流程耗时：{format_duration(recognition_end_time - recognition_start_time)}"
            )

        print("=" * 70)
        # --- 清理工作：删除临时的 WAV 文件 ---
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
            print(f"\n临时文件 '{temp_wav_path}' 已删除")

        # 使用 shutil.rmtree 来删除临时块目录及其所有内容
        if os.path.exists(temp_chunk_dir):
            shutil.rmtree(temp_chunk_dir)
            print(f"临时块目录 '{os.path.basename(temp_chunk_dir)}' 已删除")

if __name__ == "__main__":
    main()
