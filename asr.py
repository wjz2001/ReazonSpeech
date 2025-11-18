import argparse
import copy
import heapq
import os
import tempfile
import shutil
import sys
import subprocess
import torch
import torchaudio
import json
import time
import numpy as np
from pathlib import Path
from pydub import AudioSegment
from omegaconf import open_dict
from reazonspeech.nemo.asr import load_model
from reazonspeech.nemo.asr.audio import SAMPLERATE
from reazonspeech.nemo.asr.decode import find_end_of_segment, decode_hypothesis, PAD_SECONDS, SECONDS_PER_STEP, SUBWORDS_PER_SEGMENTS
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

def open_folder(path):
    """根据不同操作系统，使用默认的文件浏览器打开指定路径的文件夹"""
    if not path.is_dir():
        print(f"【调试模式】目录 '{path}' 不存在，无法自动打开")
        return
    try:
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":  # macOS
            subprocess.run(["open", path])
        else:  # Linux and other Unix-like
            subprocess.run(["xdg-open", path])
        print(f"【调试模式】已自动为您打开临时分块目录：{path}")
    except Exception as e:
        print(f"【警告】尝试自动打开目录失败：{e}，请手动访问：{path}")

# --- ONNX VAD 辅助函数 ---
def get_speech_timestamps_onnx(
    waveform,
    onnx_session,
    threshold=0.2,
    sampling_rate=SAMPLERATE,
    neg_threshold=None, #语音结束阈值（低阈值）
    min_speech_duration_ms=100, #最小语音段时长
    min_silence_duration_ms=200, #多长静音视为真正间隔
    max_speech_duration_s=float('inf'), # 智能切分阶段的软最大长度（可为 inf）
):
    """使用 ONNX 模型和后处理来获取语音时间戳"""
    # ONNX 模型需要特定的输入形状
    # ==== 默认结束阈值 ====
    if neg_threshold is None:
        neg_threshold = max(0.0, threshold - 0.15)

    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)

    # 运行 ONNX 模型
    ort_inputs = {"input_values": waveform.numpy()}
    ort_outs = onnx_session.run(None, ort_inputs)
    logits = torch.from_numpy(ort_outs[0])[0]  # 获取 logits: [num_frames, num_classes]

    # Pyannote-segmentation-3.0 的输出中，索引2是 "speech"
    speech_probs = torch.sigmoid(logits[:, 2])

    frame_duration_s = (waveform.shape[2] / sampling_rate) / logits.shape[0]

    # ==== 将 ms/s 转为帧数 ====
    min_silence_frames = min_silence_duration_ms / (frame_duration_s * 1000.0)
    max_speech_frames = (
        max_speech_duration_s / frame_duration_s
        if max_speech_duration_s != float('inf')
        else float('inf')
    )

    speeches = []
    current_speech = {}
    triggered = False
    temp_silence_start_frame = None
    possible_ends = []  # (end_time_in_s, silence_len_in_frames)

    for i, prob in enumerate(speech_probs):
        # --- 语音开始逻辑（高阈值） ---
        if not triggered and prob >= threshold:
            triggered = True
            current_speech = {'start': i * frame_duration_s}
            temp_silence_start_frame = None
            possible_ends = []
            continue

        if not triggered:
            continue

        # --- 处于语音段中时，跟踪静音区域 ---
        if prob < neg_threshold:
            # 低于结束阈值，可能是静音的开始
            if temp_silence_start_frame is None:
                temp_silence_start_frame = i
        else:
            # 回到高于 neg_threshold 的区域，检查之前是否有足够长静音
            if temp_silence_start_frame is not None:
                silence_len = i - temp_silence_start_frame
                if silence_len >= min_silence_frames:
                    # 记录为潜在的段结束位置（智能切分点）
                    end_time = temp_silence_start_frame * frame_duration_s
                    possible_ends.append((end_time, silence_len))
                temp_silence_start_frame = None

        # --- 智能最大长度切分逻辑（软约束） ---
        if max_speech_frames != float('inf'):
            current_len_frames = i - int(current_speech['start'] / frame_duration_s)
            if current_len_frames >= max_speech_frames:
                if possible_ends:
                    # 优先选择最长静音点作为切分点
                    best_split_time, _ = max(possible_ends, key=lambda x: x[1])
                    end_time = best_split_time

                    if (end_time - current_speech['start']) * 1000.0 >= min_speech_duration_ms:
                        print(f"【VAD】超过最大语音时长，在最近的静音点 {end_time:.2f} 秒处软切分")
                        current_speech['end'] = end_time
                        speeches.append(current_speech)

                    # 从切分点开始新的语音段
                    current_speech = {'start': end_time}
                    temp_silence_start_frame = None
                    possible_ends = []
                    continue
                    # 没有可用静音点：放弃软切分，继续累积，交给后续“硬切分”

        # --- 正常结束逻辑：静音持续足够长则结束当前段 ---
        if temp_silence_start_frame is not None:
            silence_len = i - temp_silence_start_frame
            if silence_len >= min_silence_frames:
                end_time = temp_silence_start_frame * frame_duration_s
                if (end_time - current_speech['start']) * 1000.0 >= min_speech_duration_ms:
                    current_speech['end'] = end_time
                    speeches.append(current_speech)
                # 重置状态
                triggered = False
                current_speech = {}
                temp_silence_start_frame = None
                possible_ends = []
                continue

    # --- 处理结尾残留的语音段 ---
    if triggered and current_speech:
        end_time = len(speech_probs) * frame_duration_s
        if (end_time - current_speech['start']) * 1000.0 >= min_speech_duration_ms:
            current_speech['end'] = end_time
            speeches.append(current_speech)

    return speeches

def format_duration(seconds):
    """将秒数格式化为 'X时Y分Z.ZZ秒' 的形式"""
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    parts = []
    if hours > 0:
        parts.append(f"{int(hours)} 小时 ")
    if minutes > 0:
        parts.append(f"{int(minutes)} 分 ")

    # 总是显示秒，并保留两位小数
    parts.append(f"{secs:.2f} 秒")

    return "".join(parts)


def create_precise_segments_from_subwords(
    all_subwords, vad_chunk_end_times_s=None, no_chunk=False, build_map=False
):
    # 如果没有提供 vad_chunk_end_times_s，创建一个新的空列表
    if vad_chunk_end_times_s is None:
        vad_chunk_end_times_s = []

    if not all_subwords:
        return [], [], []

    # 预处理：为每个子词计算其VAD块的结束时间
    subword_vad_end_times = []
    if no_chunk or not vad_chunk_end_times_s:
        subword_vad_end_times = [float("inf")] * len(all_subwords)
    else:
        vad_cursor = 0
        for sub in all_subwords:
            # 移动游标找到当前子词所属的VAD块
            while (vad_cursor < len(vad_chunk_end_times_s) and
                   sub.seconds > vad_chunk_end_times_s[vad_cursor]):
                vad_cursor += 1
            
            if vad_cursor < len(vad_chunk_end_times_s):
                subword_vad_end_times.append(vad_chunk_end_times_s[vad_cursor])
            else:
                subword_vad_end_times.append(float("inf"))

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
    MIN_SAMPLES_FOR_THRESHOLD = SUBWORDS_PER_SEGMENTS * 2 
    if len(durations) > MIN_SAMPLES_FOR_THRESHOLD and positive_durations:
        pause_threshold = np.percentile(positive_durations, 95) 

    subword_end_seconds = []
    for i, sub in enumerate(all_subwords):
        # 统一计算潜在的结束时间
        potential_end_time = (
            all_subwords[i + 1].seconds
            if i < len(all_subwords) - 1
            else sub.seconds + average_duration
        )
    
        # 直接从缓存中读取VAD边界，不再需要游标
        vad_boundary_end_s = subword_vad_end_times[i]
    
        end_time = min(potential_end_time, vad_boundary_end_s)
    
        # 最后修正：确保结束时间总是在开始时间之后
        if end_time <= sub.seconds:
            end_time = sub.seconds + SECONDS_PER_STEP
    
        subword_end_seconds.append(end_time)

    # 使用 VAD 优先、find_end_of_segment 其次、基于语速/停顿边界补充的逻辑生成片段
    all_segments = []
    segment_to_subword_map = []
    start = 0
    while start < len(all_subwords):
        # 获取当前需要遵守的 VAD 边界的时间戳
        current_vad_boundary = subword_vad_end_times[start]
        # 从 start 开始，找到最后一个仍在边界内的索引 last_in_boundary_idx
        # 初始假设切分点就在 start 处
        last_in_boundary_idx = start 
        # 向后扫描，只要下一个子词仍在边界内，就更分点
        while (last_in_boundary_idx + 1 < len(all_subwords) and all_subwords[last_in_boundary_idx + 1].seconds < current_vad_boundary):
            last_in_boundary_idx += 1

        # 规则 2: 在VAD确定的范围内，使用启发式规则寻找更早的断点
        heuristic_end_idx = find_end_of_segment(all_subwords, start)

        # --- 取两个切分点中更早的那个作为最终结束点 ---
        # 确保 heuristic_end_idx 不会超过 VAD 的硬边界
        end_idx = min(last_in_boundary_idx, heuristic_end_idx)
        # 规则 3: 基于语速/停顿的边界 (最低, 作为补充)
        # 只有当前片段依然很长 (例如超过 N 个子词)，并且没有被前两种规则切分时，才考虑此规则
        current_segment_len = end_idx - start + 1
        
        if current_segment_len > MIN_SAMPLES_FOR_THRESHOLD:
            # 在 find_end_of_segment 划定的长片段内部，寻找所有显著的停顿点
            pause_split_indices = [
                i for i in range(start, end_idx)
                if (all_subwords[i + 1].seconds - all_subwords[i].seconds) > pause_threshold
            ] # 记录所有停顿点的索引
        else:
            pause_split_indices = []
        
        # 将片段的起始点和最终的结束点加入，形成完整的切分区间
        all_split_points = [start -1] + pause_split_indices + [end_idx]

        if pause_split_indices:
            # --- 构建带有 || 分隔符的完整片段预览 ---
            preview_parts = []
            # 将切分点索引放入一个集合中，以提高查找效率，set使平均时间复杂度从 O(N) 降低到 O(1)
            split_points_set = set(pause_split_indices)
            # 遍历当前长片段（从 start 到 end_idx）的所有子词
            for i in range(start, end_idx + 1):
                # 添加子词本身
                preview_parts.append(all_subwords[i].token)
                # 如果当前子词的索引是一个切分点，则在它后面添加分隔符
                if i in split_points_set:
                    preview_parts.append(" || ")
            # 将所有部分连接成一个字符串
            full_preview_text = "".join(preview_parts)
            print(f"在 {SRTWriter._format_time(all_subwords[start].seconds)} 开始的长片段：“{full_preview_text}”内找到显著停顿点")
        
        # 遍历所有切分点，生成多个片段
        for i in range(len(all_split_points) - 1):
            segment_start_idx = all_split_points[i] + 1
            segment_end_idx = all_split_points[i + 1]

            # 提取当前片段的子词和索引
            current_subwords = all_subwords[segment_start_idx: segment_end_idx + 1]
            current_indices = list(range(segment_start_idx, segment_end_idx + 1))

            # 创建 Segment 对象
            text = "".join(s.token for s in current_subwords)
    
            if text:
                all_segment = Segment(
                    start_seconds=current_subwords[0].seconds,
                    end_seconds=subword_end_seconds[segment_end_idx],
                    text=text)
                all_segments.append(all_segment)
                # 仅在需要时构建映射
                if build_map:
                    segment_to_subword_map.append(
                        (all_segment, current_subwords, current_indices)
                    )

        start = end_idx + 1

    return all_segments, segment_to_subword_map, subword_end_seconds

def merge_overlapping_intervals(intervals):
    """仅合并物理上重叠的区间。"""
    if not intervals:
        return []
    
    intervals = sorted(intervals, key=lambda x: x[0])
    
    merged = [intervals[0]]
    for current_start, current_end in intervals[1:]:
        _, last_end = merged[-1]
        
        if current_start < last_end: 
            merged[-1][1] = max(last_end, current_end)
        else:
            merged.append([current_start, current_end])
            
    return merged

def merge_overlap_dedup(chunk_subwords_list, chunk_ranges_s, prefer='next'):
    """
    合并来自不同块的子词列表，并处理重叠区域的重复子词。

    Args:
        chunk_subwords_list (list[list[Subword]]): 每个块识别出的子词列表的列表。
        chunk_ranges_s (list[tuple[float, float]]): 每个块在原始音频中的（开始，结束）时间。
        prefer (str, optional): 在重叠区遇到冲突时保留哪个版本。
                                'next' 表示保留后面块的结果。Defaults to 'next'.

    Returns:
        list[Subword]: 合并并去重后的最终子词列表。
    """
    def _overlap_interval(a, b):
        # a, b 是 (start_s, end_s)
        s = max(a[0], b[0])
        e = min(a[1], b[1])
        return (s, e) if e > s else (None, None)

    def _is_conflict(subword1, subword2):
        """判断两个子词是否冲突（token相同且时间相近）"""
        return (subword1.token_id == subword2.token_id and
                abs(subword1.seconds - subword2.seconds) <= SECONDS_PER_STEP * 1.5)
                
    if not chunk_subwords_list:
        return []
    # 初始 merged 列表就是第一个块的结果，它已经是排序的
    merged = chunk_subwords_list[0]

    for i in range(1, len(chunk_subwords_list)):
        prev_chunk_range = chunk_ranges_s[i-1]
        curr_chunk_range = chunk_ranges_s[i]
        curr_subs = chunk_subwords_list[i]

        s_ov, e_ov = _overlap_interval(prev_chunk_range, curr_chunk_range)

        # 如果没有重叠，直接使用高效的归并
        if s_ov is None:
            # heapq.merge 要求输入是已排序的，我们的数据满足这个条件
            merged = list(heapq.merge(merged, curr_subs, key=lambda x: x.seconds))
            continue

        # --- 有重叠区域，执行更复杂的合并去重逻辑 ---
        
        # 将旧结果（merged）和新结果（curr_subs）划分为重叠区和非重叠区
        prev_non_overlap = [s for s in merged if s.seconds < s_ov]
        prev_overlap = [s for s in merged if s_ov <= s.seconds <= e_ov]
        
        curr_non_overlap = [s for s in curr_subs if s.seconds > e_ov]
        curr_overlap = [s for s in curr_subs if s_ov <= s.seconds <= e_ov]

        # 处理核心的重叠区域
        final_overlap = []
        # 使用集合来存储那些因为冲突而被“淘汰”的旧 subword 的 id (内存地址)
        discarded_old_ids = set()
        
        # 为了高效查找，将 prev_overlap 的 token 存入字典
        # 键是 token_id，值是该 token_id 对应的所有 subword 列表
        prev_overlap_map = {}
        for sub in prev_overlap:
            if sub.token_id not in prev_overlap_map:
                prev_overlap_map[sub.token_id] = []
            prev_overlap_map[sub.token_id].append(sub)
            
        # 遍历新的重叠区，决定每个 subword 的去留
        for s_new in curr_overlap:
            conflicts = prev_overlap_map.get(s_new.token_id, [])
            
            found_conflict = False
            for s_old in conflicts:
                if _is_conflict(s_new, s_old):
                    found_conflict = True
                    # 如果找到冲突，根据 prefer 策略决定是否替换
                    # 我们标记旧的 subword 为 None，表示它可能被替换
                    if prefer == 'next':
                        # 记录要丢弃的旧 subword 的 id
                        discarded_old_ids.add(id(s_old))
                    break
            
            # 如果没有冲突，或者有冲突但策略是保留旧的，则新的也需要添加
            final_overlap.append(s_new)

        # 把旧的、未被标记为删除的 subword 添加回来
        for s_old in prev_overlap:
            if id(s_old) not in discarded_old_ids:
                final_overlap.append(s_old)
        
        # 对刚刚合并的重叠区进行一次排序（这部分数据量很小）
        final_overlap.sort(key=lambda x: x.seconds)

        # 3. 使用 heapq.merge 高效合并所有部分
        # 此时 prev_non_overlap, final_overlap, curr_non_overlap 都是已排序列表
        merged = list(heapq.merge(prev_non_overlap, final_overlap, curr_non_overlap, key=lambda x: x.seconds))

    return merged


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

    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式，处理结束后不删除临时文件，并自动打开临时分块目录",
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
            if not (4 <= ivalue <= 64):
                raise ValueError() # 故意引发一个值错误，被下面的 except 捕获
        except ValueError:
            # 如果无法转换为整数，则引发错误
            raise argparse.ArgumentTypeError(f"beam_size 必须为 4 到 64 之间的整数，您提供的 {value} 不正确")
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
    # ==== 新增 VAD 结束阈值（双阈值滞回） ====
    parser.add_argument(
        "--vad_end_threshold",
        type=float,
        default=None,
        help="【VAD】判断为语音结束后静音的置信度阈值（0-1），默认值是vad_threshold的值减去0.15",
    )
    parser.add_argument(
        "--min_speech_duration_ms",
        type=float,
        default=100,
        help="【过滤器】移除短于此时长（毫秒）的语音块",
    )
     # ==== 新增静音最小时长，用于智能合并/分段 ====
    parser.add_argument(
        "--min_silence_duration_ms",
        type=float,
        default=200,
        help="【VAD】短于此时长（毫秒）的语音块不被视为间隔",
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
    # ==== VAD 阈值参数校验和默认 ====
    if args.vad_end_threshold is None:
        args.vad_end_threshold = max(0.0, args.vad_threshold - 0.15)
    if not (0.0 <= args.vad_threshold <= 1.0):
        raise ValueError(f"vad_threshold 必须在（0-1）范围内，当前值错误")
    if not (0.0 <= args.vad_end_threshold <= 1.0):
        raise ValueError(f"vad_end_threshold 必须在（0-1）范围内，当前值错误")
    if args.vad_end_threshold > args.vad_threshold:
        raise ValueError(
            f"vad_end_threshold 不能大于 vad_threshold"
        )

    if not args.no_chunk:
        if onnxruntime is None:
            print("【错误】智能分块功能需要 onnxruntime，请运行 'pip install onnxruntime'")
            return

        local_onnx_model_path = Path(__file__).resolve().parent / "models" / "model_quantized.onnx"

        if not local_onnx_model_path.exists():
            print(
                f"【错误】Pyannote-segmentation-3.0 模型未在 '{local_onnx_model_path}' 中找到"
            )
            print("请下载 model_quantized.onnx 并放入 models 文件夹")
            return

    # --- 准备路径和临时文件 ---
    input_path = Path(args.input_file)
    output_dir = input_path.parent.resolve()
    base_name = input_path.stem

    # 创建一个临时的 WAV 文件
    # delete=False 确保在 with 块外使用它，最后手动删除
    temp_wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav_path = Path(temp_wav_file.name)
    temp_wav_file.close()  # 关闭文件句柄，以便 ffmpeg 可以写入

    temp_chunk_dir = None

    # --- 执行核心的语音识别流程 ---
    try:
        # --- ffmpeg 预处理：将输入文件转换为标准 WAV ---
        print(f"正在转换输入文件 '{input_path}' 为临时 WAV 文件……")
        audio = AudioSegment.from_file(input_path)
        # 转换为单声道，16kHz采样率，这是ASR模型的标准格式
        audio = audio.set_channels(1).set_frame_rate(SAMPLERATE)
        audio.export(temp_wav_path, format="wav")
        print("转换完成")

        # --- 加载模型 (只需一次) ---
        print("正在加载模型……")
        asr_model_load_start = time.time()  # <--- 计时开始
        model = load_model()
        asr_model_load_end = time.time()  # <--- 计时结束
        print(f"模型加载完成")

        # 获取模型最大允许输入语音块长度
        MAX_SPEECH_DURATION_S = model.cfg.train_ds.max_duration

        if args.beam != model.cfg.decoding.beam.beam_size:
            # 使用深拷贝来完全复制原始配置，确保它不受任何后续修改的影响
            original_decoding_cfg = copy.deepcopy(model.cfg.decoding)
            
            # 同样使用深拷贝创建新配置，确保它与原始配置完全独立
            new_decoding_cfg = copy.deepcopy(original_decoding_cfg)
            with open_dict(new_decoding_cfg): # 使用 open_dict 使配置可修改
                new_decoding_cfg.beam.beam_size = args.beam
            
            # 在所有识别任务开始前，应用这个新的解码策略
            print(f"正在应用新的解码策略：集束搜索宽度为 {args.beam} ……")
            model.change_decoding_strategy(new_decoding_cfg)

        # --- 逐块识别并校正时间戳 ---
        all_subwords = []
        vad_chunk_end_times_s = []  # 用于存储VAD块的结束时间

        # --- 计时开始：核心识别流程 ---
        recognition_start_time = time.time()

        # 使用 pydub 创建静音段用于填充
        # PAD_SECONDS 是以秒为单位，pydub 需要毫秒
        silence_padding = AudioSegment.silent(duration=PAD_SECONDS * 1000, frame_rate=SAMPLERATE)

        if args.no_chunk:
            # --- 不分块的逻辑 ---
            print("未使用VAD，一次性处理整个文件……")
            (silence_padding + audio + silence_padding).export(temp_wav_path, format="wav")
            hyp, _ = model.transcribe(
                [str(temp_wav_path)],
                return_hypotheses=True,
                verbose=True,
            )
            if hyp and hyp[0]:
                ret = decode_hypothesis(model, hyp[0])
                all_subwords = ret.subwords

        else:
            temp_chunk_dir = Path(tempfile.mkdtemp())
            print("正在从本地路径加载 Pyannote-segmentation-3.0 模型……")
            vad_model_load_start = time.time()  # <--- 计时开始
            onnx_session = onnxruntime.InferenceSession(
                str(local_onnx_model_path), providers=["CPUExecutionProvider"]
            )
            vad_model_load_end = time.time()  # <--- 计时结束
            print(f"Pyannote-segmentation-3.0 模型加载完成，将在 CPU 上运行")
            print("正在使用 Pyannote-segmentation-3.0 侦测语音活动……")

            waveform, _ = torchaudio.load(temp_wav_path) # torchaudio.load 返回的是 waveform, sample_rate
            # ==== 先做 VAD（双阈值 + 静音 + min_speech），不在此强制 30s ====
            smart_speeches = get_speech_timestamps_onnx(
                waveform,
                onnx_session,
                threshold=args.vad_threshold,
                sampling_rate=SAMPLERATE,
                neg_threshold=args.vad_end_threshold,
                min_speech_duration_ms=int(args.min_speech_duration_ms),
                min_silence_duration_ms=int(args.min_silence_duration_ms),
                max_speech_duration_s=MAX_SPEECH_DURATION_S,  # 智能切分阶段限制 30s
            )

            if not smart_speeches:
                print("【警告】未侦测到语音活动")
                return

            # 先把 VAD 段转成毫秒区间（不做硬切分）
            base_ranges_ms = [
                [int(seg["start"] * 1000.0), int(seg["end"] * 1000.0)]
                for seg in smart_speeches
            ]

            # 合并所有重叠的 VAD 段
            print("正在合并重叠的 VAD 语音块……")
            merged_ranges_ms = merge_overlapping_intervals(base_ranges_ms)
            if len(base_ranges_ms) != len(merged_ranges_ms):
                print(f"原 {len(base_ranges_ms)} 个 VAD 语音块已合并为 {len(merged_ranges_ms)} 个 VAD 语音块")
            else:
                print(f"没有需要合并的 VAD 语音块")

            # 记录这个VAD块在原始音频中的精确结束时间
            vad_chunk_end_times_s = [end_ms / 1000.0 for _, end_ms in merged_ranges_ms]

            # 对超长块再做硬切分（保留 1 秒重叠）
            nonsilent_ranges_ms = []
            max_len_ms = int(MAX_SPEECH_DURATION_S * 1000)

            for start_ms, end_ms in merged_ranges_ms:
                cur_start = start_ms
                while cur_start < end_ms:
                    cur_end = min(cur_start + max_len_ms, end_ms)
                    nonsilent_ranges_ms.append([cur_start, cur_end])

                    if cur_end >= end_ms:
                        break

                    # 保留 1 秒重叠
                    next_start = cur_end - 1000
                    if next_start <= cur_start:
                        next_start = cur_start + 500  # 兜底，避免死循环
                    cur_start = next_start

            print(
                f"VAD 侦测到 {len(smart_speeches)} 个语音块，拆分超过 {MAX_SPEECH_DURATION_S} 秒的部分后，实际需要处理 {len(nonsilent_ranges_ms)} 个语音块"
            )
            if not nonsilent_ranges_ms:
                print("【警告】经过过滤后无有效语音块")
                return

            chunk_ranges_s = []
            chunk_subwords_list = []

            for i, (unpadded_start_ms, unpadded_end_ms) in enumerate(nonsilent_ranges_ms):
                start_ms = max(0, unpadded_start_ms - args.keep_silence)
                end_ms = min(len(audio), unpadded_end_ms + args.keep_silence)

                # 使用 pydub 切分音频块，将静音段添加到块的前后，完成填充
                chunk = silence_padding + audio[start_ms:end_ms] + silence_padding

                chunk_ranges_s.append((start_ms / 1000.0, end_ms / 1000.0))
                chunk_path = temp_chunk_dir / f"chunk_{i + 1}.wav"
                print(
                    f"正在处理语音块 {i + 1}/{len(nonsilent_ranges_ms)} （该块起止时间：{SRTWriter._format_time(unpadded_start_ms / 1000.0)} --> {SRTWriter._format_time(unpadded_end_ms / 1000.0)}，持续时间：{(unpadded_end_ms - unpadded_start_ms) / 1000.0:.2f} 秒）"
                )
                chunk.export(chunk_path, format="wav")
                
                hyp, _ = model.transcribe(
                    [str(chunk_path)],
                    return_hypotheses=True,
                    verbose=False,
                )
                if hyp and hyp[0]:
                    ret = decode_hypothesis(model, hyp[0])
                    if ret.subwords:
                        cur_chunk_subs = [
                            Subword(
                                seconds=sub.seconds + start_ms / 1000.0,
                                token_id=sub.token_id,
                                token=sub.token,
                            )
                            for sub in ret.subwords
                        ] # 额外记录当前块的子词
                        chunk_subwords_list.append(cur_chunk_subs)

            if not chunk_subwords_list:
                all_subwords = []
            elif len(chunk_subwords_list) == 1:
                # 如果只有一个块，无需合并
                all_subwords = chunk_subwords_list[0]
            else:
                # 所有块处理完后，再执行一次合并
                print("所有语音块处理完毕，正在去重……")
                all_subwords = merge_overlap_dedup(chunk_subwords_list, chunk_ranges_s)
                print(f"语音块去重完毕")

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
                all_subwords, vad_chunk_end_times_s, args.no_chunk, build_map=args.kass  # 根据用户是否选择kass来决定是否构建映射
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

        if args.text or not file_output_requested:
            full_text = " ".join([seg.text for seg in all_segments])

        # 只有在用户完全没有指定任何输出参数时，才在控制台打印
        if not file_output_requested:
            print("\n识别结果（完整文本）：")
            print(full_text)
            print("=" * 70)
            print("提示：未指定输出参数，结果将打印至控制台")
            print("请使用 -text，-segment2srt，-kass 等参数将结果保存为文件")

        if args.text:
            output_path = output_dir / f"{base_name}.txt"
            with output_path.open("w", encoding="utf-8") as f: # Path 对象自带 open 方法
                f.write(full_text)
            print(f"完整的识别文本已保存为：{output_path}")

        if args.segment:
            output_path = output_dir / f"{base_name}.segments.txt"
            with output_path.open("w", encoding="utf-8") as f:
                writer = TextWriter(f)
                writer.write_header()  # 虽然为空，但保持接口一致性
                for seg in all_segments:
                    writer.write(seg)
            print(f"带时间戳的文本片段已保存为：{output_path}")

        if args.segment2srt:
            output_path = output_dir / f"{base_name}.srt"
            with output_path.open("w", encoding="utf-8") as f:
                writer = SRTWriter(f)
                writer.write_header()
                for seg in all_segments:
                    writer.write(seg)
            print(f"文本片段 SRT 字幕文件已保存为：{output_path}")

        if args.segment2vtt:
            output_path = output_dir / f"{base_name}.vtt"
            with output_path.open("w", encoding="utf-8") as f:
                writer = VTTWriter(f)
                writer.write_header()
                for seg in all_segments:
                    writer.write(seg)
            print(f"文本片段 WebVTT 字幕文件已保存为：{output_path}")

        if args.segment2tsv:
            output_path = output_dir / f"{base_name}.tsv"
            with output_path.open("w", encoding="utf-8") as f:
                writer = TSVWriter(f)
                writer.write_header()
                for seg in all_segments:
                    writer.write(seg)
            print(f"文本片段 TSV 文件已保存为：{output_path}")

        if args.subword:
            output_path = output_dir / f"{base_name}.subwords.txt"
            with output_path.open("w", encoding="utf-8") as f:
                for sub in all_subwords:
                    f.write(
                        f"[{SRTWriter._format_time(sub.seconds)}] {sub.token.replace(' ', '')}\n"
                    )
            print(f"带时间戳的所有子词信息已保存为：{output_path}")

        if args.subword2srt:
            output_path = output_dir / f"{base_name}.subwords.srt"
            with output_path.open("w", encoding="utf-8") as f:
                if subword_end_seconds:
                    for i, sub in enumerate(all_subwords):
                        start_time_str = SRTWriter._format_time(sub.seconds)
                        end_time_str = SRTWriter._format_time(subword_end_seconds[i])
                        text = sub.token

                        f.write(f"{i + 1}\n")
                        f.write(f"{start_time_str} --> {end_time_str}\n")
                        f.write(f"{text}\n\n")

            print(f"所有子词信息的 SRT 文件已保存为：{output_path}")

        if args.subword2json:
            output_path = output_dir / f"{base_name}.subwords.json"
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(
                    [{"token": sub.token, "timestamp": sub.seconds} for sub in all_subwords],
                    f, ensure_ascii=False, indent=4
                )
            print(f"所有子词信息的 JSON 文件已保存为：{output_path}")

        if args.kass:
            output_path = output_dir / f"{base_name}.ass"
            with output_path.open("w", encoding="utf-8-sig") as f:

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
                        karaoke_text += f"{{\\k{duration_cs}}}{sub.token}"

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
        # 恢复原始解码策略，以防后续有其他操作
        if 'model' in locals() and 'original_decoding_cfg' in locals():
             print("正在恢复原始解码策略……")
             model.change_decoding_strategy(original_decoding_cfg)

        print("=" * 70)
        # 安全地获取 ReazonSpeech 模型加载的起止时间，如果不存在则默认为 0
        asr_model_load_start = locals().get("asr_model_load_start", 0)
        asr_model_load_end = locals().get("asr_model_load_end", 0)
        if asr_model_load_end - asr_model_load_start > 0:
            print(
                f"ReazonSpeech模型加载耗时：{format_duration(asr_model_load_end - asr_model_load_start)}"
            )

        # 安全地获取 VAD 加载的起止时间，如果不存在则默认为 0
        vad_model_load_start = locals().get("vad_model_load_start", 0)
        vad_model_load_end = locals().get("vad_model_load_end", 0)
        # 安全地获取识别的起止时间，如果不存在则默认为 0
        recognition_start_time = locals().get("recognition_start_time", 0)
        recognition_end_time = locals().get("recognition_end_time", 0)
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
        if args.debug:
            print("调试模式已启用，临时文件和目录将被保留")
            print(f"临时 WAV 文件位于: {temp_wav_path}")
            if temp_chunk_dir and temp_chunk_dir.exists():
                print(f"临时分块目录位于: {temp_chunk_dir}")
                open_folder(temp_chunk_dir) # 调用新函数打开文件夹

        else:
            # --- 清理工作：删除临时的 WAV 文件 ---
            if temp_wav_path.exists():
                temp_wav_path.unlink()
                print(f"\n临时文件 '{temp_wav_path}' 已删除")

        # 使用 shutil.rmtree 来删除临时块目录及其所有内容
            if temp_chunk_dir and temp_chunk_dir.exists():
                shutil.rmtree(temp_chunk_dir)
                print(f"临时块目录 '{temp_chunk_dir.name}' 已删除")

if __name__ == "__main__":
    main()
