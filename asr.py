import argparse
import os
import tempfile
import shutil
from pydub import AudioSegment
from pydub.silence import detect_nonsilent # 使用更精确的静音侦测
from reazonspeech.nemo.asr import load_model, transcribe, audio_from_path, TranscribeConfig
# 导入数据类，用于校正时间戳
from reazonspeech.nemo.asr.interface import Segment, Subword

def format_srt_time(seconds):
    """将秒数格式化为 SRT 时间戳格式 (HH:MM:SS,ms)"""
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    seconds = milliseconds // 1_000
    milliseconds %= 1_000

    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def main():
    # --- 设置命令行参数解析 ---
    parser = argparse.ArgumentParser(
        description="使用 ReazonSpeech 模型进行语音识别，并按指定格式输出结果。基于静音的智能分块方式识别长音频，以保证准确率并解决显存问题。"
    )
    
    # 添加一个必须的位置参数：音频文件路径
    parser.add_argument(
        "input_file", 
        help="需要进行语音识别的音频文件路径"
    )

    # --- --no-chunk 参数 ---
    parser.add_argument(
        "--no-chunk", 
        action="store_true", 
        help="禁用智能分块功能，一次性处理整个音频文件"
    )

    # 创建一个互斥的参数组，因为一次只能选择一种输出格式
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "-text", 
        action="store_true", 
        help="仅输出完整的识别文本并保存到 .txt 文件 (默认选项)"
    )
    output_group.add_argument(
        "-segment", 
        action="store_true", 
        help="输出带时间戳的文本片段 (Segment)并保存到 .segments.txt 文件"
    )
    output_group.add_argument(
        "-segment2srt", 
        action="store_true", 
        help="将文本片段 (Segment) 转换为 SRT 字幕文件并保存"
    )
    output_group.add_argument(
        "-subword", 
        action="store_true", 
        help="输出所有的子词 (Subword) 及其时间戳并保存到保存到 .subwords.txt 文件"
    )
    output_group.add_argument(
        "-subword2srt", 
        action="store_true", 
        help="将子词 (Subword) 转换为 SRT 字幕文件并保存"
    )

    # --- 智能分块参数 ---
    parser.add_argument("--min_silence_len", type=int, default=700, help="最小静音时长（毫秒）。")
    parser.add_argument("--silence_thresh", type=int, default=-50, help="静音音量阈值（dBFS）。")
    parser.add_argument("--keep_silence", type=int, default=300, help="分割块前后保留的静音时长（毫秒）。")

    args = parser.parse_args()

    # --- 准备路径和临时文件 ---
    input_path = args.input_file
    output_dir = os.path.dirname(os.path.abspath(input_path))
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    # 创建一个临时的 WAV 文件
    # delete=False 确保我们可以在 with 块外使用它，最后手动删除
    temp_wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav_path = temp_wav_file.name
    temp_wav_file.close() # 关闭文件句柄，以便 ffmpeg 可以写入

    temp_chunk_dir = tempfile.mkdtemp()


    # --- 执行核心的语音识别流程 ---
    try:
        # --- FFmpeg 预处理：将输入文件转换为标准 WAV ---
        print(f"正在转换输入文件 '{input_path}' 为临时 WAV 文件...")
        audio = AudioSegment.from_file(input_path)
        # 转换为单声道，16kHz采样率，这是ASR模型的标准格式
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(temp_wav_path, format="wav")
        print("转换完成。")

        # --- 加载模型 (只需一次) ---
        print("正在加载模型...")
        model = load_model(device='cuda')
        print("模型加载完成。")

        # --- 逐块识别并校正时间戳 ---
        all_segments = []
        all_subwords = []

        if args.no_chunk:
            # --- 不分块的逻辑 ---
            print("检测到 --no-chunk 参数，将一次性处理整个文件...")
            audio_data = audio_from_path(temp_wav_path)
            ret = transcribe(model, audio_data, config=TranscribeConfig(verbose=True)) # 长时间任务显示进度条
            all_segments = ret.segments
            all_subwords = ret.subwords

        else:
            # --- 智能分块的逻辑 ---
            print("正在侦测音频中的语音活动以进行智能分块...")
            wav_audio = AudioSegment.from_wav(temp_wav_path)
            
            nonsilent_ranges = detect_nonsilent(
                wav_audio,
                min_silence_len=args.min_silence_len,
                silence_thresh=args.silence_thresh,
            )
            if not nonsilent_ranges:
                print("警告：未侦测到语音活动。将尝试处理整个文件。")
                nonsilent_ranges = [[0, len(wav_audio)]]
                
            print(f"音频被智能分割成 {len(nonsilent_ranges)} 个块。")

            for i, time_range in enumerate(nonsilent_ranges):
                start_ms, end_ms = time_range
                start_ms = max(0, start_ms - args.keep_silence)
                end_ms = min(len(wav_audio), end_ms + args.keep_silence)
                chunk = wav_audio[start_ms:end_ms]
                time_offset_s = start_ms / 1000.0

                chunk_path = os.path.join(temp_chunk_dir, f"chunk_{i}.wav")
                print(f"正在处理块 {i+1}/{len(nonsilent_ranges)} (时间: {format_srt_time(time_offset_s)})...")
                chunk.export(chunk_path, format="wav")
                
                audio_chunk_data = audio_from_path(chunk_path)
                ret = transcribe(model, audio_chunk_data, config=TranscribeConfig(verbose=False))
                
                if ret.segments:
                    for seg in ret.segments:
                        all_segments.append(Segment(
                            start_seconds=seg.start_seconds + time_offset_s,
                            end_seconds=seg.end_seconds + time_offset_s,
                            text=seg.text
                        ))
                if ret.subwords:
                    for sub in ret.subwords:
                        all_subwords.append(Subword(
                            seconds=sub.seconds + time_offset_s,
                            token_id=sub.token_id,
                            token=sub.token
                        ))

        # --- 根据参数生成输出文件 ---
        print("-" * 20)
        print("识别完成，正在生成输出文件...")

        # 如果没有任何参数，则默认在控制台打印全文
        no_output_flag = not any([args.text, args.segment, args.segment2srt, args.subword, args.subword2srt])

        if args.text or no_output_flag:
            full_text = " ".join([seg.text for seg in all_segments])
            if args.text:
                output_path = os.path.join(output_dir, f"{base_name}.txt")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(full_text)
                print(f"完整文本已保存到: {output_path}")
            if no_output_flag:
                print("\n识别结果 (完整文本):")
                print(full_text)

        if args.segment:
            output_path = os.path.join(output_dir, f"{base_name}.segments.txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                for seg in all_segments:
                    f.write(f"[{format_srt_time(seg.start_seconds)} --> {format_srt_time(seg.end_seconds)}] {seg.text}\n")
            print(f"文本片段已保存到: {output_path}")

        if args.segment2srt:
            output_path = os.path.join(output_dir, f"{base_name}.srt")
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, seg in enumerate(all_segments, 1):
                    f.write(f"{i}\n")
                    f.write(f"{format_srt_time(seg.start_seconds)} --> {format_srt_time(seg.end_seconds)}\n")
                    f.write(f"{seg.text}\n\n")
            print(f"SRT 字幕文件已保存到: {output_path}")

        if args.subword:
            output_path = os.path.join(output_dir, f"{base_name}.subwords.txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                for sub in all_subwords:
                    f.write(f"[{format_srt_time(sub.seconds)}] {sub.token.replace(' ', '')}\n")
            print(f"子词信息已保存到: {output_path}")
        
        if args.subword2srt:
            output_path = os.path.join(output_dir, f"{base_name}.subwords.srt")
            with open(output_path, 'w', encoding='utf-8') as f:
                if len(all_subwords) > 1:
                    for i, sub in enumerate(all_subwords[:-1], 1):
                        f.write(f"{i}\n")
                        f.write(f"{format_srt_time(sub.seconds)} --> {format_srt_time(all_subwords[i].seconds)}\n")
                        f.write(f"{sub.token.replace(' ', '')}\n\n")
                if all_subwords:
                    last_sub = all_subwords[-1]
                    f.write(f"{len(all_subwords)}\n")
                    f.write(f"{format_srt_time(last_sub.seconds)} --> {format_srt_time(last_sub.seconds + 0.2)}\n")
                    f.write(f"{last_sub.token.replace(' ', '')}\n\n")
            print(f"子词 SRT 文件已保存到: {output_path}")

    finally:
        # --- 清理工作：删除临时的 WAV 文件 ---
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
            print(f"\n临时文件 '{temp_wav_path}' 已删除。")

        # 使用 shutil.rmtree 来删除临时块目录及其所有内容
        if os.path.exists(temp_chunk_dir):
            shutil.rmtree(temp_chunk_dir)
            print(f"临时块目录 '{os.path.basename(temp_chunk_dir)}' 已删除。")

if __name__ == "__main__":
    main()