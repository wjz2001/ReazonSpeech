import argparse
import concurrent.futures
import os
import contextlib
import io
import re
import shlex
import shutil
import socket
import sys
import tempfile
import traceback
import uvicorn
from typing import Optional, Annotated
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
# 从 asr 中获取：主入口 + 构建 ArgumentParser 的函数
from asr import main as asr_main, arg_parser, get_asr_model

def get_local_ip():
    """
    获取本机在局域网内可用的 IPv4 地址。
    通过连到一个公网地址(不会真的发数据出去)来反查本机使用的 IP
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # 这里的 8.8.8.8 只是用来获取路由信息，不会真的去访问
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()

def find_free_port(host, start_port):
    """从 start_port 开始往上找一个空闲端口"""
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                port += 1

def get_args_info():
    """
    从 asr.arg_parser() 自动分析参数信息，区分：
    - config 参数：带有 --xxx 的选项（双横线）
    - output 参数：带有 -xxx 的选项（单横线）
    """
    config: dict[str, dict] = {}
    output: dict[str, dict] = {}
    config_flag_to_dest: dict[str, str] = {}
    boolean_config_flags: set[str] = set()

    for action in arg_parser()._actions:
        # 位置参数（如 input_file）没有 option_strings，跳过
        if not action.option_strings:
            continue

        # 跳过 help
        if isinstance(action, argparse._HelpAction):
            continue

        long_flags = [
            opt for opt in action.option_strings
            if opt.startswith("--")
        ]
        short_flags = [
            opt for opt in action.option_strings
            if opt.startswith("-") and not opt.startswith("--")
        ]

        if long_flags:
            config[action.dest] = {
                "flags": long_flags,
            }
            for flag in long_flags:
                config_flag_to_dest[flag] = action.dest
            if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
                boolean_config_flags.update(long_flags)

        if short_flags:
            output[action.dest] = {
                "flags": short_flags,
            }

    return {
        "config": config,
        "output": output,
        "config_flag_to_dest": config_flag_to_dest,
        "boolean_config_flags": boolean_config_flags,
    }

# 初始化参数信息
ARGS_INFO = get_args_info()

CONFIG_INFO = ARGS_INFO["config"]               # dest -> {...}
OUTPUT_INFO = ARGS_INFO["output"]               # dest -> {...}

CONFIG_DESTS = set(CONFIG_INFO.keys())          # {"no_chunk", "beam", ...}
OUTPUT_DESTS = set(OUTPUT_INFO.keys())          # {"text", "segment2srt", ...}
CONFIG_FLAG_TO_DEST = ARGS_INFO["config_flag_to_dest"]   # {"--no-chunk": "no_chunk", "--beam": "beam", ...}
CONFIG_FLAGS = set(CONFIG_FLAG_TO_DEST)                  # {"--no-chunk", "--beam", ...}
BOOLEAN_CONFIG_FLAGS = ARGS_INFO["boolean_config_flags"] # {"--debug", "--no-chunk", ...}
OPENAI_RESPONSE_FORMATS = {"json", "text", "srt", "verbose_json", "vtt"}
STARTUP_CONFIG: dict[str, object] = {}

# response_format: item(dest 名) -> CLI 短选项 flag
# 例如 "text" -> "-text", "segment2srt" -> "-segment2srt"
FMT_DEST_TO_FLAG: dict[str, str] = {
    dest: OUTPUT_INFO[dest]["flags"][0]
    for dest in OUTPUT_DESTS
}

class ConfigArgError(ValueError):
    pass


def _parse_config_argv(argv: list[str], source: str) -> dict[str, object]:
    if not argv:
        return {}

    def _normalize_config_argv_for_validation() -> tuple[list[str], set[str]]:
        normalized_argv: list[str] = []
        disabled_dests: set[str] = set()

        for token in argv:
            if token.startswith("-") and not token.startswith("--"):
                raise ConfigArgError(f"{source} 中不允许使用单横线参数 '{token}'")

            head = token.split("=", 1)[0]
            lower_head = head.lower()

            if lower_head.startswith("--no_"):
                real_flag = "--" + head[len("--no_"):]
                real_flag = real_flag.lower()
                dest = CONFIG_FLAG_TO_DEST.get(real_flag)
                if dest is None:
                    raise ConfigArgError(f"{source} 中包含未知配置参数 '{token}'")
                if real_flag not in BOOLEAN_CONFIG_FLAGS:
                    raise ConfigArgError(f"{source} 中 '{token}' 只能用于布尔参数")

                disabled_dests.add(dest)
                normalized_argv.append(real_flag + token[len(head):])
            else:
                normalized_argv.append(lower_head + token[len(head):])

        return normalized_argv, disabled_dests

    normalized_argv, disabled_dests = _normalize_config_argv_for_validation()
    parser = arg_parser()

    try:
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            args = parser.parse_args(normalized_argv)
    except SystemExit:
        message = stderr.getvalue().strip() or "参数解析失败"
        raise ConfigArgError(f"{source} 参数错误：{message}")

    if args.input_file:
        raise ConfigArgError(f"{source} 中只允许使用 -- 配置参数，不允许包含输入文件或普通文本：'{args.input_file}'")

    present_dests: set[str] = set()
    for token in argv:
        if token.startswith("--no_"):
            continue
        if token.startswith("--"):
            head = token.split("=", 1)[0].lower()
            dest = CONFIG_FLAG_TO_DEST.get(head)
            if dest is not None:
                present_dests.add(dest)

    config: dict[str, object] = {
        dest: getattr(args, dest)
        for dest in present_dests
    }
    for dest in disabled_dests:
        config[dest] = False

    return config


def _config_error_response(message: str, param: str | None = None):
    return HTTPException(
        status_code=400,
        detail={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": param,
                "code": "invalid_config_arg",
            }
        },
    )

def _parse_prompt_config(prompt: str, source: str, error_mode: str) -> dict[str, object]:
    if not prompt.strip():
        return {}
    try:
        tokens = shlex.split(prompt)
    except ValueError as e:
        message = f"{source} CLI 解析失败：{e}"
        if error_mode == "http":
            raise _config_error_response(message, source)
        print(f"【提示词错误】{message}")
        return {}

    try:
        return _parse_config_argv(tokens, source)
    except ConfigArgError as e:
        if error_mode == "exit":
            print(f"【参数错误】{e}", file=sys.stderr)
            sys.exit(2)
        if error_mode == "http":
            raise _config_error_response(str(e), source)
        print(f"【提示词错误】{e}")
        return {}

def build_argv_openai(
    input_path: str,
    config_argv: list[str],
    rf: str,  # json/text/srt/verbose_json/vtt 之一，函数内统一转小写
    timestamp_granularities: Optional[list[str]],
) -> list[str]:
    """
    OpenAI 风格的 response_format -> asr.py CLI 输出选项：
      json         -> -text
      text         -> -text
      srt          -> -segment2srt
      vtt          -> -segment2vtt
      verbose_json -> -text + (根据 timestamp_granularities 加 -segment/-subword)
    """
    rf = rf.lower()
    argv: list[str] = [input_path]
    argv.extend(config_argv)

    cli_dests: set[str] = set()

    if rf in ("json", "text"):
        cli_dests.add("text")

    elif rf == "srt":
        cli_dests.add("segment2srt")

    elif rf == "vtt":
        cli_dests.add("segment2vtt")

    elif rf == "verbose_json":
        cli_dests.add("text")
        granularities = timestamp_granularities or ["segment"]  # 官方默认 segment

        if set(granularities) - {"word", "segment"}:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            "timestamp_granularities 只支持 'word' 和 'segment'"
                        ),
                        "type": "invalid_request_error",
                        "param": "timestamp_granularities",
                        "code": "invalid_timestamp_granularity",
                    }
                },
            )
        if "segment" in granularities:
            cli_dests.add("segment")
        if "word" in granularities:
            cli_dests.add("subword")

    # 把 dest 转成 CLI 短选项
    for dest in cli_dests:
        if dest not in OUTPUT_DESTS:
            # 这是内部配置错误，不是用户错误
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "message": f"输出类型 '{dest}' 未在输出参数中找到",
                        "type": "server_error",
                        "param": None,
                        "code": "missing_output_dest",
                    }
                },
            )
        argv.append(FMT_DEST_TO_FLAG[dest].lower())

    return argv

def build_argv_legacy(
    input_path: str,
    config_argv: list[str],
    response_format: str,
) -> list[str]:
    """
    旧的 ReazonSpeech 风格：直接把 response_format 当作空格分隔的输出参数，
    例如 "-text -segment2srt"
    """
    argv: list[str] = [input_path]
    argv.extend(config_argv)

    output_flags = {
        flag.lower()
        for output in OUTPUT_INFO.values()
        for flag in output["flags"]
    }
    try:
        cli_flags = shlex.split(response_format or "")
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"response_format 解析失败：{e}",
                    "type": "invalid_request_error",
                    "param": "response_format",
                    "code": "invalid_response_format",
                }
            },
        )

    for flag in cli_flags:
        flag = flag.lower()
        if flag not in output_flags:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            f"response_format 中包含未知输出参数 '{flag}'，允许的输出参数有："
                            f"{' '.join(sorted(output_flags))}"
                        ),
                        "type": "invalid_request_error",
                        "param": "response_format",
                        "code": "unknown_output_flag",
                    }
                },
            )
        argv.append(flag)

    return argv

app = FastAPI(title="ReazonSpeech API")

@app.exception_handler(HTTPException) # 捕获 raise HTTPException
async def openai_http_exception_handler(_: Request, exc: HTTPException):
    """
    将错误拦截下来，强制转换成 OpenAI API 风格错误格式：
    {
      "error": {
        "message": "...",
        "type": "...",
        "param": "...",
        "code": "..."
      }
    }
    """
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        content = exc.detail
    else:
        content = {
            "error": {
                "message": str(exc.detail),
                "type": "server_error" if exc.status_code >= 500 else "invalid_request_error",
                "param": None,
                "code": None,
            }
        }
    return JSONResponse(status_code=exc.status_code, content=content)

@app.post("/v1/audio/transcriptions")
def transcriptions(
    file: UploadFile = File(...),
    prompt: str = Form(""), # CLI 字符串或 JSON
    response_format: str = Form(""),
    timestamp_granularities: Annotated[
        Optional[list[str]],
        Form(alias="timestamp_granularities[]")
    ] = None,
):
    # 保存上传的文件到临时路径
    input_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or "")[1] or ".wav") as tmp:
            shutil.copyfileobj(file.file, tmp)
            input_path = tmp.name
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"无法保存上传的音频文件: {e}",
                    "type": "server_error",
                    "param": "file",
                    "code": "file_save_failed",
                }
            },
        )

    try:
        # -------------------------
        # 判定使用哪种 response_format 语义：OpenAI 风格 or 旧 dest 风格（禁止混用）
        # -------------------------
        tokens = [
            t
            for t in re.split(r"[,\s]+", (response_format or "").strip()) 
            if t
        ]
        lower_tokens = [t.lower() for t in tokens]

        if not tokens:
            mode = "openai"
            rf = "text"
        else:
            if all(t in OPENAI_RESPONSE_FORMATS for t in lower_tokens):
                if len(tokens) == 1:
                    mode = "openai"
                    rf = tokens[0]
                else:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": {
                                "message": (
                                    "json / text / srt / verbose_json / vtt 作为 response_format 值时只能单选"
                                ),
                                "type": "invalid_request_error",
                                "param": "response_format",
                                "code": "multiple_openai_response_formats",
                            }
                        },
                    )
            elif any(t in OPENAI_RESPONSE_FORMATS for t in lower_tokens):
                # 同时包含官方值和自定义 dest，显式禁止混用
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": (
                                "response_format 不能同时混用 json / text / srt / verbose_json / vtt 和 ReazonSpeech 输出参数，只能二选一"
                            ),
                            "type": "invalid_request_error",
                            "param": "response_format",
                            "code": "mixed_response_format_styles",
                        }
                    },
                )
            else:
                # 全是自定义 dest 名 -> 旧模式
                mode = "legacy"
                rf = None

        # timestamp_granularities 只在 OpenAI + verbose_json 下合法
        if timestamp_granularities and not (mode == "openai" and rf == "verbose_json"):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "timestamp_granularities 只能在 response_format=verbose_json 时使用",
                        "type": "invalid_request_error",
                        "param": "timestamp_granularities",
                        "code": "timestamp_granularities_requires_verbose_json",
                    }
                },
            )

        # 组装 argv 并调用 asr.main(argv=...)
        # argv 有值 => asr.main 会进入 API 模式，并返回内存中的结果 dict
        fallback_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "reazonspeechprompt.txt"
        )
        file_config: dict[str, object] = {}
        if os.path.isfile(fallback_path):
            try:
                with open(fallback_path, "r", encoding="utf-8-sig") as f:
                    file_prompt = f.read().strip()
            except Exception as e:
                raise _config_error_response(f"读取 reazonspeechprompt.txt 失败：{e}", "reazonspeechprompt.txt")
            file_config = _parse_prompt_config(file_prompt, "reazonspeechprompt.txt", "http")

        request_config = _parse_prompt_config(prompt, "prompt", "silent")
        merged_config = {
            **STARTUP_CONFIG,
            **file_config,
            **request_config,
        }
        config_argv: list[str] = []
        for dest, value in merged_config.items():
            if value is False or value is None:
                continue
            flag = CONFIG_INFO[dest]["flags"][0].lower()
            if isinstance(value, bool):
                if value:
                    config_argv.append(flag)
            else:
                config_argv.extend([flag, str(value)])

        if mode == "openai":
            argv = build_argv_openai(input_path, config_argv, rf, timestamp_granularities)
        else:
            argv = build_argv_legacy(input_path, config_argv, response_format)

        result = asr_main(argv)

        if result is None:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "message": "识别失败或未返回结果",
                        "type": "server_error",
                        "param": None,
                        "code": "empty_result",
                    }
                },
            )

        # asr.py 内部如果返回了 {"error": {...}}，直接转成 HTTP 错误
        if "error" in result:
            raise HTTPException(status_code=500 if result["error"]["type"] == "server_error" else 400, detail=result)

        # 直接把 asr_main 的结果整包 JSON 返回
        if mode == "legacy":
            return JSONResponse(result)

        # OpenAI 模式：
        if rf == "json":
            return JSONResponse({
                "text": result["text"] or "",
                "usage": {
                    "type": "tokens",
                    "input_tokens": 0,
                    "input_token_details": {
                        "text_tokens": 0,
                        "audio_tokens": 0
                    },
                    "output_tokens": 0,
                    "total_tokens": 0
                },
                "logprobs": None  # JSON 中会变成 null
            })

        elif rf == "text":
            # 官方 text：返回纯文本，从 -text 拿数据
            return PlainTextResponse(result["text"] or "", media_type="text/plain; charset=utf-8")

        elif rf == "srt":
            # 官方 srt：返回字幕纯文本，从 -segment2srt 拿数据
            return PlainTextResponse(result["segment2srt"] or "", media_type="application/x-subrip")

        elif rf == "vtt":
            # 官方 vtt：返回 WebVTT 纯文本，从 -segment2vtt 拿数据
            return PlainTextResponse(result["segment2vtt"] or "", media_type="text/vtt")

        elif rf == "verbose_json":
            # 根据 segment / subword 的 end 推断整段时长（秒）
            duration = result["duration"] or 0.0

            # 顶层字段
            verbose: dict = {
                "task": "transcribe",
                "language": "japanese",
                "duration": duration,
                "text": result["text"] or "",
            }

            # 根据 timestamp_granularities 决定是否返回 segments / words
            granularities = timestamp_granularities or ["segment"]

            if "segment" in granularities:
                segments = []
                for idx, seg in enumerate(result["segment"] or []):
                    segments.append(
                        {
                            "id": idx,
                            "seek": 0, # 兼容 Whisper 风格，统一设为 0
                            "start": seg["start"] or 0.0,
                            "end": seg["end"] or 0.0,
                            "text": seg["text"] or "",
                            "tokens": seg["tokens_id"],
                            # 占位
                            "temperature": 0.0,
                            "avg_logprob": 0.0,
                            "compression_ratio": 0.0,
                            "no_speech_prob": 0.0,
                        }
                    )
                if segments:
                    verbose["segments"] = segments

            if "word" in granularities:
                words = []
                for w in result["subword"] or []:
                    words.append(
                        {
                            "word": w["token"] or "",
                            "start": w["start"] or 0.0,
                            "end": w["end"] or 0.0,
                        }
                    )
                if words:
                    verbose["words"] = words

            # usage：写整段音频实际时长（秒）
            verbose["usage"] = {
                "type": "duration",
                "seconds": duration,
            }

            return JSONResponse(verbose)

    except SystemExit as e:
        # argparse 报错
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "asr.py 参数解析失败，请检查 prompt 和 response_format",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": f"argparse_exit_{e.code}",
                }
            },
        )

    except HTTPException:
        # 捕获主动抛出的 HTTPException，直接原样抛出
        raise

    except Exception as e:
        print(f"Internal Error: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": "语音识别服务内部错误",
                    "type": "server_error",
                    "param": None,
                    "code": "internal_error",
                }
            },
        )

    finally:
        if input_path and os.path.exists(input_path):
            # 删除临时文件
            try:
                os.remove(input_path)
            except Exception:
                pass


def main(startup_argv: Optional[list[str]] = None):
    """给 asr.py 调用的入口：预加载模型+选择端口 + 启动 uvicorn"""
    global STARTUP_CONFIG

    print("API 服务启动中……")

    host = "0.0.0.0"
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_parse_config_argv, startup_argv or [], "API 启动参数"): "参数验证",
            executor.submit(get_asr_model): "模型加载",
            executor.submit(lambda: (get_local_ip(), find_free_port(host, 8888))): "服务地址准备",
        }

        done, _ = concurrent.futures.wait(
            futures,
            return_when=concurrent.futures.FIRST_EXCEPTION,
        )
        for future in done:
            exc = future.exception()
            if exc is None:
                continue
            task_name = futures[future]
            if isinstance(exc, ConfigArgError):
                print(f"【参数错误】{exc}", file=sys.stderr, flush=True)
                os._exit(2)
            print(f"【启动失败】{task_name}失败：{exc}", file=sys.stderr, flush=True)
            traceback.print_exception(type(exc), exc, exc.__traceback__)
            os._exit(1)

        concurrent.futures.wait(futures)
        for future, task_name in futures.items():
            exc = future.exception()
            if exc is None:
                continue
            print(f"【启动失败】{task_name}失败：{exc}", file=sys.stderr, flush=True)
            traceback.print_exception(type(exc), exc, exc.__traceback__)
            os._exit(1)

        startup_config_future, _, service_addr_future = futures.keys()
        STARTUP_CONFIG = startup_config_future.result()
        local_ip, port = service_addr_future.result()

    if port != 8888:
        print(f"端口 {8888} 已被占用，自动切换到端口 {port}")

    print("启动 ReazonSpeech API 服务：")
    print(f"  本机访问：http://127.0.0.1:{port}/v1/audio/transcriptions")
    print(f"  本机或局域网设备访问：http://{local_ip}:{port}/v1/audio/transcriptions")
    print(f"  服务监听地址：http://{host}:{port}/v1/audio/transcriptions")
    uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":
    print("请使用: reazonspeech 命令启动 API 服务，勿要直接运行本文件")
    sys.exit(1)
