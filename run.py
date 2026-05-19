from asr import arg_parser, main as asr_main
import server
import contextlib
import io
import sys

def main():
    # 只做“是否传了 input_file”的判断，然后分发
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            args, _ = arg_parser().parse_known_args()
    except SystemExit:
        # API 启动参数的具体校验交给 server.py
        server.main(sys.argv[1:])
        return

    if args.input_file:
        # 有 input_file：走 asr.py 的 CLI 模式（argv=None）
        asr_main()
    else:
        # 没有 input_file：启动 API 服务
        server.main(sys.argv[1:])

if __name__ == "__main__":
    main()
