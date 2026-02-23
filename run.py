from asr import arg_parser, main as asr_main

def main():
    # 判断是否传了 input_file 参数，然后分发
    args, _ = arg_parser().parse_known_args()

    if args.input_file:
        # 有 input_file：走 asr.py 的 CLI 模式（argv=None）
        asr_main()
    else:
        # 没有 input_file：启动 API 服务
        import server
        server.main()

if __name__ == "__main__":
    main()