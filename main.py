"""LangChain 入门项目的命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from config import BASE_DIR


DEFAULT_KNOWLEDGE_FILE = BASE_DIR / "data" / "knowledge_base.txt"
SUPPORTED_COMMANDS = {"ask", "summary", "chat", "kb", "code", "config"}


class CommandRunner:
    """命令行功能调度器。"""

    def __init__(self) -> None:
        self.app: Any | None = None

    def get_app(self) -> Any:
        """延迟创建模型客户端，避免 config 和 help 命令加载模型依赖。"""

        if self.app is None:
            from llm_app import BeginnerLangChainApp

            self.app = BeginnerLangChainApp()
        return self.app

    def run(self, argv: list[str]) -> None:
        """根据参数执行对应命令。"""

        if not argv or argv[0] not in SUPPORTED_COMMANDS:
            self._run_legacy(argv)
            return

        command = argv[0]
        command_args = argv[1:]
        if command == "ask":
            self._run_ask(command_args)
        elif command == "summary":
            self._run_summary(command_args)
        elif command == "chat":
            self._run_chat(command_args)
        elif command == "kb":
            self._run_knowledge_base(command_args)
        elif command == "code":
            self._run_codebase_qa(command_args)
        elif command == "config":
            self._run_config(command_args)

    def _run_legacy(self, argv: list[str]) -> None:
        """兼容旧用法：python main.py 问题 或 python main.py --summary 文本。"""

        parser = argparse.ArgumentParser(description="LangChain 入门项目命令行工具")
        parser.add_argument(
            "question",
            nargs="*",
            help="要询问大模型的问题。例如：python main.py LangChain 是什么？",
        )
        parser.add_argument(
            "--summary",
            help="对一段文本做摘要。例如：python main.py --summary \"这是一段长文本\"",
        )
        args = parser.parse_args(argv)

        app = self.get_app()
        if args.summary:
            print(app.summarize(args.summary))
            return

        question = " ".join(args.question).strip() or "请用初学者能理解的话解释 LangChain 是什么。"
        print(app.ask(question))

    def _run_ask(self, argv: list[str]) -> None:
        """执行单次问答。"""

        parser = argparse.ArgumentParser(description="单次问答")
        parser.add_argument("question", nargs="*", help="要询问大模型的问题")
        args = parser.parse_args(argv)

        question = " ".join(args.question).strip() or "请用初学者能理解的话解释 LangChain 是什么。"
        print(self.get_app().ask(question))

    def _run_summary(self, argv: list[str]) -> None:
        """执行文本摘要。"""

        parser = argparse.ArgumentParser(description="文本摘要")
        parser.add_argument("text", nargs="+", help="需要摘要的文本")
        args = parser.parse_args(argv)

        print(self.get_app().summarize(" ".join(args.text)))

    def _run_chat(self, argv: list[str]) -> None:
        """启动多轮聊天。"""

        from langchain_core.messages import AIMessage, HumanMessage

        parser = argparse.ArgumentParser(description="多轮聊天，输入 exit 退出")
        parser.add_argument(
            "--max-history",
            type=int,
            default=12,
            help="最多保留多少条历史消息，默认 12 条",
        )
        args = parser.parse_args(argv)

        app = self.get_app()
        history: list[Any] = []
        print("已进入多轮聊天模式，输入 exit、quit 或 q 退出。")

        while True:
            message = input("你：").strip()
            if message.lower() in {"exit", "quit", "q"}:
                print("已退出多轮聊天。")
                return
            if not message:
                continue

            answer = app.chat_once(history, message)
            print(f"AI：{answer}")
            history.extend([HumanMessage(content=message), AIMessage(content=answer)])
            if len(history) > args.max_history:
                history = history[-args.max_history :]

    def _run_knowledge_base(self, argv: list[str]) -> None:
        """执行本地知识库问答。"""

        from knowledge_base import KnowledgeBase

        parser = argparse.ArgumentParser(description="基于本地文本知识库问答")
        parser.add_argument("question", nargs="+", help="要基于知识库询问的问题")
        parser.add_argument(
            "--file",
            default=str(DEFAULT_KNOWLEDGE_FILE),
            help="知识库文本文件路径",
        )
        parser.add_argument("--top-k", type=int, default=3, help="检索片段数量，默认 3")
        args = parser.parse_args(argv)

        question = " ".join(args.question)
        knowledge_base = KnowledgeBase.from_file(Path(args.file))
        chunks = knowledge_base.retrieve(question, top_k=args.top_k)
        context = KnowledgeBase.format_context(chunks)
        print(self.get_app().answer_with_context(question, context))

    def _run_codebase_qa(self, argv: list[str]) -> None:
        """执行代码库问答。"""

        from codebase_qa import CodebaseIndex

        parser = argparse.ArgumentParser(description="基于当前代码库问答")
        parser.add_argument("question", nargs="*", help="要询问代码库的问题")
        parser.add_argument(
            "--path",
            default=str(BASE_DIR),
            help="代码库目录，默认当前项目目录",
        )
        parser.add_argument("--top-k", type=int, default=5, help="检索片段数量，默认 5")
        parser.add_argument("--chunk-lines", type=int, default=90, help="每个代码片段的最大行数")
        parser.add_argument("--overlap-lines", type=int, default=15, help="相邻片段重叠行数")
        parser.add_argument("--show-context", action="store_true", help="显示检索到的代码上下文")
        parser.add_argument("--list-files", action="store_true", help="只列出会被索引的文件")
        args = parser.parse_args(argv)

        index = CodebaseIndex.from_directory(
            root=Path(args.path),
            chunk_lines=args.chunk_lines,
            overlap_lines=args.overlap_lines,
        )

        if args.list_files:
            print("已索引文件：")
            for file_path in index.indexed_files:
                print(f"- {file_path}")
            return

        question = " ".join(args.question).strip()
        if not question:
            raise ValueError("请提供代码库问题，例如：python main.py code main.py 做什么")

        chunks = index.retrieve(question, top_k=args.top_k)
        context = CodebaseIndex.format_context(chunks)
        if args.show_context:
            print("检索到的代码上下文：")
            print(context)
            print("\n模型回答：")
        print(self.get_app().answer_codebase_question(question, context))

    def _run_config(self, argv: list[str]) -> None:
        """查看当前模型配置。"""

        from config import load_settings

        parser = argparse.ArgumentParser(description="查看当前模型配置")
        parser.parse_args(argv)

        settings = load_settings()
        print("当前配置：")
        print(f"LLM_PROVIDER={settings.provider}")
        print(f"LLM_BASE_URL={settings.base_url or 'OpenAI 官方默认地址'}")
        print(f"LLM_MODEL={settings.model}")
        print(f"LLM_TEMPERATURE={settings.temperature}")
        print(f"LLM_API_KEY={_mask_secret(settings.api_key)}")


def _mask_secret(secret: str) -> str:
    """隐藏敏感配置。"""

    if not secret:
        return "未配置"
    if len(secret) <= 8:
        return "已配置"
    return f"{secret[:4]}...{secret[-4:]}"


def _is_error(error: Exception, class_names: set[str]) -> bool:
    """按异常类名判断第三方库错误，避免帮助命令提前加载 OpenAI 依赖。"""

    return any(error_class.__name__ in class_names for error_class in error.__class__.mro())


def _print_api_status_error(error: Exception, app: Any | None) -> None:
    """打印模型服务返回的状态码错误。"""

    message = str(error)
    print("模型服务调用失败。")
    if app:
        print(f"当前模型服务地址：{app.settings.base_url or 'OpenAI 官方默认地址'}")
        print(f"当前模型名称：{app.settings.model}")

    if "model_not_found" in message or "No available channel for model" in message:
        print("失败原因：当前模型名称在这个服务中不可用，或你的账号无权使用该模型。")
        print("处理方式：请把 .env 中的 LLM_MODEL 改成服务商支持的模型名称。")
        return

    print(f"HTTP 状态码：{getattr(error, 'status_code', '未知')}")
    print(f"错误信息：{message}")


def main() -> None:
    """程序主入口。"""

    runner = CommandRunner()
    try:
        runner.run(sys.argv[1:])
    except SystemExit:
        raise
    except Exception as error:
        if _is_error(error, {"APIStatusError"}):
            _print_api_status_error(error, runner.app)
            raise SystemExit(1) from error
        if _is_error(error, {"APIConnectionError", "APITimeoutError"}):
            print("无法连接模型服务，请检查 LLM_BASE_URL、网络连接和服务是否启动。")
            print(f"错误信息：{error}")
            raise SystemExit(1) from error
        if _is_error(error, {"APIError"}):
            print("模型服务调用失败，请检查服务地址、模型名称和 API Key。")
            print(f"错误信息：{error}")
            raise SystemExit(1) from error
        if isinstance(error, (RuntimeError, ValueError, FileNotFoundError)):
            print(f"配置或输入错误：{error}")
            raise SystemExit(1) from error
        raise


if __name__ == "__main__":
    main()
