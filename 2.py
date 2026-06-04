"""独立于命令行入口的最小可运行演示脚本。

运行方式：
    python 2.py
"""

from __future__ import annotations

from llm_app import BeginnerLangChainApp


DEMO_TEXT = """
LangChain 是一个用于开发大语言模型应用的框架。它提供 Prompt 模板、模型调用、
输出解析、链式组合等能力，帮助开发者把 LLM 接入真实业务流程。
"""


def main() -> None:
    app = BeginnerLangChainApp()

    print("=== Q&A 示例 ===")
    print(app.ask("LangChain 的 Chain 是什么？请举一个生活化例子。"))

    print("\n=== 摘要示例 ===")
    print(app.summarize(DEMO_TEXT))


if __name__ == "__main__":
    main()
