"""代码问答 LLM 生成层。"""

from __future__ import annotations

from collections.abc import Iterator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import Settings, load_settings


class CodeAnswerGenerator:
    """负责基于代码上下文生成答案。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.llm = ChatOpenAI(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
            model=self.settings.model,
            temperature=self.settings.temperature,
        )

    def answer(self, question: str, context: str) -> str:
        """一次性生成完整答案。"""

        response = self.llm.invoke(self._messages(question, context))
        return str(response.content)

    def stream_answer(self, question: str, context: str) -> Iterator[str]:
        """流式生成答案。"""

        for chunk in self.llm.stream(self._messages(question, context)):
            content = getattr(chunk, "content", "")
            if content:
                yield str(content)

    def _messages(self, question: str, context: str) -> list[SystemMessage | HumanMessage]:
        """构造代码问答提示词。"""

        return [
            SystemMessage(
                content=(
                    "你是资深代码库问答助手。只能基于提供的代码片段回答问题。"
                    "回答必须包含结论、关键依据、引用的文件路径和行号。"
                    "如果上下文不足以回答，请明确说明缺少哪些文件或信息。"
                )
            ),
            HumanMessage(
                content=(
                    "代码上下文如下：\n\n"
                    f"{context}\n\n"
                    f"用户问题：{question}\n\n"
                    "请用中文回答。"
                )
            ),
        ]
