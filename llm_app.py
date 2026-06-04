"""小型 LangChain 应用示例。

本模块集中管理 LangChain 相关逻辑，让入口脚本保持简洁。
"""

from __future__ import annotations

from collections.abc import Iterable

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from config import Settings, load_settings


class BeginnerLangChainApp:
    """一个包含提示词模板和可运行链的最小 LangChain 应用。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.llm = self._build_llm()
        self.output_parser = StrOutputParser()

    def _build_llm(self) -> ChatOpenAI:
        """构建兼容 OpenAI 接口的聊天模型客户端。"""

        return ChatOpenAI(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
            model=self.settings.model,
            temperature=self.settings.temperature,
        )

    def build_qa_chain(self) -> Runnable:
        """创建一个简单的问答链。"""

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一名表达简洁的技术导师。请用中文回答，并给出清晰示例。",
                ),
                ("human", "问题：{question}"),
            ]
        )
        return prompt | self.llm | self.output_parser

    def ask(self, question: str) -> str:
        """回答一个用户问题。"""

        question = question.strip()
        if not question:
            raise ValueError("问题不能为空。")

        chain = self.build_qa_chain()
        return chain.invoke({"question": question})

    def summarize(self, text: str) -> str:
        """使用专门的提示词总结一段文本。"""

        text = text.strip()
        if not text:
            raise ValueError("文本不能为空。")

        prompt = ChatPromptTemplate.from_template(
            "请用 3 个要点总结下面的内容，并保留关键信息：\n\n{text}"
        )
        chain = prompt | self.llm | self.output_parser
        return chain.invoke({"text": text})

    def answer_with_context(self, question: str, context: str) -> str:
        """基于给定上下文回答问题。"""

        question = question.strip()
        context = context.strip()
        if not question:
            raise ValueError("问题不能为空。")
        if not context:
            raise ValueError("上下文不能为空。")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个严谨的知识库问答助手。只能根据提供的资料回答。"
                    "如果资料不足，请直接说明资料中没有相关信息。",
                ),
                (
                    "human",
                    "资料：\n{context}\n\n问题：{question}\n\n请用中文回答，并列出依据。",
                ),
            ]
        )
        chain = prompt | self.llm | self.output_parser
        return chain.invoke({"question": question, "context": context})

    def answer_codebase_question(self, question: str, context: str) -> str:
        """基于代码片段回答代码库问题。"""

        question = question.strip()
        context = context.strip()
        if not question:
            raise ValueError("问题不能为空。")
        if not context:
            raise ValueError("代码上下文不能为空。")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个严谨的代码库问答助手。只能根据提供的代码片段回答，"
                    "必须引用相关文件路径和行号。不要编造未提供的代码细节，"
                    "如果片段不足以回答，请直接说明缺少哪些信息。",
                ),
                (
                    "human",
                    "代码上下文：\n{context}\n\n问题：{question}\n\n"
                    "请用中文回答，先给结论，再列出依据文件和行号。",
                ),
            ]
        )
        chain = prompt | self.llm | self.output_parser
        return chain.invoke({"question": question, "context": context})

    def chat_once(self, history: Iterable[BaseMessage], message: str) -> str:
        """基于简单的历史消息执行一轮聊天。"""

        message = message.strip()
        if not message:
            raise ValueError("消息不能为空。")

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "你是一个有帮助的助手。请用中文回复。"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{message}"),
            ]
        )
        chain = prompt | self.llm | self.output_parser
        return chain.invoke({"history": list(history), "message": message})
