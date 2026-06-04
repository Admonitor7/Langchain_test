"""本地知识库检索工具。

这个模块不依赖向量数据库，适合入门项目演示“先检索资料，再让模型回答”的基本流程。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


COMMON_CHINESE_CHARS = set("的是了和与及在对把被一个有也就都而很还让用中为到说请问什么怎么如何")


@dataclass(frozen=True)
class KnowledgeChunk:
    """知识库中的一个文本片段。"""

    source: str
    index: int
    content: str
    score: int = 0


class KnowledgeBase:
    """基于本地文本文件的轻量知识库。"""

    def __init__(self, chunks: list[KnowledgeChunk]) -> None:
        if not chunks:
            raise ValueError("知识库内容为空，请先写入文本内容。")
        self.chunks = chunks

    @classmethod
    def from_file(cls, file_path: str | Path, chunk_size: int = 700) -> "KnowledgeBase":
        """从文本文件创建知识库。"""

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"知识库文件不存在：{path}")

        text = path.read_text(encoding="utf-8-sig").strip()
        if not text:
            raise ValueError(f"知识库文件为空：{path}")

        chunks = [
            KnowledgeChunk(source=str(path), index=index + 1, content=content)
            for index, content in enumerate(_split_text(text, chunk_size=chunk_size))
        ]
        return cls(chunks)

    def retrieve(self, query: str, top_k: int = 3) -> list[KnowledgeChunk]:
        """根据问题检索最相关的知识片段。"""

        query = query.strip()
        if not query:
            raise ValueError("问题不能为空。")

        query_tokens = _tokenize(query)
        scored_chunks: list[KnowledgeChunk] = []
        for chunk in self.chunks:
            chunk_tokens = _tokenize(chunk.content)
            score = len(query_tokens & chunk_tokens)
            if query in chunk.content:
                score += 10
            if score > 0:
                scored_chunks.append(
                    KnowledgeChunk(
                        source=chunk.source,
                        index=chunk.index,
                        content=chunk.content,
                        score=score,
                    )
                )

        scored_chunks.sort(key=lambda item: item.score, reverse=True)
        return scored_chunks[:top_k]

    @staticmethod
    def format_context(chunks: list[KnowledgeChunk]) -> str:
        """把检索结果格式化为模型可读的上下文。"""

        if not chunks:
            return "未检索到相关资料。"

        sections = []
        for chunk in chunks:
            sections.append(
                f"资料片段 {chunk.index}，来源：{chunk.source}\n{chunk.content}"
            )
        return "\n\n".join(sections)



def _split_text(text: str, chunk_size: int) -> list[str]:
    """按段落切分文本，过长段落再按固定长度切分。"""

    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
    chunks: list[str] = []
    buffer = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if buffer:
                chunks.append(buffer.strip())
                buffer = ""
            chunks.extend(_split_long_text(paragraph, chunk_size))
            continue

        candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
        if len(candidate) <= chunk_size:
            buffer = candidate
        else:
            chunks.append(buffer.strip())
            buffer = paragraph

    if buffer:
        chunks.append(buffer.strip())

    return chunks



def _split_long_text(text: str, chunk_size: int) -> list[str]:
    """切分单段超长文本。"""

    return [text[index : index + chunk_size].strip() for index in range(0, len(text), chunk_size)]



def _tokenize(text: str) -> set[str]:
    """提取用于简单检索的关键词。"""

    result = set()
    result.update(re.findall(r"[a-zA-Z0-9_]+", text.lower()))
    for char in text:
        if 0x4E00 <= ord(char) <= 0x9FFF and char not in COMMON_CHINESE_CHARS:
            result.add(char)
    return result
