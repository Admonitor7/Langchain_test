"""代码库问答服务编排层。"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from config import Settings, load_settings
from generator.llm import CodeAnswerGenerator
from indexer.embedder import CodeIndexer, IndexBuildResult
from retriever.hybrid import HybridRetriever, format_code_context
from retriever.types import RetrievedChunk


@dataclass(frozen=True)
class CodeQAResult:
    """代码库问答结果。"""

    answer: str
    sources: list[dict[str, str | int | float]]
    context: str


class CodeQAService:
    """代码库问答服务。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()

    def build_index(self, repo_path: str | Path | None = None, reset: bool = True) -> IndexBuildResult:
        """构建代码索引。"""

        return CodeIndexer(self.settings).build(repo_path=repo_path, reset=reset)

    def clear_index(self) -> dict[str, str]:
        """清空代码索引。"""

        CodeIndexer(self.settings).clear()
        return {"status": "cleared"}

    def ask(self, question: str, top_k: int = 8) -> CodeQAResult:
        """执行非流式代码库问答。"""

        chunks = self.retrieve(question, top_k=top_k)
        context = format_code_context(chunks)
        answer = CodeAnswerGenerator(self.settings).answer(question, context)
        return CodeQAResult(answer=answer, sources=_sources(chunks), context=context)

    def stream_answer(self, question: str, top_k: int = 8) -> tuple[list[dict[str, str | int | float]], str, Iterator[str]]:
        """执行流式代码库问答。"""

        chunks = self.retrieve(question, top_k=top_k)
        context = format_code_context(chunks)
        generator = CodeAnswerGenerator(self.settings).stream_answer(question, context)
        return _sources(chunks), context, generator

    def retrieve(self, question: str, top_k: int = 8) -> list[RetrievedChunk]:
        """执行混合检索。"""

        return HybridRetriever(self.settings).retrieve(question, top_k=top_k)



def _sources(chunks: list[RetrievedChunk]) -> list[dict[str, str | int | float]]:
    """生成可序列化的引用来源。"""

    return [
        {
            "chunk_id": chunk.chunk_id,
            "path": chunk.path,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "symbol_name": chunk.symbol_name,
            "symbol_type": chunk.symbol_type,
            "score": chunk.score,
            "source": chunk.source,
        }
        for chunk in chunks
    ]


def to_dict(result: IndexBuildResult | CodeQAResult) -> dict:
    """把 dataclass 转成字典。"""

    return asdict(result)
