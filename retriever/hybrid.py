"""混合检索入口。"""

from __future__ import annotations

from config import Settings, load_settings
from retriever.bm25_search import BM25Searcher
from retriever.fusion import RRFFusion
from retriever.types import RetrievedChunk
from retriever.vector_search import VectorSearcher


class HybridRetriever:
    """向量检索 + BM25 + RRF 的混合检索器。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.vector_searcher = VectorSearcher(self.settings)
        self.bm25_searcher = BM25Searcher(self.settings)
        self.fusion = RRFFusion()

    def retrieve(self, query: str, top_k: int = 8) -> list[RetrievedChunk]:
        """执行混合检索。"""

        vector_results = self.vector_searcher.search(query, top_k=top_k * 2)
        bm25_results = self.bm25_searcher.search(query, top_k=top_k * 2)
        fused_results = self.fusion.fuse([vector_results, bm25_results], top_k=top_k * 2)
        return _boost_path_and_symbol_matches(query, fused_results)[:top_k]


def format_code_context(chunks: list[RetrievedChunk]) -> str:
    """把检索结果组装成 LLM 上下文。"""

    if not chunks:
        return "未检索到相关代码片段。"

    sections = []
    for chunk in chunks:
        sections.append(
            "# {path}:{start}-{end} | {symbol_type} {symbol_name} | 来源：{source}\n{content}".format(
                path=chunk.path,
                start=chunk.start_line,
                end=chunk.end_line,
                symbol_type=chunk.symbol_type,
                symbol_name=chunk.symbol_name,
                source=chunk.source,
                content=chunk.content.rstrip(),
            )
        )
    return "\n\n".join(sections)


def _boost_path_and_symbol_matches(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """对明确命中文件路径或符号名的结果做二次加权。"""

    from dataclasses import replace

    query_lower = query.lower().replace("\\", "/")
    boosted = []
    for chunk in chunks:
        score = chunk.score
        path_lower = chunk.path.lower()
        symbol_lower = chunk.symbol_name.lower()
        if path_lower in query_lower:
            score += 0.20
        if path_lower.split("/")[-1] in query_lower:
            score += 0.08
        if symbol_lower and symbol_lower in query_lower:
            score += 0.10
        for part in path_lower.replace(".", "/").split("/"):
            if len(part) > 1 and part in query_lower:
                score += 0.01
        boosted.append(replace(chunk, score=score))

    boosted.sort(key=lambda item: item.score, reverse=True)
    return boosted
