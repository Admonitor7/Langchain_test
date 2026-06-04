"""检索结果融合。"""

from __future__ import annotations

from dataclasses import replace

from retriever.types import RetrievedChunk


class RRFFusion:
    """使用 Reciprocal Rank Fusion 融合多路检索结果。"""

    def __init__(self, k: int = 60) -> None:
        self.k = k

    def fuse(self, result_sets: list[list[RetrievedChunk]], top_k: int = 8) -> list[RetrievedChunk]:
        """融合多个排序结果。"""

        scores: dict[str, float] = {}
        chunks: dict[str, RetrievedChunk] = {}
        sources: dict[str, set[str]] = {}

        for results in result_sets:
            for rank, chunk in enumerate(results, start=1):
                scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (self.k + rank)
                chunks.setdefault(chunk.chunk_id, chunk)
                sources.setdefault(chunk.chunk_id, set()).add(chunk.source)

        ranked_ids = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
        fused = []
        for chunk_id in ranked_ids[:top_k]:
            chunk = chunks[chunk_id]
            fused.append(
                replace(
                    chunk,
                    score=scores[chunk_id],
                    source="+".join(sorted(sources[chunk_id])),
                )
            )
        return fused
