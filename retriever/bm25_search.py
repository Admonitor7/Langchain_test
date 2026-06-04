"""关键词检索。"""

from __future__ import annotations

import re

from config import Settings, load_settings
from indexer.embedder import load_indexed_chunks
from retriever.types import RetrievedChunk


class BM25Searcher:
    """基于 rank_bm25 的关键词检索器。"""

    def __init__(self, settings: Settings | None = None) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as error:
            raise RuntimeError("缺少 rank_bm25，请先执行 pip install -r requirements.txt。") from error

        self.chunks = load_indexed_chunks(settings or load_settings())
        self._corpus_tokens = [_tokenize(_document_text(chunk)) for chunk in self.chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def search(self, query: str, top_k: int = 8) -> list[RetrievedChunk]:
        """执行 BM25 关键词检索。"""

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0。")

        scores = self._bm25.get_scores(_tokenize(query))
        ranked_indexes = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        results: list[RetrievedChunk] = []
        for index in ranked_indexes[:top_k]:
            score = float(scores[index])
            if score <= 0:
                continue
            chunk = self.chunks[index]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    path=chunk.path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    content=chunk.content,
                    language=chunk.language,
                    symbol_name=chunk.symbol_name,
                    symbol_type=chunk.symbol_type,
                    score=score,
                    source="bm25",
                )
            )
        return results



def _document_text(chunk) -> str:
    """生成关键词检索文本。"""

    return f"{chunk.path}\n{chunk.symbol_name}\n{chunk.symbol_type}\n{chunk.language}\n{chunk.content}"



def _tokenize(text: str) -> list[str]:
    """提取 BM25 使用的词元。"""

    tokens = []
    for raw_token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", text):
        tokens.extend(_split_identifier(raw_token))
    tokens.extend(char for char in text if 0x4E00 <= ord(char) <= 0x9FFF)
    return [token for token in tokens if token]



def _split_identifier(value: str) -> list[str]:
    """拆分代码标识符。"""

    value = value.strip("_").lower()
    if not value:
        return []
    parts = re.split(r"_+", value)
    tokens = {value}
    for part in parts:
        tokens.update(re.findall(r"[a-z]+|\d+", part))
    return [token for token in tokens if len(token) > 1 or token.isdigit()]
