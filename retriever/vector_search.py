"""向量检索。"""

from __future__ import annotations

from config import Settings, load_settings
from indexer.embedder import build_embedding_client, load_indexed_chunks
from retriever.types import RetrievedChunk


class VectorSearcher:
    """基于 Chroma 的向量检索器。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.embedding_client = build_embedding_client(self.settings)
        self._chunk_map = {chunk.chunk_id: chunk for chunk in load_indexed_chunks(self.settings)}

    def search(self, query: str, top_k: int = 8) -> list[RetrievedChunk]:
        """执行向量相似度检索。"""

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0。")

        collection = _get_collection(self.settings)
        query_embedding = self.embedding_client.embed_query(query)
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        chunks: list[RetrievedChunk] = []
        for index, chunk_id in enumerate(ids):
            indexed_chunk = self._chunk_map.get(chunk_id)
            if indexed_chunk is None:
                continue
            distance = float(distances[index]) if index < len(distances) else 1.0
            chunks.append(
                RetrievedChunk(
                    chunk_id=indexed_chunk.chunk_id,
                    path=indexed_chunk.path,
                    start_line=indexed_chunk.start_line,
                    end_line=indexed_chunk.end_line,
                    content=indexed_chunk.content,
                    language=indexed_chunk.language,
                    symbol_name=indexed_chunk.symbol_name,
                    symbol_type=indexed_chunk.symbol_type,
                    score=1.0 / (1.0 + max(distance, 0.0)),
                    source="vector",
                )
            )
        return chunks



def _get_collection(settings: Settings):
    """读取 Chroma 集合。"""

    try:
        import chromadb
    except ImportError as error:
        raise RuntimeError("缺少 chromadb，请先执行 pip install -r requirements.txt。") from error

    client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    return client.get_collection(settings.chroma_collection)
