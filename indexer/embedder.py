"""代码索引构建：Embedding 向量化并写入 Chroma。"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from config import Settings, load_settings
from indexer.parser import CodeParser, ParsedChunk


@dataclass(frozen=True)
class IndexedChunk:
    """已经分配稳定 ID 的索引片段。"""

    chunk_id: str
    path: str
    start_line: int
    end_line: int
    content: str
    language: str
    symbol_name: str
    symbol_type: str


@dataclass(frozen=True)
class IndexBuildResult:
    """索引构建结果。"""

    repo_path: str
    indexed_files: int
    indexed_chunks: int
    chroma_collection: str
    bm25_store: str


class EmbeddingClient(Protocol):
    """Embedding 客户端协议。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化文档。"""

    def embed_query(self, text: str) -> list[float]:
        """向量化查询。"""


class LocalHashEmbeddingClient:
    """本地哈希 Embedding，适合无外部 Embedding 服务时演示完整流程。"""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _tokenize_for_hash(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimensions
            sign = 1.0 if value & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(item * item for item in vector))
        if norm == 0:
            return vector
        return [item / norm for item in vector]


class OpenAICompatibleEmbeddingClient:
    """OpenAI-compatible Embedding 客户端。"""

    def __init__(self, settings: Settings) -> None:
        from langchain_openai import OpenAIEmbeddings

        self.client = OpenAIEmbeddings(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.client.embed_query(text)


class CodeIndexer:
    """代码库索引构建器。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.embedding_client = build_embedding_client(self.settings)

    def build(self, repo_path: str | Path | None = None, reset: bool = True) -> IndexBuildResult:
        """构建代码库索引。"""

        repo = Path(repo_path or self.settings.default_repo_path).resolve()
        parser = CodeParser(repo)
        parsed_chunks = parser.parse_repository()
        indexed_files = parser.list_files()
        chunks = [_to_indexed_chunk(repo, chunk) for chunk in parsed_chunks]

        self.settings.code_index_dir.mkdir(parents=True, exist_ok=True)
        self.settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)

        if reset:
            self.clear()
            self.settings.code_index_dir.mkdir(parents=True, exist_ok=True)
            self.settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)

        self._write_bm25_store(chunks)
        self._write_index_manifest(repo, indexed_files, chunks)
        self._write_chroma(chunks)

        return IndexBuildResult(
            repo_path=str(repo),
            indexed_files=len(indexed_files),
            indexed_chunks=len(chunks),
            chroma_collection=self.settings.chroma_collection,
            bm25_store=str(self._bm25_store_path()),
        )

    def clear(self) -> None:
        """清空当前代码索引。"""

        if self.settings.code_index_dir.exists():
            shutil.rmtree(self.settings.code_index_dir)
        self.settings.code_index_dir.mkdir(parents=True, exist_ok=True)

        client = _create_chroma_client(self.settings)
        try:
            client.delete_collection(self.settings.chroma_collection)
        except Exception:
            pass

    def _write_chroma(self, chunks: list[IndexedChunk]) -> None:
        """写入 Chroma 向量库。"""

        client = _create_chroma_client(self.settings)
        collection = client.get_or_create_collection(
            name=self.settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

        for batch in _batch(chunks, size=64):
            documents = [_document_text(chunk) for chunk in batch]
            embeddings = self.embedding_client.embed_documents(documents)
            collection.add(
                ids=[chunk.chunk_id for chunk in batch],
                embeddings=embeddings,
                documents=documents,
                metadatas=[_metadata(chunk) for chunk in batch],
            )

    def _write_bm25_store(self, chunks: list[IndexedChunk]) -> None:
        """写入关键词检索需要的片段存储。"""

        store_path = self._bm25_store_path()
        with store_path.open("w", encoding="utf-8") as file:
            for chunk in chunks:
                file.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    def _write_index_manifest(
        self,
        repo: Path,
        indexed_files: list[str],
        chunks: list[IndexedChunk],
    ) -> None:
        """写入索引元信息。"""

        manifest = {
            "repo_path": str(repo),
            "indexed_files": indexed_files,
            "indexed_chunks": len(chunks),
            "embedding_provider": self.settings.embedding_provider,
            "embedding_model": self.settings.embedding_model,
            "chroma_collection": self.settings.chroma_collection,
        }
        self._manifest_path().write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _bm25_store_path(self) -> Path:
        return self.settings.code_index_dir / "chunks.jsonl"

    def _manifest_path(self) -> Path:
        return self.settings.code_index_dir / "manifest.json"



def build_embedding_client(settings: Settings) -> EmbeddingClient:
    """根据配置创建 Embedding 客户端。"""

    if settings.embedding_provider == "local_hash":
        return LocalHashEmbeddingClient()
    return OpenAICompatibleEmbeddingClient(settings)



def load_indexed_chunks(settings: Settings | None = None) -> list[IndexedChunk]:
    """读取已索引片段。"""

    settings = settings or load_settings()
    store_path = settings.code_index_dir / "chunks.jsonl"
    if not store_path.exists():
        raise FileNotFoundError("代码索引不存在，请先调用 POST /index 或运行索引构建。")

    chunks = []
    with store_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            chunks.append(IndexedChunk(**json.loads(line)))
    return chunks



def _create_chroma_client(settings: Settings):
    """创建 Chroma 持久化客户端。"""

    try:
        import chromadb
    except ImportError as error:
        raise RuntimeError("缺少 chromadb，请先执行 pip install -r requirements.txt。") from error

    return chromadb.PersistentClient(path=str(settings.chroma_persist_dir))



def _to_indexed_chunk(repo: Path, chunk: ParsedChunk) -> IndexedChunk:
    """给解析片段生成稳定 ID。"""

    identity = f"{repo}:{chunk.path}:{chunk.start_line}:{chunk.end_line}:{chunk.symbol_name}"
    chunk_id = hashlib.sha1(identity.encode("utf-8")).hexdigest()
    return IndexedChunk(
        chunk_id=chunk_id,
        path=chunk.path,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        content=chunk.content,
        language=chunk.language,
        symbol_name=chunk.symbol_name,
        symbol_type=chunk.symbol_type,
    )



def _document_text(chunk: IndexedChunk) -> str:
    """生成用于向量化和检索的文档文本。"""

    return (
        f"文件：{chunk.path}\n"
        f"行号：{chunk.start_line}-{chunk.end_line}\n"
        f"符号：{chunk.symbol_name}\n"
        f"类型：{chunk.symbol_type}\n"
        f"语言：{chunk.language}\n"
        f"代码：\n{chunk.content}"
    )



def _metadata(chunk: IndexedChunk) -> dict[str, str | int]:
    """生成 Chroma 元数据。"""

    return {
        "chunk_id": chunk.chunk_id,
        "path": chunk.path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "language": chunk.language,
        "symbol_name": chunk.symbol_name,
        "symbol_type": chunk.symbol_type,
    }



def _batch(items: list[IndexedChunk], size: int) -> list[list[IndexedChunk]]:
    """按固定大小批量处理。"""

    return [items[index : index + size] for index in range(0, len(items), size)]



def _tokenize_for_hash(text: str) -> list[str]:
    """提取本地哈希 Embedding 使用的词元。"""

    import re

    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[\u4e00-\u9fff]", text.lower())
    return [token for token in tokens if token.strip()]
