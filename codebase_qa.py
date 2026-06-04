"""代码库问答检索工具。

本模块负责扫描代码库、切分文件、检索相关代码片段，并明确跳过敏感文件。
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path


TEXT_FILE_EXTENSIONS = {
    ".cfg",
    ".css",
    ".gitignore",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".py",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
TEXT_FILE_NAMES = {".gitignore", "requirements.txt", "readme", "readme.md"}
EXCLUDED_DIRECTORIES = {
    ".git",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "build",
    "dist",
    "node_modules",
    "__pycache__",
}
EXCLUDED_FILE_PATTERNS = {
    ".env",
    ".env.*",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.log",
    "*.sqlite",
    "*.db",
}
COMMON_CHINESE_CHARS = set("的是了和与及在对把被一个有也就都而很还让用中为到说请问什么怎么如何哪里哪些当前项目代码文件功能")


@dataclass(frozen=True)
class CodeChunk:
    """代码库中的一个代码片段。"""

    path: str
    start_line: int
    end_line: int
    content: str
    score: int = 0


class CodebaseIndex:
    """轻量代码库索引。"""

    def __init__(self, root: Path, chunks: list[CodeChunk], indexed_files: list[str]) -> None:
        if not chunks:
            raise ValueError("没有可索引的代码文件，请检查目录或排除规则。")
        self.root = root
        self.chunks = chunks
        self.indexed_files = indexed_files

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        chunk_lines: int = 90,
        overlap_lines: int = 15,
        max_file_size: int = 300_000,
    ) -> "CodebaseIndex":
        """从目录创建代码库索引。"""

        root_path = Path(root).resolve()
        if not root_path.exists():
            raise FileNotFoundError(f"代码库目录不存在：{root_path}")
        if not root_path.is_dir():
            raise ValueError(f"代码库路径不是目录：{root_path}")
        if chunk_lines <= 0:
            raise ValueError("chunk_lines 必须大于 0。")
        if overlap_lines < 0 or overlap_lines >= chunk_lines:
            raise ValueError("overlap_lines 必须大于等于 0 且小于 chunk_lines。")

        chunks: list[CodeChunk] = []
        indexed_files: list[str] = []
        for path in sorted(root_path.rglob("*")):
            if not _should_index_file(root_path, path, max_file_size=max_file_size):
                continue

            relative_path = _to_relative_path(root_path, path)
            text = _read_text(path)
            if text is None or not text.strip():
                continue

            indexed_files.append(relative_path)
            chunks.extend(
                _split_code_file(
                    relative_path=relative_path,
                    text=text,
                    chunk_lines=chunk_lines,
                    overlap_lines=overlap_lines,
                )
            )

        return cls(root=root_path, chunks=chunks, indexed_files=indexed_files)

    def retrieve(self, query: str, top_k: int = 5) -> list[CodeChunk]:
        """根据问题检索相关代码片段。"""

        query = query.strip()
        if not query:
            raise ValueError("问题不能为空。")
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0。")

        query_tokens = _tokenize(query)
        scored_chunks: list[CodeChunk] = []
        for chunk in self.chunks:
            score = _score_chunk(query, query_tokens, chunk)
            if score > 0:
                scored_chunks.append(
                    CodeChunk(
                        path=chunk.path,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                        content=chunk.content,
                        score=score,
                    )
                )

        scored_chunks.sort(key=lambda item: item.score, reverse=True)
        return scored_chunks[:top_k]

    @staticmethod
    def format_context(chunks: list[CodeChunk]) -> str:
        """把代码片段格式化成模型可读的上下文。"""

        if not chunks:
            return "未检索到相关代码片段。"

        sections = []
        for chunk in chunks:
            sections.append(
                "代码片段：{path}:{start}-{end}，匹配分数：{score}\n```\n{content}\n```".format(
                    path=chunk.path,
                    start=chunk.start_line,
                    end=chunk.end_line,
                    score=chunk.score,
                    content=chunk.content.rstrip(),
                )
            )
        return "\n\n".join(sections)



def _should_index_file(root: Path, path: Path, max_file_size: int) -> bool:
    """判断文件是否应该进入代码库索引。"""

    if not path.is_file():
        return False
    if any(part in EXCLUDED_DIRECTORIES for part in path.parts):
        return False
    relative_path = _to_relative_path(root, path)
    file_name = path.name.lower()
    if any(fnmatch.fnmatch(file_name, pattern) for pattern in EXCLUDED_FILE_PATTERNS):
        return False
    if any(fnmatch.fnmatch(relative_path.lower(), pattern) for pattern in EXCLUDED_FILE_PATTERNS):
        return False
    if path.stat().st_size > max_file_size:
        return False
    if path.suffix.lower() in TEXT_FILE_EXTENSIONS:
        return True
    if file_name in TEXT_FILE_NAMES:
        return True
    return False



def _read_text(path: Path) -> str | None:
    """读取文本文件，无法解码时返回空值。"""

    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            text = path.read_text(encoding=encoding)
            if "\x00" in text:
                return None
            return text
        except UnicodeDecodeError:
            continue
    return None



def _split_code_file(
    relative_path: str,
    text: str,
    chunk_lines: int,
    overlap_lines: int,
) -> list[CodeChunk]:
    """按行切分代码文件。"""

    lines = text.splitlines()
    if not lines:
        return []

    chunks: list[CodeChunk] = []
    step = chunk_lines - overlap_lines
    for start_index in range(0, len(lines), step):
        end_index = min(start_index + chunk_lines, len(lines))
        content = "\n".join(
            f"{line_number}: {line}"
            for line_number, line in enumerate(
                lines[start_index:end_index],
                start=start_index + 1,
            )
        )
        chunks.append(
            CodeChunk(
                path=relative_path,
                start_line=start_index + 1,
                end_line=end_index,
                content=content,
            )
        )
        if end_index == len(lines):
            break
    return chunks



def _score_chunk(query: str, query_tokens: set[str], chunk: CodeChunk) -> int:
    """计算问题和代码片段的相关度分数。"""

    content_lower = chunk.content.lower()
    path_lower = chunk.path.lower()
    query_lower = query.lower()
    chunk_tokens = _tokenize(f"{chunk.path}\n{chunk.content}")

    score = len(query_tokens & chunk_tokens)
    if query_lower and query_lower in content_lower:
        score += 25
    if query_lower and query_lower in path_lower:
        score += 20
    for token in query_tokens:
        if token in path_lower:
            score += 4
        if token in content_lower:
            score += 1
    return score



def _tokenize(text: str) -> set[str]:
    """提取适合代码检索的关键词。"""

    result: set[str] = set()
    for raw_token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", text):
        for token in _split_identifier(raw_token):
            if token:
                result.add(token)
    for char in text:
        if 0x4E00 <= ord(char) <= 0x9FFF and char not in COMMON_CHINESE_CHARS:
            result.add(char)
    return result



def _split_identifier(value: str) -> set[str]:
    """拆分 snake_case、camelCase 和普通英文标识符。"""

    value = value.strip("_").lower()
    if not value:
        return set()

    parts = re.split(r"_+", value)
    tokens = {value}
    for part in parts:
        tokens.update(re.findall(r"[a-z]+|\d+", part))
    return {token for token in tokens if len(token) > 1 or token.isdigit()}



def _to_relative_path(root: Path, path: Path) -> str:
    """生成统一使用正斜杠的相对路径。"""

    return path.resolve().relative_to(root).as_posix()
