"""检索结果类型。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    """检索得到的代码片段。"""

    chunk_id: str
    path: str
    start_line: int
    end_line: int
    content: str
    language: str
    symbol_name: str
    symbol_type: str
    score: float
    source: str

    @property
    def location(self) -> str:
        """返回文件位置。"""

        return f"{self.path}:{self.start_line}-{self.end_line}"
