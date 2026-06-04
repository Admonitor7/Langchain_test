"""聊天历史存储。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True)
class ChatRecord:
    """一条代码问答记录。"""

    chat_id: str
    question: str
    answer: str
    sources: list[dict[str, str | int | float]]
    created_at: str


class ChatStore:
    """基于本地 JSON 文件的聊天历史存储。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create_id(self) -> str:
        """创建聊天 ID。"""

        return uuid4().hex

    def save(self, chat_id: str, question: str, answer: str, sources: list[dict[str, str | int | float]]) -> ChatRecord:
        """保存聊天记录。"""

        record = ChatRecord(
            chat_id=chat_id,
            question=question,
            answer=answer,
            sources=sources,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._record_path(chat_id).write_text(
            json.dumps(asdict(record), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return record

    def get(self, chat_id: str) -> ChatRecord | None:
        """读取聊天记录。"""

        path = self._record_path(chat_id)
        if not path.exists():
            return None
        return ChatRecord(**json.loads(path.read_text(encoding="utf-8")))

    def _record_path(self, chat_id: str) -> Path:
        return self.root / f"{chat_id}.json"
