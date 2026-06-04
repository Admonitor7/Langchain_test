"""LangChain 入门项目的应用配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
PLACEHOLDER_KEYS = {"", "your_api_key_or_dummy_key", "your_openai_api_key_here"}


@dataclass(frozen=True)
class Settings:
    """从环境变量加载的运行时配置。"""

    provider: str
    api_key: str
    model: str
    temperature: float = 0.2
    base_url: str | None = None
    embedding_provider: str = "local_hash"
    embedding_api_key: str = ""
    embedding_model: str = "local-hash"
    embedding_base_url: str | None = None
    chroma_persist_dir: Path = BASE_DIR / "data" / "chroma"
    chroma_collection: str = "code_chunks"
    code_index_dir: Path = BASE_DIR / "data" / "code_index"
    default_repo_path: Path = BASE_DIR


def _to_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default


def _get_path(*names: str, default: Path) -> Path:
    value = _get_env(*names)
    if not value:
        return default
    return Path(value)


def load_settings() -> Settings:
    """从 .env 和系统环境变量中加载配置。"""

    load_dotenv(ENV_FILE)

    provider = _get_env("LLM_PROVIDER", default="openai_compatible")
    base_url = _get_env("LLM_BASE_URL", "OPENAI_BASE_URL") or None
    api_key = _get_env("LLM_API_KEY", "OPENAI_API_KEY")
    model = _get_env("LLM_MODEL", "OPENAI_MODEL", default="gpt-4o-mini")

    if provider != "openai_compatible":
        raise RuntimeError(
            "当前仅支持 LLM_PROVIDER=openai_compatible。"
            "请使用 LLM_BASE_URL 指向你自己的模型服务。"
        )

    if base_url and api_key in PLACEHOLDER_KEYS:
        api_key = "dummy-key"

    if not base_url and api_key in PLACEHOLDER_KEYS:
        raise RuntimeError(
            "LLM_API_KEY 未配置。如果使用 OpenAI，请设置真实 API Key。"
            "如果使用自己的模型服务，请设置 LLM_BASE_URL，并可使用占位 API Key。"
        )

    embedding_provider = _get_env("EMBEDDING_PROVIDER", default="local_hash")
    embedding_base_url = _get_env("EMBEDDING_BASE_URL") or base_url
    embedding_api_key = _get_env("EMBEDDING_API_KEY") or api_key
    embedding_model = _get_env("EMBEDDING_MODEL", default="local-hash")

    if embedding_provider not in {"local_hash", "openai_compatible"}:
        raise RuntimeError("EMBEDDING_PROVIDER 仅支持 local_hash 或 openai_compatible。")

    if embedding_provider == "openai_compatible" and embedding_api_key in PLACEHOLDER_KEYS:
        raise RuntimeError("使用 openai_compatible Embedding 时必须配置 EMBEDDING_API_KEY。")

    return Settings(
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=_to_float(
            _get_env("LLM_TEMPERATURE", "OPENAI_TEMPERATURE", default="0.2"),
            0.2,
        ),
        embedding_provider=embedding_provider,
        embedding_api_key=embedding_api_key,
        embedding_model=embedding_model,
        embedding_base_url=embedding_base_url,
        chroma_persist_dir=_get_path(
            "CHROMA_PERSIST_DIR",
            default=BASE_DIR / "data" / "chroma",
        ),
        chroma_collection=_get_env("CHROMA_COLLECTION", default="code_chunks"),
        code_index_dir=_get_path("CODE_INDEX_DIR", default=BASE_DIR / "data" / "code_index"),
        default_repo_path=_get_path("INDEX_REPO_PATH", default=BASE_DIR),
    )
