"""代码文件解析与结构化分块。"""

from __future__ import annotations

import ast
import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_EXTENSIONS = {
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
SUPPORTED_FILE_NAMES = {".gitignore", "requirements.txt", "readme", "readme.md"}
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
    "*.db",
    "*.log",
    "*.pyc",
    "*.pyd",
    "*.pyo",
    "*.sqlite",
}
EXCLUDED_RELATIVE_PREFIXES = {
    "data/chroma/",
    "data/code_index/",
}


@dataclass(frozen=True)
class ParsedChunk:
    """解析得到的代码片段。"""

    path: str
    start_line: int
    end_line: int
    content: str
    language: str
    symbol_name: str
    symbol_type: str


class CodeParser:
    """代码库解析器。"""

    def __init__(self, root: str | Path, max_file_size: int = 500_000) -> None:
        self.root = Path(root).resolve()
        self.max_file_size = max_file_size

    def parse_repository(self) -> list[ParsedChunk]:
        """解析整个代码库。"""

        if not self.root.exists():
            raise FileNotFoundError(f"代码库目录不存在：{self.root}")
        if not self.root.is_dir():
            raise ValueError(f"代码库路径不是目录：{self.root}")

        chunks: list[ParsedChunk] = []
        for path in sorted(self.root.rglob("*")):
            if not self.should_parse(path):
                continue
            chunks.extend(self.parse_file(path))
        return chunks

    def list_files(self) -> list[str]:
        """列出会被解析的文件。"""

        return [_relative_path(self.root, path) for path in sorted(self.root.rglob("*")) if self.should_parse(path)]

    def should_parse(self, path: Path) -> bool:
        """判断文件是否应该解析。"""

        if not path.is_file():
            return False
        if any(part in EXCLUDED_DIRECTORIES for part in path.parts):
            return False
        if path.stat().st_size > self.max_file_size:
            return False

        relative_path = _relative_path(self.root, path).lower()
        file_name = path.name.lower()
        if any(relative_path.startswith(prefix) for prefix in EXCLUDED_RELATIVE_PREFIXES):
            return False
        if any(fnmatch.fnmatch(file_name, pattern) for pattern in EXCLUDED_FILE_PATTERNS):
            return False
        if any(fnmatch.fnmatch(relative_path, pattern) for pattern in EXCLUDED_FILE_PATTERNS):
            return False
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return True
        return file_name in SUPPORTED_FILE_NAMES

    def parse_file(self, path: Path) -> list[ParsedChunk]:
        """解析单个文件。"""

        text = _read_text(path)
        if text is None or not text.strip():
            return []

        relative_path = _relative_path(self.root, path)
        suffix = path.suffix.lower()
        if suffix == ".py":
            return _parse_python(relative_path, text)
        if suffix == ".md":
            return _parse_markdown(relative_path, text)
        if suffix in {".yaml", ".yml", ".toml", ".ini", ".cfg"}:
            return _parse_config_file(relative_path, text, language=suffix.lstrip("."))
        return _parse_generic_text(relative_path, text, language=suffix.lstrip(".") or path.name)



def _parse_python(relative_path: str, text: str) -> list[ParsedChunk]:
    """按 Python 函数和类解析代码。"""

    lines = text.splitlines()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _parse_generic_text(relative_path, text, language="python")

    parent_map = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and hasattr(node, "lineno")
        and hasattr(node, "end_lineno")
    ]
    nodes.sort(key=lambda item: (item.lineno, item.end_lineno or item.lineno))

    chunks: list[ParsedChunk] = []
    covered_ranges: list[tuple[int, int]] = []
    for node in nodes:
        start_line = _node_start_line(node)
        end_line = int(node.end_lineno or node.lineno)
        covered_ranges.append((start_line, end_line))
        symbol_name = _qualified_name(node, parent_map)
        symbol_type = _python_symbol_type(node, parent_map)
        chunks.append(
            ParsedChunk(
                path=relative_path,
                start_line=start_line,
                end_line=end_line,
                content=_slice_lines(lines, start_line, end_line),
                language="python",
                symbol_name=symbol_name,
                symbol_type=symbol_type,
            )
        )

    module_chunk = _module_level_python_chunk(relative_path, lines, covered_ranges)
    if module_chunk:
        chunks.insert(0, module_chunk)
    if not chunks:
        return _parse_generic_text(relative_path, text, language="python")
    return chunks



def _parse_markdown(relative_path: str, text: str) -> list[ParsedChunk]:
    """按 Markdown 标题分块。"""

    lines = text.splitlines()
    header_lines = [index + 1 for index, line in enumerate(lines) if re.match(r"^#{1,6}\s+", line)]
    if not header_lines:
        return _parse_generic_text(relative_path, text, language="markdown")

    chunks: list[ParsedChunk] = []
    for index, start_line in enumerate(header_lines):
        end_line = header_lines[index + 1] - 1 if index + 1 < len(header_lines) else len(lines)
        title = lines[start_line - 1].lstrip("#").strip() or "未命名标题"
        chunks.append(
            ParsedChunk(
                path=relative_path,
                start_line=start_line,
                end_line=end_line,
                content=_slice_lines(lines, start_line, end_line),
                language="markdown",
                symbol_name=title,
                symbol_type="heading",
            )
        )
    return chunks



def _parse_config_file(relative_path: str, text: str, language: str) -> list[ParsedChunk]:
    """按配置文件的顶层键或段落分块。"""

    lines = text.splitlines()
    start_lines = []
    for index, line in enumerate(lines, start=1):
        if re.match(r"^[A-Za-z0-9_.-]+\s*[:=]", line) or re.match(r"^\[[^\]]+\]", line):
            start_lines.append(index)
    if not start_lines:
        return _parse_generic_text(relative_path, text, language=language)

    chunks: list[ParsedChunk] = []
    for index, start_line in enumerate(start_lines):
        end_line = start_lines[index + 1] - 1 if index + 1 < len(start_lines) else len(lines)
        name = lines[start_line - 1].strip()[:80]
        chunks.append(
            ParsedChunk(
                path=relative_path,
                start_line=start_line,
                end_line=end_line,
                content=_slice_lines(lines, start_line, end_line),
                language=language,
                symbol_name=name,
                symbol_type="config_section",
            )
        )
    return chunks



def _parse_generic_text(relative_path: str, text: str, language: str) -> list[ParsedChunk]:
    """按空行形成的逻辑段落分块，必要时再按行数兜底。"""

    lines = text.splitlines()
    chunks: list[ParsedChunk] = []
    start_line = 1
    buffer: list[str] = []

    for line_number, line in enumerate(lines, start=1):
        if not line.strip() and buffer:
            chunks.append(_generic_chunk(relative_path, language, start_line, line_number - 1, buffer))
            buffer = []
            start_line = line_number + 1
            continue
        if not buffer and line.strip():
            start_line = line_number
        if line.strip() or buffer:
            buffer.append(line)
        if len(buffer) >= 80:
            chunks.append(_generic_chunk(relative_path, language, start_line, line_number, buffer))
            buffer = []
            start_line = line_number + 1

    if buffer:
        chunks.append(_generic_chunk(relative_path, language, start_line, len(lines), buffer))
    return chunks



def _generic_chunk(relative_path: str, language: str, start_line: int, end_line: int, lines: list[str]) -> ParsedChunk:
    """创建通用文本片段。"""

    return ParsedChunk(
        path=relative_path,
        start_line=start_line,
        end_line=end_line,
        content="\n".join(lines),
        language=language,
        symbol_name=f"{relative_path}:{start_line}",
        symbol_type="block",
    )



def _module_level_python_chunk(
    relative_path: str,
    lines: list[str],
    covered_ranges: list[tuple[int, int]],
) -> ParsedChunk | None:
    """提取 Python 文件中的模块级代码。"""

    covered = set()
    for start_line, end_line in covered_ranges:
        covered.update(range(start_line, end_line + 1))

    module_lines = []
    first_line = None
    last_line = None
    for line_number, line in enumerate(lines, start=1):
        if line_number in covered:
            continue
        if not line.strip() and not module_lines:
            continue
        if line.strip():
            first_line = first_line or line_number
            last_line = line_number
        if first_line is not None:
            module_lines.append(line)

    if first_line is None or last_line is None or not "\n".join(module_lines).strip():
        return None
    return ParsedChunk(
        path=relative_path,
        start_line=first_line,
        end_line=last_line,
        content="\n".join(module_lines).strip(),
        language="python",
        symbol_name="<module>",
        symbol_type="module",
    )



def _qualified_name(node: ast.AST, parent_map: dict[ast.AST, ast.AST]) -> str:
    """生成函数或类的限定名。"""

    names = [getattr(node, "name", "<unknown>")]
    parent = parent_map.get(node)
    while parent is not None:
        if isinstance(parent, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(parent.name)
        parent = parent_map.get(parent)
    return ".".join(reversed(names))



def _python_symbol_type(node: ast.AST, parent_map: dict[ast.AST, ast.AST]) -> str:
    """识别 Python 代码符号类型。"""

    if isinstance(node, ast.ClassDef):
        return "class"
    parent = parent_map.get(node)
    while parent is not None:
        if isinstance(parent, ast.ClassDef):
            return "method"
        parent = parent_map.get(parent)
    return "function"



def _node_start_line(node: ast.AST) -> int:
    """获取节点起始行，包含装饰器。"""

    line_numbers = [int(getattr(node, "lineno", 1))]
    decorators = getattr(node, "decorator_list", [])
    line_numbers.extend(int(decorator.lineno) for decorator in decorators if hasattr(decorator, "lineno"))
    return min(line_numbers)



def _slice_lines(lines: list[str], start_line: int, end_line: int) -> str:
    """按 1 基行号截取文本。"""

    return "\n".join(lines[start_line - 1 : end_line])



def _read_text(path: Path) -> str | None:
    """读取文本文件，无法解码或疑似二进制时返回空值。"""

    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            text = path.read_text(encoding=encoding)
            if "\x00" in text:
                return None
            return text
        except UnicodeDecodeError:
            continue
    return None



def _relative_path(root: Path, path: Path) -> str:
    """生成统一使用正斜杠的相对路径。"""

    return path.resolve().relative_to(root).as_posix()
