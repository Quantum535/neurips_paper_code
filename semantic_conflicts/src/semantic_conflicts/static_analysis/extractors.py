"""Language-aware feature extraction from patches or file text.

Tree-sitter is optional. Regex extractors cover Python, JS/TS, Go, Java, Rust, C#.
Generic fallback extracts identifiers, env vars, config-like keys, and import paths.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MIN_LEN = 3

IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
IDENT_EXT = r"[A-Za-z_][A-Za-z0-9_\-]*"


@dataclass
class SymbolEvent:
    kind: str
    name: str
    file: str
    lang: str
    extra: dict = field(default_factory=dict)


def lang_of(path: str) -> str:
    p = path.lower()
    if p.endswith(".py"):
        return "python"
    if p.endswith((".ts", ".tsx")):
        return "typescript"
    if p.endswith((".js", ".jsx", ".mjs", ".cjs")):
        return "javascript"
    if p.endswith(".go"):
        return "go"
    if p.endswith(".java"):
        return "java"
    if p.endswith(".rs"):
        return "rust"
    if p.endswith((".cs",)):
        return "csharp"
    if p.endswith((".yml", ".yaml")):
        return "yaml"
    if p.endswith((".json", ".toml", ".ini", ".cfg", ".env")):
        return "config"
    if p.endswith((".sql",)):
        return "sql"
    return "generic"


def _events(kind: str, names: list[str], file: str, lang: str, **extra) -> list[SymbolEvent]:
    out = []
    for n in names:
        n = n.strip()
        if len(n) < MIN_LEN:
            continue
        out.append(SymbolEvent(kind=kind, name=n, file=file, lang=lang, extra=dict(extra)))
    return out


def parse_unified_diff(patch: str) -> tuple[str, str]:
    """Return (added_text, deleted_text) from a unified diff / GitHub patch."""
    added, deleted = [], []
    for line in (patch or "").splitlines():
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+"):
            added.append(line[1:])
        elif line.startswith("-"):
            deleted.append(line[1:])
    return "\n".join(added), "\n".join(deleted)


def extract_python(text: str, file: str, *, deleted: bool = False) -> list[SymbolEvent]:
    lang = "python"
    prefix = "deleted_" if deleted else ""
    ev = []
    ev += _events(prefix + "def", re.findall(rf"^\s*def\s+({IDENT})\s*\(", text, re.M), file, lang)
    ev += _events(prefix + "class", re.findall(rf"^\s*class\s+({IDENT})\b", text, re.M), file, lang)
    ev += _events("import", re.findall(rf"^\s*import\s+({IDENT}(?:\.{IDENT})*)", text, re.M), file, lang)
    ev += _events("import", re.findall(rf"^\s*from\s+({IDENT}(?:\.{IDENT})*)\s+import", text, re.M), file, lang)
    ev += _events("use", re.findall(rf"\b({IDENT})\s*\(", text), file, lang, via="call")
    ev += _events("env", re.findall(r"os\.environ(?:\.get)?\(\s*['\"]([A-Z0-9_]+)['\"]", text), file, lang)
    ev += _events("config_key", re.findall(r"['\"]([A-Za-z][A-Za-z0-9_\.]+)['\"]", text), file, lang)
    ev += _events("route", re.findall(r"@(?:app|router|bp)\.(?:route|get|post|put|delete|patch)\(\s*['\"]([^'\"]+)['\"]", text), file, lang)
    return ev


def extract_js(text: str, file: str, *, deleted: bool = False) -> list[SymbolEvent]:
    lang = lang_of(file)
    prefix = "deleted_" if deleted else ""
    ev = []
    ev += _events(prefix + "def", re.findall(rf"(?:export\s+)?(?:async\s+)?function\s+({IDENT})\s*\(", text), file, lang)
    ev += _events(prefix + "def", re.findall(rf"(?:export\s+)?(?:const|let|var)\s+({IDENT})\s*=\s*(?:async\s*)?\(", text), file, lang)
    ev += _events(prefix + "class", re.findall(rf"(?:export\s+)?class\s+({IDENT})\b", text), file, lang)
    ev += _events("export", re.findall(rf"export\s+(?:async\s+)?(?:function|class|const|let|var|default)\s+({IDENT})", text), file, lang)
    ev += _events("import", re.findall(r"""from\s+['"]([^'"]+)['"]""", text), file, lang)
    ev += _events("import", re.findall(r"""require\(\s*['"]([^'"]+)['"]\s*\)""", text), file, lang)
    ev += _events("use", re.findall(rf"\b({IDENT})\s*\(", text), file, lang, via="call")
    ev += _events("env", re.findall(r"process\.env\.([A-Z0-9_]+)", text), file, lang)
    ev += _events("route", re.findall(r"""(?:app|router)\.(?:get|post|put|delete|patch|use)\(\s*['"]([^'"]+)['"]""", text), file, lang)
    return ev


def extract_go(text: str, file: str, *, deleted: bool = False) -> list[SymbolEvent]:
    prefix = "deleted_" if deleted else ""
    ev = []
    ev += _events(prefix + "def", re.findall(rf"func\s+(?:\([^)]+\)\s+)?({IDENT})\s*\(", text), file, "go")
    ev += _events(prefix + "class", re.findall(rf"type\s+({IDENT})\s+struct", text), file, "go")
    ev += _events("import", re.findall(r'"([^"]+)"', text), file, "go")
    ev += _events("use", re.findall(rf"\b({IDENT})\s*\(", text), file, "go")
    return ev


def extract_java(text: str, file: str, *, deleted: bool = False) -> list[SymbolEvent]:
    prefix = "deleted_" if deleted else ""
    ev = []
    ev += _events(prefix + "class", re.findall(rf"(?:class|interface|enum)\s+({IDENT})\b", text), file, "java")
    ev += _events(prefix + "def", re.findall(rf"(?:public|private|protected|static|final|\s)+({IDENT})\s*\(", text), file, "java")
    ev += _events("import", re.findall(r"import\s+([a-zA-Z0-9_.]+);", text), file, "java")
    ev += _events("use", re.findall(rf"\b({IDENT})\s*\(", text), file, "java")
    return ev


def extract_rust(text: str, file: str, *, deleted: bool = False) -> list[SymbolEvent]:
    prefix = "deleted_" if deleted else ""
    ev = []
    ev += _events(prefix + "def", re.findall(rf"(?:pub\s+)?fn\s+({IDENT})\s*\(", text), file, "rust")
    ev += _events(prefix + "class", re.findall(rf"(?:pub\s+)?struct\s+({IDENT})\b", text), file, "rust")
    ev += _events("use", re.findall(rf"\b({IDENT})\s*\(", text), file, "rust")
    ev += _events("import", re.findall(r"use\s+([a-zA-Z0-9_:]+)", text), file, "rust")
    return ev


def extract_csharp(text: str, file: str, *, deleted: bool = False) -> list[SymbolEvent]:
    prefix = "deleted_" if deleted else ""
    ev = []
    ev += _events(prefix + "class", re.findall(rf"(?:class|interface|record|struct)\s+({IDENT})\b", text), file, "csharp")
    ev += _events(prefix + "def", re.findall(rf"(?:public|private|protected|internal|static|\s)+({IDENT})\s*\(", text), file, "csharp")
    ev += _events("use", re.findall(rf"\b({IDENT})\s*\(", text), file, "csharp")
    return ev


def extract_config(text: str, file: str, *, deleted: bool = False) -> list[SymbolEvent]:
    prefix = "deleted_" if deleted else ""
    ev = []
    ev += _events(prefix + "config_key", re.findall(rf"^[\s\-]*({IDENT_EXT})\s*:", text, re.M), file, lang_of(file))
    ev += _events(prefix + "config_key", re.findall(rf'^\s*"({IDENT_EXT})"\s*:', text, re.M), file, lang_of(file))
    ev += _events(prefix + "env", re.findall(r"^([A-Z][A-Z0-9_]+)=", text, re.M), file, lang_of(file))
    ev += _events(prefix + "schema", re.findall(rf"\b(?:CREATE|ALTER|DROP)\s+TABLE\s+({IDENT})", text, re.I), file, "sql")
    return ev


def extract_generic(text: str, file: str, *, deleted: bool = False) -> list[SymbolEvent]:
    prefix = "deleted_" if deleted else ""
    ev = _events(prefix + "ident", re.findall(rf"\b({IDENT})\b", text), file, lang_of(file))
    ev += extract_config(text, file, deleted=deleted)
    return ev


EXTRACTORS = {
    "python": extract_python,
    "javascript": extract_js,
    "typescript": extract_js,
    "go": extract_go,
    "java": extract_java,
    "rust": extract_rust,
    "csharp": extract_csharp,
    "yaml": extract_config,
    "config": extract_config,
    "sql": extract_config,
}


def extract_text(text: str, file: str, *, deleted: bool = False) -> list[SymbolEvent]:
    fn = EXTRACTORS.get(lang_of(file), extract_generic)
    return fn(text, file, deleted=deleted)


def extract_from_patch(patch: str, file: str) -> list[SymbolEvent]:
    added, deleted = parse_unified_diff(patch)
    ev = extract_text(added, file, deleted=False)
    ev += extract_text(deleted, file, deleted=True)
    return ev


def extract_from_files(file_texts: dict[str, str]) -> list[SymbolEvent]:
    ev: list[SymbolEvent] = []
    for path, text in file_texts.items():
        ev.extend(extract_text(text, path))
    return ev


def try_tree_sitter(text: str, file: str) -> list[SymbolEvent] | None:
    """Optional tree-sitter path. Returns None if the library/grammar is unavailable."""
    try:
        import tree_sitter  # noqa: F401
    except ImportError:
        return None
    return None
