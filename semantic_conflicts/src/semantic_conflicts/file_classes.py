"""Changed-file classification used by pool summaries and the file-overlap verifier."""

from __future__ import annotations

from semantic_conflicts.config import FileClassesConfig, Settings, load_settings

SOURCE = "source"
CI_WORKFLOW = "ci_workflow"
LOCKFILE = "lockfile"
MANIFEST = "manifest"
DOCKER = "docker"
README = "readme"
OTHER_DOCS = "other_docs"
OTHER_CONFIG = "other_config"


def classify_path(path: str, cfg: FileClassesConfig | None = None) -> str:
    if cfg is None:
        cfg = load_settings().file_classes
    p = path.replace("\\", "/").lower()
    base = p.rsplit("/", 1)[-1]
    if ".github/workflows" in p:
        return CI_WORKFLOW
    if base.endswith(".lock") or base in {b.lower() for b in cfg.lockfile_basenames}:
        return LOCKFILE
    if base.startswith("requirements") or base in {b.lower() for b in cfg.manifest_basenames}:
        return MANIFEST
    if base.startswith("dockerfile") or base in {b.lower() for b in cfg.docker_basenames}:
        return DOCKER
    if base.startswith("readme"):
        return README
    if base.startswith("changelog") or base.endswith((".md", ".rst", ".txt")):
        return OTHER_DOCS
    if (
        base.startswith(".gitignore")
        or base.startswith(".env")
        or base.endswith((".yml", ".yaml", ".toml", ".ini", ".cfg", ".json"))
    ):
        return OTHER_CONFIG
    return SOURCE


def is_spec_config(cls: str, settings: Settings | None = None) -> bool:
    settings = settings or load_settings()
    return cls in set(settings.file_classes.spec_config_classes)


def is_pure_config(classes: list[str], settings: Settings | None = None) -> bool:
    settings = settings or load_settings()
    spec = set(settings.file_classes.spec_config_classes)
    return bool(classes) and all(c in spec for c in classes)


def has_strict_source(classes: list[str]) -> bool:
    return any(c == SOURCE for c in classes)
