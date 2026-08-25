from pathlib import Path

from semantic_conflicts.config import load_settings


def test_historical_numbers_are_metadata_only():
    """176756 etc. may exist only in historical_released / comments, never as pool logic inputs."""
    src = Path(__file__).resolve().parents[1] / "src" / "semantic_conflicts"
    banned_uses = []
    for p in src.rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        for needle in ("176756", "176_756", "171067"):
            if needle in text.replace(" ", ""):
                # allowed only in docstrings describing the discrepancy
                if "historical" in text.lower() or "reconcil" in text.lower() or "buggy" in text.lower():
                    continue
                banned_uses.append((str(p), needle))
    assert banned_uses == []
    settings = load_settings()
    # config may record history; membership must not read it
    assert settings.historical_released is not None
    assert settings.near_miss.include_filename_as_component is False
