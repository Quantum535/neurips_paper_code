"""Tiny-fixture deterministic pipeline + schema validation."""


from semantic_conflicts.pipeline import run_deterministic
from semantic_conflicts.pools import add_pool_flags
from semantic_conflicts.schemas import validate_pool_flags


def test_tiny_pipeline(tmp_path, tiny_tables, settings_ci, monkeypatch):
    from semantic_conflicts import pipeline as P

    def fake_load(root=None, **kwargs):
        return tiny_tables

    monkeypatch.setattr(P, "load_source_tables", fake_load)
    monkeypatch.setattr("semantic_conflicts.pools.load_source_tables", fake_load)
    # expected_n_pairs is None for ci
    result = run_deterministic(settings_ci, out_dir=tmp_path)
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "pool_flags.csv.gz").exists()
    assert (tmp_path / "near_miss_reconciliation.json").exists()
    assert (tmp_path / "judging_frame.csv.gz").exists()
    assert (tmp_path / "WAITING_FOR_HUMAN_LABELS").exists()
    flags = add_pool_flags(tiny_tables["pairs"], tiny_tables["texts"], tiny_tables["files"], settings_ci, validate=False)
    validate_pool_flags(flags, strict=True)
    assert result["counts"]["n_pairs"] == len(tiny_tables["pairs"])
