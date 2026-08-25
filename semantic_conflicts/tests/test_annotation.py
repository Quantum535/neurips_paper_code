import pandas as pd
import pytest

from semantic_conflicts.annotation.io_labels import adjudicate, blinded_export, import_labels
from semantic_conflicts.schemas import InvariantError, VALID_CATEGORIES


def test_invalid_labels_rejected(tmp_path, settings_ci):
    p = tmp_path / "bad.csv"
    p.write_text("frame_id,label_category\n1,Q\n", encoding="utf-8")
    with pytest.raises(InvariantError):
        import_labels(p, tmp_path / "out.csv", annotator="a1", settings=settings_ci)


def test_blinded_export_omits_leakage(settings_ci):
    df = pd.DataFrame(
        {
            "frame_id": [1],
            "repo": ["r"],
            "pr_a": [1],
            "pr_b": [2],
            "title_a": ["t"],
            "merged_a": [True],
            "closed_b": ["x"],
            "pools": ["dup"],
            "pi_enriched": [0.1],
            "llm_category": ["D"],
            "primary_stratum": ["dup"],
        }
    )
    out = blinded_export(df, settings_ci)
    leak = [c for c in out.columns if any(k in c.lower() for k in ("merged", "closed", "pool", "llm", "pi_", "stratum"))]
    assert leak == []
    assert "title_a" in out.columns


def test_independent_annotations_and_adjudication(settings_ci):
    a1 = pd.DataFrame({"frame_id": [1, 2, 3], "label_category": ["D", "none", "H"]})
    a2 = pd.DataFrame({"frame_id": [1, 2, 3], "label_category": ["D", "B", "H"]})
    gold = adjudicate([a1, a2], None, settings_ci)
    assert set(gold.loc[gold.agreed, "label_category"]) <= VALID_CATEGORIES
    assert 1 in set(gold.frame_id) and 3 in set(gold.frame_id)
    # disagreement on 2 has no gold without adjudication
    assert 2 not in set(gold.frame_id)
    adj = pd.DataFrame({"frame_id": [2], "gold": ["none"], "adjudicator_notes": ["resolved"]})
    gold2 = adjudicate([a1, a2], adj, settings_ci)
    assert set(gold2.frame_id) == {1, 2, 3}
    assert gold2.loc[gold2.frame_id == 2, "label_category"].iloc[0] == "none"
