from semantic_conflicts.evaluation.file_overlap_verifier import file_overlap_confusion
import pandas as pd


def test_file_overlap_confusion_on_gold():
    gold = pd.DataFrame(
        {
            "conflict": [True, True, False, False, True],
            "label_category": ["H", "B", "D", "none", "I"],
        }
    )
    r = file_overlap_confusion(gold)
    # harmful = D,C,H,I. rows: H+fo TP, B+fo FP, D+no FN, none+no TN, I+fo TP
    assert r["table"]["TP"] == 2
    assert r["table"]["FP"] == 1
    assert r["table"]["FN"] == 1
    assert r["table"]["TN"] == 1
    assert r["label_source"] == "human-adjudicated-gold"


def test_waiting_without_gold():
    gold = pd.DataFrame({"conflict": [True]})
    r = file_overlap_confusion(gold)
    assert "NOT AVAILABLE" in r["status"]
