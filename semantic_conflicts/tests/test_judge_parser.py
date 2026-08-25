from semantic_conflicts.judges.parser import parse_judge_output


def test_valid_json():
    t = '{"category": "D", "confidence": 2, "evidence": "same title"}'
    p = parse_judge_output(t)
    assert p["category"] == "D" and p["confidence"] == 2


def test_extra_prose():
    t = 'Here you go:\n{"category": "none", "confidence": 1, "evidence": "unrelated"}\nThanks.'
    p = parse_judge_output(t)
    assert p["category"] == "none"


def test_malformed():
    assert parse_judge_output("not json") is None
    assert parse_judge_output("{") is None


def test_invalid_class():
    assert parse_judge_output('{"category": "Q", "confidence": 1, "evidence": "x"}') is None


def test_confidence_out_of_range():
    assert parse_judge_output('{"category": "D", "confidence": 9, "evidence": "x"}') is None
    assert parse_judge_output('{"category": "D", "confidence": 0, "evidence": "x"}') is None
