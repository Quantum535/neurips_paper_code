from semantic_conflicts.static_analysis.detector import detect_from_file_texts, pair_prediction


def _patch(old_line: str, new_line: str) -> str:
    return f"--- a/x\n+++ b/x\n@@ -1 +1 @@\n-{old_line}\n+{new_line}\n"


def test_python_def_use_flags():
    a = {
        "lib.py": "def load_collection(name):\n    return name\n",
    }
    b = {
        "app.py": "from lib import load_collection\n\ndef main():\n    load_collection('x')\n",
    }
    pred = pair_prediction(detect_from_file_texts(a, b))
    assert pred["prediction"] is True
    assert pred["symbol"] == "load_collection"
    assert pred["direction"] in {"A->B", "both"}


def test_unrelated_python_does_not_flag():
    a = {"a.py": "def alpha_helper():\n    return 1\n"}
    b = {"b.py": "def beta_helper():\n    return 2\n"}
    pred = pair_prediction(detect_from_file_texts(a, b))
    assert pred["prediction"] is False


def test_ts_export_rename_flags():
    a = {"api.ts": "export function fetchUser() { return 1 }\n"}
    b = {"page.ts": "import { fetchUser } from './api'\nfetchUser()\n"}
    pred = pair_prediction(detect_from_file_texts(a, b))
    assert pred["prediction"] is True
    assert pred["symbol"] == "fetchUser"


def test_lockfile_unrelated_source_no_semantic_dep():
    a = {"package-lock.json": '"lodash": "4.17.21"\n'}
    b = {"src/main.py": "def greet_user():\n    return 'hi'\n"}
    pred = pair_prediction(detect_from_file_texts(a, b))
    assert pred["prediction"] is False


def test_config_key_consumer_flags():
    a = {"config.yaml": "database_url: postgres://old\n"}
    b = {"settings.py": "url = config['database_url']\n"}
    pred = pair_prediction(detect_from_file_texts(a, b))
    assert pred["prediction"] is True
    assert "database_url" in pred["symbol"] or any(
        e.get("symbol") == "database_url" for e in pred["evidence"]
    )
