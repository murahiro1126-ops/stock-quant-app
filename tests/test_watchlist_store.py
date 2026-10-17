import json

import pytest

from config import WATCHLIST
from watchlist_store import load_watchlist, save_watchlist, validate_watchlist


def test_load_watchlist_uses_defaults_when_file_does_not_exist(tmp_path):
    assert load_watchlist(tmp_path / "missing.json") == WATCHLIST


def test_save_and_load_watchlist_persists_changes(tmp_path):
    path = tmp_path / "watchlist.json"

    saved = save_watchlist(
        {"7203": "トヨタ自動車", "6758.T": "ソニーグループ"}, path
    )

    assert saved == {"7203.T": "トヨタ自動車", "6758.T": "ソニーグループ"}
    assert load_watchlist(path) == saved
    assert json.loads(path.read_text(encoding="utf-8")) == saved


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ({"72.T": "短いコード"}, "銘柄コード"),
        ({"7203.T": ""}, "銘柄名"),
        ({"7203": "トヨタ", "7203.T": "重複"}, "重複"),
    ],
)
def test_validate_watchlist_rejects_invalid_entries(entries, message):
    with pytest.raises(ValueError, match=message):
        validate_watchlist(entries)


def test_load_watchlist_reports_corrupt_file(tmp_path):
    path = tmp_path / "watchlist.json"
    path.write_text("{invalid", encoding="utf-8")

    with pytest.raises(ValueError, match="読み込めません"):
        load_watchlist(path)
