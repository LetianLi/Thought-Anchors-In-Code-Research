from pathlib import Path

from thought_anchors_code.data_loading import (
    _is_saved_dataset_dir,
    _resolve_dataset_payload_path,
)


def test_resolve_dataset_payload_path_finds_nested_cache_dir(tmp_path: Path) -> None:
    nested = tmp_path / "openai_humaneval" / "0.0.0" / "hash"
    nested.mkdir(parents=True)
    (nested / "dataset_info.json").write_text("{}", encoding="utf-8")
    (nested / "openai_humaneval-test.arrow").write_text("placeholder", encoding="utf-8")

    resolved = _resolve_dataset_payload_path(tmp_path / "openai_humaneval")
    assert resolved == tmp_path / "openai_humaneval"


def test_is_saved_dataset_dir_detects_state_json(tmp_path: Path) -> None:
    (tmp_path / "state.json").write_text("{}", encoding="utf-8")
    assert _is_saved_dataset_dir(tmp_path)
