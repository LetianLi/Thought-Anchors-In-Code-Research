from pathlib import Path

from thought_anchors_code.config import canonical_dataset_name, resolve_dataset_dir


def test_canonical_dataset_name_aliases() -> None:
    assert canonical_dataset_name("humaneval") == "openai_humaneval"
    assert canonical_dataset_name("human_eval") == "openai_humaneval"
    assert canonical_dataset_name("mbpp") == "mbpp"


def test_resolve_dataset_dir_uses_legacy_alias(tmp_path: Path) -> None:
    legacy_dir = tmp_path / "human_eval"
    legacy_dir.mkdir()
    assert resolve_dataset_dir("humaneval", tmp_path) == legacy_dir
