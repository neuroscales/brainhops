"""Replacing a local directory, as the writers of pyramid stores do."""

import os

import pytest
import typing_extensions as tx

from brainhops.io.base._utils_files import replace_directory


def _fill(names: tx.Dict[str, bytes]) -> tx.Callable[[str], None]:
    """Return a writer that fills a directory with files of given content."""

    def write(directory: str) -> None:
        assert os.listdir(directory) == []
        for name, content in names.items():
            with open(os.path.join(directory, name), "wb") as file:
                file.write(content)

    return write


def _tree(directory: tx.Any) -> tx.Dict[str, bytes]:
    return {
        name: (directory / name).read_bytes()
        for name in sorted(os.listdir(directory))
    }


def test_a_new_directory_is_written(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "new.store"
    replace_directory(target, _fill({"a": b"1"}))
    assert _tree(target) == {"a": b"1"}
    assert os.listdir(tmp_path) == ["new.store"]


def test_an_existing_directory_is_replaced_whole(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "old.store"
    target.mkdir()
    (target / "a").write_bytes(b"old")
    (target / "b").write_bytes(b"gone")

    # The writer can read the old files while it fills the new directory.
    def write(directory: str) -> None:
        content = (target / "a").read_bytes()
        with open(os.path.join(directory, "a"), "wb") as file:
            file.write(content + b"+new")

    replace_directory(str(target), write)
    assert _tree(target) == {"a": b"old+new"}
    assert os.listdir(tmp_path) == ["old.store"]


def test_a_failed_write_leaves_the_directory_as_it_was(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "kept.store"
    target.mkdir()
    (target / "a").write_bytes(b"kept")

    def write(directory: str) -> None:
        _fill({"a": b"half"})(directory)
        raise RuntimeError("the writer failed")

    with pytest.raises(RuntimeError, match="the writer failed"):
        replace_directory(target, write)
    assert _tree(target) == {"a": b"kept"}
    assert os.listdir(tmp_path) == ["kept.store"]


def test_a_symbolic_link_is_followed(tmp_path) -> None:  # noqa: ANN001
    real = tmp_path / "real.store"
    real.mkdir()
    (real / "a").write_bytes(b"old")
    link = tmp_path / "link.store"
    link.symlink_to(real, target_is_directory=True)
    replace_directory(link, _fill({"a": b"new"}))
    assert link.is_symlink()
    assert _tree(real) == {"a": b"new"}
    assert sorted(os.listdir(tmp_path)) == ["link.store", "real.store"]


def test_a_remote_path_is_refused() -> None:
    def write(directory: str) -> None:
        raise AssertionError("nothing is written")

    with pytest.raises(ValueError, match="local"):
        replace_directory("memory://bucket/remote.store", write)
