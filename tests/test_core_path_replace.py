"""Checking and replacing files, as the readers and writers do."""

import pytest
import typing_extensions as tx

from brainhops._core.path import (
    Path,
    is_local,
    is_same_file,
    local_path,
    replace_directory,
    replace_file,
    write_file,
)


def _fill(names: tx.Dict[str, bytes]) -> tx.Callable[[Path], None]:
    """Return a writer that fills a directory with files of given content."""

    def write(directory: Path) -> None:
        assert list(directory.iterdir()) == []
        for name, content in names.items():
            (directory / name).write_bytes(content)

    return write


def _tree(directory: tx.Any) -> tx.Dict[str, bytes]:
    return {
        file.name: file.read_bytes() for file in sorted(directory.iterdir())
    }


def _names(directory: tx.Any) -> tx.List[str]:
    return sorted(file.name for file in directory.iterdir())


# ----------------------------------------------------------------------
#   LOCAL AND REMOTE PATHS
# ----------------------------------------------------------------------


def test_local_and_remote_paths_are_told_apart(tmp_path) -> None:  # noqa: ANN001
    assert is_local(tmp_path)
    assert is_local(str(tmp_path))
    assert is_local(f"file://{tmp_path}")
    assert not is_local("memory://bucket/file.nii")


def test_only_an_existing_local_file_has_a_local_path(tmp_path) -> None:  # noqa: ANN001
    file = tmp_path / "file.bin"
    file.write_bytes(b"1")
    assert local_path(file) == Path(file)
    assert local_path(str(file)) == Path(file)
    assert local_path(bytes(str(file), "utf-8")) == Path(file)
    assert local_path(tmp_path / "missing.bin") is None
    assert local_path(tmp_path) is None
    assert local_path("memory://bucket/file.bin") is None
    with file.open("rb") as stream:
        assert local_path(stream) is None


def test_is_same_file_resolves_links_and_refuses_streams(tmp_path) -> None:  # noqa: ANN001
    real = tmp_path / "real.bin"
    real.write_bytes(b"1")
    link = tmp_path / "link.bin"
    link.symlink_to(real)
    assert is_same_file(real, link)
    assert is_same_file(str(real), tmp_path / "." / "real.bin")
    assert is_same_file(tmp_path / "new.bin", tmp_path / "new.bin")
    assert not is_same_file(real, tmp_path / "other.bin")
    assert not is_same_file(
        "memory://bucket/real.bin", "memory://bucket/real.bin"
    )
    with real.open("rb") as stream:
        assert not is_same_file(stream, real)


# ----------------------------------------------------------------------
#   REPLACING FILES
# ----------------------------------------------------------------------


def test_a_file_is_replaced_whole(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "image.nii.gz"
    target.write_bytes(b"old")
    names = []

    def write(name: Path) -> None:
        names.append(name.name)
        assert target.read_bytes() == b"old"
        name.write_bytes(b"new")

    replace_file(target, write)
    assert target.read_bytes() == b"new"
    assert names[0].endswith(".image.nii.gz")
    assert _names(tmp_path) == ["image.nii.gz"]


def test_a_failed_file_write_leaves_the_file_as_it_was(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "kept.bin"
    target.write_bytes(b"kept")

    def write(name: Path) -> None:
        name.write_bytes(b"half")
        raise RuntimeError("the writer failed")

    with pytest.raises(RuntimeError, match="the writer failed"):
        replace_file(target, write)
    assert target.read_bytes() == b"kept"
    assert _names(tmp_path) == ["kept.bin"]


def test_an_error_names_the_target_rather_than_the_new_file(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "missing" / "file.bin"

    def write(name: Path) -> None:
        with name.open("xb") as f:
            f.write(b"1")

    with pytest.raises(FileNotFoundError) as info:
        replace_file(target, write)
    assert info.value.filename == str(target)


def test_write_file_writes_local_and_remote_files(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "file.bin"
    target.write_bytes(b"old")
    write_file(target, b"new")
    assert target.read_bytes() == b"new"
    write_file(str(target), b"newer")
    assert target.read_bytes() == b"newer"
    assert _names(tmp_path) == ["file.bin"]


# ----------------------------------------------------------------------
#   REPLACING DIRECTORIES
# ----------------------------------------------------------------------


def test_a_new_directory_is_written(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "new.store"
    replace_directory(target, _fill({"a": b"1"}))
    assert _tree(target) == {"a": b"1"}
    assert _names(tmp_path) == ["new.store"]


def test_an_existing_directory_is_replaced_whole(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "old.store"
    target.mkdir()
    (target / "a").write_bytes(b"old")
    (target / "b").write_bytes(b"gone")

    # The writer can read the old files while it fills the new directory.
    def write(directory: Path) -> None:
        content = (target / "a").read_bytes()
        (directory / "a").write_bytes(content + b"+new")

    replace_directory(str(target), write)
    assert _tree(target) == {"a": b"old+new"}
    assert _names(tmp_path) == ["old.store"]


def test_a_failed_write_leaves_the_directory_as_it_was(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "kept.store"
    target.mkdir()
    (target / "a").write_bytes(b"kept")

    def write(directory: Path) -> None:
        _fill({"a": b"half"})(directory)
        raise RuntimeError("the writer failed")

    with pytest.raises(RuntimeError, match="the writer failed"):
        replace_directory(target, write)
    assert _tree(target) == {"a": b"kept"}
    assert _names(tmp_path) == ["kept.store"]


def test_a_symbolic_link_is_followed(tmp_path) -> None:  # noqa: ANN001
    real = tmp_path / "real.store"
    real.mkdir()
    (real / "a").write_bytes(b"old")
    link = tmp_path / "link.store"
    link.symlink_to(real, target_is_directory=True)
    replace_directory(link, _fill({"a": b"new"}))
    assert link.is_symlink()
    assert _tree(real) == {"a": b"new"}
    assert _names(tmp_path) == ["link.store", "real.store"]


def test_a_remote_path_is_refused() -> None:
    def write(directory: Path) -> None:
        raise AssertionError("nothing is written")

    with pytest.raises(ValueError, match="local"):
        replace_directory("memory://bucket/remote.store", write)
