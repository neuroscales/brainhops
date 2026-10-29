"""Checking and replacing files, as the readers and writers do."""

import errno
import io
import os

import pytest
import typing_extensions as tx
from bagof.paths import CommitError

from brainhops._core.path import (
    Path,
    as_path_or_stream,
    is_same_file,
    replace_directory,
    replace_file,
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
    # The readers and writers tell them apart with `Path.is_local`.
    assert Path(tmp_path).is_local()
    assert Path(str(tmp_path)).is_local()
    assert Path(f"file://{tmp_path}").is_local()
    assert not Path("memory://bucket/file.nii").is_local()


def test_a_file_name_is_converted_to_a_path_once(tmp_path) -> None:  # noqa: ANN001
    file = tmp_path / "file.bin"
    assert as_path_or_stream(file) == Path(file)
    assert isinstance(as_path_or_stream(file), Path)
    assert isinstance(as_path_or_stream(str(file)), Path)
    assert as_path_or_stream(str(file)) == Path(file)
    assert as_path_or_stream("memory://bucket/file.bin") == Path(
        "memory://bucket/file.bin"
    )
    # A path is returned as it is, and any other value is unchanged.
    already = Path(file)
    assert as_path_or_stream(already) is already
    stream = io.BytesIO(b"1")
    assert as_path_or_stream(stream) is stream
    assert as_path_or_stream(b"1") == b"1"
    assert as_path_or_stream(None) is None


def test_is_same_file_resolves_links(tmp_path) -> None:  # noqa: ANN001
    real = tmp_path / "real.bin"
    real.write_bytes(b"1")
    link = tmp_path / "link.bin"
    link.symlink_to(real)
    (tmp_path / "sub").mkdir()
    assert is_same_file(Path(real), Path(link))
    assert is_same_file(Path(str(real)), Path(tmp_path / "." / "real.bin"))
    assert is_same_file(Path(tmp_path / "new.bin"), Path(tmp_path / "new.bin"))
    # A file that does not exist yet is compared by its resolved name.
    assert is_same_file(
        Path(tmp_path / "new.bin"), Path(tmp_path / "sub" / ".." / "new.bin")
    )
    assert not is_same_file(Path(real), Path(tmp_path / "other.bin"))
    assert not is_same_file(
        Path("memory://bucket/real.bin"), Path("memory://bucket/real.bin")
    )
    assert not is_same_file(Path(real), Path("memory://bucket/real.bin"))


# ----------------------------------------------------------------------
#   REPLACING FILES
# ----------------------------------------------------------------------


def test_a_file_is_replaced_whole(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "image.nii.gz"
    target.write_bytes(b"old")
    names = []

    def write(name: Path) -> None:
        names.append(name)
        assert target.read_bytes() == b"old"
        name.write_bytes(b"new")

    replace_file(target, write)
    assert target.read_bytes() == b"new"
    # The new file has the name of the target, in another directory.
    assert names[0].name == "image.nii.gz"
    assert names[0].parent != Path(tmp_path)
    assert _names(tmp_path) == ["image.nii.gz"]


def test_a_replaced_file_keeps_its_permissions(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "kept.bin"
    target.write_bytes(b"old")
    os.chmod(target, 0o600)
    replace_file(target, lambda name: name.write_bytes(b"new"))
    assert target.read_bytes() == b"new"
    assert os.stat(target).st_mode & 0o777 == 0o600


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

    # An error of the writer that names the new file names the target.
    target = tmp_path / "file.bin"

    def fail(name: Path) -> None:
        (name.parent / "missing" / name.name).write_bytes(b"1")

    with pytest.raises(FileNotFoundError) as info:
        replace_file(target, fail)
    assert info.value.filename == str(target)
    assert _names(tmp_path) == []


def test_a_writer_that_writes_nothing_fails(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "kept.bin"
    target.write_bytes(b"kept")
    with pytest.raises(FileNotFoundError) as info:
        replace_file(target, lambda name: None)
    assert info.value.filename == str(target)
    assert target.read_bytes() == b"kept"
    assert _names(tmp_path) == ["kept.bin"]


def test_a_file_never_replaces_a_directory(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "kept.store"
    target.mkdir()
    (target / "a").write_bytes(b"kept")
    with pytest.raises(IsADirectoryError) as info:
        replace_file(target, lambda name: name.write_bytes(b"new"))
    assert info.value.filename == str(target)
    assert _tree(target) == {"a": b"kept"}
    assert _names(tmp_path) == ["kept.store"]


def test_an_atomic_write_replaces_a_local_file(tmp_path) -> None:  # noqa: ANN001
    # The writers that hold the whole content rely on this method.
    target = Path(tmp_path / "file.bin")
    target.write_bytes(b"old")
    target.write_bytes(b"new", atomic=True)
    assert target.read_bytes() == b"new"
    assert _names(tmp_path) == ["file.bin"]
    with pytest.raises(IsADirectoryError):
        Path(tmp_path).write_bytes(b"new", atomic=True)
    with pytest.raises(FileNotFoundError) as info:
        Path(tmp_path / "missing" / "file.bin").write_bytes(b"1", atomic=True)
    assert info.value.filename == str(tmp_path / "missing" / "file.bin")
    assert _names(tmp_path) == ["file.bin"]


# ----------------------------------------------------------------------
#   REPLACING DIRECTORIES
# ----------------------------------------------------------------------


def test_a_new_directory_is_written(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "new.store"
    names = []

    def write(directory: Path) -> None:
        names.append(directory)
        _fill({"a": b"1"})(directory)

    replace_directory(target, write)
    assert _tree(target) == {"a": b"1"}
    assert _names(tmp_path) == ["new.store"]
    # The new directory has the name of the target, in another directory.
    assert names[0].name == "new.store"
    assert names[0].parent != Path(tmp_path)


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


def test_a_directory_never_replaces_a_file(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "kept.store"
    target.write_bytes(b"kept")

    def write(directory: Path) -> None:
        raise AssertionError("nothing is written")

    with pytest.raises(NotADirectoryError):
        replace_directory(target, write)
    assert target.read_bytes() == b"kept"
    assert _names(tmp_path) == ["kept.store"]


def test_a_file_that_appears_during_the_write_is_kept(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "kept.store"

    def write(directory: Path) -> None:
        target.write_bytes(b"kept")
        _fill({"a": b"new"})(directory)

    with pytest.raises(OSError) as info:
        replace_directory(target, write)
    assert info.value.errno == errno.ENOTDIR
    assert target.read_bytes() == b"kept"
    assert _names(tmp_path) == ["kept.store"]


def test_the_old_directory_is_kept_if_it_cannot_be_moved_back(
    tmp_path,  # noqa: ANN001
    monkeypatch,  # noqa: ANN001
) -> None:
    # bagof-paths 0.3.1 removed the old directory in this case
    # (bagofseeds/bagof-paths#57), and 0.3.2 keeps it.
    target = tmp_path / "kept.store"
    target.mkdir()
    (target / "a").write_bytes(b"kept")
    rename = os.rename
    calls = []

    def fail_after_the_first_move(source: tx.Any, destination: tx.Any) -> None:
        calls.append(source)
        if len(calls) > 1:
            raise OSError("the move failed")
        rename(source, destination)

    monkeypatch.setattr(os, "rename", fail_after_the_first_move)
    with pytest.raises(CommitError, match="the move failed") as info:
        replace_directory(target, _fill({"a": b"new"}))
    monkeypatch.undo()
    # The old directory was moved aside and is still there, in a hidden
    # directory of its own that the error names, and the new directory
    # is kept too, since it may be the only complete version.
    previous = Path(info.value.previous)
    assert previous.name == "kept.store"
    assert previous.parent.parent == Path(tmp_path)
    assert _tree(previous) == {"a": b"kept"}
    assert _tree(Path(info.value.staged)) == {"a": b"new"}
    assert not target.exists()
