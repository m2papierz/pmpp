import inspect
from pathlib import Path
from typing import Iterable, Optional, Sequence

from torch.utils.cpp_extension import load as _load


def _find_parent_named(path: Path, name: str) -> Optional[Path]:
    """Return the first parent in `path.parents` whose name equals `name`."""
    return next((p for p in path.parents if p.name == name), None)


def resolve_ext_build_dir(
    chapters_dir_name: str = "chapters",
    build_root_name: str = "build",
    ext_subdir_name: str = "torch_extensions",
    anchor_path: Optional[str | Path] = None,
) -> str:
    """
    Compute and create a chapter-specific build directory for C++/CUDA extensions.

    The resulting directory has the form:
        ``<project_root>/<build_root_name>/<chapters_dir_name>/<chapter>/<ext_subdir_name>``

    The `<chapter>` (e.g., ``"ch05"``) is inferred from the caller's file path, which must
    reside under ``<chapters_dir_name>/<chapter>/``. If needed, directories are created.

    Parameters
    ----------
    chapters_dir_name : str, default "chapters"
        Name of the directory containing all chapter folders.
    build_root_name : str, default "build"
        Name of the build root at the project level.
    ext_subdir_name : str, default "torch_extensions"
        Subdirectory under the chapter-specific build dir for torch extensions.
    anchor_path : str or Path, optional
        Path used to resolve the chapter. If ``None``, the direct caller's file is used.

    Returns
    -------
    str
        Absolute path to the created build directory.
    """
    # Determine anchor: the caller's file (or a provided path)
    anchor = (
        Path(anchor_path).resolve()
        if anchor_path is not None
        else Path(inspect.stack()[1].filename).resolve()
    )

    anchor_dir = anchor if anchor.is_dir() else anchor.parent

    # Locate the "<project_root>/<chapters_dir_name>" directory
    chapters_root = _find_parent_named(anchor_dir, chapters_dir_name)
    if chapters_root is None:
        raise RuntimeError(
            f"Could not find '{chapters_dir_name}' in parents of {anchor_dir}"
        )

    # Infer the chapter folder (the first segment under chapters/)
    rel = anchor_dir.relative_to(chapters_root)
    if not rel.parts:
        raise RuntimeError(
            f"Expected a subfolder under '{chapters_dir_name}' "
            f"(e.g., '{chapters_dir_name}/ch07'), got {anchor_dir}"
        )
    chapter = rel.parts[0]

    # Build path: <project_root>/<build_root_name>/<chapters_dir_name>/<chapter>/<ext_subdir_name>
    project_root = chapters_root.parent
    build_dir = (
        project_root / build_root_name / chapters_dir_name / chapter / ext_subdir_name
    ).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    return str(build_dir)


def load_cuda_extension(
    sources: Optional[Sequence[str]],
    extra_cflags: Optional[Iterable[str]] = ("-O3",),
    extra_cuda_cflags: Optional[Iterable[str]] = ("-O3",),
    verbose: bool = False,
):
    """
    Build and load a CUDA extension with sources resolved relative to the caller module.

    The extension is compiled into a chapter-specific build directory returned by
    :func:`resolve_ext_build_dir`.

    Parameters
    ----------
    sources : sequence of str
        Source file names
    extra_cflags : iterable of str, optional
        Extra C/C++ compiler flags. Defaults to ``("-O3",)``.
    extra_cuda_cflags : iterable of str, optional
        Extra NVCC compiler flags. Defaults to ``("-O3",)``.
    verbose : bool, default False
        If ``True``, shows build output.

    Returns
    -------
    torch.library.Library or torch._C.Module
        The loaded extension module.

    Notes
    -----
    The build directory is isolated per chapter to avoid rebuild conflicts across chapters.
    """
    caller_file = Path(inspect.stack()[1].filename).resolve()
    caller_dir = caller_file.parent

    # Resolve all sources against the caller's directory
    src_paths = [str((caller_dir / s).resolve()) for s in sources]

    # Use a stable, chapter-specific build directory
    build_dir = resolve_ext_build_dir(anchor_path=caller_file)

    return _load(
        name="cuda_extension",
        sources=src_paths,
        extra_cflags=list(extra_cflags or ()),
        extra_cuda_cflags=list(extra_cuda_cflags or ()),
        build_directory=build_dir,
        verbose=verbose,
    )
