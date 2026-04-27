#!/usr/bin/env python3
"""Build, stage, and package the AMD GUI + backend.

ZIP mode (default):
    python3 scripts/package.py [options]

Windows installer mode (requires MSVC, Qt, Python, PyInstaller, and Inno Setup 6):
    py -3 scripts/package.py --installer [options]

From Ubuntu, use the GitHub Actions workflow instead:
    .github/workflows/build-installer.yml
"""
from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Path separator used in PyInstaller --add-data (';' on Windows, ':' on Unix).
_DATA_SEP = ";" if platform.system() == "Windows" else ":"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def _cmake_build(root: Path, build_dir: Path, args: argparse.Namespace) -> None:
    cmake_args = ["cmake", "-S", str(root), "-B", str(build_dir)]
    if args.generator:
        cmake_args += ["-G", args.generator]
    if args.arch:
        cmake_args += ["-A", args.arch]
    if args.qt_prefix:
        cmake_args += [f"-DCMAKE_PREFIX_PATH={args.qt_prefix}"]
    run(cmake_args)

    build_args = ["cmake", "--build", str(build_dir)]
    if args.config:
        build_args += ["--config", args.config]
    run(build_args)


def _cmake_install(build_dir: Path, prefix: Path, config: str) -> None:
    install_args = ["cmake", "--install", str(build_dir), "--prefix", str(prefix)]
    if config:
        install_args += ["--config", config]
    run(install_args)


def _freeze_backend(root: Path, dist_dir: Path) -> Path:
    """Run PyInstaller to produce a standalone backend_server.exe."""
    entry = root / "scripts" / "backend_entry.py"
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name", "backend_server",
        "--hidden-import", "flask_cors",
        "--collect-all", "timm",
        "--add-data", f"backend{_DATA_SEP}backend",
        str(entry),
    ]
    run(cmd, cwd=root)
    frozen = root / "dist" / "backend_server.exe"
    if not frozen.exists():
        raise FileNotFoundError(
            f"PyInstaller did not produce {frozen}. "
            "Check the output above for errors."
        )
    dest = dist_dir / "backend_server.exe"
    shutil.copy2(frozen, dest)
    print(f"Frozen backend staged: {dest}")
    return dest


def _run_inno_setup(root: Path, staging: Path, output: Path, version: str) -> None:
    iscc_candidates = [
        r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        r"C:\Program Files\Inno Setup 6\ISCC.exe",
    ]
    iscc: str | None = next(
        (p for p in iscc_candidates if Path(p).exists()), None
    )
    if iscc is None:
        iscc = shutil.which("iscc")
    if iscc is None:
        raise RuntimeError(
            "Inno Setup compiler (ISCC.exe) not found. "
            "Install Inno Setup 6 from https://jrsoftware.org/isdl.php"
        )
    iss_script = root / "installer" / "AMD_GUI.iss"
    run([
        iscc,
        f"/DStagingDir={staging}",
        f"/DOutputDir={output}",
        f"/DAppVersion={version}",
        str(iss_script),
    ])


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    is_windows = platform.system().lower().startswith("win")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--build-dir", default="build/package", help="CMake build folder.")
    parser.add_argument(
        "--install-dir", default="dist/staging", help="Staging/install folder."
    )
    parser.add_argument("--generator", help="CMake generator (e.g. 'Visual Studio 17 2022').")
    parser.add_argument("--arch", help="CMake generator architecture (e.g. x64).")
    parser.add_argument("--qt-prefix", help="Qt install root for CMAKE_PREFIX_PATH.")
    parser.add_argument(
        "--config",
        default="Release" if is_windows else "",
        help="Build configuration. Default: Release on Windows.",
    )
    parser.add_argument("--zip-name", help="Zip archive name (default: AMD_GUI-<OS>.zip).")
    parser.add_argument(
        "--installer",
        action="store_true",
        help=(
            "Produce a Windows .exe installer via Inno Setup instead of a ZIP. "
            "Requires PyInstaller and Inno Setup 6. Windows only."
        ),
    )
    parser.add_argument(
        "--version", default="1.0.0", help="Application version string (default: 1.0.0)."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing build/install folders before packaging.",
    )
    args = parser.parse_args()

    if args.installer and not is_windows:
        parser.error("--installer can only be run on Windows. "
                     "Use the GitHub Actions workflow from Ubuntu instead:\n"
                     "  .github/workflows/build-installer.yml")

    build_dir = root / args.build_dir
    staging_dir = root / args.install_dir
    dist_dir = staging_dir.parent

    if args.clean:
        shutil.rmtree(build_dir, ignore_errors=True)
        shutil.rmtree(staging_dir, ignore_errors=True)

    # ── Step 1: CMake build ──────────────────────────────────────────────────
    _cmake_build(root, build_dir, args)

    # ── Step 2: Stage files ──────────────────────────────────────────────────
    staging_bin = staging_dir / "bin"
    staging_bin.mkdir(parents=True, exist_ok=True)

    if is_windows and args.installer:
        # Copy AMD_GUI.exe + Qt DLLs from the build output directory.
        # (windeployqt runs as a POST_BUILD step, so DLLs are beside the exe.)
        build_bin = build_dir / "bin" / args.config
        if build_bin.exists():
            for item in build_bin.iterdir():
                dest = staging_bin / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)
        # Python backend source
        backend_dest = staging_dir / "backend"
        if backend_dest.exists():
            shutil.rmtree(backend_dest)
        shutil.copytree(root / "backend", backend_dest,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copy2(root / "requirements.txt", staging_dir)

        # ── Step 3a: PyInstaller ─────────────────────────────────────────────
        _freeze_backend(root, staging_bin)

        # ── Step 3b: Inno Setup installer ────────────────────────────────────
        _run_inno_setup(root, staging_dir, dist_dir, args.version)
        print(f"\nInstaller written to: {dist_dir}")
    else:
        # ZIP mode: use cmake --install then zip the result.
        _cmake_install(build_dir, staging_dir, args.config)

        dist_dir.mkdir(parents=True, exist_ok=True)
        zip_name = args.zip_name or f"AMD_GUI-{platform.system()}.zip"
        zip_path = dist_dir / zip_name
        if zip_path.exists():
            zip_path.unlink()
        shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=staging_dir)
        print(f"Packaged: {zip_path}")


if __name__ == "__main__":
    main()
