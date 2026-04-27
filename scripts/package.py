#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    is_windows = platform.system().lower().startswith("win")

    parser = argparse.ArgumentParser(
        description="Build, install, and package the AMD GUI + backend."
    )
    parser.add_argument("--build-dir", default="build/package", help="CMake build folder.")
    parser.add_argument(
        "--install-dir", default="dist/AMD_GUI", help="Install/output staging folder."
    )
    parser.add_argument("--generator", help="CMake generator (e.g. Visual Studio 17 2022).")
    parser.add_argument("--arch", help="CMake generator architecture (e.g. x64).")
    parser.add_argument(
        "--qt-prefix", help="Qt install root for CMAKE_PREFIX_PATH (if needed)."
    )
    parser.add_argument(
        "--config",
        default="Release" if is_windows else "",
        help="Build configuration (Release/Debug). Default is Release on Windows.",
    )
    parser.add_argument("--zip-name", help="Zip file name (default: AMD_GUI-<OS>.zip).")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing build/install folders before packaging.",
    )
    args = parser.parse_args()

    build_dir = root / args.build_dir
    install_dir = root / args.install_dir
    dist_dir = install_dir.parent

    if args.clean:
        shutil.rmtree(build_dir, ignore_errors=True)
        shutil.rmtree(install_dir, ignore_errors=True)

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

    install_args = ["cmake", "--install", str(build_dir), "--prefix", str(install_dir)]
    if args.config:
        install_args += ["--config", args.config]
    run(install_args)

    dist_dir.mkdir(parents=True, exist_ok=True)
    zip_name = args.zip_name or f"AMD_GUI-{platform.system()}.zip"
    zip_path = dist_dir / zip_name
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=install_dir)
    print(f"Packaged: {zip_path}")


if __name__ == "__main__":
    main()
