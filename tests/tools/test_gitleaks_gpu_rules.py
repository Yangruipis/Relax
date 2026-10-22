# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise GPU disclosure rules through the real tracked-file scan."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
HOOK = ROOT / ".pre-commit-hooks/gitleaks_tracked.py"
GPU_MODEL = "B300"  # gitleaks:allow -- synthetic test input


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    if shutil.which("gitleaks") is None:
        pytest.skip("Run with the pinned pre-commit gitleaks executable on PATH")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    shutil.copy2(ROOT / ".gitleaks.toml", tmp_path / ".gitleaks.toml")
    shutil.copy2(ROOT / ".gitleaksignore", tmp_path / ".gitleaksignore")
    subprocess.run(["git", "add", ".gitleaks.toml", ".gitleaksignore"], cwd=tmp_path, check=True)
    return tmp_path


def _stage(repo: Path, name: str, content: str | bytes) -> None:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode() if isinstance(content, str) else content)
    subprocess.run(["git", "add", "--", name], cwd=repo, check=True)


def _scan(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(HOOK)], cwd=repo, capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("prefix,number", [("B", 300), ("H", 100), ("H", 800), ("H", 20), ("GB", 300)])
def test_gitleaks_blocks_new_gpu_content(repo: Path, prefix: str, number: int) -> None:
    model = f"{prefix}{number}"
    _stage(repo, "training.sh", f'EXP_NAME="run-128x{model.lower()}"\n# measured on {model}x8\n')
    result = _scan(repo)
    assert result.returncode == 1
    assert "nvidia-gpu-model" in result.stdout + result.stderr


@pytest.mark.parametrize("kind", ["empty", "binary", "directory", "symlink", "extensionless"])
def test_gitleaks_blocks_gpu_filenames(repo: Path, kind: str) -> None:
    name = f"run-128x{GPU_MODEL.lower()}.sh"
    if kind == "symlink":
        _stage(repo, "target.txt", "safe\n")
        (repo / name).symlink_to("target.txt")
        subprocess.run(["git", "add", "--", name], cwd=repo, check=True)
    else:
        if kind == "directory":
            name = f"runs/{GPU_MODEL}/train.sh"
        if kind == "binary":
            name = f"usage-{GPU_MODEL.lower()}.png"
        if kind == "extensionless":
            name = GPU_MODEL
        content = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR" + b"\0" * 100 if kind == "binary" else b""
        _stage(repo, name, content)
    _stage(repo, "other.txt", "safe\n")
    result = _scan(repo)
    assert result.returncode == 1
    assert "nvidia-gpu-model" in result.stdout + result.stderr


@pytest.mark.parametrize("content", ["96G Hopper", "280G Blackwell", "GLM5-744B-A40B", "910C", "P800"])
def test_gitleaks_accepts_profiles_and_non_nvidia_names(repo: Path, content: str) -> None:
    _stage(repo, "profile.txt", content)
    result = _scan(repo)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "name",
    [
        "docs/draft/distributed_checkpoint_service.md",
        "relax/utils/training/flops_counter.py",
        "relax/utils/external/command_utils.py",
        "scripts/models/deepseek-v4-flash.sh",
        "tests/utils/test_flops_counter.py",
    ],
)
def test_gitleaks_allows_annotated_sources_after_line_numbers_shift(repo: Path, name: str) -> None:
    _stage(repo, name, "\n" + (ROOT / name).read_text())
    result = _scan(repo)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("change", ["append", "remove_annotation", "copy_unannotated"])
def test_gitleaks_still_blocks_unannotated_records(repo: Path, change: str) -> None:
    name = "relax/utils/training/flops_counter.py"
    original = (ROOT / name).read_text()
    old_line = next(line for line in original.splitlines() if f'"{GPU_MODEL}"' in line)
    if change == "append":
        content = original + f"\n# training used {GPU_MODEL}\n"
    elif change == "remove_annotation":
        content = original.replace(old_line, old_line.replace("gitleaks:allow", ""))
    else:
        name, content = "new_experiment.py", old_line.split("# gitleaks:allow")[0] + "\n"
    _stage(repo, name, content)
    result = _scan(repo)
    assert result.returncode == 1
    assert "nvidia-gpu-model" in result.stdout + result.stderr


@pytest.mark.parametrize(
    "name",
    [
        ".github/ISSUE_TEMPLATE/bug_report.md",
        ".github/workflows/unittest.yml",
        ".github/workflows/integration-test.yml",
    ],
)
def test_gitleaks_native_ignore_keeps_existing_github_records(repo: Path, name: str) -> None:
    original = subprocess.run(
        ["git", "show", f"HEAD:{name}"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    _stage(repo, name, original)
    result = _scan(repo)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "comment",
    ["# gitleaks:allow -- synthetic fixture", "<!-- gitleaks:allow -- public example -->"],
)
def test_gitleaks_native_annotation_applies_only_to_its_line(repo: Path, comment: str) -> None:
    _stage(repo, "fixture.txt", f"{GPU_MODEL} {comment}\n")
    result = _scan(repo)
    assert result.returncode == 0, result.stdout + result.stderr
    _stage(repo, "fixture.txt", f"{GPU_MODEL} {comment}\n{GPU_MODEL}\n")
    result = _scan(repo)
    assert result.returncode == 1
    assert "nvidia-gpu-model" in result.stdout + result.stderr


def test_gitleaks_still_detects_secrets_outside_annotated_lines(repo: Path) -> None:
    token = "AKID" + "a" * 20
    _stage(repo, "new_config.txt", f"# {GPU_MODEL} -- gitleaks:allow\n{token}\n")
    result = _scan(repo)
    assert result.returncode == 1
    assert "token-akid" in result.stdout + result.stderr
    assert token not in result.stdout + result.stderr


def test_gitleaks_still_ignores_untracked_files(repo: Path) -> None:
    (repo / f"run-{GPU_MODEL.lower()}.sh").write_text(f"# {GPU_MODEL}\n")
    result = _scan(repo)
    assert result.returncode == 0, result.stdout + result.stderr


def test_gitleaks_manifest_does_not_overwrite_tracked_content(repo: Path) -> None:
    _stage(repo, ".gitleaks-tracked-paths-user", f"# measured on {GPU_MODEL}\n")
    result = _scan(repo)
    assert result.returncode == 1
    assert "nvidia-gpu-model" in result.stdout + result.stderr
