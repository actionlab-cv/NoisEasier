from __future__ import annotations

import fcntl
import os
import re
from collections.abc import Callable, Iterator
from pathlib import Path


VBenchPromptIterator = Callable[[str], Iterator[tuple[str, list[str]]]]
CompBenchPromptIterator = Callable[[str], Iterator[tuple[str, str, str, list[str]]]]
VBENCH_SAMPLES_PER_PROMPT = 5


def _prompt_files(prompts_dir: str | os.PathLike[str]) -> list[Path]:
    root = Path(prompts_dir)
    if root.is_file():
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"Prompt directory not found: {root}")
    return sorted(path for path in root.iterdir() if path.suffix.lower() in {".json", ".txt"})


def sanitize_filename_component(name: str, *, max_len: int = 180) -> str:
    sanitized = re.sub(r'[\\/:*?"<>|]', "", name)
    sanitized = re.sub(r"\s+", " ", sanitized).strip().strip(".")
    return (sanitized[:max_len].strip() or "sample")


def vbench_video_filename(prompt: str, index: int | str) -> str:
    return f"{prompt}-{index}.mp4"


def compbench_video_filename(sample_id: str | int) -> str:
    return f"{sanitize_filename_component(str(sample_id), max_len=80)}.mp4"


def build_vbench_tasks(
    prompts_dir: str | os.PathLike[str],
    prompt_iterator: VBenchPromptIterator,
) -> list[tuple[int, str, str, list[str], int]]:
    tasks: list[tuple[int, str, str, list[str], int]] = []
    for prompt_file in _prompt_files(prompts_dir):
        dimension = prompt_file.stem
        for prompt, neg_prompts in prompt_iterator(str(prompt_file)):
            for index in range(VBENCH_SAMPLES_PER_PROMPT):
                tasks.append((len(tasks), dimension, prompt, neg_prompts, index))
    return tasks


def build_compbench_tasks(
    prompts_dir: str | os.PathLike[str],
    prompt_iterator: CompBenchPromptIterator,
) -> list[tuple[int, str, str, str, list[str]]]:
    tasks: list[tuple[int, str, str, str, list[str]]] = []
    for prompt_file in _prompt_files(prompts_dir):
        for sample_id, prompt, category, neg_prompts in prompt_iterator(str(prompt_file)):
            tasks.append((len(tasks), sample_id, prompt, category, neg_prompts))
    return tasks


def _complete_video(path: Path, min_bytes: int) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= min_bytes
    except OSError:
        return False


def existing_video_path(
    *,
    run_root: str | os.PathLike[str],
    rank_video_dir: str | os.PathLike[str],
    filename: str,
    subdir: str | None = None,
    scope: str = "run",
    min_bytes: int = 1,
) -> str | None:
    relative = Path(subdir) / filename if subdir else Path(filename)
    rank_candidate = Path(rank_video_dir) / relative
    if _complete_video(rank_candidate, min_bytes):
        return str(rank_candidate)
    if scope == "rank":
        return None
    if scope != "run":
        raise ValueError(f"Unknown skip-existing scope: {scope}")
    for shard_dir in sorted(Path(run_root).glob("shard_*")):
        candidate = shard_dir / "all_videos" / relative
        if _complete_video(candidate, min_bytes):
            return str(candidate)
    return None


def claim_next_task(queue_dir: str | os.PathLike[str], name: str, total: int) -> int | None:
    queue = Path(queue_dir)
    queue.mkdir(parents=True, exist_ok=True)
    next_path = queue / f"{name}.next"
    lock_path = queue / f"{name}.lock"

    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            try:
                next_id = int(next_path.read_text(encoding="utf-8").strip() or "0")
            except FileNotFoundError:
                next_id = 0
            if next_id >= total:
                return None
            next_path.write_text(str(next_id + 1), encoding="utf-8")
            return next_id
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
