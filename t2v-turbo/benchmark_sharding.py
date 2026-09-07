from __future__ import annotations

import fcntl
import glob
import os
from collections.abc import Callable, Iterable

VBenchTask = tuple[int, str, str, list[str], int]
CompBenchTask = tuple[int, str, str, str, list[str]]


def parse_vbench_dimensions(values: Iterable[str] | str | None) -> list[str] | None:
    if values is None:
        return None

    if isinstance(values, str):
        raw_values = [values]
    else:
        raw_values = list(values)

    dimensions = []
    seen = set()
    for value in raw_values:
        for dim in value.replace(",", " ").split():
            key = dim.strip().lower()
            if key and key not in seen:
                dimensions.append(key)
                seen.add(key)

    return dimensions or None


def vbench_video_filename(prompt: str, index: int | str) -> str:
    return f"{prompt}-{index}.mp4"


def compbench_video_filename(sample_id: str) -> str:
    return f"{sample_id}.mp4"


def build_vbench_tasks(
    prompts_dir: str,
    iter_prompt_file: Callable[[str], Iterable[tuple[str, list[str]]]],
    dimensions: Iterable[str] | str | None = None,
) -> list[VBenchTask]:
    selected_dimensions = parse_vbench_dimensions(dimensions)
    selected_set = set(selected_dimensions) if selected_dimensions else None
    prompt_files = sorted(glob.glob(os.path.join(prompts_dir, "*.json")))
    if not prompt_files:
        prompt_files = sorted(glob.glob(os.path.join(prompts_dir, "*.txt")))
    if not prompt_files:
        raise FileNotFoundError(f"No VBench prompt files found in {prompts_dir}")

    tasks = []
    available_dims = []
    for prompt_file in prompt_files:
        dim = os.path.splitext(os.path.basename(prompt_file))[0]
        dim_key = dim.lower()
        available_dims.append(dim)
        if selected_set is not None and dim_key not in selected_set:
            continue
        n = 25 if dim.lower() == "temporal_flickering" else 5
        for ori_prompt, neg_prompts in iter_prompt_file(prompt_file):
            for idx in range(n):
                tasks.append((len(tasks), dim, ori_prompt, neg_prompts, idx))
    if selected_set is not None:
        available = ", ".join(available_dims)
        available_set = {dim.lower() for dim in available_dims}
        missing = [dim for dim in selected_dimensions if dim not in available_set]
        if missing:
            requested = ", ".join(missing)
            raise ValueError(
                f"Unknown VBench dimension(s): {requested}. "
                f"Available dimensions: {available}"
            )
        if not tasks:
            requested = ", ".join(selected_dimensions)
            raise ValueError(
                f"No VBench prompts matched requested dimension(s): {requested}. "
                f"Available dimensions: {available}"
            )
    return tasks


def build_compbench_tasks(
    prompts_dir: str,
    iter_prompt_file: Callable[[str], Iterable[tuple[str, str, str, list[str]]]],
) -> list[CompBenchTask]:
    prompt_files = sorted(glob.glob(os.path.join(prompts_dir, "*.json")))
    if not prompt_files:
        raise FileNotFoundError(f"No T2V-CompBench JSON files found in {prompts_dir}")

    tasks = []
    for prompt_file in prompt_files:
        for sample_id, prompt, category, neg_prompts in iter_prompt_file(prompt_file):
            tasks.append((len(tasks), sample_id, prompt, category, neg_prompts))
    return tasks


def claim_next_task(queue_dir: str, queue_name: str, total_tasks: int) -> int | None:
    os.makedirs(queue_dir, exist_ok=True)
    lock_path = os.path.join(queue_dir, f"{queue_name}.lock")
    next_path = os.path.join(queue_dir, f"{queue_name}.next")
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            try:
                with open(next_path, "r") as f:
                    next_task = int(f.read().strip() or "0")
            except FileNotFoundError:
                next_task = 0

            if next_task >= total_tasks:
                return None

            with open(next_path, "w") as f:
                f.write(f"{next_task + 1}\n")
                f.flush()
                os.fsync(f.fileno())
            return next_task
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def existing_video_path(
    *,
    run_root: str,
    rank_video_dir: str,
    filename: str,
    subdir: str | None = None,
    scope: str = "run",
    min_bytes: int = 1,
) -> str | None:
    if scope == "rank":
        candidates = [os.path.join(rank_video_dir, subdir, filename) if subdir else os.path.join(rank_video_dir, filename)]
    else:
        candidates = [
            os.path.join(video_dir, subdir, filename) if subdir else os.path.join(video_dir, filename)
            for video_dir in glob.glob(os.path.join(run_root, "shard_*", "all_videos"))
        ]

    for path in candidates:
        if os.path.isfile(path) and os.path.getsize(path) >= min_bytes:
            return path
    return None
