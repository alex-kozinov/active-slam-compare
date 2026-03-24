#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import shutil
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


MOVE_FORWARD_ACTION = "1"
AVG_ITER_RE = re.compile(r"Average Mapping/Iteration Time:\s*([0-9eE+.\-]+)\s*ms")
AVG_FRAME_RE = re.compile(r"Average Mapping/Frame Time:\s*([0-9eE+.\-]+)\s*s")
FAIL_PATTERNS = (
    "process has died",
    "traceback (most recent call last):",
    "error: cannot launch node",
    "windowlesscontext: unable to create windowless context",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def as_bool_flag(value: bool) -> str:
    return "1" if value else "0"


def slug(text: str, limit: int = 48) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return (s or "run")[:limit]


def resolve_path(path_str: str, base: Path) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def first_existing(paths: Iterable[Path]) -> str | None:
    for path in paths:
        if path.exists():
            return str(path)
    return None


def ffmpeg_png_to_mp4(src_dir: Path, dst_file: Path, fps: int = 10) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not available in PATH")
    if not src_dir.exists() or not any(src_dir.glob("*.png")):
        raise FileNotFoundError(f"PNG frames not found in {src_dir}")
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        str(src_dir / "*.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(dst_file),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


@dataclass(frozen=True)
class Runtime:
    experiment_name: str
    dataset_format: str
    base_config_path: Path
    user_config_path: Path
    dataset_root: Path
    result_root: Path
    result_db_path: Path
    records_root: Path
    gpu_id: int
    mode: str
    mapper: str
    hide_mapper_windows: bool
    hide_planner_windows: bool
    save_runtime_data: bool
    parallelized: bool
    debug: bool
    activesplat_root: Path
    runner_artifacts_root: Path


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    scene_id: str
    scene_scale_class: str
    step_budget: int
    seed_id: int
    remark: str
    save_video: bool
    save_map: bool


class RunDB:
    def __init__(self, path: Path, table: str, columns: list[str], primary_key: list[str]) -> None:
        self.path = path
        self.table = table
        self.columns = columns
        self.primary_key = primary_key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._ensure_table()

    def close(self) -> None:
        self.conn.close()

    def _sql_type(self, column: str) -> str:
        integer_columns = {"gpu_id", "seed_id", "step_budget", "exit_code"}
        if column in integer_columns:
            return "INTEGER"
        if column.endswith(("_cm", "_m", "_ms", "_sec", "_percent")):
            return "REAL"
        return "TEXT"

    def _ensure_table(self) -> None:
        defs = []
        for col in self.columns:
            piece = f'"{col}" {self._sql_type(col)}'
            if col in self.primary_key and len(self.primary_key) == 1:
                piece += " PRIMARY KEY"
            defs.append(piece)
        if len(self.primary_key) > 1:
            defs.append(f"PRIMARY KEY ({', '.join(self.primary_key)})")
        sql = f'CREATE TABLE IF NOT EXISTS "{self.table}" ({", ".join(defs)})'
        self.conn.execute(sql)
        existing = {
            row["name"]
            for row in self.conn.execute(f'PRAGMA table_info("{self.table}")').fetchall()
        }
        for col in self.columns:
            if col not in existing:
                self.conn.execute(
                    f'ALTER TABLE "{self.table}" ADD COLUMN "{col}" {self._sql_type(col)}'
                )
        self.conn.commit()

    def get_status(self, run_id: str) -> str | None:
        row = self.conn.execute(
            f'SELECT status FROM "{self.table}" WHERE run_id = ?',
            (run_id,),
        ).fetchone()
        return None if row is None else row["status"]

    def upsert(self, row: dict[str, Any]) -> None:
        payload = {col: row.get(col) for col in self.columns}
        cols = list(payload)
        placeholders = ", ".join("?" for _ in cols)
        updates = ", ".join(f'"{c}" = excluded."{c}"' for c in cols if c not in self.primary_key)
        quoted_cols = ", ".join(f'"{c}"' for c in cols)
        quoted_pk = ", ".join(f'"{c}"' for c in self.primary_key)
        sql = (
            f'INSERT INTO "{self.table}" ({quoted_cols}) '
            f"VALUES ({placeholders}) "
            f"ON CONFLICT({quoted_pk}) DO UPDATE SET {updates}"
        )
        values = [self._normalize(payload[c]) for c in cols]
        self.conn.execute(sql, values)
        self.conn.commit()

    @staticmethod
    def _normalize(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=True)
        if isinstance(value, bool):
            return int(value)
        return value


def build_runtime(cfg: dict[str, Any], setup_path: Path) -> Runtime:
    meta = cfg["experiment_meta"]
    rt = cfg["runtime_defaults"]
    base_dir = setup_path.parent
    base_config_path = resolve_path(rt["base_config_path"], base_dir)
    result_root = resolve_path(rt["result_root"], base_dir)
    result_db_path = resolve_path(rt["result_db_path"], result_root)
    records_root = resolve_path(rt["records_path"], result_root)
    activesplat_root = base_config_path.parents[2]
    runner_artifacts_root = result_root / "_runner"
    return Runtime(
        experiment_name=meta["experiment_name"],
        dataset_format=meta["dataset_format"],
        base_config_path=base_config_path,
        user_config_path=resolve_path(rt["user_config_path"], base_dir),
        dataset_root=resolve_path(rt["dataset_root"], base_dir),
        result_root=result_root,
        result_db_path=result_db_path,
        records_root=records_root,
        gpu_id=int(rt["gpu_id"]),
        mode=rt["mode"],
        mapper=rt["mapper"],
        hide_mapper_windows=bool(rt["hide_mapper_windows"]),
        hide_planner_windows=bool(rt["hide_planner_windows"]),
        save_runtime_data=bool(rt["save_runtime_data"]),
        parallelized=bool(rt["parallelized"]),
        debug=bool(rt["debug"]),
        activesplat_root=activesplat_root,
        runner_artifacts_root=runner_artifacts_root,
    )


def build_runs(cfg: dict[str, Any], scenes: set[str] | None, seeds: set[int] | None, limit: int | None) -> list[RunSpec]:
    exp = cfg["experiment_meta"]["experiment_name"]
    runs: list[RunSpec] = []
    for scene_id, scene in cfg["scenes_description"].items():
        if not scene.get("enabled", True):
            continue
        if scenes and scene_id not in scenes:
            continue
        for seed_id in scene["seed_ids"]:
            seed_id = int(seed_id)
            if seeds and seed_id not in seeds:
                continue
            run_id = f"{slug(exp, 24)}__{scene_id}__seed_{seed_id}__steps_{int(scene['step_budget'])}"
            runs.append(
                RunSpec(
                    run_id=run_id,
                    scene_id=scene_id,
                    scene_scale_class=scene["scene_scale_class"],
                    step_budget=int(scene["step_budget"]),
                    seed_id=seed_id,
                    remark=slug(run_id, 60),
                    save_video=seed_id in set(scene.get("save_video_seed_ids", [])),
                    save_map=seed_id in set(scene.get("save_map_seed_ids", [])),
                )
            )
            if limit is not None and len(runs) >= limit:
                return runs
    return runs


def db_columns(cfg: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    scheme = cfg["result_db"]["db_scheme"]
    groups = [
        "primary_key",
        "run_identity",
        "config_snapshot",
        "timing",
        "status",
        "artifact_paths",
        "metrics",
    ]
    seen, cols = set(), []
    for group in groups:
        for col in scheme[group]:
            if col not in seen:
                seen.add(col)
                cols.append(col)
    return scheme["table_name"], cols, scheme["primary_key"]


def resolve_env_config(base_config_path: Path, cfg: dict[str, Any]) -> Path:
    rel = cfg["env"]["config"]
    return (base_config_path.parent.parent / rel).resolve()


def forward_step_size(base_config_path: Path, cfg: dict[str, Any]) -> float | None:
    env_cfg = resolve_env_config(base_config_path, cfg)
    if not env_cfg.exists():
        return None
    try:
        return float(load_yaml(env_cfg)["habitat"]["simulator"]["forward_step_size"])
    except Exception:
        return None


def make_run_config(runtime: Runtime, run: RunSpec) -> Path:
    cfg = load_json(runtime.base_config_path)
    cfg.setdefault("dataset", {})
    cfg["dataset"]["scene_id"] = run.scene_id
    cfg["dataset"]["step_num"] = run.step_budget
    cfg["dataset"]["seed_id"] = run.seed_id
    base_root = runtime.activesplat_root
    for section, key in (("env", "config"), ("sensor", "config")):
        if section in cfg and key in cfg[section]:
            cfg[section][key] = str(resolve_path(cfg[section][key], base_root))
    if "mapper" in cfg and "splatam_cfg_path" in cfg["mapper"]:
        cfg["mapper"]["splatam_cfg_path"] = str(resolve_path(cfg["mapper"]["splatam_cfg_path"], runtime.activesplat_root))
    path = runtime.runner_artifacts_root / "configs" / f"{run.run_id}.json"
    dump_json(path, cfg)
    return path


def _roslaunch_argv(runtime: Runtime, run: RunSpec, run_config_path: Path) -> list[str]:
    return [
        "roslaunch",
        "activesplat",
        "habitat.launch",
        f"mapper:={runtime.mapper}",
        f"config:={run_config_path}",
        f"scene_id:={run.scene_id}",
        f"user_config:={runtime.user_config_path}",
        f"gpu_id:={runtime.gpu_id}",
        f"mode:={runtime.mode}",
        f"parallelized:={as_bool_flag(runtime.parallelized)}",
        f"hide_mapper_windows:={as_bool_flag(runtime.hide_mapper_windows)}",
        f"hide_planner_windows:={as_bool_flag(runtime.hide_planner_windows)}",
        f"save_runtime_data:={as_bool_flag(runtime.save_runtime_data or run.save_video or run.save_map)}",
        f"debug:={as_bool_flag(runtime.debug)}",
        f"seed_id:={run.seed_id}",
        f"remark:={run.remark}",
    ]


def _wrap_roslaunch_for_catkin_conda(runtime: Runtime, ros_argv: list[str]) -> tuple[list[str], str]:
    """
    Same shell recipe as a working interactive run:
    source conda.sh && conda activate ActiveSplat && source devel/setup.bash && roslaunch ...

    Uses bash -i -lc (interactive login) so ~/.bashrc / profile hooks run — on RunPod
    GPU/EGL vars are often only set there; plain -lc matches a «bare» subprocess and can
    reproduce EGL_BAD_PARAMETER even when the same command works in your terminal.
    """
    catkin_ws = Path(os.environ.get("ACTIVE_CATKIN_WS", str(runtime.activesplat_root.parent.parent)))
    conda_sh = Path(os.environ.get("ACTIVE_CONDA_SH", "/opt/conda/etc/profile.d/conda.sh"))
    conda_env = os.environ.get("ACTIVE_CONDA_ENV", "ActiveSplat")
    devel_setup = catkin_ws / "devel" / "setup.bash"
    ros_line = " ".join(shlex.quote(a) for a in ros_argv)
    if devel_setup.is_file() and conda_sh.is_file():
        inner = (
            f"set -euo pipefail; "
            f"source {shlex.quote(str(conda_sh))}; "
            f"conda activate {shlex.quote(conda_env)}; "
            f"source {shlex.quote(str(devel_setup))}; "
            f"cd {shlex.quote(str(runtime.activesplat_root))}; "
            f"exec {ros_line}"
        )
        cmd = ["bash", "-i", "-lc", inner]
        note = f"conda+devel wrapper, interactive bash (catkin_ws={catkin_ws}, env={conda_env})"
    else:
        cmd = ros_argv
        note = "plain roslaunch (missing conda.sh or devel/setup.bash; set ACTIVE_CATKIN_WS / ACTIVE_CONDA_SH)"
    return cmd, note


def run_command(runtime: Runtime, run: RunSpec, run_config_path: Path, log_path: Path) -> subprocess.CompletedProcess[str]:
    ros_argv = _roslaunch_argv(runtime, run, run_config_path)
    cmd, wrapper_note = _wrap_roslaunch_for_catkin_conda(runtime, ros_argv)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write(f"# {wrapper_note}\n")
        log_cmd = " ".join(shlex.quote(x) for x in cmd)
        log.write("COMMAND: " + log_cmd + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=runtime.activesplat_root,
            env=os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        forced_failure = False
        while True:
            code = proc.poll()
            if code is not None:
                return subprocess.CompletedProcess(cmd, code)
            if detect_launch_failure(log_path) is not None:
                forced_failure = True
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                for _ in range(20):
                    code = proc.poll()
                    if code is not None:
                        return subprocess.CompletedProcess(cmd, code)
                    time.sleep(0.25)
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                code = proc.wait()
                return subprocess.CompletedProcess(cmd, code if not forced_failure else max(code, 1))
            time.sleep(0.5)


def find_result_dir(runtime: Runtime, run: RunSpec) -> Path | None:
    if not runtime.result_root.exists():
        return None
    candidates = [
        p for p in runtime.result_root.iterdir()
        if p.is_dir()
        and f"_{runtime.dataset_format}_{run.scene_id}_seed_{run.seed_id}" in p.name
        and p.name.endswith(f"_{run.remark}")
    ]
    return max(candidates, key=lambda p: p.stat().st_mtime, default=None)


def parse_log_metrics(log_path: Path) -> dict[str, float | None]:
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    iter_match = AVG_ITER_RE.search(text)
    frame_match = AVG_FRAME_RE.search(text)
    return {
        "avg_mapping_iteration_ms": float(iter_match.group(1)) if iter_match else None,
        "avg_mapping_frame_sec": float(frame_match.group(1)) if frame_match else None,
    }


def detect_launch_failure(log_path: Path) -> str | None:
    if not log_path.exists():
        return "runner log was not created"
    text = log_path.read_text(errors="replace")
    low = text.lower()
    for pattern in FAIL_PATTERNS:
        if pattern in low:
            lines = [line.strip() for line in text.splitlines() if pattern in line.lower()]
            return lines[-1] if lines else pattern
    return None


def path_length_m(actions_path: Path | None, step_size: float | None) -> float | None:
    if actions_path is None or step_size is None or not actions_path.exists():
        return None
    forward_count = sum(1 for line in actions_path.read_text().splitlines() if line.strip() == MOVE_FORWARD_ACTION)
    return forward_count * step_size


def collect_artifacts(runtime: Runtime, run: RunSpec, result_dir: Path | None) -> tuple[dict[str, str | None], str | None]:
    artifacts = {
        "result_dir": str(result_dir) if result_dir else None,
        "actions_path": None,
        "video_rgb_path": None,
        "video_keyframes_path": None,
        "topdown_map_path": None,
        "visited_map_path": None,
        "gaussians_path": None,
        "transforms_path": None,
    }
    if result_dir is None or not result_dir.exists():
        return artifacts, "result directory was not found after roslaunch finished"

    actions_path = result_dir / "actions.txt"
    if actions_path.exists():
        artifacts["actions_path"] = str(actions_path)

    artifacts["visited_map_path"] = first_existing([result_dir / "visited_map.png"])
    artifacts["topdown_map_path"] = first_existing([result_dir / "topdown_free_map.png"])

    gaussians_dir = result_dir / "gaussians_data"
    if gaussians_dir.exists():
        artifacts["gaussians_path"] = str(first_existing([gaussians_dir / "params.npz", gaussians_dir]) or gaussians_dir)
        artifacts["transforms_path"] = first_existing([gaussians_dir / "transforms.json"])

    cleanup_errors: list[str] = []
    if run.save_video:
        record_mp4 = runtime.records_root / run.scene_id / str(run.seed_id) / "record.mp4"
        try:
            ffmpeg_png_to_mp4(result_dir / "render_rgbd", record_mp4)
            artifacts["video_rgb_path"] = str(record_mp4)
        except Exception as e:
            cleanup_errors.append(f"rgb video: {e}")
        keyframes_mp4 = runtime.records_root / run.scene_id / str(run.seed_id) / "keyframes.mp4"
        try:
            ffmpeg_png_to_mp4(gaussians_dir / "keyframes", keyframes_mp4)
            artifacts["video_keyframes_path"] = str(keyframes_mp4)
        except Exception:
            pass
    return artifacts, "; ".join(cleanup_errors) or None


def row_template(runtime: Runtime, run: RunSpec) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "experiment_name": runtime.experiment_name,
        "scene_id": run.scene_id,
        "scene_scale_class": run.scene_scale_class,
        "seed_id": run.seed_id,
        "step_budget": run.step_budget,
        "dataset_format": runtime.dataset_format,
        "base_config_path": str(runtime.base_config_path),
        "user_config_path": str(runtime.user_config_path),
        "dataset_root": str(runtime.dataset_root),
        "gpu_id": runtime.gpu_id,
        "mapper": runtime.mapper,
        "mode": runtime.mode,
        "started_at": None,
        "finished_at": None,
        "elapsed_sec": None,
        "status": "pending",
        "exit_code": None,
        "error_message": None,
        "cleanup_error": None,
        "result_dir": None,
        "actions_path": None,
        "video_rgb_path": None,
        "video_keyframes_path": None,
        "topdown_map_path": None,
        "visited_map_path": None,
        "gaussians_path": None,
        "transforms_path": None,
        "accuracy_cm": None,
        "completion_cm": None,
        "completion_ratio_percent": None,
        "path_length_m": None,
        "avg_mapping_iteration_ms": None,
        "avg_mapping_frame_sec": None,
    }


def launch_run(db: RunDB, cfg: dict[str, Any], runtime: Runtime, run: RunSpec, rerun: bool) -> None:
    existing = db.get_status(run.run_id)
    if existing in {"completed", "completed_with_cleanup_error"} and not rerun:
        print(f"[skip] {run.run_id} ({existing})")
        return

    row = row_template(runtime, run)
    row["status"] = "running"
    row["started_at"] = now_iso()
    db.upsert(row)

    t0 = time.time()
    run_cfg = make_run_config(runtime, run)
    log_path = runtime.runner_artifacts_root / "logs" / f"{run.run_id}.log"
    proc = run_command(runtime, run, run_cfg, log_path)
    row["finished_at"] = now_iso()
    row["elapsed_sec"] = round(time.time() - t0, 3)
    row["exit_code"] = proc.returncode

    result_dir = find_result_dir(runtime, run)
    artifacts, cleanup_error = collect_artifacts(runtime, run, result_dir)
    row.update(artifacts)
    row["cleanup_error"] = cleanup_error

    base_cfg = load_json(runtime.base_config_path)
    step_size = forward_step_size(runtime.base_config_path, base_cfg)
    actions_path = Path(artifacts["actions_path"]) if artifacts["actions_path"] else None
    row["path_length_m"] = path_length_m(actions_path, step_size)
    row.update(parse_log_metrics(log_path))
    launch_error = detect_launch_failure(log_path)

    if launch_error is not None:
        row["status"] = "failed"
        row["error_message"] = launch_error
    elif proc.returncode == 0:
        row["status"] = "completed_with_cleanup_error" if cleanup_error else "completed"
    else:
        row["status"] = "failed"
        row["error_message"] = f"roslaunch exited with code {proc.returncode}; see {log_path}"

    db.upsert(row)
    print(f"[{row['status']}] {run.run_id}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ActiveSplat experiments from setup.json")
    p.add_argument("config", type=Path, help="Path to experiment setup.json")
    p.add_argument("--scene", action="append", default=[], help="Only run selected scene_id (repeatable)")
    p.add_argument("--seed", action="append", type=int, default=[], help="Only run selected seed_id (repeatable)")
    p.add_argument("--limit", type=int, default=None, help="Stop after N concrete runs")
    p.add_argument("--rerun", action="store_true", help="Rerun completed rows")
    p.add_argument("--dry-run", action="store_true", help="Only print the expanded run list")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_json(args.config.resolve())
    runtime = build_runtime(cfg, args.config.resolve())
    table, columns, primary_key = db_columns(cfg)
    runs = build_runs(
        cfg,
        scenes=set(args.scene) or None,
        seeds=set(args.seed) or None,
        limit=args.limit,
    )

    if not runs:
        print("No runs selected.")
        return 0

    print(f"Selected {len(runs)} runs.")
    for run in runs:
        print(f" - {run.run_id}")
    if args.dry_run:
        return 0

    db = RunDB(runtime.result_db_path, table, columns, primary_key)
    try:
        for run in runs:
            launch_run(db, cfg, runtime, run, rerun=args.rerun)
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
