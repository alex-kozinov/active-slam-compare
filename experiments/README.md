# Experiment Setup

This directory stores experiment configuration files for batch runs on the `Replica` dataset.

## Files

- `setup.json`: main experiment definition

## Design Goals

The setup file is designed to support:

- batch execution across many scenes
- multiple seeded runs per scene
- geometry-only evaluation
- result tracking in a SQLite database
- explicit artifact bookkeeping for videos, maps, and saved Gaussian outputs

## Top-Level Sections

### `experiment_meta`

General metadata for the whole benchmark:

- experiment name
- dataset name and format
- creation date
- short description
- owner

### `runtime_defaults`

Shared runtime settings used unless a scene overrides them.

Typical fields:

- base config path
- user config path
- dataset root
- result root
- result database path
- records path
- GPU id
- mapper and mode
- UI flags
- step budget policy for `small` and `medium` scenes

### `result_db`

Defines how the run table is structured.

`runtime_defaults.result_db_path` is intended to be interpreted relative to `runtime_defaults.result_root`.
`runtime_defaults.records_path` is also intended to be interpreted relative to `runtime_defaults.result_root`.

Recorded videos are expected to follow this layout:

- `SCENE_ID/SEED_ID/record.mp4`

The schema is grouped into:

- `run_identity`
- `config_snapshot`
- `timing`
- `status`
- `artifact_paths`
- `metrics`

## Metrics

This setup intentionally excludes rendering-quality metrics such as `PSNR`, `SSIM`, `LPIPS`, and `Depth L1`.

The experiment is geometry-focused and only tracks:

- `accuracy_cm`
- `completion_cm`
- `completion_ratio_percent`

The completion threshold is fixed to:

- `completion_threshold_cm = 5`

## Scene Definition

Each entry inside `scenes_description` is keyed by `scene_id`.

Each scene contains:

- `scene_scale_class`
- `step_budget`
- `seed_ids`
- `save_video_seed_ids`
- `save_map_seed_ids`
- `enabled`

## Seed Semantics

`seed_id` is intended to be a first-class run parameter.

The intended meaning is:

- `seed_id` controls the start state of the environment
- `seed_id` controls mapper-side random number generators
- `seed_id` controls simulator-side random number generators

In other words, each `(scene_id, seed_id, step_budget)` tuple should describe one reproducible run.

## Artifact Paths

The database schema includes paths for generated outputs such as:

- RGB run video
- keyframe video
- topdown map
- visited map
- saved Gaussian parameters
- transforms JSON

## Notes

This file is a benchmark template, not an executor.

The execution layer is expected to:

- read `setup.json`
- generate concrete runs
- pass `seed_id` into the runtime
- compute the geometry metrics
- write one row per run into the result database
