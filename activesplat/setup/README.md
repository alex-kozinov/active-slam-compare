# ActiveSplat Setup Notes

This directory keeps the local setup work that was previously done inside the ephemeral `/opt` workspace on RunPod.

## What is stored here

- `setup.sh`: restores the patched files into the working tree
- `resources/ActiveSplat/...`: patched ActiveSplat source and config files

## What the setup changes do

The stored patches cover the following work:

- add `Replica` dataset support files
- fix Replica Habitat mesh loading to use `habitat/mesh_semantic.ply`
- add `seed_id` support through ROS launch, mapper CLI, mapper RNGs, and Habitat start state
- provide a default `user_config.json` that points Replica to `/workspace/datasets/replica`

## Restore the patched files

From this directory run:

```bash
bash setup.sh
```

By default the script restores files into:

- `ACTIVESPLAT_ROOT=/opt/activesplat_ws/src/ActiveSplat`

You can override it if needed:

```bash
ACTIVESPLAT_ROOT=/some/other/ActiveSplat bash setup.sh
```

## Expected Replica dataset layout

The restored config assumes:

- dataset root: `/workspace/datasets/replica`
- scene layout: `/workspace/datasets/replica/SCENE_ID/...`

At minimum each scene must contain the files used by the patched loader:

- `mesh.ply`
- `habitat/mesh_semantic.ply`

## Suggested restore flow after a pod restart

1. Recreate or re-clone `ActiveSplat` into `/opt/activesplat_ws/src/ActiveSplat`.
2. Activate the environment that contains the project dependencies.
3. Run `bash setup.sh`.
4. Make sure the Replica dataset exists under `/workspace/datasets/replica`.
5. Verify that `config/user_config.json` points to `/workspace/datasets/replica`.

## Example launch command

```bash
roslaunch activesplat habitat.launch \
  config:=/opt/activesplat_ws/src/ActiveSplat/config/datasets/replica.json \
  user_config:=/opt/activesplat_ws/src/ActiveSplat/config/user_config.json \
  scene_id:=frl_apartment_0 \
  mode:=AUTO_PLANNING \
  mapper:=SplaTAM \
  gpu_id:=0 \
  hide_mapper_windows:=1 \
  hide_planner_windows:=1 \
  save_runtime_data:=0 \
  parallelized:=0 \
  debug:=0 \
  seed_id:=1 \
  remark:=replica_smoke_test
```

## Output behavior

The patched dataset loader writes results into:

- `/opt/activesplat_ws/src/ActiveSplat/results`

The result directory name includes:

- dataset format
- scene id
- `seed_id`
- optional remark

This makes repeated seeded runs easier to compare and archive.

## Scope

This setup intentionally does not restore anything into the shared top-level `experiments` area of your comparison repository.

That part stays owned by your main `active-slam-compare` project structure.
