# TaskDog FloAff dataset generation

This package builds the two-stage AnchorAxisNet/FloAff dataset without editing
the existing N2M or SysNav implementation.

## Dataset units

- **placement**: one physical cube pose;
- **instance**: a placement conditioned on one operation edge;
- **M**: one manipulation base pose and its repeated grasp outcomes;
- **O**: one front-camera observation with MuJoCo-truth target/anchor clouds.

A corner placement creates two instances. Both use the corner O profile, so
their camera sectors remain disjoint.

The initial configuration generates 20 left-short-edge placements and 30
left/top-corner placements. Optional `transfer_*` counts in the YAML generate
the other short edge, the other three corners and both long edges; these
placements are automatically assigned to `transfer_test`.

## Stages

Run commands from the repository root. The default configuration is
`custom_envs/scripts/FloAff/configs/data_generation.yaml`.

To collect and process the complete pre-training dataset without manually
starting each stage, first start `moveit_mujoco.launch.py` in another terminal
and then run:

```bash
conda activate env_isaaclab
cd /home/mojie/taskdog
custom_envs/scripts/FloAff/collect_dataset.sh
```

The wrapper checks that MoveIt is ready, generates the deterministic manifests,
collects all M with `--resume`, and starts all O with `--resume` only after M
exits successfully. It then builds the placement-level splits, validates the
raw collection, generates dense affordance fields, builds the fixed-size
AnchorAxisNet samples, and writes the final validation report. It also prevents
automatic suspend with `systemd-inhibit`; screen blanking remains enabled. Pass
`--allow-sleep` to disable that behavior.

This produces everything needed to train AnchorAxisNet. The O-specific final
FloAff samples are intentionally built later: they must use predictions from a
trained, frozen AnchorAxisNet rather than ground-truth axes.

After the initial training collection is complete, append the held-out transfer
split without recollecting existing samples:

```bash
conda activate env_isaaclab
cd /home/mojie/taskdog
custom_envs/scripts/FloAff/collect_transfer_test.sh
```

The default transfer plan contains 4 placements on the unseen short edge, 3
placements in each of the other three corners, and 4 placements on each long
edge: 21 placements and 30 edge-conditioned instances in total. Each transfer
instance uses 40 M and 20 O, while the existing training instances retain 100 M
and 40 O. The command is resumable and never recollects completed training
instances.

```bash
python -m custom_envs.scripts.FloAff.data_generation.generate_instances \
  --config custom_envs/scripts/FloAff/configs/data_generation.yaml

# Generate M candidates only (no MuJoCo/MoveIt execution).
python -m custom_envs.scripts.FloAff.data_generation.collect_manipulation_labels \
  --config custom_envs/scripts/FloAff/configs/data_generation.yaml \
  --dry-run

# Collect manipulation labels. Every non-deterministic M is attempted three
# times; mixed 1/3 or 2/3 outcomes receive two additional attempts.
python -m custom_envs.scripts.FloAff.data_generation.collect_manipulation_labels \
  --config custom_envs/scripts/FloAff/configs/data_generation.yaml \
  --grasp-checkpoint /home/mojie/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar \
  --resume

python -m custom_envs.scripts.FloAff.data_generation.capture_observations \
  --config custom_envs/scripts/FloAff/configs/data_generation.yaml \
  --resume --save-previews

python -m custom_envs.scripts.FloAff.data_generation.build_splits \
  --config custom_envs/scripts/FloAff/configs/data_generation.yaml

python -m custom_envs.scripts.FloAff.data_generation.affordance_labels \
  --config custom_envs/scripts/FloAff/configs/data_generation.yaml

python -m custom_envs.scripts.FloAff.data_generation.build_axis_dataset \
  --config custom_envs/scripts/FloAff/configs/data_generation.yaml

python -m custom_envs.scripts.FloAff.data_generation.validate_dataset \
  --config custom_envs/scripts/FloAff/configs/data_generation.yaml \
  --require-complete
```

After AnchorAxisNet is trained, create a JSONL file containing one row per O:

```json
{"observation_id":"...","axis_xy_base":[0.12,0.99],"confidence":0.94}
```

Then build O-specific FloAff samples with the frozen predictions:

```bash
python -m custom_envs.scripts.FloAff.data_generation.build_floaff_dataset \
  --config custom_envs/scripts/FloAff/configs/data_generation.yaml \
  --axis-predictions /path/to/axis_predictions.jsonl
```

## Output layout

```text
data/dataset_v1/
├── placements.jsonl
├── instances.jsonl
├── dataset_config.json
├── manipulation/
│   ├── candidates.jsonl
│   ├── rollouts.jsonl
│   └── summaries.jsonl
├── observations/
│   ├── manifest.jsonl
│   ├── rejected.jsonl
│   └── clouds/*.npz
├── affordance_fields/*.npz
├── splits.json
└── processed/
    ├── axis/
    └── floaff/
```

The raw M rollouts and O clouds remain authoritative. Point counts, query-grid
resolution, kernel bandwidth and train splits can be rebuilt without repeating
the MuJoCo experiments.
