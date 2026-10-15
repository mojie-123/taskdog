# Anonymous room-map generator

This tool derives anonymous room regions from structural MuJoCo wall geometry.
It only reads direct `worldbody` box geoms named `wall_*`; XML comments,
furniture bodies and task objects are not inspected.

```bash
python custom_envs/scripts/map/generate_room_id_map.py \
  --scene custom_envs/mujoco/scene_house_N2M.xml \
  --map-yaml custom_envs/maps/map_house_mujoco_nav2.yaml \
  --out-prefix custom_envs/maps/map_house_mujoco_rooms
```

The command writes:

- `map_house_mujoco_rooms.yaml`: runtime entry point and anonymous room graph;
- `map_house_mujoco_rooms.npz`: room, searchability, wall and portal rasters;
- `map_house_mujoco_rooms_debug.png`: inspection-only colour overlay.

Load the runtime artifact with:

```python
from custom_envs.scripts.map import RoomMap

room_map = RoomMap.load("custom_envs/maps/map_house_mujoco_rooms.yaml")
room_id = room_map.room_id_at_world(x, y)
```

The generated artifacts intentionally contain only `room_N` and `portal_N`
identifiers. Room semantics must be inferred later from robot observations.
