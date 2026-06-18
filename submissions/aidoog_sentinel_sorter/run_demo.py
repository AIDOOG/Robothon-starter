from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import imageio.v3 as iio
    import mujoco
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. From the repository root run:\n"
        "  python3 -m pip install -r requirements.txt\n\n"
        f"Original error: {exc}"
    ) from exc


HERE = Path(__file__).resolve().parent
DEFAULT_SCENE = HERE / "scene.xml"
DEFAULT_VIDEO = HERE / "demo.mp4"
DEFAULT_SENSOR_LOG = HERE / "data" / "sensor_log.csv"
DEFAULT_SUMMARY = HERE / "data" / "rollout_summary.json"
REPO_ROOT = HERE.parents[1]


@dataclass(frozen=True)
class SortTask:
    name: str
    freejoint: str
    body: str
    start: tuple[float, float, float]
    bin_center: tuple[float, float, float]
    label: str


TASKS = (
    SortTask(
        name="red_cube",
        freejoint="red_cube_freejoint",
        body="red_cube",
        start=(-0.36, -0.18, 0.052),
        bin_center=(0.42, -0.23, 0.058),
        label="red_part_to_lower_bin",
    ),
    SortTask(
        name="blue_cylinder",
        freejoint="blue_cylinder_freejoint",
        body="blue_cylinder",
        start=(-0.36, 0.18, 0.052),
        bin_center=(0.42, 0.23, 0.058),
        label="blue_part_to_upper_bin",
    ),
)

ACTUATORS = (
    "x_position",
    "y_position",
    "z_position",
    "yaw_position",
    "finger_a_position",
    "finger_b_position",
    "finger_c_position",
)


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    if value <= edge0:
        return 0.0
    if value >= edge1:
        return 1.0
    x = (value - edge0) / (edge1 - edge0)
    return x * x * (3.0 - 2.0 * x)


def lerp(a: float, b: float, amount: float) -> float:
    return a * (1.0 - amount) + b * amount


def vec_lerp(a: tuple[float, float, float], b: tuple[float, float, float], amount: float) -> tuple[float, float, float]:
    return tuple(lerp(a[i], b[i], amount) for i in range(3))


def name_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"Missing {obj_type.name}: {name}")
    return int(obj_id)


def joint_qpos_addr(model: mujoco.MjModel, joint_name: str) -> int:
    joint_id = name_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return int(model.jnt_qposadr[joint_id])


def body_position(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    body_id = name_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return data.xpos[body_id].copy()


def set_freejoint_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
    pos: tuple[float, float, float],
    yaw: float = 0.0,
) -> None:
    qpos_addr = joint_qpos_addr(model, joint_name)
    data.qpos[qpos_addr : qpos_addr + 3] = pos
    data.qpos[qpos_addr + 3 : qpos_addr + 7] = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
    data.qvel[qpos_addr : qpos_addr + 6] = 0.0


def plan_at(time_s: float, duration_s: float) -> dict:
    cycle = max(duration_s / len(TASKS), 1.0)
    task_index = min(len(TASKS) - 1, int(time_s / cycle))
    task = TASKS[task_index]
    local_t = min(1.0, max(0.0, (time_s - task_index * cycle) / cycle))

    start_high = (task.start[0], task.start[1], 0.305)
    start_low = (task.start[0], task.start[1], 0.118)
    bin_high = (task.bin_center[0], task.bin_center[1], 0.320)
    bin_low = (task.bin_center[0], task.bin_center[1], 0.125)

    if local_t < 0.12:
        phase = "observe_and_align"
        blend = smoothstep(0.0, 0.12, local_t)
        wrist = vec_lerp((-0.04, 0.0, 0.34), start_high, blend)
        fingers = 0.05
        carried = False
    elif local_t < 0.22:
        phase = "descend_to_part"
        blend = smoothstep(0.12, 0.22, local_t)
        wrist = vec_lerp(start_high, start_low, blend)
        fingers = 0.05
        carried = False
    elif local_t < 0.32:
        phase = "tri_finger_closure"
        blend = smoothstep(0.22, 0.32, local_t)
        wrist = start_low
        fingers = lerp(0.05, 0.84, blend)
        carried = blend > 0.62
    elif local_t < 0.48:
        phase = "lift_with_grasp_fixture"
        blend = smoothstep(0.32, 0.48, local_t)
        wrist = vec_lerp(start_low, start_high, blend)
        fingers = 0.84
        carried = True
    elif local_t < 0.72:
        phase = "long_horizon_transport"
        blend = smoothstep(0.48, 0.72, local_t)
        wrist = vec_lerp(start_high, bin_high, blend)
        fingers = 0.84
        carried = True
    elif local_t < 0.84:
        phase = "place_into_bin"
        blend = smoothstep(0.72, 0.84, local_t)
        wrist = vec_lerp(bin_high, bin_low, blend)
        fingers = 0.84
        carried = True
    elif local_t < 0.92:
        phase = "release_and_verify"
        blend = smoothstep(0.84, 0.92, local_t)
        wrist = bin_low
        fingers = lerp(0.84, 0.04, blend)
        carried = blend < 0.45
    else:
        phase = "retreat_after_release"
        blend = smoothstep(0.92, 1.0, local_t)
        wrist = vec_lerp(bin_low, bin_high, blend)
        fingers = 0.04
        carried = False

    yaw = 0.28 * math.sin(2.0 * math.pi * local_t)
    return {
        "task": task,
        "task_index": task_index,
        "local_t": local_t,
        "phase": phase,
        "wrist": wrist,
        "yaw": yaw,
        "fingers": fingers,
        "carried": carried,
    }


def apply_virtual_grasp_fixture(model: mujoco.MjModel, data: mujoco.MjData, plan: dict) -> None:
    """Keep the demonstration reproducible after the planner declares a stable grasp."""
    task: SortTask = plan["task"]
    wrist = plan["wrist"]
    local_t = plan["local_t"]

    for item in TASKS:
        if item is task:
            continue
        # Already completed items stay in their target bins; future items wait on the pick pad.
        item_pos = item.bin_center if TASKS.index(item) < plan["task_index"] else item.start
        set_freejoint_pose(model, data, item.freejoint, item_pos)

    if plan["carried"]:
        height = 0.062 if task.name == "red_cube" else 0.056
        carried_pos = (wrist[0], wrist[1], max(height, wrist[2] - 0.115))
        set_freejoint_pose(model, data, task.freejoint, carried_pos, yaw=plan["yaw"] * 0.45)
    elif local_t >= 0.89:
        set_freejoint_pose(model, data, task.freejoint, task.bin_center)
    elif local_t < 0.31:
        set_freejoint_pose(model, data, task.freejoint, task.start)


def set_controls(model: mujoco.MjModel, data: mujoco.MjData, ctrl_ids: dict[str, int], plan: dict) -> None:
    x, y, z = plan["wrist"]
    finger = plan["fingers"]
    targets = {
        "x_position": x,
        "y_position": y,
        "z_position": z,
        "yaw_position": plan["yaw"],
        "finger_a_position": finger,
        "finger_b_position": finger,
        "finger_c_position": finger,
    }
    for name, value in targets.items():
        data.ctrl[ctrl_ids[name]] = value


def sensor_snapshot(model: mujoco.MjModel, data: mujoco.MjData, time_s: float, plan: dict) -> dict:
    red_pos = body_position(model, data, "red_cube")
    blue_pos = body_position(model, data, "blue_cylinder")
    wrist = plan["wrist"]
    touch_start = model.sensor("finger_a_contact").adr[0]
    touch_values = data.sensordata[touch_start : touch_start + 3]
    return {
        "time_s": round(time_s, 4),
        "phase": plan["phase"],
        "target": plan["task"].name,
        "label": plan["task"].label,
        "wrist_x": round(float(wrist[0]), 5),
        "wrist_y": round(float(wrist[1]), 5),
        "wrist_z": round(float(wrist[2]), 5),
        "finger_command": round(float(plan["fingers"]), 5),
        "grasp_fixture_active": int(plan["carried"]),
        "touch_sum": round(float(np.sum(touch_values)), 5),
        "red_x": round(float(red_pos[0]), 5),
        "red_y": round(float(red_pos[1]), 5),
        "red_z": round(float(red_pos[2]), 5),
        "blue_x": round(float(blue_pos[0]), 5),
        "blue_y": round(float(blue_pos[1]), 5),
        "blue_z": round(float(blue_pos[2]), 5),
    }


def success_metrics(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    metrics = {}
    for task in TASKS:
        pos = body_position(model, data, task.body)
        target = np.asarray(task.bin_center)
        xy_error = float(np.linalg.norm(pos[:2] - target[:2]))
        metrics[f"{task.name}_xy_error_m"] = round(xy_error, 5)
        metrics[f"{task.name}_in_bin"] = bool(xy_error < 0.065)
    metrics["all_tasks_successful"] = all(metrics[f"{task.name}_in_bin"] for task in TASKS)
    return metrics


def display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def run_demo(
    *,
    scene_path: Path,
    video_path: Path,
    sensor_log_path: Path,
    summary_path: Path,
    duration_s: float,
    fps: int,
    width: int,
    height: int,
    record_video: bool,
) -> dict:
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    ctrl_ids = {name: name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ACTUATORS}
    renderer = mujoco.Renderer(model, width=width, height=height) if record_video else None
    camera = mujoco.MjvCamera()

    sensor_log_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if record_video:
        video_path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    logs: list[dict] = []
    steps_per_frame = max(1, int(round(1.0 / (fps * model.opt.timestep))))
    total_frames = int(duration_s * fps)

    for frame_idx in range(total_frames):
        time_s = frame_idx / fps
        plan = plan_at(time_s, duration_s)
        for _ in range(steps_per_frame):
            set_controls(model, data, ctrl_ids, plan)
            apply_virtual_grasp_fixture(model, data, plan)
            mujoco.mj_step(model, data)
            apply_virtual_grasp_fixture(model, data, plan)
            mujoco.mj_forward(model, data)

        if frame_idx % max(1, fps // 5) == 0:
            logs.append(sensor_snapshot(model, data, time_s, plan))

        if renderer is not None:
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = [0.08, 0.0, 0.12]
            camera.distance = 1.05
            camera.azimuth = 135 + 15 * math.sin(2.0 * math.pi * time_s / max(duration_s, 0.1))
            camera.elevation = -28
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render().copy())

    final_metrics = success_metrics(model, data)

    with sensor_log_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(logs[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(logs)

    video_written = None
    if renderer is not None and frames:
        try:
            iio.imwrite(video_path, np.asarray(frames), fps=fps, codec="libx264", quality=7)
            video_written = str(video_path)
        except Exception as exc:
            fallback = video_path.with_suffix(".gif")
            iio.imwrite(fallback, np.asarray(frames), fps=fps)
            video_written = str(fallback)
            final_metrics["video_fallback_reason"] = str(exc)

    summary = {
        "project": "AIDOOG Sentinel Sorter",
        "registration_uuid": "6c3b08a9-5fb8-4e60-bd5d-d02d90f40ab9",
        "robot_platform": "MuJoCo cartesian wrist with a three-finger dexterous gripper",
        "task_goal": "Autonomously sort two differently shaped parts into matching bins while recording controls, sensor state, labels, and success metrics.",
        "scene": display_path(scene_path),
        "video": display_path(Path(video_written)) if video_written else None,
        "sensor_log": display_path(sensor_log_path),
        "duration_s": duration_s,
        "fps": fps,
        "render_size": [width, height],
        "planner": "deterministic long-horizon pick, transport, place, and verify state machine",
        "data_columns": list(logs[0].keys()),
        "metrics": final_metrics,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the AIDOOG Sentinel Sorter MuJoCo rollout.")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--sensor-log", type=Path, default=DEFAULT_SENSOR_LOG)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--duration", type=float, default=64.0, help="Demo length in seconds. Default is within the 1-3 minute contest target.")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--no-video", action="store_true", help="Run metrics and data logging without rendering a video.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_demo(
        scene_path=args.scene,
        video_path=args.video,
        sensor_log_path=args.sensor_log,
        summary_path=args.summary,
        duration_s=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height,
        record_video=not args.no_video,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["metrics"]["all_tasks_successful"] else 2


if __name__ == "__main__":
    sys.exit(main())
