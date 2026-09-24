"""Diagnostics helpers for navigate_mujoco.py.

All functions in this module are read-only with respect to robot control/state
(except writing optional PNG debug artifacts). They intentionally keep verbose
instrumentation out of the main navigation/manipulation state machine.
"""

import math
import os

import numpy as np

DEBUG_IMAGE_DIR = "/home/mojie/taskdog/custom_envs/tmp_pictures"


def capture_body_pose(env, body_name):
    """Return (world_pos, world_rotmat) for a body, or (None, None)."""
    try:
        import mujoco
        bid = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            return None, None
        pos = env._data.xpos[bid].astype(np.float64).copy()
        rot = env._data.xmat[bid].reshape(3, 3).astype(np.float64).copy()
        return pos, rot
    except Exception:
        return None, None


def finger_contact_info(env, obj_name):
    """Return first finger-object contact details for link7/link8 (diagnostic only)."""
    out = {"link7": None, "link8": None}
    try:
        import mujoco
        model, data = env._model, env._data
        fingers = {
            name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("link7", "link8")
        }
        obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        if obj < 0:
            return out
        obj_pos = data.xpos[obj].copy()
        obj_rot = data.xmat[obj].reshape(3, 3)
        for ci in range(int(data.ncon)):
            contact = data.contact[ci]
            b1 = int(model.geom_bodyid[int(contact.geom1)])
            b2 = int(model.geom_bodyid[int(contact.geom2)])
            if obj not in (b1, b2):
                continue
            obj_geom = int(contact.geom1) if b1 == obj else int(contact.geom2)
            for name, bid in fingers.items():
                if bid >= 0 and bid in (b1, b2) and out[name] is None:
                    world = data.contact[ci].pos.copy()
                    out[name] = dict(
                        world=world,
                        local=(world - obj_pos) @ obj_rot,
                        obj_geom=obj_geom,
                        obj_geom_name=(mujoco.mj_id2name(
                            model, mujoco.mjtObj.mjOBJ_GEOM, obj_geom
                        ) or f"id{obj_geom}"),
                        dist=float(data.contact[ci].dist),
                    )
    except Exception:
        pass
    return out


def close_contact_diag(env, obj_name):
    """Collect per-finger contact-force diagnostics without changing control."""
    out = {
        "link7": dict(ncon=0, min_dist=None, normal_force=0.0,
                      tangential_force=0.0, others=set()),
        "link8": dict(ncon=0, min_dist=None, normal_force=0.0,
                      tangential_force=0.0, others=set()),
    }
    try:
        import mujoco
        model, data = env._model, env._data
        fingers = {
            name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("link7", "link8")
        }
        obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        for ci in range(int(data.ncon)):
            contact = data.contact[ci]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            b1 = int(model.geom_bodyid[g1])
            b2 = int(model.geom_bodyid[g2])
            for name, finger_bid in fingers.items():
                if finger_bid < 0 or finger_bid not in (b1, b2):
                    continue
                other_bid = b2 if b1 == finger_bid else b1
                if other_bid == obj:
                    item = out[name]
                    item["ncon"] += 1
                    dist = float(contact.dist)
                    if item["min_dist"] is None or dist < item["min_dist"]:
                        item["min_dist"] = dist
                    force = np.zeros(6, dtype=np.float64)
                    mujoco.mj_contactForce(model, data, ci, force)
                    item["normal_force"] += max(0.0, float(force[0]))
                    item["tangential_force"] += float(np.hypot(force[1], force[2]))
                else:
                    body_name = mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_BODY, other_bid)
                    if body_name:
                        out[name]["others"].add(body_name)
        for name in out:
            out[name]["others"] = sorted(out[name]["others"])
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def finger_deep_diag(env, obj_name):
    """Collect detailed gripper actuator/contact diagnostics (read-only)."""
    out = {"link7": {}, "link8": {}}
    try:
        import mujoco
        model, data = env._model, env._data
        obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        obj_geoms = [g for g in range(int(model.ngeom))
                     if int(model.geom_bodyid[g]) == int(obj_bid)] if obj_bid >= 0 else []

        for name, pad_name in (("link7", "link7_pad"), ("link8", "link8_pad")):
            joint_name = "joint7" if name == "link7" else "joint8"
            motor_name = "joint7_motor" if name == "link7" else "joint8_motor"
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, pad_name)
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, motor_name)
            qadr = int(model.jnt_qposadr[jid]) if jid >= 0 else -1
            dadr = int(model.jnt_dofadr[jid]) if jid >= 0 else -1
            item = dict(
                q=(float(data.qpos[qadr]) if qadr >= 0 else float("nan")),
                qd=(float(data.qvel[dadr]) if dadr >= 0 else float("nan")),
                qacc=(float(data.qacc[dadr]) if dadr >= 0 else float("nan")),
                ctrl=(float(data.ctrl[aid]) if aid >= 0 else float("nan")),
                qfrc_actuator=(float(data.qfrc_actuator[dadr]) if dadr >= 0 else float("nan")),
                qfrc_constraint=(float(data.qfrc_constraint[dadr]) if dadr >= 0 else float("nan")),
                qfrc_passive=(float(data.qfrc_passive[dadr]) if dadr >= 0 else float("nan")),
                cube_contact=False,
                geom_gap=None,
                fromto=None,
                nearest_obj_geom=None,
                contacts=[],
            )

            if gid >= 0 and obj_geoms and hasattr(mujoco, "mj_geomDistance"):
                best = None
                for obj_gid in obj_geoms:
                    try:
                        fromto = np.zeros(6, dtype=np.float64)
                        dist = float(mujoco.mj_geomDistance(
                            model, data, int(gid), int(obj_gid), 0.05, fromto))
                        if best is None or dist < best[0]:
                            best = (dist, fromto.copy(), int(obj_gid))
                    except Exception:
                        continue
                if best is not None:
                    item["geom_gap"] = best[0]
                    item["fromto"] = best[1]
                    item["nearest_obj_geom"] = (mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, best[2]) or f"id{best[2]}")

            for ci in range(int(data.ncon)):
                contact = data.contact[ci]
                g1, g2 = int(contact.geom1), int(contact.geom2)
                b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
                if bid < 0 or bid not in (b1, b2):
                    continue
                other_b = b2 if b1 == bid else b1
                other_g = g2 if b1 == bid else g1
                if other_b == obj_bid:
                    item["cube_contact"] = True
                force = np.zeros(6, dtype=np.float64)
                try:
                    mujoco.mj_contactForce(model, data, ci, force)
                except Exception:
                    pass
                self_g = g1 if b1 == bid else g2
                item["contacts"].append(dict(
                    idx=int(ci),
                    self_geom=(mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, self_g) or f"id{self_g}"),
                    other_geom=(mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, other_g) or f"id{other_g}"),
                    other_body=(mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_BODY, other_b) or f"id{other_b}"),
                    dist=float(contact.dist),
                    Fn=max(0.0, float(force[0])),
                    Ft=float(np.hypot(force[1], force[2])),
                    pos=np.asarray(contact.pos, dtype=np.float64).copy(),
                ))
            out[name] = item
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def print_finger_deep_diag(env, tag, step_label, obj_name, force_contacts=False):
    """Print one detailed gripper diagnostic frame."""
    diag = finger_deep_diag(env, obj_name)
    if "error" in diag:
        print(f"[{tag}] step={step_label:03d} diagnostic error: {diag['error']}", flush=True)
        return diag

    bits = [int(bool(diag[name].get("cube_contact", False)))
            for name in ("link7", "link8")]
    print(f"[{tag}] step={step_label:03d} cube_contact={bits[0]}/{bits[1]}", flush=True)

    for name in ("link7", "link8"):
        item = diag[name]
        gap = "n/a" if item.get("geom_gap") is None else f"{item['geom_gap']*1000:+.3f}mm"
        print(
            f"[{tag}]   {name}: q={item['q']*1000:+.3f}mm "
            f"qd={item['qd']*1000:+.3f}mm/s qacc={item['qacc']:+.3f}m/s^2 "
            f"ctrl={item['ctrl']:+.3f}N act={item['qfrc_actuator']:+.3f}N "
            f"constraint={item['qfrc_constraint']:+.3f}N "
            f"passive={item['qfrc_passive']:+.3f}N gap={gap}",
            flush=True,
        )
        if item.get("fromto") is not None:
            ft = np.asarray(item["fromto"], dtype=np.float64).reshape(2, 3)
            print(
                f"[{tag}]     nearest: pad={np.round(ft[0],6)} "
                f"obj={np.round(ft[1],6)} obj_geom={item.get('nearest_obj_geom')}",
                flush=True,
            )

        stalled = (abs(float(item.get("ctrl", 0.0))) > 0.5 and
                   abs(float(item.get("qd", 0.0))) < 0.002 and
                   not bool(item.get("cube_contact", False)))
        if force_contacts or stalled:
            contacts = item.get("contacts", [])
            if not contacts:
                print(f"[{tag}]     all_contacts: NONE", flush=True)
            else:
                for contact in contacts:
                    print(
                        f"[{tag}]     contact#{contact['idx']} "
                        f"{contact['self_geom']} <-> "
                        f"{contact['other_body']}/{contact['other_geom']} "
                        f"dist={contact['dist']*1000:+.3f}mm "
                        f"Fn={contact['Fn']:.3f}N Ft={contact['Ft']:.3f}N "
                        f"pos={np.round(contact['pos'],6)}",
                        flush=True,
                    )
    return diag


def print_close_step_diag(env, obj_name, step, contact_flags, q7, q8, obj_pose0):
    """Print the compact CLOSE diagnostic block previously embedded in main()."""
    try:
        diag = close_contact_diag(env, obj_name)
        obj_p, obj_R = capture_body_pose(env, obj_name)
        p0, R0 = obj_pose0
        dp_mm = ((obj_p - p0) * 1000.0
                 if obj_p is not None and p0 is not None
                 else np.array([np.nan, np.nan, np.nan]))
        if obj_R is not None and R0 is not None:
            dR = obj_R @ R0.T
            cth = float(np.clip((np.trace(dR) - 1.0) * 0.5, -1.0, 1.0))
            dtheta_deg = math.degrees(math.acos(cth))
        else:
            dtheta_deg = float("nan")

        target = env.get_gripper_target()
        command = np.asarray(getattr(env, "_grip_cmd", target), dtype=np.float64).copy()
        ctrl = env._data.ctrl[np.asarray(env._grip_ctrl_idx, dtype=np.int32)].copy()
        mode = "FORCE" if getattr(env, "gripper_force_mode", False) else "POS"

        def fmt_side(name):
            item = diag[name]
            dist = "--" if item["min_dist"] is None else f"{item['min_dist']*1000:+.3f}mm"
            others = ",".join(item["others"]) if item["others"] else "-"
            return (f"{name}:c={item['ncon']} dist={dist} "
                    f"Fn={item['normal_force']:.2f}N Ft={item['tangential_force']:.2f}N "
                    f"other={others}")

        print(
            f"[CDIAG] CLOSE step={step:03d} contact="
            f"{int(contact_flags['link7'])}/{int(contact_flags['link8'])} "
            f"q=[{q7:+.4f},{q8:+.4f}] mode={mode} "
            f"target=[{target[0]:+.4f},{target[1]:+.4f}] "
            f"cmd=[{command[0]:+.4f},{command[1]:+.4f}] "
            f"ctrl=[{ctrl[0]:+.2f},{ctrl[1]:+.2f}]N "
            f"obj_dmm={np.round(dp_mm,3)} obj_dR={dtheta_deg:.3f}deg",
            flush=True,
        )
        print(f"[CDIAG]   {fmt_side('link7')} | {fmt_side('link8')}", flush=True)
    except Exception as exc:
        print(f"[CDIAG] CLOSE diagnostic error: {exc}", flush=True)


def print_reach_done_diag(env, obj_name, jpos, arm_indices, target_q):
    """Print final REACH pose/joint diagnostics."""
    try:
        import mujoco
        gb_bid = mujoco.mj_name2id(
            env._model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
        j7_bid = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_BODY, "link7")
        gb_actual = env._data.xpos[gb_bid].copy()
        j7_actual = env._data.xpos[j7_bid].copy() if j7_bid >= 0 else None
        print(f"[DIAG] REACH done gb_actual_world  = {np.round(gb_actual,4)}  ← MuJoCo实际", flush=True)
        if j7_actual is not None:
            print(f"[DIAG] REACH done j7_actual_world  = {np.round(j7_actual,4)}  ← MuJoCo实际", flush=True)
    except Exception as exc:
        print(f"[DIAG] REACH done xpos read err: {exc}", flush=True)

    try:
        real_obj = env.get_object_pos(obj_name)
        print(f"[DIAG] REACH done object real world={np.round(real_obj,4)}", flush=True)
    except Exception:
        pass

    try:
        cur_q = np.asarray(jpos)[np.asarray(arm_indices)].copy()
        tgt_q = np.asarray(target_q, dtype=np.float64)
        err_q = cur_q - tgt_q
        print(f"[DIAG] REACH done joint_actual = {np.round(cur_q,4)}", flush=True)
        print(f"[DIAG] REACH done joint_target = {np.round(tgt_q,4)}", flush=True)
        print(f"[DIAG] REACH done joint_err    = {np.round(err_q,4)}  "
              f"(deg={np.round(np.degrees(err_q),2)})", flush=True)
    except Exception as exc:
        print(f"[DIAG] REACH done joint read err: {exc}", flush=True)


def save_scan_raw(scan_rgb, output_dir=DEBUG_IMAGE_DIR):
    """Save SCAN RGB frame as scan_raw.png; diagnostic artifact only."""
    try:
        if scan_rgb is None:
            return
        import cv2
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "scan_raw.png")
        cv2.imwrite(path, cv2.cvtColor(scan_rgb, cv2.COLOR_RGB2BGR))
        print(f"[SM] scan_raw.png 已保存: {path}", flush=True)
    except Exception as exc:
        print(f"[SM] scan_raw.png 保存失败: {exc}", flush=True)


def save_filter_image(scan_rgb, valid_mask, ransac_keep, vivid_mask,
                      output_dir=DEBUG_IMAGE_DIR):
    """Save filter.png showing depth/table/color filtering."""
    try:
        if scan_rgb is None:
            return
        import cv2
        H, W = scan_rgb.shape[:2]
        mask_flat = np.asarray(valid_mask, dtype=bool).ravel()
        valid_idx = np.where(mask_flat)[0]
        image = scan_rgb.copy()
        flat = image.reshape(-1, 3)
        flat[~mask_flat] = 0
        flat[valid_idx[~np.asarray(ransac_keep, dtype=bool)]] = 0
        kept_idx = valid_idx[np.asarray(ransac_keep, dtype=bool)]
        flat[kept_idx[~np.asarray(vivid_mask, dtype=bool)]] = 0
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "filter.png")
        cv2.imwrite(path, cv2.cvtColor(flat.reshape(H, W, 3), cv2.COLOR_RGB2BGR))
        print(f"[SM] filter.png 已保存: {path}", flush=True)
    except Exception as exc:
        print(f"[SM] filter.png 保存失败: {exc}", flush=True)


def save_choice_image(scan_rgb, t_cam, R_cam, output_dir=DEBUG_IMAGE_DIR,
                      fx=616.0, fy=616.0):
    """Save choice.png with selected grasp point and closing-axis arrow."""
    try:
        import cv2
        if scan_rgb is not None:
            image = cv2.cvtColor(scan_rgb, cv2.COLOR_RGB2BGR).copy()
        else:
            image = np.zeros((480, 640, 3), dtype=np.uint8)
        H, W = image.shape[:2]
        cx, cy = W / 2.0, H / 2.0
        t_cam = np.asarray(t_cam, dtype=np.float64)
        R_cam = np.asarray(R_cam, dtype=np.float64)
        u0 = int(t_cam[0] / t_cam[2] * fx + cx)
        v0 = int(t_cam[1] / t_cam[2] * fy + cy)
        closing_end = t_cam + R_cam[:, 1] * 0.05
        u1 = int(closing_end[0] / closing_end[2] * fx + cx)
        v1 = int(closing_end[1] / closing_end[2] * fy + cy)
        cv2.circle(image, (u0, v0), 6, (0, 0, 255), -1)
        cv2.arrowedLine(image, (u0, v0), (u1, v1), (0, 255, 0), 2, tipLength=0.3)
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "choice.png")
        cv2.imwrite(path, image)
        print(f"[SM] choice.png 已保存: {path}  "
              f"grasp_pt=({u0},{v0}) closing=({u1},{v1})", flush=True)
    except Exception as exc:
        print(f"[SM] choice.png 保存失败: {exc}", flush=True)
