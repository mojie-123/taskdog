#!/usr/bin/env python3
"""grasp_worker.py — AnyGrasp 子进程抓取检测脚本。

Protocol:
  INPUT  : /tmp/pointcloud.npz  {points: (N,3) float32, colors: (N,3) float32}
  OUTPUT : /tmp/grasp_result.npz {
               translations: (K,3) float32,
               rotations:    (K,3,3) float32,
               widths:       (K,) float32,
               scores:       (K,) float32,
           }

Note:
  gsnet.so 在加载 License 时会在当前工作目录下找 license/ 文件夹，
  因此本脚本启动后会先 os.chdir 到 SDK_DETECTION_DIR。
  navigate_to_goal.py 的输入/输出协议与 graspnet-baseline 版完全相同，无需改动。
"""

import os
import sys
import argparse
# NOTE: numpy must NOT be imported at module level before gsnet/MinkowskiEngine.
# MinkowskiEngine was compiled against numpy 2.4.6; importing numpy 1.26.0 first
# causes ABI mismatch inside gsnet.so. numpy is imported inside main() after gsnet.

# ── AnyGrasp SDK 路径 ────────────────────────────────────────────────────────
SDK_DETECTION_DIR = "/home/mojie/anygrasp_sdk/grasp_detection"
# gsnet.so 和 license/ 子目录必须在同一个工作目录下
sys.path.insert(0, SDK_DETECTION_DIR)
os.chdir(SDK_DETECTION_DIR)

INPUT_PATH  = "/tmp/pointcloud.npz"
OUTPUT_PATH = "/tmp/grasp_result.npz"

# Piper 手爪参数（单位：米）
MAX_GRIPPER_WIDTH = 0.08   # 最大张开宽度
GRIPPER_HEIGHT    = 0.06   # 手指高度（用于碰撞检测）


def main():
    parser = argparse.ArgumentParser("grasp_worker")
    parser.add_argument("--checkpoint", required=True,
                        help="AnyGrasp checkpoint 路径（checkpoint_detection.tar）")
    parser.add_argument("--topk", type=int, default=5,
                        help="保存的抓取候选数")
    parser.add_argument("--max_gripper_width", type=float, default=MAX_GRIPPER_WIDTH,
                        help="手爪最大张开宽度（米）")
    parser.add_argument("--gripper_height", type=float, default=GRIPPER_HEIGHT,
                        help="手指高度（米）")
    parser.add_argument("--no_collision", action="store_true",
                        help="禁用碰撞过滤（保留更多候选）")
    parser.add_argument("--dense", action="store_true",
                        help="启用密集预测模式（更多候选但质量略低）")
    args = parser.parse_args()

    # ── 导入 AnyGrasp（必须在 import numpy 之前！）────────────────────────────
    # MinkowskiEngine 在 numpy 2.4.6 下编译，若先 import numpy 1.26.0 会导致
    # C ABI 不兼容：gsnet.so 内部出现 AttributeError: 'dict' has no attr 'endswith'
    from argparse import Namespace
    from gsnet import create_detector

    # ── 在 gsnet 加载之后再 import numpy ─────────────────────────────────────
    import numpy as np

    # ── 初始化检测器 ──────────────────────────────────────────────────────────
    config = Namespace(
        checkpoint_path=args.checkpoint,
        max_gripper_width=args.max_gripper_width,
        gripper_height=args.gripper_height,
    )
    print(f"[grasp_worker] Loading AnyGrasp detector from: {args.checkpoint}")
    detector = create_detector(config)
    if detector is None:
        print("[grasp_worker] ERROR: AnyGrasp 初始化失败（License 验证未通过或 checkpoint 错误）")
        sys.exit(1)
    print("[grasp_worker] AnyGrasp detector ready")

    # ── 读取点云输入 ──────────────────────────────────────────────────────────
    if not os.path.exists(INPUT_PATH):
        print(f"[grasp_worker] ERROR: 输入文件不存在: {INPUT_PATH}")
        sys.exit(1)

    data   = np.load(INPUT_PATH)
    points = data["points"].astype(np.float32)   # (N, 3)
    # colors 字段保留接口兼容，AnyGrasp 新版 API 只接收 points
    print(f"[grasp_worker] 输入点云: {len(points)} 个点")

    if len(points) < 50:
        print("[grasp_worker] ERROR: 点云点数过少（< 50），退出")
        sys.exit(1)

    # ── 推理 ──────────────────────────────────────────────────────────────────
    optional_params = {
        "collision_detection": not args.no_collision,
        "dense_grasp": args.dense,
    }
    gg = detector.get_grasp(points, optional_params)

    if gg is None or len(gg) == 0:
        print("[grasp_worker] WARN: 未检测到有效抓取")
        np.savez(OUTPUT_PATH,
                 translations=np.zeros((0, 3), dtype=np.float32),
                 rotations=np.zeros((0, 3, 3), dtype=np.float32),
                 widths=np.zeros(0, dtype=np.float32),
                 scores=np.zeros(0, dtype=np.float32))
        return

    # ── NMS + 排序 ────────────────────────────────────────────────────────────
    print(f"[grasp_worker] 原始抓取数: {len(gg)}")
    if not args.dense:
        gg = gg.nms()
    gg = gg.sort_by_score()

    # ── 截取 topk ─────────────────────────────────────────────────────────────
    topk = min(args.topk, len(gg))
    if topk == 0:
        print("[grasp_worker] WARN: NMS 后无有效抓取")
        np.savez(OUTPUT_PATH,
                 translations=np.zeros((0, 3), dtype=np.float32),
                 rotations=np.zeros((0, 3, 3), dtype=np.float32),
                 widths=np.zeros(0, dtype=np.float32),
                 scores=np.zeros(0, dtype=np.float32))
        return

    gg = gg[:topk]

    # ── 提取结果并保存 ────────────────────────────────────────────────────────
    # GraspGroup 属性：translations (K,3), rotation_matrices (K,3,3),
    #                  widths (K,), scores (K,)
    translations = gg.translations.astype(np.float32)       # (K, 3)
    rotations    = gg.rotation_matrices.astype(np.float32)  # (K, 3, 3)
    widths       = gg.widths.astype(np.float32)             # (K,)
    scores       = gg.scores.astype(np.float32)             # (K,)

    np.savez(OUTPUT_PATH,
             translations=translations,
             rotations=rotations,
             widths=widths,
             scores=scores)

    print(f"[grasp_worker] 保存 {topk} 个抓取到 {OUTPUT_PATH}")
    for i in range(topk):
        print(f"  [{i}] score={scores[i]:.3f}  t={np.round(translations[i], 3)}  w={widths[i]:.3f}")


if __name__ == "__main__":
    main()
