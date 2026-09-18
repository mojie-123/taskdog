#!/usr/bin/env python3
"""程序生成碗的碰撞体（凸块拼装），并从视觉网格实测的剖面重写 scene.xml 的对应块。

为什么不用 mesh 当碰撞体
    实测（/tmp/probe44.py）MuJoCo 的 **碰撞** 几何 = 该 mesh 的凸包（只有射线用真面）：
    任何"空心碗"的 mesh 碰撞体都会退化成实心凸包（带盖），内侧指板插不进腔内 → 夹不住。
    凸块拼装没有这个问题：每个 geom 都是凸原始体，腔体由"环"而不是"面"围出来。

做法（全部数值从 meshes/bowl/textured.obj 实测，不手抄）
    1. 以视觉网格的**外表面**为准，取 r_out(z)（分层最大半径, 0.25mm 网格 + 平滑）
    2. 用"弦误差 ≤ tol"把剖面贪心切成若干段（每段最多 band_max 高），每段一个倾斜方块环
       box 局部 x = 剖面外法线（厚 thick）、y = 切向、z = 沿壁方向；16 段/环用 <replicate> 展开
    3. 底部用实心圆柱（足）。总质量默认 0.5kg（与原 bowl_geom 一致），沿轴线对称
    4. 外表面严格贴视觉剖面（加厚全部向内，指板不会"提前"碰到碗壁）

用法:
    python generate_bowl_collision.py [--dry-run] [--nseg 16] [--thick 0.008]
                                      [--tol 0.0005] [--band-max 0.008] [--mass 0.5]
"""
import argparse
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # /home/mojie/taskdog
SCENE = os.path.join(ROOT, 'custom_envs', 'mujoco', 'scene.xml')
VIS = os.path.join(ROOT, 'custom_envs', 'mujoco', 'meshes', 'bowl', 'textured.obj')
BEGIN = '<!-- BOWL_COLLISION_BEGIN -->'
END = '<!-- BOWL_COLLISION_END -->'


def load_visual_mesh(path):
    """读视觉 OBJ 顶点（该 OBJ 自身就是 z 向上的正立坐标系，无需再变换）。"""
    v = []
    with open(path) as f:
        for ln in f:
            if ln.startswith('v '):
                v.append([float(x) for x in ln.split()[1:4]])
    return np.asarray(v, dtype=np.float64)


def outer_profile(V, z0, z1, fine=0.00025, win=0.00060):
    """r_out(z): 细网格上分层最大半径 + 5 点滑动平均。"""
    r = np.hypot(V[:, 0], V[:, 1])
    z = V[:, 2]
    zg = np.arange(z0, z1 + 1e-9, fine)
    rg = np.full_like(zg, np.nan)
    for i, zc in enumerate(zg):
        m = np.abs(z - zc) <= win
        if m.sum() >= 3:
            rg[i] = r[m].max()
    # 补齐（顶部/底部可能稀疏）
    good = ~np.isnan(rg)
    if good.sum() < 10:
        raise RuntimeError('剖面点太少: %d' % good.sum())
    rg = np.interp(zg, zg[good], rg[good])
    k = 5
    ker = np.ones(k) / k
    rg = np.convolve(np.pad(rg, k // 2, mode='edge'), ker, mode='valid')
    return zg, rg


def segment(zg, rg, tol, band_max):
    """贪心分段: 每段用一条直线近似外剖面, 弦误差(法向距离) <= tol, 高度 <= band_max。"""
    segs = []
    i = 0
    n = len(zg)
    while i < n - 1:
        best_j = i + 1
        for j in range(i + 1, n):
            if zg[j] - zg[i] > band_max:
                break
            p0 = np.array([rg[i], zg[i]])
            p1 = np.array([rg[j], zg[j]])
            d = p1 - p0
            L = np.linalg.norm(d)
            if L < 1e-9:
                continue
            dn = d / L
            nvec = np.array([dn[1], -dn[0]])           # 法向
            pts = np.stack([rg[i:j + 1], zg[i:j + 1]], axis=1) - p0
            err = np.abs(pts @ nvec).max()
            if err > tol:
                break
            best_j = j
        segs.append((i, best_j))
        i = best_j
    return segs


def seg_geometry(zg, rg, i, j, thick):
    """一段 → 方块的位置/朝向/半尺寸。返回 (rc, zc, psi, half_len)"""
    r0, z0 = rg[i], zg[i]
    r1, z1 = rg[j], zg[j]
    d = np.array([r1 - r0, z1 - z0])
    L = float(np.linalg.norm(d))
    t = d / L                                          # 沿壁方向(向上向外)
    nvec = np.array([t[1], -t[0]])                     # 外法线
    if nvec[0] < 0:                                    # 保证朝外
        nvec = -nvec
    mid = np.array([(r0 + r1) / 2.0, (z0 + z1) / 2.0])
    inner = mid - nvec * thick                         # 加厚全部向内
    rc, zc = (mid + inner) / 2.0
    psi = math.atan2(nvec[1], nvec[0])                 # 外法线在 (r,z) 平面的倾角
    return rc, zc, psi, L / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true', help='只打印, 不写 scene.xml')
    ap.add_argument('--nseg', type=int, default=16, help='每环方块数（环向细分）')
    ap.add_argument('--thick', type=float, default=0.008, help='壁厚 m（全部向内加厚）')
    ap.add_argument('--tol', type=float, default=0.0005, help='外剖面弦误差上限 m')
    ap.add_argument('--band-max', type=float, default=0.008, help='单段最大高度 m')
    ap.add_argument('--mass', type=float, default=0.5, help='碰撞体总质量 kg')
    ap.add_argument('--foot-mass', type=float, default=0.30, help='底部圆柱质量 kg')
    args = ap.parse_args()

    V = load_visual_mesh(VIS)
    z0, z1 = float(V[:, 2].min()), float(V[:, 2].max())
    r_all = np.hypot(V[:, 0], V[:, 1])
    print('视觉网格: %d 顶点, z %.2f..%.2fmm (高 %.1fmm), 半径 %.2f..%.2fmm'
          % (len(V), z0 * 1000, z1 * 1000, (z1 - z0) * 1000,
             r_all.min() * 1000, r_all.max() * 1000))

    # 底部圆柱: 覆盖 z_min 起 foot_h 高（视觉底盘 r≈45.8@-27.5 → 47.8@-24）
    foot_h = 0.004
    foot_r = float(np.hypot(V[np.abs(V[:, 2] - z0) < 0.0015, 0],
                            V[np.abs(V[:, 2] - z0) < 0.0015, 1]).max()) if \
        (np.abs(V[:, 2] - z0) < 0.0015).sum() else 0.045
    z_wall0 = z0 + foot_h

    zg, rg = outer_profile(V, z_wall0, z1)
    segs = segment(zg, rg, args.tol, args.band_max)
    print('剖面 %.1f..%.1fmm → %d 段（弦误差<=%.2fmm, 每段<=%.1fmm）'
          % (z_wall0 * 1000, z1 * 1000, len(segs), args.tol * 1000, args.band_max * 1000))

    nbox = len(segs) * args.nseg
    box_mass = max((args.mass - args.foot_mass) / nbox, 1e-6)

    lines = []
    # 底部: 实心圆柱（足）
    lines.append('      <geom name="bowl_foot" type="cylinder" size="%.5f %.5f" pos="0 0 %.5f"'
                 % (foot_r, foot_h / 2.0, z0 + foot_h / 2.0))
    lines.append('            mass="%.4f" friction="5.0 0.5 0.5" solref="0.004 1"'
                 % args.foot_mass)
    lines.append('            solimp="0.99 0.999 0.001" rgba="0.9 0.85 0.7 1" group="3"'
                 ' contype="1" conaffinity="1" />')
    # 壁: 每段一个倾斜方块环
    print('\n 段 | 高度范围(mm) | 外半径mm 起→止 | 倾角° | 方块: 中心r,z(mm) 半长mm')
    for si, (i, j) in enumerate(segs):
        rc, zc, psi, half = seg_geometry(zg, rg, i, j, args.thick)
        # 弦面补偿: 方块外表面是 nseg 边形的一条弦, 角点比中点远 (1/cos-1) 倍;
        # 把环心向内挪半个矢高, 使外表面在目标半径上下各 ±(矢高/2) 起伏
        fac = 1.0 / math.cos(math.pi / args.nseg) - 1.0
        rc_eff = rc - (rc + args.thick / 2.0) * fac / 2.0
        half_w = 1.25 * rc_eff * math.sin(math.pi / args.nseg)
        # 方块绕切向轴的倾角: 要让局部 x 轴指向外法线 (n_r, n_z)。
        # MuJoCo 的 quat=Ry(ψ) 把 +X 映射到 (cosψ, -sinψ) 于 (r,z) 平面
        # → 取 ψ = -atan2(n_z, n_r), 即 qy 取负号（否则倾角整体镜像, 方块躺错方向）。
        qw, qy = math.cos(psi / 2.0), -math.sin(psi / 2.0)
        print(' %2d | %6.1f..%6.1f | %6.2f → %6.2f | %5.1f | %.2f,%.2f  %.2f'
              % (si, zg[i] * 1000, zg[j] * 1000, rg[i] * 1000, rg[j] * 1000,
                 math.degrees(psi), rc_eff * 1000, zc * 1000, half * 1000))
        lines.append('      <replicate count="%d" euler="0 0 %.8f">' % (args.nseg, 360.0 / args.nseg))
        lines.append('        <geom type="box" size="%.5f %.5f %.5f" pos="%.5f 0 %.5f"'
                     % (args.thick / 2.0, half_w, half, rc_eff, zc))
        lines.append('              quat="%.6f 0 %.6f 0" mass="%.5f" friction="5.0 0.5 0.5"'
                     % (qw, qy, box_mass))
        lines.append('              solref="0.004 1" solimp="0.99 0.999 0.001"'
                     ' rgba="0.9 0.85 0.7 1" group="3" contype="1" conaffinity="1" />')
        lines.append('      </replicate>')
    block = '\n'.join(lines)
    print('\n合计: 1 足圆柱 + %d 环 x %d 块 = %d 个 geom, 总质量 %.3fkg (块 %.5fkg/个)'
          % (len(segs), args.nseg, nbox + 1, args.foot_mass + nbox * box_mass, box_mass))

    if args.dry_run:
        print('\n[dry-run] 不写 scene.xml')
        return 0
    with open(SCENE) as f:
        txt = f.read()
    a, b = txt.index(BEGIN), txt.index(END)
    new = txt[:a + len(BEGIN)] + '\n' + block + '\n      ' + txt[b:]
    with open(SCENE, 'w') as f:
        f.write(new)
    print('\n已写入 %s (%d 行)' % (SCENE, block.count('\n') + 1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
