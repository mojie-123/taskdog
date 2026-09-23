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

内侧直筒（bore）—— 第五轮，针对"抓的位置/深度都对了但还是抓不起来"
    实测指板（probe，gb 系）: **沿 approach 长 76.5mm、张合方向厚 26.5mm、宽 56mm**，
    指尖（= pinch 中心 C = gb+0.1358·a）是板的最低点，板身从指尖向阳掌心方向长 76.5mm。
    抓碗时（approach 朝下）指板的夹持面 = 76.5(竖直) × 56(切向)，其**两个竖直棱**到碗轴
    的距离 = sqrt(d² + 28.05²)（d = 板面的径向位置）：
      a) 对**内凹**的碗内壁，56mm 宽的平板永远只有棱先碰到（矢高 r−sqrt(r²−28.05²)
         ≈5.9~7.1mm），所以内侧指板本来就是"棱接触"——不是缺陷，棱线接触面积照样够；
      b) 真正的故障是**棱在低于 collar 的高度撞上收口的碗内壁**：approach 有倾角时
         指板棱会下探到 mesh z<+8（collar 带以下），那里的碰撞内表面 = 视觉外剖面−8mm
         （z=0 时 59.3、z=−8 时 54.8、z=−16 时 49.0）→ 指板棱（半径 ~59）直接插进壁里。
         实测（/tmp/probe_diag.py）: link8 ↔ 碗穿模 −23mm、接触力 73N、碗被顶出去、
         q8 卡在全开 → "抓不住"。
    修复 = 把碰撞体的**内表面做成从碗口到碗底的一根直筒**（半径 = collar 内表面
    有效值 ~59.2mm），壁厚在 collar 带内仍是 21mm（外 81），带以下随视觉外剖面收薄
    （r_out−bore_r，最薄处自动消失）。这样指板棱在**任意高度**都有 ≥26.5mm 的让位，
    棱线能沿着直筒从上到下连续接触 → 夹持面积更大、任何倾角/抓取高度都不再被卡住。
    视觉代价（明示）：mesh z 低于 ~−8 处视觉内壁（半径 49~59）会落在指板棱的内侧，
    指板棱看上去"切进"视觉碗壁 ≤10mm —— 与第四轮已接受的 ~10mm 量级相同。
    用户诊断"碗沿加宽了但碗壁没有加宽"正是这个：壁的内表面必须与 collar 齐平地加宽。

抓取带（collar）—— 第四轮，针对"碗沿加宽了但碗壁没加宽，抓不稳"
    实测：AnyGrasp 给的抓取中心在碗壁中部（mesh z≈+14mm，视觉外表面 r≈72.2、
    视觉壁厚只有 ~2.5mm），而视觉碗沿在 z≥20.7mm 外扩到 r=81。
      a) 指板是 26.5mm 厚的平板：夹 72mm 处的薄壁时它的板身（r≥72.2 向外 26.5mm）
         正好压在碗沿外扩段（73.5..81）的高度范围内 → 物理上撞沿、几何上必然打架；
      b) 更要命的是力：薄壁+闭死命令下 PD 误差只有 ~7.9mm→1.58N/指，容量 15.8N
         ≈ 碗重 15.7N → 边缘打滑（用户"抓不稳"）。
    所以把 **z≥band_from 的碰撞剖面外半径抬平到 band_r（=视觉碗沿最大半径 81）、
    厚度加到 band_thick（外 81 / 内 60，中线 70.5 ≈ AnyGrasp 中心）**：
      → 形成一圈"领口"，两个指板都能落在它上面（外板面 81 与视觉沿口外缘齐平、
        内板面 60 在腔内）；指板板身（r≥81）完全让开视觉沿口，不再撞沿；
      → collar 中线 ≈ AnyGrasp 抓取中心 → 两侧同时接触的概率最大；即使只先碰一侧，
        CLOSE 的逐指定力冻结会把碗推到两侧夹住（详见 navigate_mujoco.py）。
    视觉代价：抓取带那一段碰撞外表面比视觉壁面外鼓 ~9mm（视觉不动），指板看上去
    是在沿口外缘那一带夹住碗 —— 从侧面看被沿口自身遮挡，可接受。

削沿（rim-cut）—— 第七轮，替代 collar
    实测（生产场景、去狗）：collar 把 z>=8mm 的外半径抬到 81（16 弦角点实测 82.15mm），
    比**视觉沿口**还粗 1mm、比抓取高度处的视觉碗壁（z=+17.7 外 72.97 / 内 69.76）粗 ~9mm
    —— 外侧指板碰的是"看不见的领口"，而视觉上沿口在 z>=20.5 才外扩到 81.09（+24 最大）
    ⇒ 用户观测"碗沿加宽了但碗壁没有加宽"。
    改法（--rim-cut R --rim-cut-from z0）：z>=z0 的外剖面 clip 到 min(r_out(z), R)，
    取 R = 壁带最大半径 0.0735（= 视觉 z=+20 的 73.34 量级）⇒ 抓取带变成一段**竖直**外壁：
      外表面 = 视觉碗壁本身（指板碰的就是看得见的壁），只有 z>=20.5 的外扩段被削平；
      壁厚 = --thick（默认 8mm，比视觉壁 ~3mm 厚）⇒ 内表面 = R-thick ≈ 65.5mm
      （配 --bore-r 0.0655 让内侧直筒与抓取带内表面齐平 → 内表面也竖直，指板在任意高度
       都有同一个内表面可夹，不再出现"带内 65.5 / 带下 61"的台阶）。
    两指板能同时夹住的条件 ρ_内 < r_C < ρ_外（r_C = AnyGrasp 抓取中心径向位置，实测
    良性值 70.8）仍成立，且 r_C 在 [65.5, 73.5] 内都不需要单侧推 >4mm。
    视觉代价（明示）：z>=20.5 的视觉沿口外扩段（79.8~81.1）没有碰撞，指板会视觉上
    切入沿口 ≤7.5mm；内侧指板切进视觉内壁 ~4mm —— 与第四轮已接受的量级相同。

用法:
    python generate_bowl_collision.py [--dry-run] [--nseg 16] [--thick 0.008]
                                      [--tol 0.0005] [--band-max 0.008] [--mass 0.5]
                                      [--rim-cut 0.0735 --rim-cut-from 0.008 --bore-r 0.0655]
                                      （旧 collar 模式: --collar-from 0.008 --collar-r 0.081
                                        --collar-thick 0.021；给 --rim-cut 后 collar 自动关闭）
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
    ap.add_argument('--foot-mass', type=float, default=0.30, help='底部足块(box)质量 kg')
    ap.add_argument('--collar-from', type=float, default=0.008, help='抓取带起始高度 m (mesh z, 0=碗中部)')
    ap.add_argument('--collar-r', type=float, default=0.081, help='抓取带外半径 m (默认=视觉沿口最大半径)')
    ap.add_argument('--collar-thick', type=float, default=0.021, help='抓取带厚度 m (外->内)')
    ap.add_argument('--rim-cut', type=float, default=0.0,
                    help='削沿半径 m: z>=--rim-cut-from 的外半径 clip 到该值（0=不削; >0 时关闭 collar）')
    ap.add_argument('--rim-cut-from', type=float, default=0.008, help='削沿起始高度 m (mesh z)')
    ap.add_argument('--bore-r', type=float, default=0.0600,
                    help='内侧直筒半径 m（名义值; 16 面弦心内移 → 有效内表面 ≈59.2mm。'
                         '削沿模式下应取 = 抓取带内表面（--rim-cut 减 --thick）⇒ 内表面竖直）')
    ap.add_argument('--min-thick', type=float, default=0.0006,
                    help='段最小厚度 m（更薄的段直接丢掉: 该高度不再有碰撞壁）')
    ap.add_argument('--friction', type=str, default='5.0 0.01 0.01', help='摩擦串')
    args = ap.parse_args()

    V = load_visual_mesh(VIS)
    z0, z1 = float(V[:, 2].min()), float(V[:, 2].max())
    r_all = np.hypot(V[:, 0], V[:, 1])
    print('视觉网格: %d 顶点, z %.2f..%.2fmm (高 %.1fmm), 半径 %.2f..%.2fmm'
          % (len(V), z0 * 1000, z1 * 1000, (z1 - z0) * 1000,
             r_all.min() * 1000, r_all.max() * 1000))

    # 底部足: 覆盖 z_min 起 foot_h 高（视觉底盘 r≈45.8@-27.5 → 47.8@-24）
    foot_h = 0.004
    foot_r = float(np.hypot(V[np.abs(V[:, 2] - z0) < 0.0015, 0],
                            V[np.abs(V[:, 2] - z0) < 0.0015, 1]).max()) if \
        (np.abs(V[:, 2] - z0) < 0.0015).sum() else 0.045
    z_wall0 = z0 + foot_h

    # 抓取带: 削沿（rim-cut）或抬平到 collar_r（其余剖面严格贴视觉）
    collar = (args.collar_from < z1) and args.rim_cut <= 0.0
    if args.rim_cut > 0.0 and args.collar_from < z1:
        print('削沿模式: collar 关闭（不再把外半径抬到 %.2fmm）' % (args.collar_r * 1000))
    zg, rg = outer_profile(V, z_wall0, z1)
    if collar:
        m = zg >= args.collar_from
        rg = np.where(m, args.collar_r, rg)
        print('抓取带: z>=%.1fmm 外半径抬平到 %.2fmm, 厚 %.1fmm (内 %.2fmm)'
              % (args.collar_from * 1000, args.collar_r * 1000, args.collar_thick * 1000,
                 (args.collar_r - args.collar_thick) * 1000))
    if args.rim_cut > 0.0:
        m = zg >= args.rim_cut_from
        over = m & (rg > args.rim_cut)
        if over.any():
            print('削沿: z %.1f..%.1fmm 外半径由视觉 %.2f~%.2fmm 削到 %.2fmm'
                  % (zg[over][0] * 1000, zg[over][-1] * 1000, rg[over].min() * 1000,
                     rg[over].max() * 1000, args.rim_cut * 1000))
        else:
            print('削沿: z>=%.1fmm 的视觉剖面本来就没超过 %.2fmm（无改动）'
                  % (args.rim_cut_from * 1000, args.rim_cut * 1000))
        rg = np.where(m, np.minimum(rg, args.rim_cut), rg)
        print('削沿后抓取带: 外 %.2fmm / 内 %.2fmm（厚 %.1fmm, 受 bore_r %.2fmm 封顶）'
              % (args.rim_cut * 1000, (args.rim_cut - args.thick) * 1000, args.thick * 1000,
                 args.bore_r * 1000))
    segs = segment(zg, rg, args.tol, args.band_max)
    print('剖面 %.1f..%.1fmm → %d 段（弦误差<=%.2fmm, 每段<=%.1fmm）'
          % (z_wall0 * 1000, z1 * 1000, len(segs), args.tol * 1000, args.band_max * 1000))


    def thick_at(zmid, i, j, psi):
        """该段该用多厚：先按带内/带外取目标厚度, 再用"内表面不侵入 --bore-r 直筒"封顶。

        段是绕切向倾斜的方块, 外表面沿剖面直线, 内表面 = 外表面向内法线偏 th。
        内表面的两个角点半径 = r_end − th·cosψ（ψ = 外法线倾角）—— 取两端较小者,
        要求它 ≥ bore_r（指板棱的半径就是 collar 内表面那一圈）。解不出来的段返回 ~0
        （丢掉, 该高度不再有碰撞壁）。
        """
        want = args.collar_thick if (collar and zmid >= args.collar_from) else args.thick

        def ok(th):
            return all(r_e - th * math.cos(psi) >= args.bore_r
                       for r_e in (rg[i], rg[j]))

        if ok(want):
            return want
        lo, hi = 0.0, want
        for _ in range(24):
            mid = (lo + hi) / 2.0
            if ok(mid):
                lo = mid
            else:
                hi = mid
        return lo

    # 先按"直筒封顶"定下每段厚度, 丢掉被直筒吃光/太薄的段
    kept, dropped = [], []
    for (i, j) in segs:
        zmid = (zg[i] + zg[j]) / 2.0
        _rc0, _zc0, psi, _h0 = seg_geometry(zg, rg, i, j, args.thick)   # psi 与厚度无关
        th = thick_at(zmid, i, j, psi)
        (kept if th >= args.min_thick else dropped).append((i, j, th, None))
    if dropped:
        print('直筒封顶: %d/%d 段被丢掉 (mesh z %.1f..%.1fmm 不再有碰撞壁, 只保留视觉)'
              % (len(dropped), len(segs), zg[dropped[0][0]] * 1000, zg[dropped[-1][1]] * 1000))
    print('直筒: 碰撞内表面一律 ≥ %.2fmm（collar 内表面那一圈; 指板棱 r≈59.2 在任意高度都不再被挡）'
          % (args.bore_r * 1000))

    nbox = len(kept) * args.nseg
    box_mass = max((args.mass - args.foot_mass) / nbox, 1e-6)

    lines = []
    # 底部: 平底方足。**必须是 box，不能再用 cylinder**（实测原因）:
    #   MuJoCo 的 cylinder 是多边形棱柱, cylinder↔box 无解析解、走通用 convex(GJK),
    #   平底圆柱压在平桌(box)上只生成 1 个接触点, 且落在底棱上(实测离轴 44.9mm ≈ 足半径)
    #   ⇒ 1.61kg 的碗平衡在一个刀刃点上: |w|=2.07rad/s 摇摆永不衰减 + 0.04rad/s 偏航自转
    #   (静置 10s 里 z 波动 0.107mm、倾斜 1.15°)。
    #   同底平面的方足走 box↔box 解析解, 出 4 个角点 ⇒ |w|≡0.0000、倾斜 0.01°、无自转;
    #   1000 步耗时 0.154→0.175s(+14%)。底平面 z 与圆柱完全相同(都是 z0..z0+foot_h)。
    #   边长取 foot_r = 视觉底盘半径 ⇒ 最坏方向(边法向)支撑半径 = 视觉足半径, 不会更易倒。
    lines.append('      <geom name="bowl_foot" type="box" size="%.5f %.5f %.5f" pos="0 0 %.5f"'
                 % (foot_r, foot_r, foot_h / 2.0, z0 + foot_h / 2.0))
    lines.append('            mass="%.4f" friction="%s" solref="0.004 1"'
                 % (args.foot_mass, args.friction))
    lines.append('            solimp="0.99 0.999 0.001" rgba="0.9 0.85 0.7 1" group="3"'
                 ' contype="1" conaffinity="1" />')
    # 壁: 每段一个倾斜方块环
    print('\n 段 | 高度范围(mm) | 外半径mm 起→止 | 倾角° | 厚mm | 内表面mm | 方块: 中心r,z(mm) 半长mm')
    for si, (i, j, th, _rseg) in enumerate(kept):
        rc, zc, psi, half = seg_geometry(zg, rg, i, j, th)
        # 弦面补偿: 方块外表面是 nseg 边形的一条弦, 角点比中点远 (1/cos-1) 倍;
        # 把环心向内挪半个矢高, 使外表面在目标半径上下各 ±(矢高/2) 起伏
        fac = 1.0 / math.cos(math.pi / args.nseg) - 1.0
        rc_eff = rc - (rc + th / 2.0) * fac / 2.0
        half_w = 1.25 * rc_eff * math.sin(math.pi / args.nseg)
        # 方块绕切向轴的倾角: 要让局部 x 轴指向外法线 (n_r, n_z)。
        # MuJoCo 的 quat=Ry(ψ) 把 +X 映射到 (cosψ, -sinψ) 于 (r,z) 平面
        # → 取 ψ = -atan2(n_z, n_r), 即 qy 取负号（否则倾角整体镜像, 方块躺错方向）。
        qw, qy = math.cos(psi / 2.0), -math.sin(psi / 2.0)
        print(' %2d | %6.1f..%6.1f | %6.2f → %6.2f | %5.1f | %5.1f | %7.2f | %.2f,%.2f  %.2f'
              % (si, zg[i] * 1000, zg[j] * 1000, rg[i] * 1000, rg[j] * 1000,
                 math.degrees(psi), th * 1000, (rc_eff - th / 2.0) * 1000,
                 rc_eff * 1000, zc * 1000, half * 1000))
        lines.append('      <replicate count="%d" euler="0 0 %.8f">' % (args.nseg, 360.0 / args.nseg))
        lines.append('        <geom type="box" size="%.5f %.5f %.5f" pos="%.5f 0 %.5f"'
                     % (th / 2.0, half_w, half, rc_eff, zc))
        lines.append('              quat="%.6f 0 %.6f 0" mass="%.5f" friction="%s"'
                     % (qw, qy, box_mass, args.friction))
        lines.append('              solref="0.004 1" solimp="0.99 0.999 0.001"'
                     ' rgba="0.9 0.85 0.7 1" group="3" contype="1" conaffinity="1" />')
        lines.append('      </replicate>')
    # 注意: XML 注释里不允许出现 "--"，所以记录命令时把参数的前导双横线去掉
    # （每个参数前补回 "--" 就是原命令；直接写 "--rim-cut" 会让整个 scene.xml 变成非法 XML）。
    _argv_txt = ' '.join(a.lstrip('-') if a.startswith('--') else a for a in sys.argv[1:])
    block = ('      <!-- 生成命令（可原样重跑还原本块; 参数前补回双横线）: python %s %s -->\n'
             % (os.path.basename(__file__), _argv_txt) +
             '\n'.join(lines))
    print('\n合计: 1 平底方足 + %d 环 x %d 块 = %d 个 geom, 总质量 %.3fkg (块 %.5fkg/个)'
          % (len(kept), args.nseg, nbox + 1, args.foot_mass + nbox * box_mass, box_mass))

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
