#!/usr/bin/env python3
"""苹果碰撞体"平顶化"：把 collision.obj 的顶点截到竖直平面 + 把底面拍平，写 collision_flat.obj。

为什么（第四轮实测/机制）
    机器狗的指板是 26.5mm 厚的刚性平板；接触一个**圆球**时接触法线沿球面径向，
    只要指板中心与球心有几毫米偏差，法线就带竖直分量 → 挤压力产生"向上/向下"的
    分力把球顶走；同时球在桌面上纯滚动不需要克服滑动摩擦（μ=5 也没用，纯滚动的
    阻力属 condim=6 滚动摩擦，本轮不启用）。用户的观测正是"苹果太圆了，滚动走了"。

做法（数值全部来自 meshes/apple/collision.obj 本文件实测，不手抄）
    1. 求 x/y 的 bbox 中心 (cx, cy)（注意原点不在几何中心）
    2. x/y 顶点相对 (cx, cy) 截到 ±A（A 默认 0.0365m）→ 得到两片竖直平面：
       夹持面法线恒水平 → 深度/横向误差不再产生竖直分力；面宽 2A（×0.8 后 58.4mm）
       仍留 ~5.8mm/侧 张开余量（夹爪最大 70mm）
    3. z 截到 z_min（底面 z_min = -1.0mm）→ 球底变成平底圆盘（半径 ~21mm），
       桌面上几何上滚不动（凸包的支撑面是一整个圆盘，不是一点）
    4. 拓扑（f 行）原样保留，只改 v；MuJoCo 编译时自动重算凸包

为什么还要**密采样重采样**（第六轮；probe 实测）
    源文件只有 64 个凸包顶点：落在每个 ±x/±y 极面附近的只有 5~7 个 → clamp 之后的
    "夹持平面"是这几个点连成的多边形（实测 x 面 21.1×19.1mm、y 面 22.0×23.0mm），
    **比指板足迹 26.4×12.58mm 窄** → 指板两侧边缘压在球面/斜面（实测 0.5~29.5° 法线）
    → 100N 级挤压力（env.py: kp=200 × 0.028rad = 5.6N·m ÷ 力臂 49.5mm ≈ 113N/指）
    产生横向分力 → "苹果滚动走了"。
    改法：把顶点换成同一 bbox 的密采样椭球（--resample N 条纬带），clamp 后平面就是
    **解析椭圆截面**：半宽 = sqrt(a² - half²)·0.8，宽度随 --half 连续可控（见下表），
    而夹持总宽 2·half·0.8 与视觉网格的失配仍只有几毫米。
      --half 0.035 → 面 25.0mm（现状 21.1）  宽 56.0mm（不变）
      --half 0.0335→ 面 29.8mm               宽 53.6mm（-2.4mm）
      --half 0.033 → 面 31.2mm               宽 52.8mm（-3.2mm）
    ⇒ 取 0.0335：两面都 ≥ 指板足迹 26.4mm 且余量 ~1.7mm/侧，视觉失配只多 1.2mm/侧。

为什么还要**底面极浅穹顶**（第七轮；生产场景实测）
    平底面与桌面盒接触时 MuJoCo 只给 **1 个接触点**，而这个点是"385 个共面顶点里谁最深"
    的数值抽签 → 实测相邻步接触点跳变 47.73mm、离轴 26.75mm，法线恒竖直 ⇒ 竖直反力
    挂在随机力臂上 = 每步随机力矩泵 ⇒ 苹果自激滚动（实测 30s xy 漂移 18.7~19.6mm、
    倾角 10.3°、|w| 涨到 4.6 rad/s）；图元（sphere/box/cylinder）与胶囊链是解析/多接触
    ⇒ 静置 |w| ≡ 0。
    修法：平底顶点抬成极浅穹顶 z = z0 + sag·(r/r_max)²（--bottom-dome sag，mesh 单位）
    ⇒ 最低点唯一 ⇒ 接触点确定（跳变 47.73 → 0.14mm）；再配 scene.xml 里该 geom 的
    condim="6"（friction 已是 5.0 0.01 0.01）让滚动摩擦生效 ⇒ 实测 |w| ≤ 1e-6 rad/s、
    倾角 0.00°、30s xy 漂移 0.78mm（其中 0.59mm 是 2.6e-5 m/s 的匀速残余滑移）。
    两者缺一不可：只 condim6 残留 1.12 rad/s 的自激、只穹顶残留 ≤0.9 rad/s 的原地小晃。

用法:
    python generate_apple_collision.py [--dry-run] [--half 0.0335] [--flat-h 0.008]
                                       [--resample 32] [--bottom-dome 0.0025]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # /home/mojie/taskdog
SRC = os.path.join(ROOT, 'custom_envs', 'mujoco', 'meshes', 'apple', 'collision.obj')
DST = os.path.join(ROOT, 'custom_envs', 'mujoco', 'meshes', 'apple', 'collision_flat.obj')


def read_obj(path):
    vs, fs, head = [], [], []
    with open(path) as f:
        for ln in f:
            if ln.startswith('v '):
                vs.append([float(x) for x in ln.split()[1:4]])
            elif ln.startswith('f '):
                fs.append(ln.rstrip('\n'))
            else:
                if not vs:
                    head.append(ln.rstrip('\n'))
    return np.asarray(vs, dtype=np.float64), fs, head


def resample_ellipsoid(V, nlat):
    """把顶点换成"同一 bbox 的密采样椭球"（返回 verts, faces 文本行）。

    nlat = 纬带数（经度用 2*nlat 份）→ 顶点数 2 + (nlat-1)*2*nlat。
    取 bbox 半轴当椭球半轴（外接）：夹持面/底面由 clamp 决定，与源点数无关；
    圆面比 64 点凸包"胖"≤1.5mm（源凸包在 64 点之间的凹处），本轮接受（见文件头）。
    """
    lo, hi = V.min(0), V.max(0)
    c, ax = (lo + hi) / 2.0, (hi - lo) / 2.0
    M = 2 * nlat
    pts = [c + [0.0, 0.0, ax[2]]]                      # 北极（1 号顶点）
    for i in range(1, nlat):
        th = np.pi * i / nlat
        for j in range(M):
            ph = 2.0 * np.pi * j / M
            pts.append(c + [ax[0] * np.sin(th) * np.cos(ph),
                            ax[1] * np.sin(th) * np.sin(ph),
                            ax[2] * np.cos(th)])
    pts.append(c + [0.0, 0.0, -ax[2]])                 # 南极（最后 1 号）
    W = np.asarray(pts, dtype=np.float64)
    vid = lambda i, j: 2 + (i - 1) * M + (j % M)       # i=1..nlat-1 纬线, j 经度（1 基）
    F = []
    for j in range(M):                                 # 北极三角扇
        F.append('f 1 %d %d' % (vid(1, j), vid(1, j + 1)))
    for i in range(1, nlat - 1):                       # 中间四边形 → 两三角
        for j in range(M):
            a, b = vid(i, j), vid(i + 1, j)
            F.append('f %d %d %d' % (a, b, vid(i + 1, j + 1)))
            F.append('f %d %d %d' % (a, vid(i + 1, j + 1), vid(i, j + 1)))
    n = len(W)
    for j in range(M):                                 # 南极三角扇
        F.append('f %d %d %d' % (n, vid(nlat - 1, j + 1), vid(nlat - 1, j)))
    return W, F


def boundary_rings(c, ax, half, flat_h, K=96):
    """5 个 clamp 平面（±x, ±y, 底面）的**解析边界环**：点在椭球面上 → 不撑大外形。

    clamp 之后这些环就成了平面区域的边界 ⇒ 平面尺寸 = 解析椭圆
      ±x 面: 半轴 (ay,az)·sqrt(1-(half/ax)²)   ±y 面: (ax,az)·sqrt(1-(half/ay)²)
      底面  : (ax,ay)·sqrt(1-((az-flat_h)/az)²)
    为什么需要：只用经纬网格采样时，落在 clamp 面上的点集边界随 --half 跳变
    （实测 0.034→23.5mm、0.0335→28.9mm 一跳），尺寸不可连续控制；补环后连续且精确。
    """
    ph = np.arange(K) * 2.0 * np.pi / K
    rings = []
    for sgn in (1.0, -1.0):
        k = np.sqrt(max(1.0 - (half / ax[0]) ** 2, 0.0))
        rings.append(np.stack([np.full(K, c[0] + sgn * half),
                               c[1] + ax[1] * k * np.sin(ph), c[2] + ax[2] * k * np.cos(ph)], 1))
        k = np.sqrt(max(1.0 - (half / ax[1]) ** 2, 0.0))
        rings.append(np.stack([c[0] + ax[0] * k * np.sin(ph), np.full(K, c[1] + sgn * half),
                               c[2] + ax[2] * k * np.cos(ph)], 1))
    k = np.sqrt(max(1.0 - ((ax[2] - flat_h) / ax[2]) ** 2, 0.0))
    rings.append(np.stack([c[0] + ax[0] * k * np.sin(ph), c[1] + ax[1] * k * np.cos(ph),
                           np.full(K, c[2] - ax[2] + flat_h)], 1))
    return np.vstack(rings)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true', help='只打印, 不写文件')
    ap.add_argument('--half', type=float, default=0.033, help='x/y 相对 bbox 中心的截断半宽 m')
    ap.add_argument('--min-face', type=float, default=0.0264,
                    help='夹持平面最小边长(米, **缩放后**)警戒 = 指板足迹 26.4mm；小于它就说明指板边缘会压到斜面')
    ap.add_argument('--scale', type=float, default=0.8,
                    help='scene.xml 里该 mesh geom 的 scale（只用于打印/告警，不写进文件）')
    ap.add_argument('--flat-h', type=float, default=0.008, help='底面拍平高度 m（z<=zmin+flat_h → z=zmin）')
    ap.add_argument('--resample', type=int, default=24,
                    help='0=用源 64 顶点(旧行为); N>0=椭球(同 bbox)采样的纬带数, 默认 24 → 1106 顶点')
    ap.add_argument('--bottom-dome', type=float, default=0.0,
                    help='底面穹顶 sag（mesh 单位, ×scale 后是物理高度）: 平底顶点 z = z0 + sag·(r/r_max)², '
                         '轴心最低 ⇒ 最低点唯一、接触点不再抽签。0=平底（旧行为）')
    args = ap.parse_args()

    V, F, head = read_obj(SRC)
    print('源 %s: %d 顶点, %d 面' % (os.path.basename(SRC), len(V), len(F)))
    x0, x1 = float(V[:, 0].min()), float(V[:, 0].max())
    y0, y1 = float(V[:, 1].min()), float(V[:, 1].max())
    z0, z1 = float(V[:, 2].min()), float(V[:, 2].max())
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    print('bbox x %.2f..%.2f  y %.2f..%.2f  z %.2f..%.2f mm'
          % (x0 * 1000, x1 * 1000, y0 * 1000, y1 * 1000, z0 * 1000, z1 * 1000))
    print('bbox 中心 (%.2f, %.2f) mm; 半径 x %.2f y %.2f (相对中心)'
          % (cx * 1000, cy * 1000, max(abs(x0 - cx), abs(x1 - cx)) * 1000,
             max(abs(y0 - cy), abs(y1 - cy)) * 1000))
    if args.resample > 0:
        V, F = resample_ellipsoid(V, args.resample)
        ax = (V.max(0) - V.min(0)) / 2.0
        ring = boundary_rings(np.array([cx, cy, (z0 + z1) / 2.0]), ax, args.half, args.flat_h)
        V = np.vstack([V, ring])
        print('重采样: %d 纬带网格 %d 顶点 + 5 个 clamp 面解析边界环 %d 点 = %d 顶点 %d 面'
              % (args.resample, len(V) - len(ring), len(ring), len(V), len(F)))

    W = V.copy()
    # 1) x/y 截到 ±half（相对 bbox 中心）
    for k, c in ((0, cx), (1, cy)):
        d = W[:, k] - c
        W[:, k] = c + np.clip(d, -args.half, args.half)
    # 2) 底面拍平（+ 可选极浅穹顶：轴心最低 ⇒ 最低点唯一）
    flat = W[:, 2] <= z0 + args.flat_h
    if args.bottom_dome > 0.0:
        rr_flat = np.hypot(W[flat, 0] - cx, W[flat, 1] - cy)
        W[flat, 2] = z0 + args.bottom_dome * (rr_flat / rr_flat.max()) ** 2
        r_edge = float(rr_flat.max()) * args.scale * 1000
        lo_i = int(np.argmin(W[:, 2]))
        r_lo = float(np.hypot(W[lo_i, 0] - cx, W[lo_i, 1] - cy)) * args.scale * 1000
        print('底面穹顶: sag %.4f mesh = %.3f mm 物理（底面边缘抬高）; 轴心最低点距轴 %.2fmm ×%g'
              % (args.bottom_dome, args.bottom_dome * args.scale * 1000, r_lo, args.scale))
        print('  平底圆盘半径 %.2fmm（物理）: 盘内(x/y 平面)最高处比轴心高 %.3fmm'
              % (r_edge, args.bottom_dome * args.scale * 1000))
    else:
        W[flat, 2] = z0
    # 3) 顶部也削一点（球顶半径小, 对夹持无用; 保留原样会留一个尖, 无害但无意义）
    #    实测保持原样即可 → 不动

    moved = int(np.sum(np.any(np.abs(W - V) > 1e-9, axis=1)))
    n_xy = int(np.sum(np.any(np.abs(W[:, :2] - V[:, :2]) > 1e-9, axis=1)))
    # 夹持平面实测: 落在 x=cx±half (或 y=cy±half) 平面上的顶点 → 它们凸包 = 平面区域
    for k, c, name, other in ((0, cx, 'x', 1), (1, cy, 'y', 0)):
        on = np.where(np.abs(np.abs(W[:, k] - c) - args.half) < 1e-9)[0]
        if len(on) >= 3:
            e0 = float(W[on, other].max() - W[on, other].min())
            e1 = float(W[on, 2].max() - W[on, 2].min())
            sm = min(e0, e1) * args.scale
            flag = ('' if sm >= args.min_face else
                    '  <<< 平面过小(缩放后 %.1fmm < 指板足迹 %.1fmm)：指板边缘会压到斜面!'
                    % (sm * 1000, args.min_face * 1000))
            print('  %s 平面: %d 顶点, 尺寸(缩放前) %.1f x %.1f mm -> ×%g = %.1f x %.1f mm%s'
                  % (name, len(on), e0 * 1000, e1 * 1000, args.scale,
                     e0 * args.scale * 1000, e1 * args.scale * 1000, flag))
        else:
            print('  %s 平面: %d 顶点 <<< 没切出平面, 需要减小 --half!' % (name, len(on)))
    mx0, mx1 = float(W[:, 0].min()), float(W[:, 0].max())
    my0, my1 = float(W[:, 1].min()), float(W[:, 1].max())
    print('改动顶点 %d/%d（其中 x/y 段 %d, 底平 %d）' % (moved, len(V), n_xy, int(flat.sum())))
    print('新 bbox x %.2f..%.2f (宽 %.2fmm)  y %.2f..%.2f (宽 %.2fmm)  z %.2f..%.2f'
          % (mx0 * 1000, mx1 * 1000, (mx1 - mx0) * 1000,
             my0 * 1000, my1 * 1000, (my1 - my0) * 1000, W[:, 2].min() * 1000, W[:, 2].max() * 1000))
    # 平底圆盘: 底圈顶点的 xy 质心 = 轴心, 取最大半径
    bi = np.where(flat)[0]
    if len(bi) >= 3:
        # 圆心取平底点集的 bbox 中心（不是质心：补了边界环后点分布不均匀，质心会偏移，
        # 曾把 ⌀47.4 测成 48.3）。平底是解析椭圆，长轴半径 = max(ax,ay)*sqrt(1-((az-flat_h)/az)²)
        bcx = (W[bi, 0].min() + W[bi, 0].max()) / 2.0
        bcy = (W[bi, 1].min() + W[bi, 1].max()) / 2.0
        br = float(np.hypot(W[bi, 0] - bcx, W[bi, 1] - bcy).max())
    else:
        br, bcx, bcy = float('nan'), float('nan'), float('nan')
    print('×%g 缩放后: 夹持宽 %.2fmm (夹爪最大 70mm → 张开余量 %.2fmm/侧), 底圆盘直径 %.1fmm%s'
          % (args.scale, (mx1 - mx0) * args.scale * 1000,
             (70.0 - (mx1 - mx0) * args.scale * 1000) / 2.0, 2 * br * args.scale * 1000,
             '（穹顶版: 盘边比轴心高 %.2fmm, 仍远小于落地下沉 ~0.5mm → 只有轴心接触）'
             % (args.bottom_dome * args.scale * 1000) if args.bottom_dome > 0 else ''))

    if args.dry_run:
        print('\n[dry-run] 不写文件')
        return 0

    out = list(head) if head else ['o convex_flat']
    if not out[0].startswith('o '):
        out.insert(0, 'o convex_flat')
    out.append('# 平顶化: x/y 相对 bbox 中心 (%.4f, %.4f) 截到 ±%.4f; z <= %.4f 拍平到 %.4f%s'
               % (cx, cy, args.half, z0 + args.flat_h, z0,
                  ('+底穹 %.4f·(r/r_max)²' % args.bottom_dome) if args.bottom_dome > 0 else ''))
    for p in W:
        out.append('v %.6f %.6f %.6f' % tuple(p))
    out.extend(F)
    with open(DST, 'w') as f:
        f.write('\n'.join(out) + '\n')
    print('\n已写入 %s (%d 行)' % (DST, len(out)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
