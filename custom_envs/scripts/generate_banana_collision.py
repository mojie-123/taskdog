#!/usr/bin/env python3
"""程序生成香蕉的碰撞体（沿中心线的胶囊链），重写 scene.xml 的 BANANA_COLLISION 块。

为什么（第四轮实测）
    旧碰撞体是 3 根手调胶囊：抓取站附近（body-x ≈ -45.7）实际生效的是"左弯曲段"，
    它的圆心在 y=-20.6mm、半径 10mm；而视觉网格该站的中心线在 y≈-28.1mm、
    管半径 17.7mm（z 向）→ 碰撞管偏 7.5mm、细 43%。夹爪夹一根又偏又细的管子：
    两侧指板接触时刻/深度不对称 → 挤压力矩 + 粘滑 → 用户看到的"奇怪的振动"。

做法（数值全部从 meshes/banana/textured.obj 实测，不手抄）
    1. 沿 body-x 取薄片（半厚 1.5mm），每片算：
         中心 y = (ymin+ymax)/2, 中心 z = (zmin+zmax)/2   ← 管截面的极值中点（对偏斜稳健）
         半径 r = (zmax-zmin)/2                            ← 轴向在 x-y 平面内弯曲, z 向极差就是直径
         半径 ry = (ymax-ymin)/2                           ← **夹持方向**的半宽（见下）
    2. 相邻站心之间放一根胶囊（半径 = 两端站均值）→ 链式贴合弯曲的中心线
    3. 每站 ±y 两侧各加一块**薄板**（夹持面"拍平"），外表面落在 y = cy ± ry
       （= 视觉最宽处），板厚 4mm、z 向半高 0.45·ry。
    4. 总质量默认 0.068kg（与旧 3 胶囊一致，按 geom 数均分）

为什么夹持方向是 ±y（不是假设，是几何必然）
    夹爪最大开口 70mm，香蕉长 180mm → 沿长轴根本夹不住（要 ≥180mm 开口）；
    而抓取是**从上方**接近（approach 沿 -z），两指板只能水平闭合 → 只有 ±y 这一个解。
    香蕉 body 没有 quat（body 系 = 世界系）⇒ body-y 就是两指板的法线方向。

为什么"拍平"（第五轮；用户观测：夹住时奇怪振动）
    · 指板是**严格平行的平板**（probe 实测：任意 q 下板面都平行于 xz 平面、只沿 y 刚性平移），
      圆圆地夹一根管子 → 接触点/法线随相对姿态漂移；拍平后法线恒为 ±y，与夹持深度无关。
    · friction 5.0 → 1.2：MuJoCo 接触摩擦=两 geom **逐元素取 max**，手指默认 1.0，
      所以旧的有效 μ = 5.0。5.0 配上"kp=200 的 PD + 1mm/步速率限制的目标"= 典型**粘滑振动**。

为什么拍平要**整个抓取窗共用一对平面**（第六轮；probe 实测的残留漏洞）
    指板近侧面实测 26.40mm(切向) × 12.58mm(approach)，而站距只有 12mm → **一块指板永远
    跨 2~3 个站**。逐站拍平只保证"每站各自是一个平面"，站与站之间仍有台阶：
    实测 l7 侧三块板面在 T = -28.08 / -23.30 / -19.11mm（台阶 4.2~4.8mm，因为香蕉中线
    每 12mm 站心 y 差 3.4~3.8mm），l8 侧 id95 在 +22.03 → 两指落在**不同**的板上，
    夹持中心偏捏合线 3.03mm；而板的 x 端面是一个 90° 的面，CLOSE 触后仍命令 28mm 行程
    （kp=200×0.028rad）→ 指板压过台阶、法线翻到端面上 → 沿香蕉长轴的推力（粘滑振动源）。
    ⇒ 抓取窗（|x - grasp_x| ≤ grip_window）内所有板的两个面各共用**一个**平面：
      平面位置 = 窗口内全部圆管极值的最外包络 + pad_lead（保证圆管仍被板挡住），
      于是窗口内板面共面且相接（台阶=0），两指接触时刻由同一对平面决定（不再偏心）。
      窗口外仍按逐站拍平/挡管，视觉代价只是窗口内板面与视觉表面差几毫米（碰撞≠视觉）。

为什么改成**按指板足迹拆板**（第九轮；按固定半宽选窗的两个毛病）
    (a) 窗比足迹小 → 指板边缘落在窗外那块板上（那块板按自己那段摆面，可能高出 4.8mm）：
        指板先顶台阶、其余悬空 → 第一轮的踩台阶失效模式回归（14mm 窗实测：共面区 24mm
        < 足迹 26.4mm，指板左缘 2.4mm 踩 4.8mm 台阶）。
    (b) 窗比足迹大 → 共面区被撑到 48mm，平面被窗口两端（弯香蕉的粗端）顶高 →
        窗内板面相对视觉凸出最大 ~14mm（27mm 窗实测）。
    ⇒ 共面区 COP 直接由足迹定义：|x - grasp_x| ≤ footprint/2 + cop_margin（默认 26.4/2 + 2mm），
      与 COP 相交的站按 x 拆成 外段/共面段/外段（<1mm 的段丢掉）：共面段用 COP 统一平面，
      外段按**本段 x 跨度**的视觉外形 + 相邻圆管极值摆面（_visual_reach/_link_reach）。
      于是：指板整片落在同一平面；台阶全部推到 COP 边界（足迹之外）；凸出只留在 COP 内
      （指板正下方，夹持时被夹爪遮住）。视觉凸出量 = COP 内香蕉表面的起伏（消不掉的几何代价）。
      --grasp-tol / --grip-window 自本轮起停用（保留参数只为不破坏旧命令行）。

为什么拍平板要**盖过同段胶囊**（第一版拍平板的漏洞，probe 实测）
    胶囊是"圆柱 + 两端半球"。板面只按**本站**视觉半宽 ry 摆放时，**邻站那根胶囊的端帽半球**
    会跨过接缝，在本板 x 跨度内仍凸出近整个半径：实测 link3（-62.5↔-50.5）在 -62.5 端的
    端帽凸到体 y = -6.73，而 -50.5 站的板面在 y = -7.41 → **圆管比板外 0.68mm**，
    夹爪先碰圆管，法线是斜的（MuJoCo 实测 21.8°/29.5°，另有 9.5°/1.7°/0.9° 等 10 处胶囊接触），
    拍平板形同虚设 → 粘滑振动照旧。侧壁（倾斜圆柱）也有份但小得多。
    ⇒ 板面取 max(视觉表面, 相邻两根胶囊在本板 x 跨度内的 y 极值) + pad_lead(默认 1mm)，
      极值按 _link_reach 算（侧壁 r·sqrt(1−u_y²) + 两端半球）。
      这样板面在整个夹持区都是最外层，圆管被板挡在后面（见下表 lead 列，全部 >0）。

用法:
    python generate_banana_collision.py [--dry-run] [--step 0.012] [--slab 0.0015]
                                        [--mass 0.068] [--friction "1.2 0.01 0.01"]
                                        [--no-pad] [--pad-thick 0.004] [--pad-zfrac 0.45]
                                        [--pad-min-r 0.008] [--pad-lead 0.001]
                                        [--grip-window 0.027] [--pinch-y -0.0287]
                                        [--pad-zmin 0.010]

--grip-window 取 27mm = 指板切向足迹 26.4mm 的一半(13.2) + 10mm 定位余量，四舍五入；
窗口内 4 块板（x -68.5..-20.5，48mm 连续平面）远大于足迹 ⇒ 指板落点 ±10mm 内都在共面区。

为什么整体内缩进视觉（第十二轮；用户实测：抓取仍失败，根因是"看不见的障碍"）
    前几轮板面摆在"同段圆管 + 1mm"处，而共面区(COP)为了盖住整片指板足迹用的是**窗口包络**
    —— 香蕉是弯的，包络取最粗处 ⇒ 板面相对局部视觉凸出 +1.0..+9.7mm（见 dry-run"超视觉"列）。
    碰撞体在 group=3（相机看不见），而下探/规划按视觉算 ⇒ 撞上不可见板。用户要求：全部缩进
    视觉内部，**接受穿模**（夹爪夹进香蕉视觉体比抓不到好）。
    ⇒ --inset δ（默认 11mm）：板面与圆管半径**同步内缩** δ，于是
       (a) 板面 ≤ 视觉包络 − (δ−pad_lead)，COP 最窄处仍内缩 ~1.3mm → 全程无凸出；
       (b) 板面与圆管间距（挡管余量）逐点不变 ⇒ 不会退回"圆管先接触→斜法线→粘滑振动"；
       (c) 面取 max(挡管要求, 视觉−δ) 两者中更靠外的那个 —— 端头圆管被 rmin 托住时，
           板仍在管外 1mm（此时视觉约束已不可能满足，但该处不在抓取窗内）。
    z 向：板半高不再用"0.45·ry 或 10mm"，改成**盒角恰好贴住椭圆管截面**
       h = 0.95·r·sqrt(1−(dy/ry)²)（dy = 板面到站心的 y 偏移）。矩形盒放进椭圆截面时若直接取
       h=r，盒角会凸出视觉 3.8mm；本式让四角落在视觉内（dry-run"角凸"列应 ≤0）。
       抓取区 h≈12.3mm：指板 12.58mm 的 approach 足迹仍被整片覆盖，且板底更低 → 香蕉沉桌量减小。
    代价（用户已接受）：CLOSE 时指板穿入香蕉视觉体 ~4..10mm 才碰到板面（板面就是碰撞面，
    接触间隙 = COP 两侧面距离 26.6mm，夹持中心不变）。
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # /home/mojie/taskdog
SCENE = os.path.join(ROOT, 'custom_envs', 'mujoco', 'scene.xml')
VIS = os.path.join(ROOT, 'custom_envs', 'mujoco', 'meshes', 'banana', 'textured.obj')
BEGIN = '<!-- BANANA_COLLISION_BEGIN -->'
END = '<!-- BANANA_COLLISION_END -->'


def load_visual_mesh(path):
    v = []
    with open(path) as f:
        for ln in f:
            if ln.startswith('v '):
                v.append([float(x) for x in ln.split()[1:4]])
    return np.asarray(v, dtype=np.float64)


def station(V, x, slab, min_v=6):
    """一个站的 (cy, cz, r, ry)；顶点太少返回 None。

    cy = 薄片顶点在 (y,z) 平面圆拟合(Kasa)的圆心 y —— 管截面是整圈面点, 拟合比
         "极值中点"稳健（端头/斜切截面不会跑飞）;
    cz = (zmin+zmax)/2, r = (zmax-zmin)/2 —— 中心线在 x-y 平面内弯曲, z 向极差
         就是**内接**管直径: 用它当半径可保证胶囊不凸出视觉表面（圆拟合的半径是
         平均半径, 端头会偏大 → 香蕉会"踮着脚尖"站着）。
    另算极值中点做交叉校验。
    """
    m = np.abs(V[:, 0] - x) < slab
    if int(m.sum()) < min_v:
        return None
    S = V[m]
    y, z = S[:, 1], S[:, 2]
    A = np.stack([y, z, np.ones_like(y)], axis=1)
    b = -(y * y + z * z)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None
    D, E, F = sol
    cy, cz_fit = -D / 2.0, -E / 2.0
    cz = (z.min() + z.max()) / 2.0
    r = float((z.max() - z.min()) / 2.0)
    cy2 = (y.min() + y.max()) / 2.0
    cz2 = (z.min() + z.max()) / 2.0
    ry = float((y.max() - y.min()) / 2.0)     # 夹持方向(±y)的半宽 → 拍平板的外表面
    return cy, cz, r, ry, int(m.sum()), (cy - cy2, cz_fit - cz2)


def _link_reach(sts, link_r, j, x_lo, x_hi):
    """第 j 根胶囊在 [x_lo, x_hi] 这个 x 跨度内的 (min_y, max_y)。

    侧壁：轴段与跨度求交，极值 = 轴心 y 极值 ± r·sqrt(1−u_y²)（倾斜圆柱在 y 向的半伸）。
    端帽：两个半球，各取跨度上**离端心最近**的那点算 sqrt(r²−d²)。
          ← 第一版只算侧壁，漏了端帽：邻站端帽的半球跨过接缝、在本板 x 跨度内仍能凸出
            近整个半径（probe 实测：link3 在 -62.53 端帽凸到体 y=-6.73，而 -50.53 站板面
            在 -7.41 → 圆管比板外 0.68mm，夹爪先碰圆管，法线 21.8°/29.5° 斜的）。
    """
    x_a, y_a, z_a = sts[j][0], sts[j][1], sts[j][2]
    x_b, y_b, z_b = sts[j + 1][0], sts[j + 1][1], sts[j + 1][2]
    r = link_r[j]
    d = np.array([x_b - x_a, y_b - y_a, z_b - z_a])
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return None
    u = d / L
    lo, hi = 1e9, -1e9
    ov_lo, ov_hi = max(min(x_a, x_b), x_lo), min(max(x_a, x_b), x_hi)
    if ov_lo <= ov_hi:
        if abs(u[0]) > 1e-9:                          # 侧壁（倾斜圆柱）
            ys = [y_a + u[1] * (t - x_a) / u[0] for t in (ov_lo, ov_hi)]
            half = r * np.sqrt(max(1.0 - u[1] ** 2, 0.0))
            lo, hi = min(ys) - half, max(ys) + half
        else:                                         # 轴几乎 ⊥ x：整根都在跨度内
            lo, hi = min(y_a, y_b) - r, max(y_a, y_b) + r
    for xe, ye in ((x_a, y_a), (x_b, y_b)):           # 端帽（半球）
        dd = abs(min(max(xe, x_lo), x_hi) - xe)
        if dd < r:
            reach = np.sqrt(r * r - dd * dd)
            lo, hi = min(lo, ye - reach), max(hi, ye + reach)
    return None if lo > hi else (float(lo), float(hi))


def _station_at(sts, x):
    """按站线性插值出 x 处的 (cy, cz, r, ry)；超出两端就夹到端点。

    第十二轮定板 z 半高用：香蕉是弯的、尖部截面是斜切（段中心站的椭圆与实际相差很大，
    例如 x=-92 处 tube 半径只有 8.9mm 而段中心站是 12.0mm），所以沿整段 x 采样取最严值。
    与胶囊链用的是同一套插值模型（站心连线 + 内接半径）。
    """
    xs = [s[0] for s in sts]
    if x <= xs[0]:
        k = 0
    elif x >= xs[-1]:
        k = len(sts) - 2
    else:
        k = max(i for i in range(len(sts) - 1) if xs[i] <= x)
    xa, xb = sts[k][0], sts[k + 1][0]
    t = 0.0 if abs(xb - xa) < 1e-12 else (x - xa) / (xb - xa)
    t = min(max(t, 0.0), 1.0)
    return tuple(sts[k][i] + (sts[k + 1][i] - sts[k][i]) * t for i in (1, 2, 3, 4))


def _visual_reach(V, x_lo, x_hi):
    """视觉网格在 [x_lo, x_hi] 跨度内的 (min_y, max_y)。

    第九轮拆板用：板面按**本段实际 x 跨度**内的视觉外形摆放（比按整站 cy±ry 更贴），
    逐段贴合后共面区外的板不会再白白凸出（站级极值对整个 12mm 站都保守）。
    """
    m = (V[:, 0] >= x_lo) & (V[:, 0] <= x_hi)
    if not m.any():
        return None
    return float(V[m, 1].min()), float(V[m, 1].max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true', help='只打印, 不写 scene.xml')
    ap.add_argument('--slab', type=float, default=0.0015, help='取样薄片半厚 m')
    ap.add_argument('--step', type=float, default=0.012, help='相邻站间距 m（scene 现状 12mm）')
    ap.add_argument('--mass', type=float, default=0.068, help='碰撞体总质量 kg')
    ap.add_argument('--rmin', type=float, default=0.006, help='半径下限 m（端头太薄会抖）')
    ap.add_argument('--rmargin', type=float, default=0.0,
                    help='半径外扩 m（0 = 严格贴视觉表面; >0 会视觉切入）')
    ap.add_argument('--inset', type=float, default=0.011,
                    help='碰撞体整体内缩 m（第十二轮：板面与圆管同步内缩，消除凸出视觉的不可见障碍）')
    ap.add_argument('--friction', type=str, default='1.2 0.01 0.01',
                    help='摩擦串（第五轮 5.0→1.2：有效 μ=max(手指1.0, 本值)，5.0 会粘滑振动）')
    ap.add_argument('--no-pad', action='store_true', help='不加 ±y 夹持拍平板（退回纯胶囊链）')
    ap.add_argument('--pad-thick', type=float, default=0.004, help='拍平板厚度 m（沿 y）')
    ap.add_argument('--pad-zfrac', type=float, default=0.45,
                    help='[第十二轮起停用] 旧 z 半高 = zfrac·ry；现按椭圆盒角公式定 h')
    ap.add_argument('--pad-min-r', type=float, default=0.008,
                    help='ry 小于此值的站不加板（端头太薄，板会被胶囊盖住）')
    ap.add_argument('--pad-lead', type=float, default=0.001,
                    help='板面相对同段胶囊外凸的最小余量 m（保证先碰板、法线恒 ±y）')
    ap.add_argument('--pad-min-h', type=float, default=0.005,
                    help='盒角装不进视觉的段（算出的 z 半高 < 此值）就不放板，退回圆管链')
    ap.add_argument('--grasp-x', type=float, default=-0.0457,
                    help='记录抓取站的体坐标 x（来自管线记录；只有指板够得着的站才加宽挡管）')
    ap.add_argument('--grasp-tol', type=float, default=0.014,
                    help='[第八轮起停用] 旧的挡管区半宽；现按 footprint/cop-margin 决定')
    ap.add_argument('--grip-window', type=float, default=0.027,
                    help='[第八轮起停用] 旧的按站选窗半宽；共面区改由 footprint/cop-margin 定')
    ap.add_argument('--footprint', type=float, default=0.0264,
                    help='指板切向足迹 m（文件头 probe 实测 26.4mm；共面区 = 此值 + 两侧余量）')
    ap.add_argument('--cop-margin', type=float, default=0.002,
                    help='共面区(COP)在足迹两侧的外扩余量 m（吸收落点定位误差；原窗设计留 10mm）')
    ap.add_argument('--pinch-y', type=float, default=-0.0287,
                    help='捏合线在体坐标的 y m（= 夹爪中心线；只用于打印对称性，不参与摆面）')
    ap.add_argument('--pad-zmin', type=float, default=0.010,
                    help='[第十二轮起停用] 旧窗口板 z 半高下限；现按椭圆盒角公式定 h')
    args = ap.parse_args()

    V = load_visual_mesh(VIS)
    x0, x1 = float(V[:, 0].min()), float(V[:, 0].max())
    print('视觉网格 %d 顶点, x %.2f..%.2fmm, y %.2f..%.2f, z %.2f..%.2f'
          % (len(V), x0 * 1000, x1 * 1000, V[:, 1].min() * 1000, V[:, 1].max() * 1000,
             V[:, 2].min() * 1000, V[:, 2].max() * 1000))

    # 候选站（从两端各留 3mm）→ 保留顶点足够的
    xs = np.arange(x0 + 0.003, x1 - 0.003 + 1e-9, args.step)
    sts = []
    for x in xs:
        s = station(V, x, args.slab)
        if s is None:
            print('  站 x=%+.3f 顶点太少, 跳过' % x)
            continue
        sts.append((float(x),) + s)
    if len(sts) < 3:
        raise RuntimeError('有效站太少: %d' % len(sts))

    print('\n 站 | x(mm)   | 顶点 | 圆心 y,z(mm)     | 半径mm | 夹持半宽ry | 与极值中点差(mm)')
    _dmax = 0.0
    for x, cy, cz, r, ry, n, d in sts:
        _dmax = max(_dmax, abs(d[0]), abs(d[1]))
        print('   | %+7.2f | %4d | %+7.2f,%+7.2f | %6.2f | %7.2f | %+.2f,%+.2f'
              % (x * 1000, n, cy * 1000, cz * 1000, r * 1000, ry * 1000, d[0] * 1000, d[1] * 1000))
    print('圆拟合 vs 极值中点 最大差 %.2fmm（>3mm 说明该站截面不干净）' % (_dmax * 1000))

    # 拍平板（第九轮拆板）：共面区(COP) = 指板切向足迹(footprint) + 两侧 cop_margin。
    # 与 COP 相交的站按 x 拆成 外段/共面段/外段：共面段共用一对平面（台阶=0，盖满指板落点），
    # 外段各自贴视觉+挡管（台阶被推到 COP 边界以外）。见文件头"为什么共面区必须盖住指板足迹"。
    pads = [] if args.no_pad else [k for k, s in enumerate(sts) if s[4] >= args.pad_min_r]
    ncap = len(sts) - 1
    # 每根胶囊的半径（两端视觉半径取小 = 内接）与该段在**接缝处**的轴心 y：
    # 板面必须盖住它们，否则圆管会从板边冒出去先接触（见文件头"为什么拍平板要盖过同段胶囊"）
    link_r = [max(min(sts[i][3], sts[i + 1][3]) + args.rmargin - args.inset, args.rmin)
              for i in range(ncap)]
    hx = args.step / 2.0
    cop_lo = args.grasp_x - args.footprint / 2.0 - args.cop_margin
    cop_hi = args.grasp_x + args.footprint / 2.0 + args.cop_margin
    win = [k for k in pads
           if min(sts[k][0] + hx, cop_hi) - max(sts[k][0] - hx, cop_lo) > 1e-3]
    win_lo = win_hi = None
    rows = []       # (x, cy, cz, ry, r, sa, sb, is_cop, lo, hi, h, v_lo, v_hi, t_lo, t_hi)
    if win:
        # 视觉包络与圆管包络分开记：面 = max(挡管要求, 视觉−δ) 里"更靠外"的那个（第十二轮）
        vlo, vhi, tlo, thi = 1e9, -1e9, 1e9, -1e9
        for k in win:
            x = sts[k][0]
            a, b = max(x - hx, cop_lo), min(x + hx, cop_hi)
            _v = _visual_reach(V, a, b)              # 该段视觉外形也要包进去
            if _v is not None:
                vlo, vhi = min(vlo, _v[0]), max(vhi, _v[1])
            for j in range(max(0, k - 2), min(ncap, k + 2)):
                _r = _link_reach(sts, link_r, j, a, b)
                if _r is not None:
                    tlo, thi = min(tlo, _r[0]), max(thi, _r[1])
        win_lo = min(tlo - args.pad_lead, vlo - args.pad_lead + args.inset)
        win_hi = max(thi + args.pad_lead, vhi + args.pad_lead - args.inset)
    skipped = []                    # 盒角装不进视觉的段（香蕉尖部斜截面）→ 不放板
    for k in pads:
        x, cy, cz, r, ry = sts[k][:5]
        kwin = win_lo is not None and k in win
        if kwin:
            a, b = max(x - hx, cop_lo), min(x + hx, cop_hi)
            segs = []
            if a - (x - hx) > 1e-3:
                segs.append((x - hx, a, False))
            segs.append((a, b, True))
            if (x + hx) - b > 1e-3:
                segs.append((b, x + hx, False))
        else:
            segs = [(x - hx, x + hx, False)]
        for sa, sb, is_cop in segs:
            _v = _visual_reach(V, sa, sb) or (cy - ry, cy + ry)
            v_lo, v_hi = _v
            t_lo, t_hi = None, None         # 圆管包络**只含圆管**（不能混进视觉，否则内缩失效）
            for j in range(max(0, k - 2), min(ncap, k + 2)):   # 相邻两段+隔一段（端帽会伸过来）
                _r = _link_reach(sts, link_r, j, sa, sb)
                if _r is not None:
                    t_lo = _r[0] if t_lo is None else min(t_lo, _r[0])
                    t_hi = _r[1] if t_hi is None else max(t_hi, _r[1])
            if t_lo is None:
                t_lo, t_hi = v_lo, v_hi     # 该跨度没有圆管（理论上不会）：退回视觉
            if is_cop:
                lo, hi = win_lo, win_hi     # 共面段：这对值对整段 COP 都一样
            else:
                # 板面 = max(挡管要求, 视觉−δ) 中"更靠外"的那个（见文件头第十二轮）
                lo = min(t_lo - args.pad_lead, v_lo - args.pad_lead + args.inset)
                hi = max(t_hi + args.pad_lead, v_hi + args.pad_lead - args.inset)
            # z 半高：矩形盒放进椭圆管截面，盒角恰好贴住视觉（沿整段 x 采样取最严 + 5% 余量）
            _h = 1e9
            for _xs in np.linspace(sa, sb, 25):
                _cyX, _czX, _rX, _ryX = _station_at(sts, float(_xs))
                if _rX <= 1e-6 or _ryX <= 1e-6:
                    continue
                _dyX = max(hi - _cyX, _cyX - lo)
                _h = min(_h, _rX * float(np.sqrt(max(1.0 - (_dyX / _ryX) ** 2, 0.0))))
            h = 0.95 * (0.002 if _h == 1e9 else _h)
            if h < args.pad_min_h:
                # 该段香蕉轴心沿 x 移动太快（尖部斜截面）：任何盒子都会凸出视觉 → 不放板，
                # 该处碰撞体 = 圆管链（那里没有指板落点，不需要平面）
                skipped.append((sa, sb, h))
                continue
            h = max(h, 0.002)
            rows.append((x, cy, cz, ry, r, sa, sb, is_cop, lo, hi, h, v_lo, v_hi, t_lo, t_hi))
    ngeom = ncap + len(rows)
    mg = args.mass / ngeom
    lines = []
    print('\n 胶囊 | fromto (body系, m)                                  | 半径mm | 视觉半径(两端)mm')
    for i in range(ncap):
        x_a, cy_a, cz_a, r_a = sts[i][0], sts[i][1], sts[i][2], sts[i][3]
        x_b, cy_b, cz_b, r_b = sts[i + 1][0], sts[i + 1][1], sts[i + 1][2], sts[i + 1][3]
        r = link_r[i]
        # 与视觉形状的偏差: 站心的横向偏移（旧碰撞体的问题就是圆心偏了）
        lines.append('      <geom type="capsule" group="3" fromto="%.5f %.5f %.5f  %.5f %.5f %.5f"'
                     ' size="%.5f"' % (x_a, cy_a, cz_a, x_b, cy_b, cz_b, r))
        lines.append('            mass="%.5f" friction="%s" solref="0.004 1"'
                     ' solimp="0.99 0.999 0.001" rgba="1.0 0.85 0.1 1" />'
                     % (mg, args.friction))
        print('  %2d   | %+.4f %+.4f %+.4f  %+.4f %+.4f %+.4f | %6.2f | r_visual(两端)=%5.2f,%5.2f'
              % (i, x_a, cy_a, cz_a, x_b, cy_b, cz_b, r * 1000, r_a * 1000, r_b * 1000))
    if rows:
        _ncop = sum(1 for r in rows if r[7])
        print('\n共面区(COP): x %+.2f..%+.2f mm = 指板足迹 %.1fmm + 两侧余量 %.1fmm → %d 站拆成 %d 段共面:'
              % (cop_lo * 1000, cop_hi * 1000, args.footprint * 1000,
                 args.cop_margin * 1000, len(win), _ncop))
        print('  上侧(l7)面 y = %+.3fmm, 下侧(l8)面 y = %+.3fmm；两指接触时捏合线偏移 %+.3fmm'
              '（← 共面后两指同刻接触，此值即夹持中心偏心）'
              % (win_hi * 1000, win_lo * 1000, ((win_hi + win_lo) / 2.0 - args.pinch_y) * 1000))
        print('  共面段之间 y 台阶 = 0；台阶只出现在 COP 边界（指板足迹之外）')
        print('\n拍平板 | x 范围(mm)      | 视觉 y(mm)        | 板面 y(mm)        | z半高mm | 段 | 超视觉mm | 挡管余量mm | 角凸mm')
        _minlead, _maxpoke, _minpoke, _maxcorn, _minbot = 1e9, -1e9, 1e9, -1e9, 1e9
        for x, cy, cz, ry, r, sa, sb, is_cop, lo, hi, h, vlo, vhi, tlo, thi in rows:
            lead = min(hi - thi, tlo - lo) * 1000
            poke = max(hi - vhi, vlo - lo) * 1000
            _corn = 0.0
            for _xs in np.linspace(sa, sb, 13):      # 四角线对插值椭圆截面：>0 = 盒角出视觉
                _cyX, _czX, _rX, _ryX = _station_at(sts, float(_xs))
                if _rX <= 1e-6 or _ryX <= 1e-6:
                    continue
                for _yb in (lo, hi):
                    for _zb in (cz + h, cz - h):
                        _s = float(np.sqrt(((_yb - _cyX) / _ryX) ** 2 + ((_zb - _czX) / _rX) ** 2))
                        _d = float(np.hypot(_yb - _cyX, _zb - _czX))
                        _corn = max(_corn, (_s - 1.0) * _d)
            _minlead = min(_minlead, lead)
            _maxpoke = max(_maxpoke, poke)
            _minpoke = min(_minpoke, poke)
            _maxcorn = max(_maxcorn, _corn * 1000)
            _minbot = min(_minbot, cz - h)           # 全碰撞体最低点（决定香蕉坐多高）
            print('   | %+7.2f..%+7.2f | %+7.2f / %+7.2f | %+7.2f / %+7.2f | %7.2f | %s | %+8.2f | %+6.2f | %+7.2f'
                  % (sa * 1000, sb * 1000, vhi * 1000, vlo * 1000, hi * 1000, lo * 1000, h * 1000,
                     '共' if is_cop else '外', poke, lead, _corn * 1000))
            ht = max(args.pad_thick / 2.0, (hi - lo) / 2.0)
            lines.append('      <geom type="box" group="3" size="%.5f %.5f %.5f" pos="%.5f %.5f %.5f"'
                         % ((sb - sa) / 2.0, ht, h, (sa + sb) / 2.0, (hi + lo) / 2.0, cz))
            lines.append('            mass="%.5f" friction="%s" solref="0.004 1"'
                         ' solimp="0.99 0.999 0.001" rgba="1.0 0.85 0.1 1" />'
                         % (mg, args.friction))
        print('   %d 段共面 + %d 段外贴：板面恒在圆管外 ≥%.2fmm（圆管被板挡住）；'
              '板面相对视觉 %+.2f..%+.2fmm（正 = 凸出视觉, 第十二轮目标: 全部 ≤0）'
              % (_ncop, len(rows) - _ncop, _minlead, _minpoke, _maxpoke))
        _vislo = float(V[:, 2].min())
        _capbot = min(min(sts[i][2], sts[i + 1][2]) - link_r[i] for i in range(ncap))
        _low = min(_minbot, _capbot)
        print('   盒角对椭圆截面最凸 %+.2fmm（>0 = 盒角出视觉）；碰撞体最低点 z=%+.2fmm'
              '（板 %+.2f / 圆管 %+.2f）、视觉最低点 %+.2fmm → 静止时视觉沉桌 %.2fmm'
              % (_maxcorn, _low * 1000, _minbot * 1000, _capbot * 1000, _vislo * 1000,
                 max(0.0, (_low - _vislo) * 1000)))
    if skipped:
        print('\n跳过的段（盒角装不进视觉，退回圆管链）: '
              + ', '.join('%+.2f..%+.2f(h=%+.1fmm)' % (a * 1000, b * 1000, hh * 1000)
                          for a, b, hh in skipped))
    block = '\n'.join(lines)
    print('\n合计 %d 根胶囊 + %d 块拍平板 = %d geom, 总质量 %.4fkg (%.5fkg/geom)'
          % (ncap, len(rows), ngeom, mg * ngeom, mg))

    if args.dry_run:
        print('\n[dry-run] 不写 scene.xml')
        return 0
    with open(SCENE) as f:
        txt = f.read()
    if BEGIN not in txt or END not in txt:
        print('\n[ERR] scene.xml 里没有 %s / %s 标记, 先加标记再跑' % (BEGIN, END))
        return 1
    a, b = txt.index(BEGIN), txt.index(END)
    new = txt[:a + len(BEGIN)] + '\n' + block + '\n      ' + txt[b:]
    with open(SCENE, 'w') as f:
        f.write(new)
    print('\n已写入 %s (%d 行)' % (SCENE, block.count('\n') + 1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
