"""多巴胺决定学习方向，肯扬细胞的活动决定改哪些连接。

惩罚把刚才发放过的 KC 在左内侧叶里的已有突触削弱。
奖赏把同样这些突触增强。没有教学信号就不改。
输出细胞的先后不再决定方向。多巴胺不驱动细胞。

全脑循环的每一步调用 update(kc 脉冲, dan 脉冲)。
方向 valence 在建立这个脑区的组件时给定：惩罚 -1，奖赏 +1。

  python learn.py
"""
import os

os.environ['JAX_ENABLE_X64'] = '1'

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

# 下限避开把突触写成 0，上限避开乘到几百倍。
FACTOR_MIN = 0.05
FACTOR_MAX = 2.0
# 正在发放的 KC 资格接近 1。一次惩罚大约乘 0.8，一次奖赏大约乘 1.2。
GAIN = 0.2
TAU_E_MS = 200.0
TAU_N_MS = 200.0
# 一个教学脉冲的浓度峰值是 1/200。掉到峰值的五分之一以下，这一次就结束。
TEACH_N = 0.001


class KcMbonDopamine(brainstate.nn.Module):
    """一个脑区里已经存在的 KC→MBON。

    pre_index、post_index 是这一群 KC、MBON 里的局部编号。
    syn_count 是这些边上的解剖突触数。n_dan 是接到这个脑区的 DAN 人数。
    valence 是教学信号的方向，惩罚为 -1，奖赏为 +1。它不是脑区的名字。
    """

    def __init__(self, n_pre, n_post, pre_index, post_index, syn_count, n_dan, valence, gain=GAIN):
        super().__init__()
        self.n_pre = int(n_pre)
        self.n_post = int(n_post)
        self.n_dan = int(n_dan)
        self.valence = float(valence)
        self.gain = float(gain)
        if self.n_dan < 1:
            raise ValueError('n_dan 至少为 1')
        if self.valence not in (-1.0, 1.0):
            raise ValueError('valence 只能是惩罚 -1 或奖赏 +1')
        pre_index = np.asarray(pre_index, dtype=np.int32)
        post_index = np.asarray(post_index, dtype=np.int32)
        syn_count = np.asarray(syn_count, dtype=np.float64)
        if pre_index.shape != post_index.shape or pre_index.shape != syn_count.shape:
            raise ValueError('边的编号和突触数长度不一致')
        if np.any(syn_count <= 0):
            raise ValueError('只接受原来就存在的突触')
        self.pre_index = jnp.asarray(pre_index)
        self.post_index = jnp.asarray(post_index)
        self._syn0 = jnp.asarray(syn_count)

    def init_state(self, batch_size=None, **kwargs):
        self.w = brainstate.HiddenState(self._syn0)
        self.k = brainstate.HiddenState(jnp.zeros(self.n_pre))
        self.n = brainstate.HiddenState(jnp.zeros(()))
        self.latched = brainstate.HiddenState(jnp.zeros(()))

    def update(self, kc_spike, dan_spike):
        """kc_spike (KC,)、dan_spike (DAN,)，都是本步 0 或 1。"""
        dt_ms = u.Quantity(brainstate.environ.get_dt()).to_decimal(u.ms)
        k = self.k.value * jnp.exp(-dt_ms / TAU_E_MS)
        k = k + (1.0 - k) * kc_spike
        self.k.value = k
        count = jnp.sum(dan_spike) / self.n_dan
        n = self.n.value * jnp.exp(-dt_ms / TAU_N_MS) + count / TAU_N_MS
        above = n > TEACH_N
        onset = above & (self.latched.value < 0.5)
        self.n.value = n
        self.latched.value = above.astype(jnp.float64)
        participated = k[self.pre_index]
        step_factor = jnp.clip(1.0 + self.valence * self.gain * participated, FACTOR_MIN, FACTOR_MAX)
        ratio = jnp.clip(self.w.value / self._syn0 * step_factor, FACTOR_MIN, FACTOR_MAX)
        self.w.value = jnp.where(onset, self._syn0 * ratio, self.w.value)


class _MbonDrive(brainstate.nn.Module):
    """MBON 的突触电流，以及左内侧叶里正在变化的突触数。"""

    def __init__(self, n_mbon, medial):
        super().__init__()
        self.n_mbon = int(n_mbon)
        self._medial = jnp.asarray(medial)

    def init_state(self, batch_size=None, **kwargs):
        self.g = brainstate.HiddenState(jnp.zeros(self.n_mbon))
        self.medial = brainstate.HiddenState(self._medial)


def load():
    import circuit
    conn = pd.read_csv(circuit.CONN_PATH)
    kcs, dans, _mbons, edges = circuit.load_populations(conn)
    kc_ids, _mbon_ids, weights = circuit.build_region_weights(edges)
    left_only, right_only, _bilateral = circuit.mbon_sides(weights)
    circuit.drop_stray_synapses(weights, left_only, right_only)
    is_left = circuit.kc_is_left(conn, kc_ids)
    targets = circuit.projection_neurons(conn, kc_ids, is_left, kcs)
    odors = circuit.three_odors(targets, len(kc_ids), 0)
    rows_a = np.flatnonzero(odors[0] > 0)
    rows_b = np.flatnonzero(odors[1] > 0)
    lobe = conn[conn['pre_index'].isin(dans.index) & conn['nt_type'].eq('DA')]
    dan_ids = np.sort(lobe.loc[lobe['neuropil'].eq(circuit.MEDIAL), 'pre_index'].unique())
    return weights, rows_a, rows_b, dan_ids


def _edges(matrix, rows):
    block = matrix[rows]
    local_pre, post = np.nonzero(block > 0)
    order = np.argsort(local_pre, kind='stable')
    return local_pre[order], post[order]


def _recording(n_kc):
    import lif
    brainstate.environ.set(dt=lif.DT, precision=64)
    spikes, rate, dt_s = lif.kc_recording(n_kc)
    return spikes, float(rate), float(dt_s)


def associate(weights, rows_a, rows_b, n_dan, valence, n_events=4, spacing_s=5.0, teach=True, dan_after_odor_s=None):
    """气味 A 配上教学信号。气味 B 若给出，则同样呈现，但不给教学信号。

    dan_after_odor_s 为 None 时，教学在气味的 0.5 秒。
    给出秒数时，教学改到气味结束之后，用来做未配对。
    """
    import lif
    import circuit
    spikes_a, rate_a, dt_s = _recording(len(rows_a))
    odor_steps = spikes_a.shape[0]
    gap_steps = int(round(spacing_s / dt_s))
    pieces = [('A', rows_a, spikes_a, teach)]
    rate_b = None
    if rows_b is not None:
        spikes_b, rate_b, dt_b = _recording(len(rows_b))
        if spikes_b.shape[0] != odor_steps or abs(dt_b - dt_s) > 1e-12:
            raise RuntimeError('两个气味的时间步不一致')
        pieces.append(('B', rows_b, spikes_b, False))
    rows = np.union1d(rows_a, rows_b if rows_b is not None else rows_a)
    place = {int(neuron): i for i, neuron in enumerate(rows)}
    local_a = np.array([place[int(neuron)] for neuron in rows_a], dtype=int)
    local_b = np.array([place[int(neuron)] for neuron in rows_b], dtype=int) if rows_b is not None else np.array([], dtype=int)
    in_a = np.zeros(len(rows), dtype=bool)
    in_b = np.zeros(len(rows), dtype=bool)
    in_a[local_a] = True
    in_b[local_b] = True
    slot = odor_steps + gap_steps
    period = slot * len(pieces)
    n_steps = n_events * period
    n_pre = len(rows)
    n_post = next(iter(weights.values())).shape[1]
    kc = np.zeros((n_steps, n_pre))
    dopa = np.zeros((n_steps, n_dan))
    reward_step = int(round(circuit.PAIRED_REWARD_TIME / dt_s))
    marks = []
    for event in range(n_events):
        for slot_i, (name, _global_rows, recording, taught) in enumerate(pieces):
            origin = event * period + slot_i * slot
            target = local_a if name == 'A' else local_b
            kc[origin:origin + odor_steps, target] = recording
            if taught:
                if dan_after_odor_s is None:
                    dopa[origin + reward_step, :] = 1.0
                else:
                    late = odor_steps + int(round(dan_after_odor_s / dt_s))
                    if late >= slot:
                        raise RuntimeError('未配对的教学落进了下一次气味')
                    dopa[origin + late, :] = 1.0
            marks.append((name, origin, origin + odor_steps - 1, origin + slot - 1))
    other = sum(weights[region][rows] for region in circuit.REGIONS if region != circuit.MEDIAL)
    medial0 = np.asarray(weights[circuit.MEDIAL][rows], dtype=np.float64)
    scale_a = np.asarray(circuit.odor_input_scale(weights, rows_a), dtype=np.float64)
    scale_b = scale_a if rows_b is None else np.asarray(circuit.odor_input_scale(weights, rows_b), dtype=np.float64)
    which = np.zeros(n_steps, dtype=np.float64)
    for event in range(n_events):
        for slot_i, (name, _global_rows, _spikes, _taught) in enumerate(pieces):
            if name == 'B':
                origin = event * period + slot_i * slot
                which[origin:origin + odor_steps] = 1.0
    scale = scale_a * (1.0 - which)[:, None] + scale_b * which[:, None]
    other = jnp.asarray(np.asarray(other, dtype=np.float64))
    scale = jnp.asarray(scale)
    local_pre, post = _edges(weights[circuit.MEDIAL], rows)
    module = KcMbonDopamine(
        n_pre, n_post, local_pre, post,
        weights[circuit.MEDIAL][rows[local_pre], post], n_dan, valence,
    )
    groups = {
        'A 独有': jnp.asarray(np.flatnonzero(in_a[local_pre] & ~in_b[local_pre])),
        '两边共有': jnp.asarray(np.flatnonzero(in_a[local_pre] & in_b[local_pre])),
        'B 独有': jnp.asarray(np.flatnonzero(~in_a[local_pre] & in_b[local_pre])),
    }
    drive = _MbonDrive(n_post, medial0)
    mbon = lif._neuron(n_post)
    brainstate.environ.set(dt=lif.DT, precision=64)
    brainstate.nn.init_all_states((module, drive, mbon))
    listen = {
        'A': np.asarray(circuit.odor_matrix(weights, rows_a).sum(axis=0) > 0),
        'B': np.asarray(circuit.odor_matrix(weights, rows_b).sum(axis=0) > 0) if rows_b is not None else None,
    }
    tau_syn = lif.TAU_SYN_MS
    kc, dopa = jnp.asarray(kc), jnp.asarray(dopa)
    times = jnp.arange(n_steps) * lif.DT
    def _mean(factor, index):
        if index.size == 0:
            return jnp.asarray(1.0)
        return factor[index].mean()

    def step(kc_t, dopa_t, scale_t, t):
        with brainstate.environ.context(t=t):
            current = (kc_t @ (other + drive.medial.value)) * scale_t
            drive.g.value = drive.g.value * jnp.exp(-lif.DT_MS / tau_syn) + current
            raw = mbon.update(drive.g.value * u.mA)
            spk = (jnp.asarray(u.get_mantissa(raw)) > 0.5).astype(jnp.float64)
            module.update(kc_t, dopa_t)
            drive.medial.value = drive.medial.value.at[module.pre_index, module.post_index].set(module.w.value)
            factor = module.w.value / module._syn0
            return (
                spk,
                _mean(factor, groups['A 独有']),
                _mean(factor, groups['两边共有']),
                _mean(factor, groups['B 独有']),
            )

    spikes, mean_a, mean_shared, mean_b = brainstate.transform.for_loop(step, kc, dopa, scale, times)
    spikes = np.asarray(spikes)
    mean_a = np.asarray(mean_a)
    mean_shared = np.asarray(mean_shared)
    mean_b = np.asarray(mean_b)
    rows_out = []
    for name, origin, end, gap_end in marks:
        fired = spikes[origin:origin + odor_steps][:, listen[name]]
        counts = fired.sum(axis=0)
        rows_out.append({
            'name': name,
            'a_only': float(mean_a[end]),
            'shared': float(mean_shared[end]),
            'b_only': float(mean_b[end]),
            'after': float(mean_a[gap_end]),
            'median': float(np.median(counts)) if counts.size else 0.0,
            'total': int(counts.sum()),
            'n_mbon': int(listen[name].sum()),
        })
    counts = {name: int(index.size) for name, index in groups.items()}
    return rate_a, rate_b, counts, rows_out


_RECORDINGS = {}


def _recording_cached(n_kc):
    key = int(n_kc)
    if key not in _RECORDINGS:
        _RECORDINGS[key] = _recording(key)
    return _RECORDINGS[key]


def _input_mass(weights, rows):
    import circuit
    if len(rows) == 0:
        return 0.0
    return float(sum(weights[region][rows].sum() for region in circuit.REGIONS))


def overlap_once(weights, odor_rows, taught, n_dan, n_learn=4, spacing_s=5.0):
    """三个气味都呈现。taught 这一个在 0.5 秒接受惩罚，另外两个不给。

    第 0 轮不给多巴胺，用来记学习前的输出。后面 n_learn 轮才惩罚。
    """
    import lif
    import circuit
    order = [taught] + [i for i in range(len(odor_rows)) if i != taught]
    recordings = []
    odor_steps = None
    dt_s = None
    for odor_i in order:
        spikes, _rate, dt = _recording_cached(len(odor_rows[odor_i]))
        if odor_steps is None:
            odor_steps = spikes.shape[0]
            dt_s = dt
        elif spikes.shape[0] != odor_steps or abs(dt - dt_s) > 1e-12:
            raise RuntimeError('气味的时间步不一致')
        recordings.append(spikes)
    gap_steps = int(round(spacing_s / dt_s))
    slot = odor_steps + gap_steps
    n_cycles = n_learn + 1
    period = slot * len(order)
    n_steps = n_cycles * period
    rows = np.unique(np.concatenate(odor_rows))
    place = {int(neuron): i for i, neuron in enumerate(rows)}
    local = []
    member = np.zeros((len(odor_rows), len(rows)), dtype=bool)
    for odor_i, global_rows in enumerate(odor_rows):
        loc = np.array([place[int(neuron)] for neuron in global_rows], dtype=int)
        local.append(loc)
        member[odor_i, loc] = True
    n_pre = len(rows)
    n_post = next(iter(weights.values())).shape[1]
    kc = np.zeros((n_steps, n_pre))
    dopa = np.zeros((n_steps, n_dan))
    scale = np.zeros((n_steps, n_post))
    reward_step = int(round(circuit.PAIRED_REWARD_TIME / dt_s))
    windows = []
    for cycle in range(n_cycles):
        for pos, odor_i in enumerate(order):
            origin = cycle * period + pos * slot
            kc[origin:origin + odor_steps, local[odor_i]] = recordings[pos]
            scale[origin:origin + odor_steps] = circuit.odor_input_scale(weights, odor_rows[odor_i])
            if cycle > 0 and odor_i == taught:
                dopa[origin + reward_step, :] = 1.0
            windows.append((odor_i, cycle, origin))
    other = sum(weights[region][rows] for region in circuit.REGIONS if region != circuit.MEDIAL)
    medial0 = np.asarray(weights[circuit.MEDIAL][rows], dtype=np.float64)
    other = jnp.asarray(np.asarray(other, dtype=np.float64))
    scale = jnp.asarray(scale)
    local_pre, post = _edges(weights[circuit.MEDIAL], rows)
    module = KcMbonDopamine(
        n_pre, n_post, local_pre, post,
        weights[circuit.MEDIAL][rows[local_pre], post], n_dan, -1.0,
    )
    drive = _MbonDrive(n_post, medial0)
    mbon = lif._neuron(n_post)
    brainstate.environ.set(dt=lif.DT, precision=64)
    brainstate.nn.init_all_states((module, drive, mbon))
    listen = [
        np.asarray(circuit.odor_matrix(weights, odor_rows[i]).sum(axis=0) > 0)
        for i in range(len(odor_rows))
    ]
    tau_syn = lif.TAU_SYN_MS
    kc, dopa = jnp.asarray(kc), jnp.asarray(dopa)
    times = jnp.arange(n_steps) * lif.DT

    def step(kc_t, dopa_t, scale_t, t):
        with brainstate.environ.context(t=t):
            current = (kc_t @ (other + drive.medial.value)) * scale_t
            drive.g.value = drive.g.value * jnp.exp(-lif.DT_MS / tau_syn) + current
            raw = mbon.update(drive.g.value * u.mA)
            spk = (jnp.asarray(u.get_mantissa(raw)) > 0.5).astype(jnp.float64)
            module.update(kc_t, dopa_t)
            drive.medial.value = drive.medial.value.at[module.pre_index, module.post_index].set(module.w.value)
            return spk

    spikes = np.asarray(brainstate.transform.for_loop(step, kc, dopa, scale, times))
    factor = np.asarray(module.w.value / module._syn0)
    taught_member = member[taught]
    medians = np.zeros((len(odor_rows), n_cycles))
    for odor_i, cycle, origin in windows:
        fired = spikes[origin:origin + odor_steps][:, listen[odor_i]]
        counts = fired.sum(axis=0)
        medians[odor_i, cycle] = float(np.median(counts)) if counts.size else 0.0
    held = []
    for odor_i in range(len(odor_rows)):
        if odor_i == taught:
            continue
        shared_kc = np.intersect1d(odor_rows[taught], odor_rows[odor_i])
        exclusive_kc = np.setdiff1d(odor_rows[odor_i], odor_rows[taught])
        total_mass = _input_mass(weights, odor_rows[odor_i])
        shared_mass = float(weights[circuit.MEDIAL][shared_kc].sum()) if len(shared_kc) else 0.0
        exclusive_mass = float(weights[circuit.MEDIAL][exclusive_kc].sum()) if len(exclusive_kc) else 0.0
        on_edge = member[odor_i][local_pre]
        shared_edge = on_edge & taught_member[local_pre]
        exclusive_edge = on_edge & ~taught_member[local_pre]
        held.append({
            'odor': odor_i,
            'n_kc': int(len(odor_rows[odor_i])),
            'n_shared_kc': int(len(shared_kc)),
            'shared_edges': int(shared_edge.sum()),
            'exclusive_edges': int(exclusive_edge.sum()),
            'plastic_fraction': shared_mass / total_mass if total_mass else 0.0,
            'shared_factor': float(factor[shared_edge].mean()) if shared_edge.any() else 1.0,
            'exclusive_factor': float(factor[exclusive_edge].mean()) if exclusive_edge.any() else 1.0,
            'median_before': float(medians[odor_i, 0]),
            'median_after': float(medians[odor_i, n_learn]),
            'exclusive_mass': exclusive_mass,
        })
    taught_edge = taught_member[local_pre]
    return {
        'n_kc': int(len(odor_rows[taught])),
        'factor': float(factor[taught_edge].mean()) if taught_edge.any() else 1.0,
        'median_before': float(medians[taught, 0]),
        'median_after': float(medians[taught, n_learn]),
        'held': held,
    }


def _corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom == 0:
        return float('nan')
    return float((x * y).sum() / denom)


def overlap_main(n_seed=10):
    """十个种子，每个气味轮流接受惩罚。看没被惩罚的气味被拖下去多少。"""
    import circuit
    conn = pd.read_csv(circuit.CONN_PATH)
    kcs, dans, _mbons, edges = circuit.load_populations(conn)
    _kc_ids, _mbon_ids, weights = circuit.build_region_weights(edges)
    left_only, right_only, _bilateral = circuit.mbon_sides(weights)
    circuit.drop_stray_synapses(weights, left_only, right_only)
    is_left = circuit.kc_is_left(conn, _kc_ids)
    targets = circuit.projection_neurons(conn, _kc_ids, is_left, kcs)
    lobe = conn[conn['pre_index'].isin(dans.index) & conn['nt_type'].eq('DA')]
    n_dan = len(np.sort(lobe.loc[lobe['neuropil'].eq(circuit.MEDIAL), 'pre_index'].unique()))
    print('种子 被惩罚气味 被惩罚KC 惩罚前中位数 惩罚后中位数 倍数 | 旁边气味 共用KC 可改输入比例 独有倍数 共用倍数 惩罚前 惩罚后')
    rows = []
    for seed in range(n_seed):
        odors = circuit.three_odors(targets, len(_kc_ids), seed)
        odor_rows = [np.flatnonzero(odor > 0) for odor in odors]
        for taught in range(3):
            print(f'正在跑种子 {seed}，惩罚气味 {taught} …', flush=True)
            out = overlap_once(weights, odor_rows, taught, n_dan)
            print(
                f"{seed} 气味{taught} KC{out['n_kc']} "
                f"中位数 {out['median_before']:.0f}→{out['median_after']:.0f} ×{out['factor']:.4f}"
            )
            for held in out['held']:
                print(
                    f"    气味{held['odor']} 共用KC {held['n_shared_kc']} "
                    f"可改输入 {held['plastic_fraction']:.3f} "
                    f"独有 ×{held['exclusive_factor']:.4f} 共用 ×{held['shared_factor']:.4f} "
                    f"中位数 {held['median_before']:.0f}→{held['median_after']:.0f}"
                )
                before = held['median_before']
                drop = (before - held['median_after']) / before if before else float('nan')
                rows.append({
                    'seed': seed,
                    'taught': taught,
                    'taught_kc': out['n_kc'],
                    'taught_before': out['median_before'],
                    'taught_after': out['median_after'],
                    'held_larger': held['n_kc'] > out['n_kc'],
                    **held,
                    'drop': drop,
                })
    usable = [row for row in rows if row['median_before'] > 0 and row['shared_edges'] > 0]
    print(
        f"没被惩罚的气味共 {len(rows)} 个，其中输出原本就有、且有共用边的 {len(usable)} 个。"
        f"可改输入比例和输出下降比例的相关是 {_corr([row['plastic_fraction'] for row in usable], [row['drop'] for row in usable]):.3f}。"
    )
    larger = [row for row in usable if row['held_larger']]
    if larger:
        print(
            f"旁边气味比被惩罚的更大的有 {len(larger)} 个，相关是 "
            f"{_corr([row['plastic_fraction'] for row in larger], [row['drop'] for row in larger]):.3f}。"
        )
    exclusive = np.array([row['exclusive_factor'] for row in rows])
    print(f"独有突触的倍数：最小 {exclusive.min():.4f}，最大 {exclusive.max():.4f}。")
    return rows


def _line(row):
    return (
        f"  气味 {row['name']}：A 独有 ×{row['a_only']:.4f}，两边共有 ×{row['shared']:.4f}，"
        f"B 独有 ×{row['b_only']:.4f}，MBON 中位数 {row['median']:.0f}，"
        f"共 {row['total']} 个（{row['n_mbon']} 个细胞）"
    )


def main():
    weights, rows_a, rows_b, dan_ids = load()
    overlap = len(np.intersect1d(rows_a, rows_b))
    print(f'左内侧叶 DAN {len(dan_ids)} 个。气味 A {len(rows_a)} 个 KC，气味 B {len(rows_b)} 个 KC，共有 {overlap} 个。')
    print('教学信号决定方向。刚才发放过的 KC，其左内侧叶已有突触才改。MBON 的先后不参与。')
    print(f'惩罚。学习率 {GAIN:g}。气味 A 在 0.5 秒给惩罚，气味 B 不给。新突触数下一步才用。')
    rate_a, rate_b, counts, rows = associate(weights, rows_a, rows_b, len(dan_ids), valence=-1.0)
    print(f'  气味期间 KC：A {rate_a:.1f} Hz，B {rate_b:.1f} Hz。')
    print(f"  边数：A 独有 {counts['A 独有']}，两边共有 {counts['两边共有']}，B 独有 {counts['B 独有']}。")
    for event in range(0, len(rows), 2):
        print(f'第 {event // 2 + 1} 次')
        print(_line(rows[event]))
        print(_line(rows[event + 1]))
    print('未配对。还是气味 A、还是四次惩罚，多巴胺改到气味结束后 3 秒。')
    _, _, counts_u, rows_u = associate(
        weights, rows_a, None, len(dan_ids), valence=-1.0, dan_after_odor_s=3.0,
    )
    print(f'  A 的边 {counts_u["A 独有"]} 条。气味结束时的倍数还没吃到这次教学，教学之后那一列才是写完的结果。')
    for round_i, row in enumerate(rows_u, start=1):
        print(
            f"  第 {round_i} 次：气味结束 ×{row['a_only']:.4f}，教学之后 ×{row['after']:.4f}，"
            f"MBON 中位数 {row['median']:.0f}，共 {row['total']} 个（{row['n_mbon']} 个细胞）"
        )
    print('配对对照。同一个气味 A、同样四次惩罚，多巴胺仍在气味的 0.5 秒。')
    _, _, counts_p, rows_p = associate(
        weights, rows_a, None, len(dan_ids), valence=-1.0,
    )
    print(f'  A 的边 {counts_p["A 独有"]} 条。')
    for round_i, row in enumerate(rows_p, start=1):
        print(
            f"  第 {round_i} 次：气味结束 ×{row['a_only']:.4f}，教学之后 ×{row['after']:.4f}，"
            f"MBON 中位数 {row['median']:.0f}，共 {row['total']} 个（{row['n_mbon']} 个细胞）"
        )
    print('奖赏。同一条规则，方向改成 +1。只呈现气味 A。')
    rate_r, _, counts_r, rows_r = associate(
        weights, rows_a, None, len(dan_ids), valence=1.0,
    )
    print(f'  气味期间 KC {rate_r:.1f} Hz。A 的边 {counts_r["A 独有"]} 条。')
    for round_i, row in enumerate(rows_r, start=1):
        print(
            f"  第 {round_i} 次：×{row['a_only']:.4f}，"
            f"MBON 中位数 {row['median']:.0f}，共 {row['total']} 个（{row['n_mbon']} 个细胞）"
        )


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'overlap':
        overlap_main()
    else:
        main()
