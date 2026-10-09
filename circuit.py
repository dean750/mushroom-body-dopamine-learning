"""蘑菇体回路。读本文件夹里的 FlyWire 连接，分出 KC、DAN、MBON，并抽出气味 A。

这里没有神经元。学习在 learn.py，只改已经存在的 KC→MBON。
连接表只有萼、柄、内侧叶、垂直叶，再分左右。内侧叶是 MB_ML，垂直叶是 MB_VL。
"""
from pathlib import Path

import numpy as np
import pandas as pd

CONN_PATH = Path(__file__).resolve().parent / 'data' / '783_connections_processed.csv'

ODOR_START = 0.0
ODOR_END = 1.0
PAIRED_REWARD_TIME = 0.5

MBON_INPUT = 13.0
REGIONS = ['MB_ML_L', 'MB_ML_R', 'MB_VL_L', 'MB_VL_R']
LEFT_REGIONS = ['MB_ML_L', 'MB_VL_L']
RIGHT_REGIONS = ['MB_ML_R', 'MB_VL_R']
BILATERAL_MIN = 0.2
MEDIAL = 'MB_ML_L'

PN_PER_ODOR = 5
MIN_KC = 20
MIN_SYN = 5


def load_populations(conn):
    """KC、DAN、MBON，以及叶里的 KC→MBON 突触。"""
    mb = conn[conn['neuropil'].str.contains('MB', na=False)]
    outside = conn[~conn['neuropil'].str.contains('MB', na=False)]
    lobe = mb['neuropil'].str.contains('ML|VL')

    kcs = set(mb[mb['neuropil'].str.contains('MB_CA')]['post_index']) & set(
        mb[mb['neuropil'].str.contains('PED|ML|VL') & (mb['nt_type'] == 'ACH')]['pre_index']
    )

    nt_budget = conn.groupby(['pre_index', 'nt_type'])['syn_count'].sum().reset_index()
    dominant = nt_budget.sort_values('syn_count').groupby('pre_index').tail(1)
    da_neurons = set(dominant.loc[dominant['nt_type'] == 'DA', 'pre_index'])
    lobe_da = mb[lobe & (mb['nt_type'] == 'DA') & mb['pre_index'].isin(da_neurons)]
    dan_syn = lobe_da.groupby('pre_index')['syn_count'].sum().sort_values(ascending=False)
    n_dan = int((dan_syn.cumsum() / dan_syn.sum() < 0.9).sum()) + 1
    dans = dan_syn.iloc[:n_dan]

    downstream = set(outside['pre_index']) - kcs - da_neurons
    kc_out = mb[lobe & mb['pre_index'].isin(kcs) & mb['post_index'].isin(downstream)]
    mbon_syn = kc_out.groupby('post_index')['syn_count'].sum().sort_values(ascending=False)
    n_mbon = int((mbon_syn.cumsum() / mbon_syn.sum() < 0.9).sum()) + 1
    mbons = mbon_syn.iloc[:n_mbon]

    edges = (
        mb.loc[
            lobe & mb['pre_index'].isin(kcs) & mb['post_index'].isin(mbons.index),
            ['pre_index', 'post_index', 'neuropil', 'syn_count'],
        ]
        .groupby(['pre_index', 'post_index', 'neuropil'], as_index=False)['syn_count']
        .sum()
        .rename(columns={'neuropil': 'compartment'})
    )
    return kcs, dans, mbons, edges


def build_region_weights(edges):
    """每个脑区一张 KC→MBON。"""
    kc_ids = np.sort(edges['pre_index'].unique())
    mbon_ids = np.sort(edges['post_index'].unique())
    kc_index = {neuron: i for i, neuron in enumerate(kc_ids)}
    mbon_index = {neuron: i for i, neuron in enumerate(mbon_ids)}
    weights = {region: np.zeros((len(kc_ids), len(mbon_ids))) for region in REGIONS}
    for pre, post, region, syn_count in edges[['pre_index', 'post_index', 'compartment', 'syn_count']].itertuples(index=False):
        if region in weights:
            weights[region][kc_index[pre], mbon_index[post]] += syn_count
    return kc_ids, mbon_ids, weights


def kc_is_left(conn, kc_ids):
    """KC 的左右按萼里突触后突触更多的那一侧。"""
    calyx = conn[conn['neuropil'].str.contains('MB_CA') & conn['post_index'].isin(kc_ids)]
    left = calyx[calyx['neuropil'].str.endswith('_L')].groupby('post_index')['syn_count'].sum()
    right = calyx[calyx['neuropil'].str.endswith('_R')].groupby('post_index')['syn_count'].sum()
    side = pd.DataFrame({'L': left, 'R': right}).reindex(kc_ids).fillna(0)
    return side['L'].to_numpy() >= side['R'].to_numpy()


def mbon_sides(weights):
    """较弱一侧低于 20% 就归到突触多的那一侧。"""
    left = sum(weights[region].sum(axis=0) for region in LEFT_REGIONS)
    right = sum(weights[region].sum(axis=0) for region in RIGHT_REGIONS)
    total = left + right
    minor = np.minimum(left, right) / np.maximum(total, 1e-9)
    bilateral = minor >= BILATERAL_MIN
    left_only = (left > right) & ~bilateral
    right_only = ~left_only & ~bilateral
    return left_only, right_only, bilateral


def drop_stray_synapses(weights, left_only, right_only):
    """单侧 MBON 只保留同侧脑区的 KC 输入。"""
    for region in LEFT_REGIONS:
        weights[region][:, right_only] = 0.0
    for region in RIGHT_REGIONS:
        weights[region][:, left_only] = 0.0


def projection_neurons(conn, kc_ids, is_left_kc, kcs):
    """左萼里、乙酰胆碱、而且自身不是 KC 的突触前神经元。"""
    left_ids = set(kc_ids[is_left_kc])
    calyx = conn[
        conn['neuropil'].eq('MB_CA_L')
        & conn['post_index'].isin(left_ids)
        & (conn['nt_type'] == 'ACH')
        & ~conn['pre_index'].isin(kcs)
    ]
    pairs = calyx.groupby(['pre_index', 'post_index'])['syn_count'].sum()
    pairs = pairs[pairs >= MIN_SYN].reset_index()
    fan = pairs.groupby('pre_index')['post_index'].nunique()
    keep = fan[fan >= MIN_KC].index
    pairs = pairs[pairs['pre_index'].isin(keep)]
    kc_index = {neuron: i for i, neuron in enumerate(kc_ids)}
    return {
        int(neuron): [kc_index[post] for post in posts if post in kc_index]
        for neuron, posts in pairs.groupby('pre_index')['post_index']
    }


def odor_from_pns(targets, pn_ids, n_kc):
    odor = np.zeros(n_kc)
    for neuron in pn_ids:
        odor[targets[int(neuron)]] = 1.0
    return odor


def three_odors(targets, n_kc, seed):
    """一组种子抽 15 个投射神经元，分成三个气味，每个 5 个。"""
    pn_ids = np.array(list(targets))
    rng = np.random.default_rng(seed)
    picked = rng.choice(pn_ids, size=PN_PER_ODOR * 3, replace=False)
    return [
        odor_from_pns(targets, picked[:PN_PER_ODOR], n_kc),
        odor_from_pns(targets, picked[PN_PER_ODOR:PN_PER_ODOR * 2], n_kc),
        odor_from_pns(targets, picked[PN_PER_ODOR * 2:], n_kc),
    ]


def odor_matrix(weights, odor_rows):
    return sum(weights[region][odor_rows] for region in REGIONS)


def odor_input_scale(weights, rows):
    mass = odor_matrix(weights, rows).sum(axis=0)
    return np.divide(MBON_INPUT, mass, out=np.zeros_like(mass), where=mass > 0)


