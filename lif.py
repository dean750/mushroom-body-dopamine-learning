"""闭环里的 BrainX 细胞。

KC 和 MBON 都是 brainpy.state.LIFRef。气味电流和 KC→MBON 电流先经过指数突触。
膜时间常数 20 ms，突触时间常数 5 ms，不应期 2.2 ms，阈值 1 mV，复位 0 mV。
学习不在这里。
"""
import os

os.environ['JAX_ENABLE_X64'] = '1'

import brainpy
import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

import circuit

DT_MS = 0.2
TAU_M_MS = 20.0
TAU_SYN_MS = 5.0
TAU_REF_MS = 2.2
V_TH = 1.0
KC_CURRENT = 0.045

DT = DT_MS * u.ms
brainstate.environ.set(dt=DT, precision=64)


def _neuron(n):
    return brainpy.state.LIFRef(
        n,
        R=1. * u.ohm,
        tau=TAU_M_MS * u.ms,
        tau_ref=TAU_REF_MS * u.ms,
        V_th=V_TH * u.mV,
        V_reset=0. * u.mV,
        V_rest=0. * u.mV,
        spk_reset='hard',
    )


def _synapse(n):
    return brainpy.state.Expon.desc(
        n,
        tau=TAU_SYN_MS * u.ms,
        g_initializer=braintools.init.Constant(0. * u.mA),
    )


def _run(current_fn, n_steps, n_cell):
    post = _neuron(n_cell)
    proj = brainpy.state.AlignPostProj(
        comm=current_fn,
        syn=_synapse(n_cell),
        out=brainpy.state.CUBA.desc(scale=1.),
        post=post,
    )
    brainstate.nn.init_all_states(proj)
    dt = brainstate.environ.get_dt()

    def step(i):
        with brainstate.environ.context(t=i * dt):
            proj.update(i)
            return post()

    spikes = brainstate.transform.for_loop(step, jnp.arange(n_steps))
    return np.asarray(spikes) > 0.5


def simulate_current(current):
    """current 形状 (步数, 细胞数)。返回脉冲。"""
    current = jnp.asarray(current)

    def at(_i):
        return current[_i] * u.mA

    return _run(at, current.shape[0], current.shape[1])


def kc_recording(n_odor):
    """气味期间的 KC 脉冲。电流先经过 Expon，再进入 LIFRef。"""
    dt_s = DT_MS / 1000.0
    n_steps = int(round((circuit.ODOR_END - circuit.ODOR_START) / dt_s))
    current = KC_CURRENT * np.linspace(0.85, 1.35, n_odor)
    drive = np.broadcast_to(current, (n_steps, n_odor)).copy()
    spikes = simulate_current(drive)
    rate = spikes.sum() / n_odor / (circuit.ODOR_END - circuit.ODOR_START)
    return spikes, rate, dt_s
