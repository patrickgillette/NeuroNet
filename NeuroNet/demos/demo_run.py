from __future__ import annotations
import sys, time, logging

from ..sim.sim_core import LIFConfig
from ..sim.processing_core import ProcessingCore
from ..core.interfaces import IOCoordinator, ScreenAction, ScreenActionType, SimTime
from ..core.runtime import Runtime
from ..adapters.screen.env import SimpleScreen
from ..adapters.screen.encoders import PositionEncoder
from ..adapters.screen.decoders import FirstToSpikeMoveDecoder
from ..adapters.screen.goals import CenterSeekingGoal

logging.basicConfig(filename="neuron_log.txt", level=logging.INFO, format="%(message)s")

CSI_HOME_CLEAR = "\x1b[H\x1b[J"
RENDER_CLEARS = True

W, H = 16, 9
N_IN = W * H
N_PROC = 64
N_OUT = 4

lif = LIFConfig(v_rest=0.0, v_reset=0.0, v_thresh=1.0, tau_m_ms=10.0, r_m=1.0, tau_ref_ms=2.0)
core = ProcessingCore(lif_input=lif, lif_hidden=lif, lif_output=lif)
pop_in = core.add_population("in", N_IN)
pop_proc = core.add_population("proc", N_PROC)
pop_out = core.add_population("out", N_OUT)
core.build()

# Wiring
core.dense(pop_in, pop_proc, w=(0.8, 1.2), delay_ms=0.0, plastic=True)
core.dense(pop_proc, pop_out, w=(0.6, 1.0), delay_ms=0.0, plastic=True)
core.lateral_inhibition(pop_proc, w_inh=-0.3)
core.lateral_inhibition(pop_out, w_inh=-0.6)

# Env + I/O
env = SimpleScreen(width=W, height=H)
env.apply_action(0.0, ScreenAction(kind=ScreenActionType.DRAW_DOT, x=W // 2, y=H // 2))
encoder = PositionEncoder(width=W, height=H, base_id=pop_in.start, min_interval_ms=5.0)

out_ids = list(pop_out.ids)
decoder = FirstToSpikeMoveDecoder(
    up_ids=[out_ids[0]],
    down_ids=[out_ids[1]],
    left_ids=[out_ids[2]],
    right_ids=[out_ids[3]],
    readout_period_ms=100.0,
    min_action_delay_ms=1.0,
    step=1,
)
io = IOCoordinator(env, encoder, decoder)
decoder.reset()

goal = CenterSeekingGoal()
rt = Runtime(core=core, io=io)

def render_ascii(t: SimTime):
    frame = env.observe(t)
    lines = []
    for y in range(frame.height):
        row = "".join("#" if frame.get(x, y) else "." for x in range(frame.width))
        lines.append(row)
    if RENDER_CLEARS:
        sys.stdout.write(CSI_HOME_CLEAR)
    sys.stdout.write("\n".join(lines))
    sys.stdout.write(f"\nSim time: {t:.1f} ms\n")
    sys.stdout.flush()

dt = 10.0
t = 0.0

try:
    while True:
        reward, spikes = rt.tick(t, dt_ms=dt, goal=goal, inject_scale=1.3)
        if reward is not None:
            rt.net.apply_reward(reward)  # Runtime already applies; keep if you want explicit logging only.
        if int(t) % 200 == 0:
            render_ascii(t)
        t += dt
        time.sleep(dt / 1000.0)
except KeyboardInterrupt:
    print("\nStopped.")
