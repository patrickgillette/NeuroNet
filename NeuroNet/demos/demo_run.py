import time
from typing import Optional, Tuple
import sys
import logging
import random

from sim_core import LIFConfig
from processing_core import ProcessingCore
from concrete_io import (
    SimpleScreen,
    PositionEncoder,           # use full input population
    FirstToSpikeMoveDecoder    # decoder watches the 4 output neurons
)
from screen_interface import IOCoordinator, SimTime, ScreenAction, ScreenActionType
from colorama import init as colorama_init

logging.basicConfig(filename="neuron_log.txt", level=logging.INFO, format="%(message)s")
colorama_init()  # enable ANSI on Windows


CSI_HOME_CLEAR = "\x1b[H\x1b[J"  # go home + clear screen
RENDER_CLEARS = True

# --- Setup network ---
W, H = 16, 9
N_IN = W * H          # one neuron per screen cell
N_PROC = 64 
N_OUT = 4             # up, down, left, right

lif = LIFConfig(v_rest=0.0, v_reset=0.0, v_thresh=1.0, tau_m_ms=10.0, r_m=1.0, tau_ref_ms=2.0)
core = ProcessingCore(lif_input=lif, lif_hidden=lif, lif_output=lif)
pop_in = core.add_population("in", N_IN)
pop_proc = core.add_population("proc", N_PROC)
pop_out = core.add_population("out", N_OUT)
core.build()
net = core.net  

probe_pre = pop_in.start + 0               # top-left pixel neuron
probe_post = list(pop_out.ids)[0]          # "up"
def probe_weight():
    for s in net._syn_list:
        if s.plastic and s.pre == probe_pre and s.post == probe_post:
            return s.w
    return None

# Input -> Processing (plastic)
core.dense(pop_in, pop_proc, w=(0.8, 1.2), delay_ms=0.0, plastic=True)

# Processing -> Output (plastic or fixed; start plastic to learn a mapping)
core.dense(pop_proc, pop_out, w=(0.6, 1.0), delay_ms=0.0, plastic=True)

# Competition in processing & outputs
core.lateral_inhibition(pop_proc, w_inh=-0.3, delay_ms=0.0)
core.lateral_inhibition(pop_out, w_inh=-0.6, delay_ms=0.0)


# --- Setup I/O ---
env = SimpleScreen(width=W, height=H)
env.apply_action(0.0, ScreenAction(kind=ScreenActionType.DRAW_DOT, x=W // 2, y=H // 2))

# Encode the dot's position as a single active input neuron (y*W + x)
encoder = PositionEncoder(width=W, height=H, base_id=pop_in.start, min_interval_ms=5.0)

# Decoder looks at the actual output neuron IDs (N_IN..N_IN+3)
out_ids = list(pop_out.ids)
decoder = FirstToSpikeMoveDecoder(
    up_ids=[out_ids[0]],
    down_ids=[out_ids[1]],
    left_ids=[out_ids[2]],
    right_ids=[out_ids[3]],
     readout_period_ms=100.0,
     min_action_delay_ms=1.0,
     step=1
 )
io = IOCoordinator(env, encoder, decoder)
decoder.reset()


# --- Utilities ---
def render_ascii(env, t):
    frame = env.observe(t)
    out_lines = []
    for y in range(frame.height):
        row_chars = []
        for x in range(frame.width):
            row_chars.append("#" if frame.get(x, y) else ".")
        out_lines.append("".join(row_chars))
    if RENDER_CLEARS:
        sys.stdout.write(CSI_HOME_CLEAR)
    sys.stdout.write("\n".join(out_lines))
    sys.stdout.write(f"\nSim time: {t:.1f} ms\n")
    sys.stdout.flush()

def stimulate_direction(t: SimTime, dir_id: int, strength: float = 1.2):
    net.inject_spike(t, dir_id, strength)

def find_dot(frame) -> Optional[Tuple[int,int]]:
    for y in range(frame.height):
        for x in range(frame.width):
            if frame.get(x,y) != 0:
                return (x,y)
    return None

def dist_to_wall(x, y, w, h):
    return min(x, y, w-1-x, h-1-y)

# --- Continuous simulation ---
dt = 10.0  # ms
t = 0.0


try:
    while True:
        enc_spikes = list(io.encode_observation(t))
        # 1) Encode what the screen looks like *now* and inject before stepping the network
        for nid, offset in enc_spikes:
            net.inject_spike(t + offset, nid, 1.3)  # slightly stronger than 1.1

        if enc_spikes:
            msg = f"[{t:7.1f} ms] ENCODER -> {enc_spikes}"
            #print(msg)
            logging.info(msg)

        # 2) Step network and collect spikes
        spikes = net.step(t, dt_ms=dt)

        # Feed ONLY output spikes to the decoder; harmless to pass all, decoder filters by ID sets
        for nid in spikes:
            io.on_output_spike(t, nid)
        

        


        frame_before = env.observe(t)
        pos_before = find_dot(frame_before)   
        action = io.maybe_emit_action(t)
        frame_after = env.observe(t)
        pos_after = find_dot(frame_after)

        r = 0.0
        if pos_before and pos_after:
            x0,y0 = pos_before
            x1,y1 = pos_after
            if (x1,y1) == (x0,y0):
                r = -1.0   # tried to move into wall; punish
            else:
                # reward moving to safer space: farther from walls
                w, h = frame_after.width, frame_after.height
                gain = dist_to_wall(x1,y1,w,h) - dist_to_wall(x0,y0,w,h)
                r = 0.2 + 0.1*gain  # small base reward plus bonus if farther from edges

        # apply reward to the network
        if r != 0.0:
            net.apply_reward(r)

        # 3) Render (cosmetic)
        if int(t) % 200 == 0:
            render_ascii(env, t)

        t += dt
        time.sleep(dt / 1000.0)
except KeyboardInterrupt:
    print("\nStopped.")
