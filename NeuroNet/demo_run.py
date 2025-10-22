
import time
from typing import Optional, Tuple
from sim_core import SNN, Synapse, LIFConfig
from concrete_io import SimpleScreen,  HalfPlaneDirectionEncoder, FirstToSpikeMoveDecoder
from screen_interface import IOCoordinator, SimTime, ScreenAction, ScreenActionType
import sys, time
from colorama import init as colorama_init
import logging
logging.basicConfig(filename="neuron_log.txt", level=logging.INFO, format="%(message)s")

colorama_init()  # enable ANSI on Windows


CSI_HOME_CLEAR = "\x1b[H\x1b[J"  # go home + clear screen
RENDER_CLEARS = True

# --- Setup network ---
import random
N_IN = 16*9
N_OUT = 4

lif = LIFConfig(v_rest=0.0, v_reset=0.0, v_thresh=1.0, tau_m_ms=10.0, r_m=1.0, tau_ref_ms=2.0)
net = SNN(num_neurons=N_IN + N_OUT, lif_template=lif)

# random small weights, short delay
for pre in range(N_IN):
    for d in range(N_OUT):
        post = N_IN + d
        w0 = random.uniform(-0.05, 0.05)
        net.add_synapse(Synapse(pre=pre, post=post, w=w0, delay_ms=1.0, plastic=True))


# --- Setup I/O ---
env = SimpleScreen(width=16, height=9)
env.apply_action(0.0, ScreenAction(kind=ScreenActionType.DRAW_DOT, x=8, y=4))

#encoder = NullEncoder(target_neuron_ids=[])   #old
from concrete_io import HalfPlaneDirectionEncoder
encoder = HalfPlaneDirectionEncoder(
    up_id=0, down_id=1, left_id=2, right_id=3,
    min_interval_ms=30.0  # was 80ms
)

decoder = FirstToSpikeMoveDecoder(
    up_ids=[0], down_ids=[1], left_ids=[2], right_ids=[3],
    readout_period_ms=100.0, min_action_delay_ms=1.0, step=1
)
io = IOCoordinator(env, encoder, decoder)
decoder.reset()

from sim_core import Synapse
INH = -0.6; DELAY = 5.0
pairs = [(0,1),(0,2),(0,3),(1,0),(1,2),(1,3),(2,0),(2,1),(2,3),(3,0),(3,1),(3,2)]
for pre, post in pairs:
    net.add_synapse(Synapse(pre=pre, post=post, w=INH, delay_ms=DELAY))

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
start = time.time()

try:
    while True:
        # Determine which neuron to stimulate every 500 ms
        # phase = int((t // 500) % 4)
        # if phase == 0: stimulate_direction(t, 0)
        # elif phase == 1: stimulate_direction(t, 3)
        # elif phase == 2: stimulate_direction(t, 1)
        # else: stimulate_direction(t, 2)
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
        #if spikes:
            #print(f"\nSpikes @ {t:.1f} ms:", spikes)
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
