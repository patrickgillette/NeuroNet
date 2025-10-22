import time
from typing import Optional, Tuple
import sys
import logging
import random

from sim_core import SNN, Synapse, LIFConfig
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
N_OUT = 4             # up, down, left, right
OUT0 = N_IN           # first output neuron index
OUT_IDS = [OUT0 + i for i in range(N_OUT)]  # [N_IN, N_IN+1, N_IN+2, N_IN+3]

lif = LIFConfig(v_rest=0.0, v_reset=0.0, v_thresh=1.0, tau_m_ms=10.0, r_m=1.0, tau_ref_ms=2.0)
net = SNN(num_neurons=N_IN + N_OUT, lif_template=lif)

# Random small plastic feedforward weights from all inputs -> each output
for pre in range(N_IN):
    for d in range(N_OUT):
        post = OUT_IDS[d]
        w0 = random.uniform(1.1, 1.5)
        net.add_synapse(Synapse(pre=pre, post=post, w=w0, delay_ms=0.0, plastic=True))


# Lateral inhibition among outputs to encourage a single winner
INH = -0.6
DELAY = 0.0
pairs = [(a, b) for a in OUT_IDS for b in OUT_IDS if a != b]
for pre, post in pairs:
    net.add_synapse(Synapse(pre=pre, post=post, w=INH, delay_ms=DELAY))


# --- Setup I/O ---
env = SimpleScreen(width=W, height=H)
env.apply_action(0.0, ScreenAction(kind=ScreenActionType.DRAW_DOT, x=W // 2, y=H // 2))

# Encode the dot's position as a single active input neuron (y*W + x)
encoder = PositionEncoder(width=W, height=H, base_id=0, min_interval_ms=5.0)

# Decoder looks at the actual output neuron IDs (N_IN..N_IN+3)
decoder = FirstToSpikeMoveDecoder(
    up_ids=[OUT_IDS[0]],
    down_ids=[OUT_IDS[1]],
    left_ids=[OUT_IDS[2]],
    right_ids=[OUT_IDS[3]],
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
        

        action = io.maybe_emit_action(t)


        frame_before = env.observe(t)
        pos_before = find_dot(frame_before)   

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
