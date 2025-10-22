
import time
from sim_core import SNN, Synapse, LIFConfig
from concrete_io import SimpleScreen, NullEncoder, FirstToSpikeMoveDecoder
from screen_interface import IOCoordinator, SimTime, ScreenActionType
import sys, time
from colorama import init as colorama_init

colorama_init()  # enable ANSI on Windows

CSI_HOME_CLEAR = "\x1b[H\x1b[J"  # go home + clear screen

# --- Setup network ---
N_OUT = 4  # up, down, left, right
lif = LIFConfig(v_rest=0.0, v_reset=0.0, v_thresh=1.0, tau_m_ms=10.0, r_m=1.0, tau_ref_ms=2.0)
net = SNN(num_neurons=N_OUT, lif_template=lif)

# --- Setup I/O ---
env = SimpleScreen(width=16, height=9)
encoder = NullEncoder(target_neuron_ids=[])
decoder = FirstToSpikeMoveDecoder(
    up_ids=[0], down_ids=[1], left_ids=[2], right_ids=[3],
    readout_period_ms=100.0, min_action_delay_ms=1.0, step=1
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
    sys.stdout.write(CSI_HOME_CLEAR)           # clear
    sys.stdout.write("\n".join(out_lines))     # draw
    sys.stdout.write(f"\n\nSim time: {t:.1f} ms")
    sys.stdout.flush()

def stimulate_direction(t: SimTime, dir_id: int, strength: float = 1.2):
    net.inject_spike(t, dir_id, strength)

# --- Continuous simulation ---
dt = 10.0  # ms
t = 0.0
start = time.time()

try:
    while True:
        # Determine which neuron to stimulate every 500 ms
        phase = int((t // 500) % 4)
        if phase == 0:
            stimulate_direction(t, 0)  # Up
        elif phase == 1:
            stimulate_direction(t, 3)  # Right
        elif phase == 2:
            stimulate_direction(t, 1)  # Down
        else:
            stimulate_direction(t, 2)  # Left

        # Advance network and handle spikes
        spikes = net.step(t, dt_ms=dt)
        for nid in spikes:
            io.on_output_spike(t, nid)
        io.maybe_emit_action(t)

        # Render every ~200 ms of sim time
        if int(t) % 200 == 0:
            render_ascii(env,t)

        t += dt
        time.sleep(dt / 1000.0)  # scale sim time to real time
except KeyboardInterrupt:
    print("\nStopped.")
