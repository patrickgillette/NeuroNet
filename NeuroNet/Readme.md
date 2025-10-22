Goals:
Generalized I/O architecture:
Support arbitrary input and output modalities through well-defined interfaces (Encoder, Decoder, Environment, IOCoordinator).

Minimal but extensible simulation core:
A compact LIF-based SNN capable of time-stepped simulation and future synaptic plasticity.

Experimentation sandbox:
Allow quick iteration of biologically inspired control or learning algorithms without external dependencies.

Composability:
Enable mixing and matching of encoders, decoders, and environments without code changes in the simulation core.

Transparency and introspection:
Encourage understanding of timing, neuron activity, and spike-driven dynamics through logs and ASCII visualization.

///
Integrations and Examples:
Built-in:

SimpleScreen - ASCII environment for visualization and movement.

HalfPlaneDirectionEncoder — Simple spatial encoder based on screen pixel distribution.

FirstToSpikeMoveDecoder - Decision policy based on the first neuron to spike in a window.

SNN Core - Lightweight scheduler and LIF neuron implementation.

Potential Integrations:

Vision: connect a camera feed -> encoder -> visual cortex simulation -> decoder -> robot arm commands.

Audio: microphone -> spectrogram encoder -> phoneme classifier.

Control Systems: sensor readings -> encoder -> spiking controller -> actuator output (e.g., motor signals).

Game/Simulation Interfaces: map observations from a gym-like API to SNN actions.


/// 
Current Problems / Work in Progress:

No synaptic connectivity in demo:
The network runs on external spike injections; add synapses to enable recurrent or learned behavior.

Unimplemented encoders/decoders:
Several stubs (e.g., TilePoissonEncoder, EventDeltaEncoder) are placeholders for future encoding strategies.

No learning mechanisms:
STDP or reinforcement-driven updates are not yet included.

Time resolution mismatch:
Input binning and simulation step size are coarse-grained; precise event timing may require refinement.

Duplicate classes:
Some overlap between NeuroNet.py and screen_interface.py can be consolidated.

Logging and visualization:
Expand from text logs to richer plots or spike rasters for analysis.