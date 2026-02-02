from MarkovChain import *

idle = State(
    "IDLE",
    power=0.1,
    bandwidth=0.0,
    noise_level=1e-3
)

s1 = State(
    "state 1",
    power=1.0,
    bandwidth=20e6,
    noise_level=5e-3
)

s2 = State(
    "state 2",
    power=2.0,
    bandwidth=5e6,
    noise_level=1e-3
)

s3 = State(
    "state 3",
    power=2.0,
    bandwidth=5e6,
    noise_level=1e-3
)

s4 = State(
    "state 4",
    power=2.0,
    bandwidth=5e6,
    noise_level=1e-3
)
states = [idle, s1, s2]

P = [
    [0.8, 0.2, 0.0],   
    [0.1, 0.7, 0.2],  
    [0.0, 0.3, 0.7]
]

mc = MarkovChain(states, P)

path = mc.simulate(start_state=idle, n_steps=100)

for s in path:
    print(s)