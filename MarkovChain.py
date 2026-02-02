import numpy as np
class State:
    def __init__(self, name, **variables):
        self.name = name
        self.vars = variables

    def __repr__(self):
        return f"{self.name} {self.vars}"

    def get_vars(self):
        """Return the variables dictionary for this state."""
        return self.vars

    def get_var(self, key, default=None):
        """Return a single variable by key, or default if not present."""
        return self.vars.get(key, default)


class MarkovChain:
    def __init__(self, states, transition_matrix):
        self.states = states
        self.P = np.array(transition_matrix, dtype=float)

        if not np.allclose(self.P.sum(axis=1), 1.0):
            raise ValueError("Transition matrix rows must sum to 1")

    def step(self, current_state):
        idx = self.states.index(current_state)
        next_idx = np.random.choice(len(self.states), p=self.P[idx])
        return self.states[next_idx]

    def simulate(self, start_state, n_steps):
        trajectory = [start_state]
        current = start_state

        for _ in range(n_steps):
            current = self.step(current)
            trajectory.append(current)

        return trajectory
