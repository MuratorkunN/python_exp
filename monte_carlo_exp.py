from monte_carlo_sim import MonteCarloModel, MonteCarloSimulation
import numpy as np
rng = np.random.default_rng()


class Craps(MonteCarloModel):
    def __init__(self):
        self.rolls = []
        self.win_status = None

    def run(self):
        first_roll = sum(rng.integers(1, 7, 2))
        if first_roll in (7, 11):
            self.win_status = 1
            return 1

        if first_roll in (2, 3, 12):
            self.win_status = 0
            return 0

        while True:
            next_roll = sum(rng.integers(1, 7, 2))
            if next_roll == 7:
                self.win_status = 0
                return 0

            if next_roll == first_roll:
                self.win_status = 1
                return 1

            next_roll = first_roll

    def reset(self):
        self.rolls = []
        self.win_status = None



c_model = Craps()
print(c_model.run())

sim = MonteCarloSimulation(1000, c_model)
sim.replicate()
print(sim.get_mean())
sim.display_hist()

