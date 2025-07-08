from monte_carlo_sim import MonteCarloModel, MonteCarloSimulation
import numpy as np
rng = np.random.default_rng()

class RandomWalk(MonteCarloModel):
    def __init__(self, stop_diff, p_win):
        self.stop_diff, self.p_win = stop_diff, p_win
        self.win_status = None
        self.scores = np.array([0,0])
        self.score_diff = [0]
        self.round_count = 0
    
    def run(self):
        while abs(self.score_diff[-1] < self.stop_diff):
            r = rng.random(1)
            current_round = r < self.p_win
            self.scores[1 - current_round] += 1
            self.score_diff.append(self.scores[0]- self.scores[1])
        self.round_count = self.scores.sum()
        self.win_status = self.scores[0] > self.scores[1]
        return self.round_count
    
    def reset(self):
        self.__init__(self.stop_diff, self.p_win)

    def get_history(self):
        return self.score_diff


    def get_result(self):
        if self.win_status:
            return "YOU WIN!"
    
        elif self.win_status == False:
            return "YOU LOST!"
    
        else:
            return "YOU HAVEN'T PLAYED NO GAMES YET"
    
    def __repr__(self):
        t = f"Player: {self.scores[0]}, Opposition: {self.scores[1]}, {self.get_result()}"
        return t
        


my_game = RandomWalk(10, 0.5) # Instantiate an empty game object.
print(my_game) # A game object can be printed.
output = my_game.run() # We have generated an artificial history for the game.
print(my_game) # Result of the game can, now, be printed.
print(output) # This is the total number of rounds for this run.


sim = MonteCarloSimulation(10000, my_game)
sim.replicate()
print(sim.get_mean())
sim.display_hist()