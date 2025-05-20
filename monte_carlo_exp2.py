from monte_carlo_sim import MonteCarloModel, MonteCarloSimulation
import numpy as np
rng = np.random.default_rng()



class DiceGame(MonteCarloModel):
    def __init__(self, n):
        self.rolls = []
        self.n = n
        self.score = None

    def run(self):
        half = (self.n//2)
        for i in rng.integers(1,7,self.n):
            self.rolls.append(i)

        # print("Rolls:", self.rolls)
        self.rolls.sort()

        mysum = sum(self.rolls[half:])

        myproduct = 1
        for i in self.rolls[:half]:
            myproduct *= i

        self.score = mysum-myproduct

        # print(f"Score : {self.score}")
        return self.score

    def reset(self):
        self.rolls = []
        self.score = None
        
        

c_model1 = DiceGame(2)
c_model2 = DiceGame(4)
c_model3 = DiceGame(6)
c_model4 = DiceGame(8)

s1 = MonteCarloSimulation(1000, c_model1)
s1.replicate()
s1.display_hist()
print(s1.get_mean())

s2 = MonteCarloSimulation(1000, c_model2)
s2.replicate()
s2.display_hist()
print(s2.get_mean())

s3 = MonteCarloSimulation(1000, c_model3)
s3.replicate()
s3.display_hist()
print(s3.get_mean())

s4 = MonteCarloSimulation(1000, c_model4)
s4.replicate()
s4.display_hist()
print(s4.get_mean())
