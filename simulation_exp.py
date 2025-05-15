import gurobipy as gp
import numpy as np
from matplotlib import pyplot as plt

rng = np.random.default_rng()


class MCModel():
    """An abstract class that contains methods that every Monte Carlo simulation model
           must overload."""

    def __init__(self):
        t = "Override the .__init__() method for your Monte Carlo model such that "
        t += "instantiated model stores all relevant parameters associated with the "
        t += "beginning of the scenario."
        print(t)

    def run(self):
        t = "Override the .run() method for your Monte Carlo model such that it "
        t += "generates an artificial history of the simulation scenario and returns "
        t += "a numerical output."
        print(t)

    def reset(self):
        t = "Override the .reset() method for your Monte Carlo model such that it resets "
        t += "the Monte Carlo Model to its initial state."
        print(t)


class MCSim():
    """A class that generates simulation objects."""

    def __init__(self, n, model):
        """Assumes n as a positive integer as the number of independent replications of
               the simulation model, model as a Monte Carlo model object.
           Initializes all attributes of a simulation."""
        self.n, self.model = n, model
        self.output = np.empty(n)  # An empty array that will hold replication outputs.
        self.x_bar = None  # The average of replication outputs.

    def replicate(self):
        """Re-runs the model in number of replications. Fills the .output and .x_bar
               attributes."""
        for i in range(self.n):
            self.output[i] = self.model.run()
            self.model.reset()
        self.x_bar = self.output.mean()

    def get_mean(self):
        """Returns the mean estimate of the model output if simulation is replicated."""
        if self.x_bar:
            return self.x_bar
        else:
            print("You have to replicate your model first.")

    def display_hist(self, bins=20):
        """Draws a histogram of model outputs across replications if simulation is
               replicated at least 25 times."""
        if self.x_bar:
            if self.n >= 25:
                plt.figure()
                plt.title("Histogram of Replication Outputs")
                plt.xlabel("Observation Range")
                plt.ylabel("Observed Frequencies")
                plt.hist(self.output, bins=bins, color="yellow", edgecolor="black")
                plt.axvline(self.x_bar, linestyle="dotted", linewidth=3, color="black",
                            label=f"Mean Estimate: {round(self.x_bar, 2)}")
                plt.legend()
            else:
                t = "You need at least 25 replications for a sensible histogram."
                print(t)
        else:
            print("You have to replicate your model first.")


class Craps(MCModel):
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

sim = MCSim(1000, c_model)
sim.replicate()
print(sim.get_mean())
sim.display_hist()


class DiceGame(MCModel):
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

s1 = MCSim(1000, c_model1)
s1.replicate()
s1.display_hist()
print(s1.get_mean())

s2 = MCSim(1000, c_model2)
s2.replicate()
s2.display_hist()
print(s2.get_mean())

s3 = MCSim(1000, c_model3)
s3.replicate()
s3.display_hist()
print(s3.get_mean())

s4 = MCSim(1000, c_model4)
s4.replicate()
s4.display_hist()
print(s4.get_mean())





