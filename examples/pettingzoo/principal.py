import collections
from typing import Callable


class Principal:
    """Principal class for universal mechanism design
    In this setting, the principal computes a tax on the reward based on the number of apples collected by the agent.
    The tax should be between 0 and 1.5. 0 means no tax, 1.5 means 150% tax.
    """
    def __init__(self, objective: Callable, num_players) -> None:
        self.set_objective(objective)
        self.apple_counts = [0 for _ in range(num_players)]
        self.collected_tax = 0

    def set_objective(self, objective: Callable) -> None:
        print("********\nSetting objective to", objective.__name__, "\n********")
        self.objective = objective

    def calculate_tax(self, num_apples) -> float:
        """very simple baseline principal: no tax on utilitarian, 100% tax on egalitarian if num_apples > 10"""
        if self.objective.__name__ == "utilitarian":
            return 0
        if self.objective.__name__ == "egalitarian":
            if num_apples > 10:
                return 1.5  # punish the agent for being too greedy
            else:
                return 0

    def collect_tax(self, tax: float) -> None:
        """store collected tax"""
        self.collected_tax += tax
