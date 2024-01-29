from typing import Callable

import numpy as np


def utilitarian(num_apples: dict[str, int], num_players) -> float:
    """Utilitarian objective"""
    return sum(num_apples) / num_players


def egalitarian(num_apples: dict[str, int]) -> float:
    """Egalitarian objective"""
    return min(num_apples)


def vote(player_values: np.ndarray) -> Callable:
    """Vote on objective"""
    if player_values.mean() > 0.5:
        return egalitarian
    else:
        return utilitarian
