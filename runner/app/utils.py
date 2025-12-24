import math
import random

import numpy as np


def random_with_step(min_val, max_val, step):
    steps = int(math.floor((max_val - min_val) / step))
    return min_val + step * random.randint(0, steps)


def truncated_normal(a: float, b: float, mean: float, std: float) -> float:

    if not (a <= mean <= b):
        raise ValueError("mean должен быть в диапазоне [a, b]")

    while True:
        x = np.random.normal(loc=mean, scale=std)
        if a <= x <= b:
            return x


def clamp(x: float, min_: float, max_: float) -> float:
    return max(min_, min(max_, x))

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))