import numpy as np
from typing import Callable

def newton_raphson(f : Callable[[float],float], df : Callable[[float],float], z : float, epsilon : float = 1e-10, max_iteration : int = 100):

    min_derivative=1e-10

    for i in range(max_iteration):

        value = f(z)
        derivative = df(z)

        if abs(derivative) <= min_derivative:
            raise ValueError("Derivative too small to apply the Newton-Raphson method.")

        z_new = z - value/derivative
        
        if abs(z_new - z) <= epsilon:
            return z_new

        z = z_new

    raise RuntimeError("The method does not converge.")