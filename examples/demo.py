"""
TISA Demo Script
================

This script demonstrates how to use the TISA library to compute the transform-invariant
distance between two synthetic time series.

It covers:
1. Generating synthetic financial-like data (random walk).
2. Creating a "transformed" version of the data (scaled and shifted).
3. Computing the TISA distance.
4. Extracting detailed alignment information.
"""

import numpy as np
import matplotlib.pyplot as plt
from tisa.distance import TISADistance

def main():
    # 1. Generate synthetic data
    # We'll create a random walk to simulate a stock price
    np.random.seed(42)
    n_steps = 252  # One year of daily data
    returns = np.random.randn(n_steps)
    price_a = 100 + np.cumsum(returns)

    # 2. Create a transformed version
    # Let's simulate a correlated asset that has higher volatility and a different price level
    # We'll also add some noise
    scale = 1.5
    shift = 50
    noise = np.random.randn(n_steps) * 0.5
    price_b = (price_a * scale) + shift + noise

    print(f"Series A: mean={np.mean(price_a):.2f}, std={np.std(price_a):.2f}")
    print(f"Series B: mean={np.mean(price_b):.2f}, std={np.std(price_b):.2f}")
    print("-" * 40)

    # 3. Compute TISA distance
    # TISA is invariant to the scale and shift we introduced
    tisa = TISADistance()
    distance = tisa.pairwise(price_a, price_b)
    
    print(f"TISA Distance: {distance:.4f}")
    
    # Compare with Euclidean distance (on z-normalized data for fairness)
    def z_norm(x):
        return (x - np.mean(x)) / np.std(x)
    
    euclidean_dist = np.linalg.norm(z_norm(price_a) - z_norm(price_b))
    print(f"Euclidean Distance (z-norm): {euclidean_dist:.4f}")
    print("-" * 40)

    # 4. Get detailed alignment info
    detail = tisa.detailed(price_a, price_b)
    print(f"Best Transform Mode: {detail['best_transform']}")
    print(f"Alignment Cost: {detail['distance']:.4f}")
    
    # The 'mapping' shows which indices in A align with which indices in B
    mapping = detail['mapping']
    print(f"Alignment path length: {len(mapping)}")

    print("\nDone! TISA successfully aligned the series despite the transformation.")

if __name__ == "__main__":
    main()
