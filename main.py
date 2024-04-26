import numpy as np
import kmeans
import common
import matplotlib.pyplot as plt

# Load Toy Dataset
X = np.loadtxt("toy_data.txt")

def run_kmeans():
    for K in range(1, 5):
        min_cost = None
        best_seed = None
        for seed in range(0, 5):
            mixture, post = common.init(X, K, seed)
            mixture, post, cost = kmeans.run(X, mixture, post)
            if min_cost is None or cost < min_cost:
                min_cost = cost
                best_seed = seed

        mixture, post = common.init(X, K, best_seed)
        mixture, post, cost = kmeans.run(X, mixture, post)
        title = f"K-means for K={K}, seed={best_seed}, cost={min_cost}"  # Use f-string for formatting
        print(title)
        common.plot(X, mixture, post, title)

run_kmeans()
