import numpy as np
import matplotlib.pyplot as plt

def generate_checkered_samples(grid_size, sample_per_region):
    # グリッドの作成
    x = np.linspace(0, 1, grid_size + 1)
    y = np.linspace(0, 1, grid_size + 1)
    xx, yy = np.meshgrid(x, y)
    grid = np.stack([xx, yy], axis=-1)

    # サンプルの生成
    samples = []
    labels = []
    for i in range(grid_size):
        for j in range(grid_size):
            region_samples = np.random.rand(sample_per_region, 2) * (1 / grid_size) + grid[i, j]
            region_labels = np.zeros(sample_per_region)
            if (i + j) % 2 == 1:
                region_labels = np.ones(sample_per_region)
            samples.append(region_samples)
            labels.append(region_labels)

    samples = np.concatenate(samples, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    return samples, labels


