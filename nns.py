import numpy as np
from sklearn.neighbors import NearestNeighbors

def compute_distances(N, samples, distance_type='max'):
    n = N

    # N近傍の距離を計算
    neighbors = NearestNeighbors(n_neighbors=n)
    neighbors.fit(samples)
    distances, _ = neighbors.kneighbors(samples)

    if distance_type == 'max':
        # 最大距離を計算
        max_distances = np.max(distances[:, 1:], axis=1)
        return max_distances
    elif distance_type == 'avg':
        # 平均距離を計算
        avg_distances = np.mean(distances[:, 1:], axis=1)
        return avg_distances
    elif distance_type == 'min':
        min_distaces = np.min(distances[:, 1:], axis=1)
        return min_distaces
    else:
        raise ValueError('Invalid distance_type. Choose from "max" or "avg" or "min".')

# サンプルデータ（仮のデータ）
#samples = np.array([[0, 1], [1, 0], [2, 2], [3, 3]])

# 最大距離を計算
#max_distances = compute_distances(samples, distance_type='max')
#print('Max Distances:', max_distances)

# 平均距離を計算
#avg_distances = compute_distances(samples, distance_type='avg')
#print('Avg Distances:', avg_distances)
