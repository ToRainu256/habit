import numpy as np

def gaussian_kernel(x, samples, sigma):
    diff = samples - x
    exponent = -0.5 * np.sum((diff / sigma) ** 2, axis=2)
    return np.exp(exponent) / (np.sqrt(2 * np.pi) * sigma)

def variable_sigma_kde(samples, sigma_values, eval_points):
    densities = np.zeros(eval_points.shape[1])
    for i, x in enumerate(eval_points.T):
        kernel_values = gaussian_kernel(x, samples, sigma_values)
        densities[i] = np.mean(kernel_values)
    return densities

# サンプルデータの生成（例として2次元のサンプルを生成）
samples = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]], size=1000)

# 各サンプルごとのsigma値の設定（例としてランダムな値を生成）
sigma_values = np.random.uniform(low=0.1, high=1.0, size=samples.shape[0])

# 密度推定結果の評価点の設定
grid_points = np.mgrid[-5:5:0.1, -5:5:0.1]
eval_points = np.vstack([grid_points[0].ravel(), grid_points[1].ravel()])

# カーネル密度推定の実行
density_estimation = variable_sigma_kde(samples, sigma_values, eval_points)

# 可視化
import matplotlib.pyplot as plt

plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
plt.contour(grid_points[0], grid_points[1], density_estimation.reshape(grid_points[0].shape))
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Variable Sigma KDE')
plt.show()
