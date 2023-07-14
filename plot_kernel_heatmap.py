import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(samples, sigmas):
    n = samples.shape[0]
    kernel_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            diff = samples[i] - samples[j]
            sigma_i = sigmas[i]
            sigma_j = sigmas[j]
            kernel_matrix[i, j] = np.exp(-np.dot(diff, diff) / (2 * sigma_i * sigma_j))

    return kernel_matrix

def plot_kernel_matrix(kernel_matrix):
    plt.imshow(kernel_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Index')
    plt.title('Gaussian Kernel Matrix')
    plt.show()

# ラベルが0と1のサンプルデータ（仮のデータ）
samples = np.array([[0, 1], [1, 0], [2, 2], [3, 3]])
labels = np.array([0, 1, 0, 1])

# 各サンプルごとのsigmaの値（仮の値）
sigmas = np.array([1.0, 0.5, 2.0, 1.5])

# ガウスカーネルを適用
kernel_matrix = gaussian_kernel(samples, sigmas)

# ガウスカーネル行列を描画
plot_kernel_matrix(kernel_matrix)
