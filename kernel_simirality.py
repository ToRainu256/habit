import numpy as np
from sklearn.neighbors import NearestNeighbors

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

def compute_distances(samples, distance_type='max'):
    n = samples.shape[0]

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
    else:
        raise ValueError('Invalid distance_type. Choose from "max" or "avg".')

def threshold_kernel_matrix(kernel_matrix, threshold):
    return np.where(kernel_matrix > threshold, 1, 0)

def predict_labels(train_samples, train_labels, test_samples, thresholded_kernel_matrix_train, thresholded_kernel_matrix_test):
    predicted_labels = []
    for i in range(test_samples.shape[0]):
        # i番目の未知サンプルとの類似度を計算
        similarities = np.sum(thresholded_kernel_matrix_train * thresholded_kernel_matrix_test[i][:, np.newaxis], axis=0)
        
        # 類似度が最も高いトレーニングデータのクラスを取得
        majority_class = train_labels[np.argmax(similarities)]
        
        predicted_labels.append(majority_class)

    predicted_labels = np.array(predicted_labels)
    return predicted_labels

# トレーニングデータ
train_samples = np.array([[0, 1], [1, 0], [2, 2], [3, 3]])
train_labels = np.array([0, 1, 0, 1])

# 未知のサンプル
test_samples = np.array([[4, 4], ])
test_sigmas = np.array([1.0,])

# 各サンプルごとのsigmaの値（仮の値）
sigmas = np.array([1.0, 0.5, 2.0, 1.5])

# ガウスカーネルを適用
kernel_matrix_train = gaussian_kernel(train_samples, sigmas)
kernel_matrix_test = gaussian_kernel(test_samples, test_sigmas)

# ガウスカーネル行列を閾値処理
threshold = 0.5
thresholded_kernel_matrix_train = threshold_kernel_matrix(kernel_matrix_train, threshold)
thresholded_kernel_matrix_test = threshold_kernel_matrix(kernel_matrix_test, threshold)

# 未知のサンプルのクラスを予測
predicted_labels = predict_labels(train_samples, train_labels, test_samples, thresholded_kernel_matrix_train, thresholded_kernel_matrix_test)
print('Predicted Labels:', predicted_labels)
