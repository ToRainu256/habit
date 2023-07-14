from sklearn.svm import SVC
import numpy as np

def variable_sigma_svm(X, y, sigma_values):
    svm_models = []
    for i in range(X.shape[0]):
        svm = SVC(kernel='rbf', gamma=1/(2*sigma_values[i]**2))
        svm.fit(X[i], y[i])
        svm_models.append(svm)
    return svm_models

# サンプルデータとラベルの生成
X = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
y = np.array([[0], [0], [1], [1]])
print(y[0])
# 各サンプルごとのsigma値の設定（例としてランダムな値を生成）
sigma_values = np.random.uniform(low=0.1, high=1.0, size=X.shape[0])

# サンプルごとに異なるsigmaを使用したSVMの実行
svm_models = variable_sigma_svm(X, y, sigma_values)

# サンプルデータの分類結果の表示
for i, svm in enumerate(svm_models):
    print(f"サンプル{i+1}の分類結果: {svm.predict(X[i:i+1])}")
