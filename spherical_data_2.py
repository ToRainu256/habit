import numpy as np

def generate_random_samples(num_samples, num_dimensions):
    samples = 5 * np.random.rand(num_samples, num_dimensions)
    return samples

def generate_random_spheres(num_spheres, samples):
    spheres = []
    for _ in range(num_spheres):
        center_sample = samples[np.random.randint(len(samples))]
        sphere_radius = np.random.rand()*1.1
        spheres.append((center_sample, sphere_radius))
    return spheres

def assign_labels(samples, spheres):
    labels = []
    for sample in samples:
        inside_sphere = False
        for sphere_center, sphere_radius in spheres:
            distance = np.linalg.norm(sample - sphere_center)
            if distance <= sphere_radius:
                inside_sphere = True
                break
        if inside_sphere:
            labels.append(-1)
        else:
            labels.append(1)
    return labels

def generate_labeled_samples(num_samples, num_dimensions, num_spheres):
    # サンプルの生成
    samples = generate_random_samples(num_samples, num_dimensions)

    # 球の生成
    spheres = generate_random_spheres(num_spheres, samples)

    # ラベルの割り当て
    labels = assign_labels(samples, spheres)

    return samples, labels

# パラメータの設定
#num_samples = 1000  # サンプル数
#num_dimensions = 3  # N次元
#num_spheres = 5     # 球の数

# ラベル付きサンプルの生成
#samples, labels = generate_labeled_samples(num_samples, num_dimensions, num_spheres)

# 結果の表示
#for i in range(num_samples):
#    print(f"Sample {i+1}: {samples[i]} - Label: {labels[i]}")
