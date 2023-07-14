import numpy as np
from sklearn.datasets import make_blobs

def generate_multilayer_spheres(n_samples, n_features, n_balls, radius_ratio):
    samples = []
    labels = []
    
    for i in range(n_balls):
        center = np.random.randn(n_features)
        radius = np.random.uniform(0.1, 1.0)
        ball_samples = np.random.randn(n_samples, n_features)
        ball_samples /= np.linalg.norm(ball_samples, axis=1)[:, np.newaxis]
        ball_samples *= radius
        ball_samples += center
        samples.append(ball_samples)
        labels.extend([1] * n_samples)
    
    samples = np.concatenate(samples, axis=0)
    labels = np.array(labels)
    
    outer_samples, _ = make_blobs(
        n_samples=n_samples * n_balls,
        n_features=n_features,
        centers=n_balls,
        cluster_std=radius_ratio,
        random_state=42
    )
    samples = np.concatenate((samples, outer_samples), axis=0)
    outer_labels = np.zeros(n_samples * n_balls)
    labels = np.concatenate((labels, outer_labels), axis=0)
    
    return samples, labels