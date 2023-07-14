import numpy as np


def select_random_samples(samples, num_samples):
    indices = np.random.choice(len(samples), num_samples, replace=False)
    return samples[indices]

def generate_samples_within_sphere(center, radius, num_samples):
    dim = len(center)
    samples = []
    for _ in range(num_samples):
        direction = np.random.randn(dim)
        direction /= np.linalg.norm(direction)
        r = radius * np.cbrt(np.random.rand())
        sample = center + r * direction
        samples.append(sample)
    return np.array(samples)

def generate_random_spheres_and_samples(samples, num_spheres, num_samples_per_sphere):
    selected_samples = select_random_samples(samples, num_spheres)
    spheres = []
    all_samples = samples.tolist()  # Modified to include initial samples
    for selected_sample in selected_samples:
        radius = np.random.rand()*1.1
        spheres.append((selected_sample, radius))
        samples_in_sphere = generate_samples_within_sphere(selected_sample, radius, num_samples_per_sphere)
        all_samples.extend(samples_in_sphere)
    return spheres, np.array(all_samples)

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
            labels.append(1) 
        else:
            labels.append(-1) 
    return labels

def generate_labeled_samples(samples, num_spheres, num_samples_per_sphere):
    spheres, new_samples = generate_random_spheres_and_samples(samples, num_spheres, num_samples_per_sphere)
    labels = assign_labels(new_samples, spheres)
    return new_samples, labels, spheres