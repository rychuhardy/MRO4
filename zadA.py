import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap


samples_num = 10000  # 00
dims = [3, 4, 5, 7, 13] # 5, 16, etc
radius = 1.0

# generating integers instead of floats because the function have closed interval and random_uniform does not
precision = 100000
# this is a hack to generate more border points
border_part = samples_num/10
# red, blue, green
cmap = ListedColormap(['#FF0000', '#0000FF', '#00FF00'])


def assignColor(sample):
    # 0 - red
    # 2 - green
    # 1 - blue
    if any(abs(x) >= radius*precision for x in sample):
        return 0
    if np.linalg.norm(sample) < radius*precision:
        return 2
    else:
        return 1

for dim in dims:
    # there are almost none points on the border of hypercube
    samples = np.random.random_integers(-radius*precision-border_part, radius*precision+border_part, (samples_num, dim))

    for samp in samples:
        for el in range(0, len(samp)):
            if samp[el] < -radius*precision:
                samp[el] = -radius*precision
            elif samp[el] > radius*precision:
                samp[el] = radius*precision

    color_samples = list(map(assignColor ,samples))
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(samples)

    plt.figure(dim)
    plt.scatter(reduced[:,0], reduced[:,1], c=color_samples, cmap=cmap, edgecolor='k', s=20)
    plt.title("PCA reduction from %i dimensions" % dim)

plt.show()