"""Make Gaussian distributio
"""

import numpy as np


def make_gaussian(size, sigma=3, center=None):
    """ Make a square gaussian kernel.
    Size is the length of a side of the square
    sigma is standard deviation.
    Idea from this gist: https://gist.github.com/andrewgiessel/4635563
    With additions for skewed gaussian
    original output: np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)
    """

    x = np.arange(0, size[0], 1, float)
    y = np.arange(0, size[1], 1, float)
    y = y[:, np.newaxis]

    if center is None:
        x0 = size[0] // 2
        y0 = size[1] // 2
    else:
        x0 = center[0]
        y0 = center[1]

    # choose given sigma for the shortest side
    if size[0] < size[1]:
        sigma_1 = sigma
        sigma_2 = (size[1] * sigma) // size[0]
    else:
        sigma_2 = sigma
        sigma_1 = (size[0] * sigma) // size[1]

    return np.exp(
        -(((x - x0) ** 2) / sigma_1 ** 2 + ((y - y0) ** 2) / sigma_2 ** 2)
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    blank = np.zeros([200, 200])
    gaussian_patch = make_gaussian([30, 100], sigma=8)
    blank[0:100, 0:30] = gaussian_patch
    plt.figure()
    plt.imshow(blank)
    plt.show()
