import torch
import math

def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

def get_pascal_window(N):
    w = torch.tensor([binom(N-1,i) for i in range(N)], dtype=torch.float32).view((-1,1))
    w = w/w.max()
    return w @ w.T

def gHelper(x, N, sigma):
    return torch.exp(-torch.pow(x - (N - 1) * 0.5, 2) / (4 * torch.pow(N*sigma, 2)))

def confinedGaussian1D(k, windowSize, sigma):
    const1 = gHelper(torch.tensor(-0.5).to(sigma.device), windowSize, sigma)
    const2 = gHelper(torch.tensor(-0.5).to(sigma.device) - windowSize, windowSize, sigma)
    const3 = gHelper(torch.tensor(-0.5).to(sigma.device) + windowSize, windowSize, sigma)
    denom = gHelper(k + windowSize, windowSize, sigma) + gHelper(k - windowSize, windowSize, sigma)
    return gHelper(k, windowSize, sigma) - const1 * denom / (const2 + const3)

def get_ACGW(windowSize, sigma):
    window_1D = confinedGaussian1D(torch.arange(windowSize).view(-1,1).to(sigma.device), windowSize, sigma)
    return window_1D @ window_1D.T

def get_2D_Gaussian(resolution, sigma, offset=0, normalized=False):
    kernel_size = resolution

    x_cord = torch.arange(kernel_size).to(sigma.device).float()
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size).float()
    y_grid = x_grid.t().float()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = ((kernel_size - 1) + offset) / 2.
    variance = sigma ** 2.

    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2. * variance)
                      )
    if normalized:
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    return gaussian_kernel.reshape(1, 1, kernel_size, kernel_size)
