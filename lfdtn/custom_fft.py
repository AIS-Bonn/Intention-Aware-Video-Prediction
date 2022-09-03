import torch
import torch.fft
import numpy as np
from lfdtn.complex_ops import complex_div, complex_abs, complex_conj, complex_mul,getNormalizeFFT

def custom_fft(t_r):
    return custom_fft_fun(t_r)

def custom_ifft(t_c):
    return custom_ifft_fun(t_c)


def custom_fft_fun(t_r):
    t_c = torch.stack([t_r, torch.zeros_like(t_r).detach()], dim=-1)
    t_c = torch.view_as_complex(t_c)
    result = torch.fft.fftn(t_c, dim=(-2, -1), norm="forward" if getNormalizeFFT() else "backward")
    return torch.view_as_real(result)


def custom_ifft_fun(t_c):
    t_c = torch.view_as_complex(t_c)
    result = torch.fft.ifftn(t_c, dim=(-2, -1), norm="forward" if getNormalizeFFT() else "backward")
    return result.real
