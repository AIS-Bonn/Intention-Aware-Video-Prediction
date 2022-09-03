from matplotlib.pyplot import get
from torch.functional import F
import torch.nn as nn
from asset.utils import getPhaseAdd,dmsg
from lfdtn.complex_ops import getEps
from lfdtn.custom_fft import custom_ifft, custom_fft
import torch
from packaging import version

@torch.jit.script
def fold(T, res_y: int, res_x: int, y_stride: int, x_stride: int, cell_size: int, pad_size: int):
    return nn.functional.fold(T, output_size=(res_y, res_x), kernel_size=(cell_size, cell_size), padding=(pad_size, pad_size), stride=(y_stride, x_stride))

@torch.jit.script
def calcCOM(inp):
    denom = torch.sum(inp, dim=(-2, -1))
    xs = inp.size(-2)
    ys = inp.size(-1)
    xMesh, yMesh = torch.meshgrid(torch.arange(0, xs), torch.arange(0, ys),indexing='xy')
    xMesh = xMesh.to(inp.device)
    yMesh = yMesh.to(inp.device)
    COMX = torch.sum(inp * xMesh, dim=(-2, -1)) / (denom + getEps())
    COMY = torch.sum(inp * yMesh, dim=(-2, -1)) / (denom + getEps())
    return torch.stack([COMX/xs, COMY/ys], dim=-1)

@torch.jit.script
def extract_local_windows(batch, windowSize: int, y_stride: int = 1, x_stride: int = 1, padding: int = 0):
    BS,CS,HH,WW = batch.shape
  
    windowSizeAdjust = windowSize + 2 * padding


    padSize = int(x_stride * ((windowSizeAdjust - 1) // x_stride))


    result = nn.functional.unfold(batch, kernel_size=(windowSizeAdjust, windowSizeAdjust), padding=(padSize, padSize),
                       stride=(y_stride, x_stride))

    result = result.view(BS,CS,windowSizeAdjust*windowSizeAdjust,-1)

    result = result.permute(0,3,1,2)

    return result.reshape(BS, -1, windowSizeAdjust, windowSizeAdjust)

@torch.jit.script
def LFT(batch, window, y_stride: int = 1, x_stride: int = 1, padding: int = 1,useCOM:bool=False)-> tuple:
    windowBatch = extract_local_windows(batch, window.shape[0], y_stride=y_stride, x_stride=x_stride, padding=padding)

    COM:Union[None,torch.Tensor]=None
    if useCOM:
        COM =  calcCOM(windowBatch)

    windowPadded = F.pad(window, (padding, padding, padding, padding))
    localImageWindowsSmoothedPadded = windowBatch * windowPadded

    return custom_fft(localImageWindowsSmoothedPadded),COM

@torch.jit.script
def iLFT(stft2D_result, window, T, res_y: int, res_x: int, y_stride: int = 1, x_stride: int = 1, padding: int = 1,
         is_inp_complex: bool=True,move_window_according_T: bool=True,channels: int=1) -> tuple:

    BS= stft2D_result.shape[0]
    cellSize = window.shape[0]


    padSize = int(x_stride * ((cellSize - 1) // x_stride))

    cellSizeAdjust = cellSize + 2 * padding
    padSizeAdjust = int(x_stride * ((cellSizeAdjust - 1) // x_stride))


    num_windows_y = (res_y + 2 * padSizeAdjust - cellSizeAdjust) // y_stride + 1
    num_windows_x = (res_x + 2 * padSizeAdjust - cellSizeAdjust) // x_stride + 1
    num_windows_total = num_windows_y * num_windows_x

    if is_inp_complex:
        ifft_result = custom_ifft(stft2D_result)
    else:
        ifft_result = stft2D_result.clone()



    ifft_result = ifft_result.view((BS, num_windows_y, num_windows_x,channels, cellSizeAdjust, cellSizeAdjust))

    window_big = F.pad(window, (padding, padding, padding, padding), value=0.0)
    window_big = window_big.expand(BS, num_windows_total,channels, -1, -1)

    if move_window_according_T:
        window_big_Complex = custom_fft(window_big)
        window_big_Complex = getPhaseAdd(window_big_Complex, T.view_as(window_big_Complex))
        window_big = custom_ifft(window_big_Complex)

    window_big = window_big.view(BS, num_windows_y, num_windows_x,channels, window_big.shape[3], window_big.shape[4])

    ifft_result *= window_big

    ifft_result = ifft_result.reshape(BS, -1, ifft_result.shape[4] * ifft_result.shape[5]*channels).permute(0, 2, 1)
    test = fold(ifft_result, \
                res_y=res_y, res_x=res_x, y_stride=y_stride, x_stride=x_stride, cell_size=cellSizeAdjust,
                pad_size=padSizeAdjust)

    window_big = (window_big ** 2).reshape(BS, -1, window_big.shape[4] * window_big.shape[5]*channels).permute(0, 2, 1)
    windowTracker = fold(window_big, \
                         res_y=res_y, res_x=res_x, y_stride=y_stride, x_stride=x_stride, cell_size=cellSizeAdjust,
                         pad_size=padSizeAdjust)
                        

    windowTracker += getEps()
    weighted_result = test / windowTracker
    return weighted_result, windowTracker

def compact_LFT(batch, window, config):
    return LFT(batch, window, x_stride=config.stride, y_stride=config.stride, padding=config.window_padding_constrained,useCOM=config.useCOM)

def compact_iLFT(LFT_result, window, T, config,is_inp_complex=True,move_window_according_T=True,channel=1):
    return iLFT(LFT_result, window, T, res_y=config.res_y_constrained, res_x=config.res_x_constrained, y_stride=config.stride,\
                    x_stride=config.stride, padding=config.window_padding_constrained,is_inp_complex=is_inp_complex,move_window_according_T=move_window_according_T,channels=channel)