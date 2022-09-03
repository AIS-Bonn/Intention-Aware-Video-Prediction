import datetime
import io
import json
import math
import os
import re
from datetime import datetime
from math import exp
import imageio
import cv2
#import ipykernel
import kornia
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn as nn
import wandb
#from IPython.display import display
from notebook.notebookapp import list_running_servers
from torch.autograd import Variable
from torch.functional import F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import time
import pandas as pd
from lfdtn.complex_ops import getEps,complex_div, complex_abs, complex_conj, complex_mul, getNormalizeFFT
from icecream import ic
from packaging import version
from typing import Union, List,Tuple
import seaborn as sns
ic.configureOutput(includeContext=True)
ic.disable()
wandbLog = {}
wandbC = 0

augmentData=False


def getIfAugmentData():
    global augmentData
    return augmentData

def setIfAugmentData(inp):
    global augmentData
    augmentData=inp


def show_gif(fname):
    import base64
    from IPython import display
    from IPython.core.display import display as gifshow
    with open(fname, 'rb') as fd:
        b64 = base64.b64encode(fd.read()).decode('ascii')
    gifshow(display.HTML(f'<img src="data:image/gif;base64,{b64}" />'))


def wandblog(d, commit=False,jupyter=False,txt=None):
    global wandbLog, wandbC,junckFiles
    wandbLog.update(d)
    if commit:
        wandb.log(wandbLog, step=wandbC,commit=True)
        wandbC += 1
        wandbLog = {}
        clean_junk_files()
    if jupyter:
        for it in wandbLog.items():
            if txt is not None and txt not in it[0]:
                continue
            print(it[0])
            try:
                if type(it[1])==list:
                    for it in it[1]:
                        show_gif(it._path)
                else:
                    show_gif(it[1]._path)
            except:
                pass
        wandbLog = {}
        clean_junk_files()


torch.pi = torch.acos(torch.zeros(1)).item() * 2

try:
    TOKEN = "mytoken"

    base_url = next(list_running_servers())['url']
    r = requests.get(
        url=base_url + 'api/sessions',
        headers={'Authorization': 'token {}'.format(TOKEN), })

    r.raise_for_status()
    response = r.json()

    kernel_id = re.search('kernel-(.*).json', ipykernel.connect.get_connection_file()).group(1)
    theNotebook = ({r['kernel']['id']: r['notebook']['path'] for r in response}[kernel_id]).split("/")[-1].replace(
        ".ipynb", "")
except:
    theNotebook = "Untitled"

dimension = 2
li = 0
ui = li + 1


NOTVALID = cv2.resize(cv2.imread('asset/notValid.jpg')[:, :, 1], dsize=(128 - 2, 128 - 2),
                      interpolation=cv2.INTER_CUBIC)
NOTVALID = cv2.copyMakeBorder(NOTVALID, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, 0)
NOTVALID = torch.from_numpy(NOTVALID).unsqueeze(0).float() / 255.

import inspect
from colorama import Fore

np.random.seed(123)
randColors = torch.from_numpy(np.random.randint(low=20,high=230,size=(3,20))).float() /255.
randColors = torch.cat([randColors for i in range(5)],dim=1)
def colorChannels(inp,dim=2,max=False):
    sh =list(inp.shape)
    sh[dim]=3
    if max:
        ret,_ =inp.max(dim=dim,keepdim=True)
        return ret.expand(sh)

    fsh = [1]*len(sh)
    fsh[dim] = 3
    res = torch.zeros(sh).to(inp.device)
    for i in range(inp.shape[dim]):
        tmp = torch.index_select(inp, dim, torch.tensor(i).to(inp.device)).expand(sh)
        tmp2 = randColors[:,i].to(inp.device).view(fsh)
        res+=tmp*tmp2
    return res.clamp(0,1)


def dmsg(v=None, *args):
    insp = inspect.currentframe().f_back
    (filename, line_number, function_name, lines, index) = inspect.getframeinfo(insp)
    if '/' in filename:
        sp = filename.split('/')
        filename = sp[-1] if len(sp) < 2 else sp[-2] + '/' + sp[-1]
        filename = '.../' + filename
    vtxt = ''
    args = list(args)
    args.insert(0, v)
    for v in args:
        if v is None:
            continue
        vtxt += ' ! '
        varv = None
        rest = None
        if '.' in v:
            v, rest = v.split('.')
        try:
            varv = insp.f_globals[v]
        except:
            pass
        try:
            varv = insp.f_locals[v]
        except:
            pass
        try:
            if varv is not None:
                if rest is None:
                    try:
                        vtxt += v + ' = ' + json.dumps(varv, indent=4)
                    except:
                        vtxt += v + ' = ' + str(varv)
                else:
                    if rest == 'shape':
                        vtxt += v + ' = ' + str(eval("varv" + "." + rest))
                    else:
                        try:
                            vtxt += v + "." + rest + ' = ' + json.dumps(eval("varv" + "." + rest), indent=4)
                        except:
                            vtxt += v + ' = ' + str(eval("varv" + "." + rest))
        except:
            vtxt += v + ' = ' + 'Could not find!'

    print(Fore.GREEN + datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]
          , Fore.BLUE + filename, line_number, function_name, Fore.BLACK + '|', Fore.RED + vtxt + Fore.RESET)


class timeit():
    def __enter__(self):
        self.tic = self.datetime.now()

    def __exit__(self, *args, **kwargs):
        print(Fore.GREEN + 'Runtime: {}'.format(self.datetime.now() - self.tic) + Fore.RESET)



def showSeq(normalize, step, caption, data, revert=False, oneD=False, dpi=1, save="",
            vmin=None, vmax=None, normType=matplotlib.colors.NoNorm(), verbose=True, show=True):
    if type(data) is not list:
        data = [data]
        
    for i in range(len(data)):
        if type(data[i]) is torch.Tensor:
            data[i] = data[i].detach().cpu().numpy()
    
    if len(data[0].shape)!=5:
        print("Wrong shape!")
        return
    
    B,S,C,H,W = data[0].shape
    L = len(data)
    
    if type(data[0]) == np.ndarray:
        for i in range(len(data)):
            SH = data[i].shape
            if type(data[i]) != np.ndarray:
                print("Not consistent data type")
                return
            elif (SH[0],SH[1],SH[3],SH[4])!=(B,S,H,W): 
                print("Not consistent dim")
                return
            else:
                data[i] = np.repeat(a=data[i], repeats = 4-data[i].shape[2],axis = 2) 
                if 3!=data[i].shape[2]:
                    print("Not consistent data channel shape")
                    return
        C=3
    else:
        print("Undefined Type Error")
        return
    

    if verbose and show:
        for i in range(len(data)):
            print("Data[", i, "]: min and max", data[i].min(), data[i].max())
    

    if (W == 1 or H == 1) and not oneD:
        longD = max(H,W)
        dimsqrt = int(math.sqrt(longD))
        if (dimsqrt * dimsqrt == longD):
            H=W=dimsqrt
            for i in range(len(data)):
                data[i] = data[i].reshape((B, S,C, H, W))
        else:
            print("Error while reshaping")
            return
        

    if (vmax == None and vmin == None):
        maxAbsVal = -1000000
        for i in range(len(data)):
            maxAbsVal = max(maxAbsVal, max(abs(data[i].max()), abs(data[i].min())))
            
        for i in range(len(data)):
            if normalize:
                data[i] = ((data[i] / maxAbsVal) / 2) + 0.5
            else:
                data[i] = ((data[i] / maxAbsVal))
    

    finalImg = np.stack(data,axis=1)
    padH = max(2,W//40);padW = max(2,H//40)
    finalImg = np.pad(finalImg, ((0,0),(0,0),(0,0),(0,0),(padH,padH),(padW,padW)), 'constant', constant_values=(1))
    finalImg = np.moveaxis(finalImg, (2), (4))
    finalImg = finalImg.reshape((B,L,C,padH+H+padH,-1))
    finalImg = np.moveaxis(finalImg, (2), (0))
    finalImg = finalImg.reshape((C,-1,(padW+W+padW)*S))

    if show:
        display(Markdown('<b><span style="color: #ff0000">' + caption + (" (N)" if normalize else "") +
                         (" (r)" if revert else "") + "</span></b>  " + (str(finalImg.shape) if verbose else "") + ''))

    cmap = 'viridis'

    if revert:
        if (vmax == None and vmin == None):
            if normalize:
                mid = 0.5
            else:
                mid = (finalImg.max() - finalImg.min()) / 2
            finalImg = ((finalImg - mid) * -1) + mid
        else:
            mid = (vmax - vmin) / 2
            finalImg = ((finalImg - mid) * -1) + mid
            
    dpi = 240. * dpi
    if verbose and show:
        print("Image: min and max", finalImg.min(), finalImg.max())

    if show or save != "":
        pltImg = np.moveaxis(finalImg, 0, -1)
        if show:
            plt.figure(figsize=None, dpi=dpi)
            plt.imshow(pltImg, cmap=cmap, norm=normType, vmin=vmin, vmax=vmax, aspect='equal')
            plt.axis('off')

            plt.show()
        if save != "":
            plt.imsave(save + "/" + caption + str(step) + '.png', pltImg, cmap=cmap)

    return [wandb.Image(torch.from_numpy(finalImg).cpu(), caption=caption)]



def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None)
                  for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None)
                  for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


def listToTensor(inp):
    return torch.stack(inp, dim=1)


def manyListToTensor(inp):
    res = []
    for i in inp:
        res.append(listToTensor(i))
    return res


def createarray(a1, a2, b1, b2, sp):
    return np.hstack((np.linspace(a1, a2, seq_length - sp, endpoint=False), np.linspace(b1, b2, sp, endpoint=False)))


def getItemIfList(inp, indx):
    if type(inp) == np.ndarray:
        return inp[indx]
    else:
        return inp



def showSingleImg(img, colorbar=True):
    if type(img) is torch.Tensor:
        img = img.detach().cpu().numpy()

    if type(img) is np.ndarray:
        if len(img.shape) > 3:
            raise Exception("showSingleImage is only for one image not batch!")
        elif len(img.shape) < 2:
            raise Exception("showSingleImage is for show 2D images!")
        elif len(img.shape) == 3:
            if img.shape[0] == 1:
                img = img[0, :, :]
            elif img.shape[0] == 3 or img.shape == 4:
                img = np.moveaxis(img, 0, -1)
            else:
                raise Exception("wrong number of channels!")
        else:
            pass
    else:
        raise Exception("Unknown input type!")
    plt.imshow(img)
    if colorbar:
        plt.colorbar()
    plt.show()


def visshift(data):
    for dim in range(len(data.size()) - 3, len(data.size()) - 1):
        data = roll_n(data, axis=dim, n=data.size(dim) // 2)
    return data

def justshift(data):
    for dim in range(len(data.size()) - 3, len(data.size()) - 1):
        data = roll_n(data, axis=dim, n=data.size(dim) // 2)
    return data




@torch.jit.script
def clmp(a, soft:bool=False,skew:float=10):
    if soft:
        return torch.sigmoid(-(skew/2.) + (a * skew)) 
    return a.clamp(0, 1)


def showComplex(norm, step, name, a, justRI=False, show=True, dpi=1):
    tmp = visshift(a)[li:ui].permute(0, 3, 1, 2).unsqueeze(1).cpu().detach()
    res = []
    res.append(showSeq(norm, step, name + " R&I " + str(step), [tmp[:, :, 0:1, :, :], tmp[:, :, 1:2, :, :]],
                       oneD=dimension == 1, revert=True, dpi=dpi, show=show)[0])
    if justRI:
        return
    angle = torch.atan2(tmp[:, :, 1:2, :, :], tmp[:, :, 0:1, :, :])
    absol = torch.sqrt(tmp[:, :, 1:2, :, :] * tmp[:, :, 1:2, :, :] + tmp[:, :, 0:1, :, :] * tmp[:, :, 0:1, :, :] + getEps())
    res.append(showSeq(norm, step, name + " Abs " + str(step), absol, oneD=dimension == 1, revert=True, dpi=dpi / 2.,
                       show=show)[0])
    res.append(showSeq(norm, step, name + " Angl " + str(step), angle, oneD=dimension == 1, revert=True, dpi=dpi / 2.,
                       show=show)[0])
    return res


def niceL2S(n, l):
    res = n + ": <br>"
    res += (pd.DataFrame(l.detach().cpu().numpy()).style.format("{0:.5f}")).render()
    return res


def showReal(norm, step, name, a, vmin=None, vmax=None, show=True, dpi=0.3):
    return showSeq(norm, step, name + " " + str(step), (a)[li:ui].unsqueeze(1).unsqueeze(1).cpu().detach(),
                   oneD=dimension == 1, dpi=dpi, revert=True, vmin=vmin, vmax=vmax, show=show)



@torch.jit.script
def getPhaseDiff(hf0, hf1,energy: bool= False) -> Tuple[torch.Tensor,Union[None, torch.Tensor]]:
    R = complex_mul(hf0, complex_conj(hf1))
    Energy = complex_abs(R)
    R = complex_div(R, Energy)
    if energy:
        return R,Energy[...,0:1]
    return R,None

@torch.jit.script
def getPhaseAdd(hf0, T):
    return complex_mul(hf0, T)


recIndex = None


def createPhaseDiff(avgRecX, avgRecY, shape):
    global recIndex

    BS, H, W, _ = shape

    angleX = avgRecX.unsqueeze(-1).unsqueeze(-1)
    angleY = avgRecY.unsqueeze(-1).unsqueeze(-1)

    reconstructIdx = False
    if recIndex is None:
        reconstructIdx = True
    else:
        if recIndex.shape[1] != H or recIndex.shape[2] != W:
            reconstructIdx = True

    if reconstructIdx:
        with torch.no_grad():
            recIndex = avgRecX.new_zeros((H, W, 2), requires_grad=False)
            xMid = recIndex.shape[0] // 2
            yMid = recIndex.shape[1] // 2
            for i in range(xMid, -1, -1):
                if i != xMid:
                    highI = (xMid - i) + xMid
                    lowI = i
                    if highI < recIndex.shape[0]:
                        recIndex[highI, yMid] = recIndex[highI - 1, yMid] + avgRecX.new_tensor([-1, 0])
                    recIndex[lowI, yMid] = recIndex[lowI + 1, yMid] + avgRecX.new_tensor([1, 0])

            for j in range(yMid, -1, -1):
                if j != yMid:
                    highJ = (yMid - j) + yMid
                    lowJ = j
                    if highJ < recIndex.shape[1]:
                        recIndex[:, highJ] = recIndex[:, highJ - 1] + avgRecX.new_tensor([0, -1]).unsqueeze(0).expand(
                            recIndex.shape[1], 2)
                    recIndex[:, lowJ] = recIndex[:, lowJ + 1] + avgRecX.new_tensor([0, 1]).unsqueeze(0).expand(
                        recIndex.shape[1], 2)

            recIndex = recIndex.unsqueeze(0)
            recIndex = justshift(recIndex)
    inter = (angleX * recIndex[:, :, :, 0] + angleY * recIndex[:, :, :, 1]).unsqueeze(-1)
    return torch.cat([torch.cos(inter), torch.sin(inter)], dim=-1)

@torch.jit.script
def getAvgDiff(rot, energy: Union[None,torch.Tensor]=None, step: int=1, axis: int=0, dims: Tuple[int,int] =(1, 2), untilIndex:Union[int,None] =None,variance:bool =False) -> Tuple[torch.Tensor,Union[None, torch.Tensor]]:
    assert (len(rot.shape) == 4)
    assert (dims == (1, 2))
    assert (axis == 0 or axis == 1)
    oAxis = 1 if axis == 0 else 0
    midIdx = (rot.size(dims[oAxis]) - 1) // 2
    if untilIndex is not None:
        untilIndex = max(2, untilIndex)
        rot = rot[:, 0:untilIndex, 0:untilIndex, :]
        if energy is not None:
            energy = energy[:, 0:untilIndex, 0:untilIndex, :]

    a,_ = getPhaseDiff(rot, rot.roll(-step, dims=dims[axis]))
    
    if energy is not None:
        energy = (energy) / (energy.sum(dim=list(dims)+[-1]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + getEps())
        
    if energy is not None:
        am = (a*energy).sum(dim=dims[oAxis])
    else:
        am = a.mean(dim=dims[oAxis])

    removeMid = False
    if untilIndex is not None: 
        am = am[:, 0:-1, :]
        if untilIndex > midIdx:
            removeMid = True
    else:
        removeMid = True
    if removeMid:
        amm = torch.cat([am[:, :midIdx, :], am[:, midIdx + 1:, :]],
                        dim=1)  
    else:
        amm = am

    
    if energy is not None:
        res= amm.sum(dim=1)
    else:
        res= amm.mean(dim=1)

    if variance:
        if energy is not None:
            varam = (((a-res.unsqueeze(-2).unsqueeze(-2))**2)*energy).sum(dim=dims[oAxis]).sum(dim=1).sum(dim=-1).unsqueeze(-1) 
        else:
            varam = a.var(dim=[-3,-2]).sum(dim=-1).unsqueeze(-1) 
    else:
        varam = None
        
    return res,varam


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


from IPython.display import Image
from IPython.display import Markdown


def _prepare_pytorch(x):
    if isinstance(x, torch.autograd.Variable):
        x = x.data
        x = x.cpu().numpy()
        return x


def make_np(x):
    if isinstance(x, np.ndarray):
        return x
    if np.isscalar(x):
        return np.array([x])
    if isinstance(x, torch.Tensor):
        return _prepare_pytorch(x)
    raise NotImplementedError(
        'Got {}, but numpy array, torch tensor, or caffe2 blob name are expected.'.format(type(x)))


def _prepare_video(V):
    b, t, c, h, w = V.shape

    if V.dtype == np.uint8:
        V = np.float32(V) / 255.

    def is_power2(num):
        return num != 0 and ((num & (num - 1)) == 0)


    if not is_power2(V.shape[0]):
        len_addition = int(2 ** V.shape[0].bit_length() - V.shape[0])
        V = np.concatenate(
            (V, np.zeros(shape=(len_addition, t, c, h, w))), axis=0)

    n_rows = 2 ** ((b.bit_length() - 1) // 2)
    n_cols = V.shape[0] // n_rows

    V = np.reshape(V, newshape=(n_rows, n_cols, t, c, h, w))
    V = np.transpose(V, axes=(2, 0, 4, 1, 5, 3))
    V = np.reshape(V, newshape=(t, n_rows * h, n_cols * w, c))

    return V


def _calc_scale_factor(tensor):
    converted = tensor.numpy() if not isinstance(tensor, np.ndarray) else tensor
    return 1 if converted.dtype == np.uint8 else 255



def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def make_video(tensor, fps):
    try:
        import moviepy  
    except ImportError:
        print('add_video needs package moviepy')
        return
    try:
        from moviepy import editor as mpy
    except ImportError:
        print("moviepy is installed, but can't import moviepy.editor.",
              "Some packages could be missing [imageio, requests]")
        return
    import tempfile

    t, h, w, c = tensor.shape


    clip = mpy.ImageSequenceClip(list(tensor), fps=fps)

    filename = tempfile.NamedTemporaryFile(suffix='.gif', delete=False).name
    try:  
        clip.write_gif(filename, verbose=False, logger=None)
    except TypeError:
        try:  
            clip.write_gif(filename, verbose=False, progress_bar=False)
        except TypeError:
            clip.write_gif(filename, verbose=False)

    with open(filename, 'rb') as f:
        tensor_string = f.read()

    junckFiles.append(filename)

    return tensor_string


def video(tensor, fps=10):
    tensor = make_np(tensor)
    tensor = _prepare_video(tensor)

    scale_factor = _calc_scale_factor(tensor)
    tensor = tensor.astype(np.float32)
    tensor = (tensor * scale_factor).astype(np.uint8)
    video = make_video(tensor, fps)
    return video


def showMultiVideo(tags, tensors, fps=10):
    display(Markdown('Tag: <span style="color: #ff0000; font-size: 12pt">' + tags + '</span>'))
    newList = []
    for t in tensors:
        newList.append(t)
        newList.append(torch.ones_like(t)[..., 0:1])
    tensor = torch.cat(newList, dim=-1)
    return display(Image(data=video(1 - tensor.clamp(0, 1)), format='gif', width=900))


def showVideo(tag, tensor, fps=10):
    display(Markdown('Tag: <span style="color: #ff0000; font-size: 12pt">' + tag + '</span>'))
    return display(Image(data=video(1 - tensor.clamp(0, 1)), format='gif', width=200))


def log_npList(figs, fps, vis_anim_string="animations"):
    tensor = torch.cat(figs, dim=1)
    tensor = make_np(tensor)
    tensor = _prepare_video(tensor)

    scale_factor = _calc_scale_factor(tensor)
    tensor = tensor.astype(np.float32)
    tensor = (tensor * scale_factor).astype(np.uint8)

    return log_video(tensor, fps=7, vis_anim_string=vis_anim_string)


junckFiles = []


def clean_junk_files():
    global junckFiles
    for j in junckFiles:
        try:
            os.remove(j)  
        except OSError:
            logging.warning('The temporary file used by moviepy cannot be deleted.')
    junckFiles = []


def log_video(tensor, fps, vis_anim_string='animations'):
    global junckFiles
    try:
        import moviepy  
    except ImportError:
        print('add_video needs package moviepy')
        return
    try:
        from moviepy import editor as mpy
    except ImportError:
        print("moviepy is installed, but can't import moviepy.editor.",
              "Some packages could be missing [imageio, requests]")
        return
    import tempfile

    t, h, w, c = tensor.shape


    clip = mpy.ImageSequenceClip(list(tensor), fps=fps)

    filename = tempfile.NamedTemporaryFile(suffix='.gif', delete=False).name
    try: 
        clip.write_gif(filename, verbose=False, logger=None)
    except TypeError:
        try: 
            clip.write_gif(filename, verbose=False, progress_bar=False)
        except TypeError:
            clip.write_gif(filename, verbose=False)

    tensor_video = wandb.Video(filename, fps=fps, format='gif')
    wandblog({vis_anim_string: tensor_video}, commit=False)

    junckFiles.append(filename)

    return None


def logMultiVideo(tags, tensors,seed, fps=10, vis_anim_string='animations'):
    newList = []
    for i,t in enumerate(tensors):
        sh = t.shape
        if sh[-3]==1:
            sh = [-1]*len(sh)
            sh[-3]=3
            t = t.expand(sh)
        newList.append(t)
        if i<len(tensors)-1:
            b = torch.zeros_like(t)[..., 0:1]
            b[...,seed:,0,:,:] = 1
            b[...,:seed,1,:,:] = 1
            newList.append(b)
    tensor = torch.cat(newList, dim=-1)
    tensor = 1 - tensor.clamp(0, 1)
    tensor = make_np(tensor)
    tensor = _prepare_video(tensor)
  
    scale_factor = _calc_scale_factor(tensor)
    tensor = tensor.astype(np.float32)
    tensor = (tensor * scale_factor).astype(np.uint8) 

    return log_video(tensor, fps, vis_anim_string)

@torch.jit.script
def getDistWindow(windowSize: int):
    c = windowSize // 2
    indxT, indyT = torch.meshgrid([torch.arange(windowSize).float(), torch.arange(windowSize).float()],indexing='xy')
    distMat = torch.sqrt((c - indxT) * (c - indxT) + (c - indyT) * (c - indyT))
    return distMat

@torch.jit.script
def getFlatTopWindow(windowSize: int,
                     a_flat=torch.tensor([0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368])):
    distMat = getDistWindow(windowSize)
    adjuster = torch.tensor(2 * (windowSize) ** 2).float().to(a_flat.device)
    adjuster = torch.sqrt(adjuster)
    distMatAdjust = distMat + adjuster / 2
    if a_flat.is_cuda:
        distMatSend = (distMatAdjust / adjuster).cuda()
    else:
        distMatSend = distMatAdjust / adjuster

    return a_flat[0] - a_flat[1] * torch.cos(2 * torch.pi * distMatSend) + a_flat[2] * torch.cos(
        4 * torch.pi * distMatSend) \
           - a_flat[3] * torch.cos(6 * torch.pi * distMatSend) + a_flat[4] * torch.cos(8 * torch.pi * distMatSend)

@torch.jit.script
def gH(x, N, sigma):
    return torch.exp(-(x - 0.5 * N) ** 2 / (2 * (N + 1) * sigma) ** 2)

@torch.jit.script
def confinedGaussian(r, windowSize, sigma):
    const1 = gH(torch.tensor(-0.5).to(sigma.device), windowSize, sigma)
    const2 = gH(torch.tensor(0.5).to(sigma.device) + windowSize, windowSize, sigma)
    const3 = gH(torch.tensor(-1.5).to(sigma.device) - windowSize, windowSize, sigma)
    denom = gH(r + windowSize + 1, windowSize, sigma) + gH(r - windowSize - 1, windowSize, sigma)
    return gH(r, windowSize, sigma) - const1 * denom / (const2 + const3)

@torch.jit.script
def getGaussianWindow(windowSize, sigma):
    distMat = getDistWindow(windowSize)

    wsAdjust = math.sqrt(windowSize ** 2 * 2)

    distMatSend = distMat.to(sigma.device)

    return confinedGaussian(distMatSend + wsAdjust / 2, wsAdjust, sigma)



class CombinedLossDiscounted(torch.nn.Module):
    def __init__(self, alpha=0.5, window_size=9, gamma=0.9):
        super(CombinedLossDiscounted, self).__init__()
        self.alpha = alpha
        self.window_size = window_size
        self.loss_function_structural = kornia.losses.SSIMLoss(window_size, reduction='none')
        self.loss_function_pixel_wise = nn.L1Loss(reduction='none')
        self.gamma = 0.9

    def forward(self, x, y):
        lGains = [self.gamma ** i for i in range(1, x.shape[-3] + 1, 1)]
        lGains = torch.tensor([i / sum(lGains) for i in lGains], device=x.device)
        out = (1 - self.alpha) * self.loss_function_pixel_wise(x, y).mean(-1).mean(
            -1) + self.alpha * self.loss_function_structural(x, y).mean(-1).mean(-1)
        return (out @ lGains).mean()


def generate_name():
    return datetime.now().strftime("%d-%m-%Y_%H_%M")


def make_colorwheel():

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0


    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY

    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG

    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC

    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB

    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM

    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75 
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def show_phase_diff(pd_list, gt, config, title,clear_motion=False,gtOrig=None):
    if config.useGlobalLFT and config.showArrowInGOnCom and gtOrig is not None:
        gtOrig=gtOrig[0].cpu()
        denom = torch.sum(gtOrig, dim=(-2, -1))
        xMesh, yMesh = torch.meshgrid(torch.arange(0, gtOrig.shape[-2]), torch.arange(0, gtOrig.shape[-1]),indexing='xy')
        COMX = torch.sum(gtOrig * xMesh, dim=(-2, -1)) / (denom + 1e-8)
        COMY = torch.sum(gtOrig * yMesh, dim=(-2, -1)) / (denom + 1e-8)

    N_prime = config.window_size + 2 * config.window_padding_constrained

    color_like_flow=False
    gt = gt.permute(0,1,3,4,2)
    offset = config.image_pad_size_constrained - (config.window_size + 2 * config.window_padding_constrained) // 2
    figs = []
    showStillVisualization = False
    if showStillVisualization:
        figT, axT = plt.subplots(1, config.sequence_length, figsize=(30, 3), dpi=360)
    for i in range(config.sequence_length):
        img = gt[0, i].detach().cpu()
        if config.showRField:
            img[0:config.window_size ,0]=0.5
            img[0,0:config.window_size ]=0.5
            img[-config.window_size :,-1]=0.5
            img[-1,-config.window_size :]=0.5

        cmap = None
        if img.shape[2]==1:
            img = img[:,:,0]
            cmap='gray'
        fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=360)
        ax.set_yticks([])
        ax.set_xticks([])
        if showStillVisualization:
            axT[i].set_yticks([])
            axT[i].set_xticks([])
        if i == 0:
            ax.imshow(img, cmap=cmap,vmin=0,vmax=1)
            if showStillVisualization:
                axT[i].imshow(img, cmap=cmap,vmin=0,vmax=1)
            with torch.no_grad():
                visY, visX,_,_,_ = pd_list[0]


            L_y = visY.shape[-3]
            L_x = visY.shape[-2]
            mmx = np.arange(0 - offset, config.stride * L_x - offset, config.stride) + 1
            mmy = np.arange(0 - offset, config.stride * L_y - offset, config.stride) + 1
            xx, yy = np.meshgrid(mmx, mmy, sparse=True) 
            ax.quiver(xx, yy, visX[0,:,:,0].cpu(), visY[0,:,:,0].cpu(), alpha=0.2)
            if showStillVisualization:
                axT[i].quiver(xx, yy, visX[0,:,:,0].cpu(), visY[0,:,:,0].cpu(), alpha=0.2)
        else:
            with torch.no_grad():
                visY, visX,energy,varX,varY = pd_list[i - 1]
                if energy is not None:
                    energy=energy/energy.sum(dim=[1,2]).unsqueeze(-2).unsqueeze(-2)
                else:
                    energy=torch.ones_like(visX)

                if color_like_flow:
                    colors = flow_uv_to_colors(visX[0].detach().cpu().numpy(), -visY[0].detach().cpu().numpy()).reshape((-1,3))	
                    colors = colors / 255.
                if clear_motion:
                    visX=torch.zeros_like(visX)
                    visY=torch.zeros_like(visY)
                    
            ax.imshow(img, cmap=cmap,vmin=0,vmax=1)
            if showStillVisualization:
                axT[i].imshow(img, cmap=cmap,vmin=0,vmax=1)

            chsum = (abs(visX[0,:,:,:])+abs(visY[0,:,:,:])).sum(dim=[0,1])
            for chan in range(visY.shape[-1]):
                if not color_like_flow or visY.shape[-1]>1:
                    colors=1-randColors[:,chan].numpy()
                    colors = np.array(list(colors))



                xxx=xx.copy()
                yyy=yy.copy()
                e=energy[0,:,:,chan].cpu()>0.005
                if config.showArrowInGOnCom and len(xx)==1 and len(xx[0])==1 and config.useGlobalLFT and gtOrig is not None:
                    xxx[0][0]=int(COMX[i,chan].item())
                    yyy[0][0]=int(COMY[i,chan].item())
                    colors = np.array(list(colors[:3])+[1])
                    colors[:-1]*=0.85
                    if gtOrig[i,chan,:,:].sum()<3 or gtOrig[i,chan,:,:].max()<0.08:
                        colors[-1]=0

                
                ax.quiver(xxx, yyy, visX[0,:,:,chan].cpu()*e, visY[0,:,:,chan].cpu()*e, color=colors, scale=1/config.ArrowScale,
                        units='xy',scale_units='xy')  
                if showStillVisualization:
                    axT[i].quiver(xxx, yyy, visX[0,:,:,chan].cpu()*e, visY[0,:,:,chan].cpu()*e, color=colors, scale=1/config.ArrowScale,
                                units='xy',scale_units='xy') 
            figs.append(torch.from_numpy(get_img_from_fig(fig)).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).clone())

    log_npList(figs, fps=7, vis_anim_string="GIF " + title)
    if showStillVisualization:
        wandblog({title: wandb.Image(figT, caption="T superimposed on GT \n" + "max_y = " + str(
            visY.abs().max()) + " max_x = " + str(visX.abs().max()))})
    plt.close('all')
    return None


def show_hist(data,bins='auto',dis=False,title=""):
    sns.set_theme()
    fig = plt.figure(figsize=(10,5), dpi=300)
    ax = fig.add_subplot(111)
    ax = sns.histplot(data,stat='probability',discrete=dis,bins=bins,kde=False,multiple="dodge", shrink=.8,common_bins=False)
    wandblog({"histogram "+title:  wandb.Image(fig, caption="C histogram "+title)})
    plt.close('all')
    sns.reset_orig()




def positionalencoding2d(d_model, height, width):
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)

    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe



class DenseNetLikeModel(nn.Module):
    def __init__(self, inputC,outputC, hiddenF=24, filterS=3,gain=1,nonlin='ReLU',lastNonLin=False,initWIdentity=True,bn=True):
        super().__init__()
        self.layerC = len(filterS)
        self.convS = []
        for c in range(self.layerC):
            filt = filterS[c]
            outc = hiddenF if c < (self.layerC - 1) else outputC
            seq = [
                nn.Conv2d(in_channels=inputC + (c * hiddenF),
                                        out_channels=outc,
                                        kernel_size=filt, stride=1, padding=filt//2)
            ]
            if bn and (c < (self.layerC - 1)):
                seq.append(nn.BatchNorm2d(outc))
            self.convS.append(nn.Sequential(*seq))
            if initWIdentity:
                with torch.no_grad():
                    self.convS[-1][0].weight.data *= 0.01
                    mid = self.convS[-1][0].weight.shape[2] // 2
                    self.convS[-1][0].weight.data[:, :, mid, mid] = (1. / (self.convS[-1][0].in_channels))*gain
                    self.convS[-1][0].bias.data *= 0.01
        self.convS = nn.ModuleList(self.convS)
        
        self.nonLin = eval('torch.nn.'+nonlin+"()")
        self.lastNonLin = lastNonLin
    def forward(self, inp):
        res = []
        for c in range(self.layerC):
            toAppend =  self.convS[c](torch.cat(res + [inp], dim=1))
            if self.lastNonLin or not c==self.layerC-1:
                toAppend = self.nonLin(toAppend)
            res.append(toAppend)

        return res[-1]


def double_conv(in_channels, out_channels,full=True):
    lay = [
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if full:
        lay+=[
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
    return nn.Sequential(*lay)


class UNet(nn.Module):

    def __init__(self, in_ch,out_ch,full=True,hd=16,pad=0,lastNonLin=True):
        super().__init__()
                

        self.dconv_down1 = double_conv(in_ch, hd,full)
        self.dconv_down2 = double_conv(hd, hd*2,full)
        self.dconv_down3 = double_conv(hd*2, hd*4,full)
        self.dconv_down4 = double_conv(hd*4, hd*8,full)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(hd*4 + hd*8, hd*4,full)
        self.dconv_up2 = double_conv(hd*2 + hd*4, hd*2,full)
        self.dconv_up1 = double_conv(hd*2 + hd, hd,full)
        
        self.conv_last = nn.Conv2d(hd, out_ch, 1)
        if lastNonLin:
            self.sig = torch.nn.Sigmoid()

        self.pad=pad
        self.lastNonLin=lastNonLin
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)

        if self.pad>0:
            x = torch.nn.functional.pad(x, (0,self.pad,0,self.pad), mode='constant', value=0)

        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)

        if not self.lastNonLin:
            return out
        return self.sig(out)
