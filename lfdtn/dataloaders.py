import random
import torch
import cv2
import json
import os
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from asset.utils import dmsg,getIfAugmentData
import imgaug.augmenters as iaa
import numpy as np
from tqdm import tqdm
import random as rand
from torchvision import datasets, transforms
from scipy import ndimage
from scipy.stats import multivariate_normal
import dill
import math
import pickle

from imgaug.augmentables import Keypoint, KeypointsOnImage

class PropMMNISTDataset(Dataset):

    def __init__(self, infrencePhase, seqLength, shapeX, shapeY,digitCount = 1, scale=2, foregroundScale=0.7, blurIt=True,
                 minResultSpeed=0, maxResultSpeed=2,color=False,blob=True,changeID=5):
        super(PropMMNISTDataset).__init__()
        if digitCount>2:
            raise BaseException("Too much FG requested!")
        self.shapeXOrig = shapeX
        self.shapeYOrig = shapeY
        self.seqLength = seqLength
        self.blurIt = blurIt
        self.minResultSpeed = minResultSpeed
        self.maxResultSpeed = maxResultSpeed
        self.foregroundScale = foregroundScale
        self.digitCount = digitCount
        self.scale = int(scale)
        self.shapeX = int(shapeX * scale)
        self.shapeY = int(shapeY * scale)
        self.color = color
        self.blob = blob
        self.changeID=changeID
        self.MNIST = datasets.MNIST('data', train=not infrencePhase, download=True)

    def _scaleBlur(self, arry):
        if (self.blurIt):
            arry = cv2.blur(arry, (self.scale, self.scale))

        if self.scale != 1:
            arry = cv2.resize(arry, (self.shapeYOrig, self.shapeXOrig), interpolation=cv2.INTER_NEAREST)# cv2.resize wants [shape[1],shape[0]]
            if not self.color:
                arry = arry[:,:,np.newaxis] 
        
        return np.clip(arry, a_min=0, a_max=1)

    def _blob(self, arry):
        if not self.blob:
            return arry
        center = ndimage.measurements.center_of_mass(arry[:,:,0])
        radius=arry.shape[0]*0.17
        idxs = np.meshgrid(np.arange(0, arry.shape[0]), np.arange(0, arry.shape[1]))
        idxs = np.array(idxs).T.reshape(-1,2)
        dist = multivariate_normal.pdf(idxs, center, [radius, radius])


        dist = (dist - dist.min()) / (dist.max() - dist.min())

        return dist.reshape(arry.shape)



    def _cImg(self, image, scale, original=False):
        if original == True:
            return image

        res = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)),
                         interpolation=cv2.INTER_AREA) 
        if not self.color:
            res = res[:,:,np.newaxis] 

        return res

    def __len__(self):
        return len(self.MNIST)

    def __getitem__(self, idx):
        foreground_objs = []
        random.seed(idx)
        np.random.seed(idx)
        digitY = []
        for _ in range(self.digitCount):
            firstL =True
            dy = -1
            while dy in digitY or firstL:
                mnistdig,dy=self.MNIST.__getitem__(random.randint(0, len(self.MNIST) - 1))
                firstL = False
                    
            
            mnistdig = np.array(mnistdig)[ :, :,np.newaxis]
            if self.color:
                mnistdig = np.repeat(mnistdig, 3, axis=2)
                randC = random.randint(0,2)
                mnistdig[:,:,randC] = random.randint(0,100)
            mnistdig = self._cImg(np.moveaxis(mnistdig, 0, 1) / 255., self.scale * self.foregroundScale, False)
            foreground_objs.append(mnistdig)
            digitY.append(dy)
            

        
        shapeX2 = self.shapeX // 2
        shapeY2 = self.shapeY // 2
        MINPOS = 2
        MAXPOSX = self.shapeX - MINPOS - foreground_objs[0].shape[1]
        MAXPOSY = self.shapeY - MINPOS - foreground_objs[0].shape[0]

        if not self.color:
            bg = np.zeros([self.shapeY, self.shapeX,1])
        else:
            bg = np.zeros([self.shapeY, self.shapeX,3])



        refChoice=[[1,1],[1,-1],[-1,-1],[-1,1]]

        shouldRedo=True
        while shouldRedo:
            shouldRedo=False

            positions = np.array([[MAXPOSX//2,MAXPOSY//2],[MAXPOSX//2,MAXPOSY//2]])

            velocities = np.random.randint(low=int(self.minResultSpeed * self.scale),
                                        high=(self.maxResultSpeed * self.scale) + 1, size=(2, 2))
                                        
            velocities = velocities*np.random.choice([-1,1],size=(2, 2))


            channel = 3 if self.color else 1
            ResFrame = np.empty((self.seqLength, self.shapeXOrig, self.shapeYOrig,channel), dtype=np.float32)
            ResFrameFG = np.zeros((self.seqLength, self.shapeXOrig, self.shapeYOrig,channel,10), dtype=np.float32)
            ResFrameAlpha = np.empty((self.seqLength, self.shapeXOrig, self.shapeYOrig,channel), dtype=np.float32)
            ResFrameBG = np.empty((self.seqLength, self.shapeXOrig, self.shapeYOrig,channel), dtype=np.float32)

            for frame_idx in range(self.seqLength):

                if frame_idx==self.changeID:
                    flipVIdx = int(np.random.choice([idxr for idxr in range(len(refChoice))],
                                        p=[1/len(refChoice) for idxr in range(len(refChoice))]))
                    flipVDescrete = torch.nn.functional.one_hot(torch.tensor(flipVIdx),len(refChoice)).float()
                    flipV=refChoice[flipVIdx]
                    for ax in range(self.digitCount):
                        for dimen in range(2):
                            velocities[ax,dimen] *= flipV[dimen]


                frame = np.zeros((self.shapeX, self.shapeY,channel), dtype=np.float32)
                frameFG = np.zeros((self.shapeX, self.shapeY,channel,self.digitCount), dtype=np.float32)
                frameAlpha = np.zeros((self.shapeX, self.shapeY,channel), dtype=np.float32)
                frameBG = np.zeros((self.shapeX, self.shapeY,channel), dtype=np.float32)
                frameBG = bg
                frame += bg
                ptmp=positions.copy()
                for ax in range(self.digitCount):
                    ptmp[ax] += velocities[ax]
                    for dimen in range(2):
                        if ptmp[ax,dimen] < 0:

                            shouldRedo=True
                            break
                        if ptmp[ax,dimen] > self.shapeX - foreground_objs[ax].shape[dimen]:

                            shouldRedo=True
                            break

                    if shouldRedo:
                        break
                            
                    positions[ax]+= velocities[ax]

                    IN = [positions[ax][0],
                        positions[ax][0]
                        + foreground_objs[ax].shape[0],
                        positions[ax][1],
                        positions[ax][1]
                        + foreground_objs[ax].shape[1]]

                    frame[IN[0]:IN[1], IN[2]:IN[3],:] += foreground_objs[ax]
                    frameFG[IN[0]:IN[1], IN[2]:IN[3],:,ax] = foreground_objs[ax]
                    frameAlpha[IN[0]:IN[1], IN[2]:IN[3],:] = np.ones_like(foreground_objs[ax])

                if shouldRedo:
                    break

                ResFrame[frame_idx] = self._scaleBlur(frame)
                for ax in range(self.digitCount):

                    ResFrameFG[frame_idx,:,:,:,digitY[ax]] = self._blob(self._scaleBlur(frameFG[...,ax]))
                ResFrameAlpha[frame_idx] = self._scaleBlur(frameAlpha)
                ResFrameBG[frame_idx] = self._scaleBlur(frameBG)
                del frame, frameFG, frameAlpha, frameBG

            if shouldRedo:
                continue

            ResFrame = np.moveaxis(ResFrame, [0,1,2,3] , [0,3,2,1])
            ResFrameAlpha = np.moveaxis(ResFrameAlpha, [0,1,2,3] , [0,3,2,1])
            ResFrameBG = np.moveaxis(ResFrameBG, [0,1,2,3] , [0,3,2,1])
            ResFrameFG = np.moveaxis(ResFrameFG, [0,1,2,3,4] , [0,4,3,2,1])

            result = {'GT': torch.from_numpy(ResFrame), 'A': torch.from_numpy(ResFrameAlpha), 'BG': torch.from_numpy(ResFrameBG), 'FG': torch.from_numpy(ResFrameFG),
                        "velocity": torch.tensor(velocities / self.scale)}

            return torch.from_numpy(ResFrame),flipV,flipVDescrete

sequenceFileDict = {
    'lin_3': 'canvas_down_sample_just_translation_3_18_06.pt',
    'rot_lin_2': 'canvas_down_sample_just_rotation_15_06.pt',
    'lin_1': 'canvas_down_sample_just_translation_15_06.pt',
    'challenge': 'canvas_down_sample_extreme_16_06.pt',
    'cars': 'car_sequence_257.pt',
    'small_cars': 'car_sequence_downsampled.pt',
    'more_cars': 'car_sequence_downsampled_long.pt',
    'stationary_rotation': 'canvas_down_sample_inplace_rotation_2_22_06.pt',
    'lin_2': 'canvas_down_sample_just_translation_2_21_06.pt',
    'acc_1': 'canvas_down_sample_with_acc_16_06.pt',
    'rot_lin_scale_2': 'canvas_down_sample_everything_2_24_06.pt',
    'all_cars': 'carData_inv_01_07.pt',
    'random_cars': 'carData_inv_permuted_01_07.pt',
    'augmented_cars': 'augmented_cars.pt',
    'rot_lin_2_NOSCALE': 'canvas_down_sample_no_scale_12_07.pt',
    'high_res_test': 'canvas_07_08.pt',
    'high_res_test_3': 'canvas_3_07_08.pt',
    'circle_sanity_1_px': 'canvas_circle_int_up.pt',
    'circle_sanity_2_px': 'canvas_circle_int_up_2.pt',
    'planets_3': 'planets_3.pt'
}


class sequenceData(Dataset):
    def __init__(self, config,key='rot_lin_scale_2', device=torch.device('cpu'), size=(65, 65), sequence_length=10,color=False):
        self.key = key
        self.dict = sequenceFileDict
        if key in self.dict:
            self.filePath = 'data/pickled_ds/' + self.dict[key]
        else:
            self.filePath = 'data/pickled_ds/' + key+'.pt'
        self.data = torch.load(self.filePath).float()
        self.len = self.data.shape[0]
        self.dev = device
        self.size = size[::-1]
        self.sequence_length = sequence_length
        self.color = color
        self.config =config
    def __getitem__(self, ind):
        if len(self.data.shape)==4:
            data = self.data[ind].unsqueeze(1).to(self.dev)
        else:
            data = self.data[ind].to(self.dev)
        res = torch.nn.functional.interpolate(data, size=self.size,
                                        mode='bilinear', align_corners=False)[:self.sequence_length]
        if res.shape[1] !=self.config.input_channels:
            if res.shape[1]==1 and self.color:
                res = res.expand(res.shape[0],3,res.shape[2],res.shape[3])
            elif res.shape[1]==3 and not self.color:
                res,_ = res.max(dim=1)
                res=res.unsqueeze(1)
        return res
    def __len__(self):
        return self.len

class savedData(Dataset):
    def __init__(self,data,device=torch.device('cpu'), size=(65, 65),limit=1):
        self.data = data
        self.limit=limit
        self.dev = device
        self.size = size[::-1]
    def reshape(self,inp):
        if type(inp)!=tuple and type(inp)!=list:
            inp=[inp]
        shp=inp[0].shape
        if len(shp)>4 and (shp[-2]!=self.size[0] or shp[-1]!=self.size[1]):
            inp[0] = torch.nn.functional.interpolate(inp[0].reshape(-1,1,shp[-2],shp[-1]), size=self.size,
                                            mode='bilinear', align_corners=False).reshape(shp[:-2]+self.size)
        elif len(shp)>3 and (shp[-2]!=self.size[0] or shp[-1]!=self.size[1]):
            inp[0] = torch.nn.functional.interpolate(inp[0], size=self.size,
                                            mode='bilinear', align_corners=False)
        return inp
            
    def __getitem__(self, ind):
        res = self.data[ind]
        if res is dict:
            for it in res.keys():
                res[it] = self.reshape(res[it])
            return res
        else:
            return self.reshape(res)

    def __len__(self):
        return int(len(self.data)*self.limit)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def get_data_loaders(config, key='rot_lin_scale_2', size=(65, 65), ratio=[0.1,0.1], batch_size=1,test_batch_size=1, num_workers=1,
                     device=torch.device('cpu'), limit=1, sequence_length=10):
    if config.load_ds_from_saved:
        try:
            tr_ds,val_ds, test_ds = torch.load('data/savedDS/' + key+ '.pt')
            train_loader = DataLoader(savedData(tr_ds,device, size,limit), batch_size=batch_size, num_workers=num_workers,
                                    pin_memory=True, shuffle=True,worker_init_fn=worker_init_fn)
            valid_loader = DataLoader(savedData(val_ds,device, size,limit), batch_size=test_batch_size, num_workers=num_workers,
                                    pin_memory=False, shuffle=False,worker_init_fn=worker_init_fn)
            test_loader = DataLoader(savedData(test_ds,device, size,limit), batch_size=test_batch_size, num_workers=num_workers,
                                    pin_memory=False, shuffle=False,worker_init_fn=worker_init_fn)
            print('Loaded From: ' + 'data/savedDS/'+ key+ '.pt')
            return train_loader,valid_loader, test_loader
        except Exception as ex:
            print(ex)
            print('Could not load From: ' + 'data/savedDS/' + key+ '.pt')
            print("load live... (key: {})".format(key))
            pass #Go and load it live

    randomSeed = 123
    torch.manual_seed(randomSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(randomSeed)
    rand.seed(randomSeed)
    np.random.seed(randomSeed)

    if key == 'PropMMNIST':
        dataset = PropMMNISTDataset(infrencePhase=False, seqLength=sequence_length, shapeX=size[0], shapeY=size[1]
                ,digitCount = 1, scale=2, foregroundScale=0.6, blurIt=True,
                 minResultSpeed=1, maxResultSpeed=3,color=config.input_channels==3,blob=True,changeID=config.sequence_seed)
        dlen = len(dataset)
        splitTr = int((1-(ratio[0]))*dlen)
        tr_ds,val_ds=torch.utils.data.random_split(dataset,
                                    [splitTr, dlen-(splitTr)], 
                                    generator=torch.Generator().manual_seed(randomSeed))
        tr_ds = torch.utils.data.Subset(tr_ds, range(0,int(len(tr_ds)*limit)))
        val_ds = torch.utils.data.Subset(val_ds, range(0,int(len(val_ds)*limit)))
        train_loader = DataLoader(tr_ds, batch_size=batch_size,pin_memory=config.pin_mem, num_workers=num_workers, shuffle=True,worker_init_fn=worker_init_fn)
        valid_loader = DataLoader(val_ds, batch_size=batch_size,pin_memory=False, num_workers=num_workers, shuffle=False,worker_init_fn=worker_init_fn)

        dataset = PropMMNISTDataset(infrencePhase=True, seqLength=sequence_length, shapeX=size[0], shapeY=size[1]
                ,digitCount = 1, scale=2, foregroundScale=0.6, blurIt=True,
                 minResultSpeed=1, maxResultSpeed=3,color=config.input_channels==3,blob=True,changeID=config.sequence_seed)
        dlen = len(dataset)
        limitedDlen = int(dlen * limit)
        te_ds,_=torch.utils.data.random_split(dataset,
                                    [limitedDlen, dlen-(limitedDlen)], 
                                    generator=torch.Generator().manual_seed(randomSeed))
        test_loader = DataLoader(te_ds, batch_size=test_batch_size,pin_memory=False, num_workers=num_workers, shuffle=False,worker_init_fn=worker_init_fn)
    elif key == 'Skeleton':
        datasetTr = Skeleton(config,config.ds_subjects, fCount=sequence_length,
                        size=size)
        datasetTe = Skeleton(config,[9,11], fCount=sequence_length,
                        size=size)
        dlen = len(datasetTe)
        splitTe = int((ratio[1]/(ratio[0]+ratio[1]))*dlen)
        test_ds,val_ds=torch.utils.data.random_split(datasetTe,
                                    [splitTe, dlen-(splitTe)], 
                                    generator=torch.Generator().manual_seed(randomSeed))
        tr_ds = torch.utils.data.Subset(datasetTr, range(0,int(len(datasetTr)*limit)))
        val_ds = torch.utils.data.Subset(val_ds, range(0,int(len(val_ds)*limit)))
        test_ds = torch.utils.data.Subset(test_ds, range(0,int(len(test_ds)*limit)))
        train_loader = DataLoader(tr_ds, batch_size=batch_size, num_workers=num_workers,
                                  pin_memory=config.pin_mem, shuffle=True,worker_init_fn=worker_init_fn)
        valid_loader = DataLoader(val_ds, batch_size=test_batch_size, num_workers=num_workers,
                                  pin_memory=False, shuffle=False,worker_init_fn=worker_init_fn)
        test_loader = DataLoader(test_ds, batch_size=test_batch_size, num_workers=num_workers,
                                  pin_memory=False, shuffle=False,worker_init_fn=worker_init_fn)
    else:
        dataset = sequenceData(config,key=key, device=device, size=size, sequence_length=sequence_length,color=config.input_channels==3)
        dlen = len(dataset)
        splitTr = int((1-(ratio[0]+ratio[1]))*dlen)
        splitVa = int(ratio[0]*dlen)
        tr_ds,val_ds,test_ds=torch.utils.data.random_split(dataset,
                                    [splitTr,splitVa, dlen-(splitTr+splitVa)], 
                                    generator=torch.Generator().manual_seed(randomSeed))

        tr_ds = torch.utils.data.Subset(tr_ds, range(0,int(len(tr_ds)*limit)))
        val_ds = torch.utils.data.Subset(val_ds, range(0,int(len(val_ds)*limit)))
        test_ds = torch.utils.data.Subset(test_ds, range(0,int(len(test_ds)*limit)))

        train_loader = DataLoader(tr_ds, batch_size=batch_size, num_workers=num_workers,
                                  pin_memory=True, shuffle=True,worker_init_fn=worker_init_fn)
        valid_loader = DataLoader(val_ds, batch_size=test_batch_size, num_workers=num_workers,
                                  pin_memory=False, shuffle=False,worker_init_fn=worker_init_fn)
        test_loader = DataLoader(test_ds, batch_size=test_batch_size, num_workers=num_workers,
                                  pin_memory=False, shuffle=False,worker_init_fn=worker_init_fn)

    
    if config.save_ds_and_exit:
        t = [[],[],[]]
        for i in tqdm(range(len(train_loader.dataset))):
                t[0].append(train_loader.dataset.__getitem__(i))
        for i in tqdm(range(len(valid_loader.dataset))):
                t[1].append(valid_loader.dataset.__getitem__(i))
        for i in tqdm(range(len(test_loader.dataset))):
                t[2].append(test_loader.dataset.__getitem__(i))
        torch.save(t,'data/savedDS/' + key + '.pt')
        print('Saved: ' + 'data/savedDS/'+ key + '.pt')
        import sys
        sys.exit(0)

    return train_loader, valid_loader, test_loader


class Skeleton(Dataset):
    def __init__(self, config,subjects, fCount: int = 10,
                 size: Tuple[int, int] = (160, 120)):

        self.fCount = fCount
        self.size = size
        self.length = 0
        self.frameVideo = []

        self.subjects =   subjects
        self.seqs = config.ds_sequences  
        self.cameras = config.ds_cameras  
        pnames = config.ds_joints

        avParts = {'Nose': 0, 'Head': 1, 'Neck': 2, 'Belly': 3, 'Root': 4, 'LShoulder': 5,
                   'RShoulder': 6, 'LElbow': 7, 'RElbow': 8, 'LWrist': 9, 'RWrist': 10,
                   'LHip': 11, 'RHip': 12, 'LKnee': 13, 'RKnee': 14, 'LAnkle': 15, 'RAnkle': 16}
        self.parts = [avParts[i] for i in pnames]

        dirname = 'data/h36m_skeleton_data/'

        self.frameVideo = []
        self.length = 0

        for subject in self.subjects:
            filename = dirname + 'keypoints_s' + str(subject) + '_h36m.json'
            with open(filename) as json_file:
                data = json.load(json_file)
            for camera in self.cameras:
                for seq in self.seqs:
                    heatmaps_path = os.path.join(dirname, 'heatmaps', 's{:02d}'.format(subject),
                                                    'seq{:02d}/cam_{}'.format(seq, camera))
                    kp_sequence = data['sequences'][seq]
                    tmpIdxs = []
                    for idx, frame in enumerate(kp_sequence):
                        if frame['poses_2d'] is None:
                            print('missing frame t={}!'.format(frame['time_idx']))
                            continue
                        tmpIdxs.append(idx)
                    vidLen = len(tmpIdxs) - 1
                    curStart = 0
                    curFinish = vidLen
                    curLen = (curFinish - curStart) - self.fCount
                    self.frameVideo.append([self.length, self.length + curLen, curStart,
                                            {'heatmaps_path': heatmaps_path, 'list': tmpIdxs}])
                    self.length += curLen


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        foundC = False
        curVI = None
        for ifv, fv in enumerate(self.frameVideo):
            if idx >= fv[0] and idx < fv[1]:
                foundC = True
                curVI = ifv
                break
        if not foundC:
            raise IndexError()


        heatmaps_path = self.frameVideo[curVI][3]['heatmaps_path']
        idx -= self.frameVideo[curVI][0]
        vidFrameIdx = (idx) + self.frameVideo[curVI][2]

        frames = torch.Tensor( self.fCount,len(self.parts), self.size[1], self.size[0])
        _img_size = (1000, 1000, len(self.parts))

        for i in range(self.fCount):
            hm_path = os.path.join(heatmaps_path,
                                   'hm_{:04d}.npz'.format(self.frameVideo[curVI][3]['list'][vidFrameIdx + i]))
            if os.path.exists(hm_path):
                img = np.zeros(_img_size, dtype=np.float32)
                loaded = np.load(hm_path)
                crop_info = loaded['crop_info'] 
                heatmap = loaded['heatmap'][:, :, self.parts]
                heatmap = heatmap * (heatmap > 0)


                scale = (float(round(crop_info[2])) / heatmap.shape[1], float(round(crop_info[3])) / heatmap.shape[0])
                warp_mat_upscale = np.array([[scale[0], 0., 0.] , [0., scale[1], 0.]])
                hm_crop = cv2.warpAffine(heatmap, warp_mat_upscale, (int(round(crop_info[2])), int(round(crop_info[3]))))

                x0 = int(crop_info[0] - 0.5 * crop_info[2])
                y0 = int(crop_info[1] - 0.5 * crop_info[3])
                x1 = int(crop_info[0] + 0.5 * crop_info[2])
                y1 = int(crop_info[1] + 0.5 * crop_info[3])

                hm_x = max(0, -x0), min(x1, _img_size[0]) - x0
                hm_y = max(0, -y0), min(y1, _img_size[1]) - y0
                img_x = max(0, x0), min(x1, _img_size[0])
                img_y = max(0, y0), min(y1, _img_size[1])

                img[img_y[0]:img_y[1], img_x[0]:img_x[1], :] = hm_crop[hm_y[0]:hm_y[1], hm_x[0]:hm_x[1],:]

                
                scale = (float(self.size[0])/img.shape[1], float(self.size[1])/img.shape[0])
                warp_mat_downscale = np.array([[scale[0], 0., 0.] , [0., scale[1], 0.]])
                hm_global = cv2.warpAffine(img, warp_mat_downscale, self.size)
            else:
                raise Exception("Cannot read the file %s at index %d" % (hm_path, idx))

            frames[i,:, :, :] = torch.from_numpy(hm_global).permute(2, 0, 1)

        frames = (frames - frames.min()) / (frames.max() - frames.min())

        return frames