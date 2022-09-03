import torch
import math
import torch.nn as nn
from asset.utils import getAvgDiff, createPhaseDiff,positionalencoding2d,dmsg,DenseNetLikeModel,wandblog

def mysoftmax(logits, tau=1, hard=False, dim=-1):
    nlogits = (logits) / tau
    y_soft = nlogits.softmax(dim)

    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret

class cellTransportRefine(nn.Module):
    def __init__(self, config):
        super(cellTransportRefine, self).__init__()
        self.res_y = config.res_y_constrained
        self.res_x = config.res_x_constrained
        self.y_stride = config.stride
        self.x_stride = config.stride
        self.pS = config.window_padding_constrained
        self.N = config.window_size
        self.N_prime = self.N + 2 * self.pS

        self.padSizeAdjust = int(self.x_stride * ((self.N_prime - 1) // self.x_stride))
        self.L_y = (self.res_y + 2 * self.padSizeAdjust - self.N_prime) // self.y_stride + 1
        self.L_x = (self.res_x + 2 * self.padSizeAdjust - self.N_prime) // self.x_stride + 1

        
        self.untilIndex = config.untilIndex
       
        self.denseNet = True
        self.pose_enc_level = 4
        moreC = self.pose_enc_level if config.pos_encoding else 0
        if config.futureAwareMPF!='No':
            moreC+=config.futureAwareMPFChannelNumber

        inpDim = 4 if config.use_variance else 2
        if config.useCOM:
            inpDim+=2
        if config.use_energy:
            inpDim+=1

        self.dimentionMultiplier =config.input_channels
        
        self.nonLin = eval('torch.nn.'+config.tr_non_lin+"()")

        if config.share_trans_model_and_MPF and config.futureAwareMPF=='Network':
            self.cnn = torch.nn.Sequential(DenseNetLikeModel( inputC=(inpDim *self.dimentionMultiplier * (config.history_len))+moreC,
                outputC=4*self.dimentionMultiplier, hiddenF=config.tran_hidden_unit,
                    filterS=config.tran_filter_sizes, nonlin = config.tr_non_lin,bn=config.tr_bn,lastNonLin=True),
                DenseNetLikeModel(inputC=4*self.dimentionMultiplier,
                outputC=2*self.dimentionMultiplier, hiddenF=4,
                    filterS=[1,1], nonlin = config.tr_non_lin,bn=config.tr_bn,lastNonLin=False)
            )

        else:
            self.cnn = DenseNetLikeModel( inputC=(inpDim *self.dimentionMultiplier * (config.history_len))+moreC,
                        outputC=2*self.dimentionMultiplier, hiddenF=config.tran_hidden_unit,
                            filterS=config.tran_filter_sizes, nonlin = config.tr_non_lin,bn=config.tr_bn,lastNonLin=False)

        if config.futureAwareMPF=='Network':
            if not config.share_trans_model_and_MPF:
                self.auxModel = torch.nn.Sequential(
                        DenseNetLikeModel( inputC=(inpDim *self.dimentionMultiplier * (config.futureAwareMPFHistory_len))+moreC-config.futureAwareMPFChannelNumber,
                            outputC=config.futureAwareMPFChannelNumber, hiddenF=config.tran_hidden_unit,
                                filterS=config.tran_filter_sizes, nonlin = config.tr_non_lin,bn=config.tr_bn,lastNonLin=False),
                        torch.nn.Flatten(start_dim=1, end_dim=-1),
                        torch.nn.Linear(self.L_x * self.L_y * config.futureAwareMPFChannelNumber,config.futureAwareMPFChannelNumber)
                )
            else:
                self.auxModel = torch.nn.Sequential(
                        self.cnn[0],
                        DenseNetLikeModel(inputC=4*self.dimentionMultiplier,
                            outputC=config.futureAwareMPFChannelNumber, hiddenF=4,
                                filterS=[3,3], nonlin = config.tr_non_lin,bn=config.tr_bn,lastNonLin=False),
                        torch.nn.Flatten(start_dim=1, end_dim=-1),
                        torch.nn.Linear(self.L_x * self.L_y * config.futureAwareMPFChannelNumber,config.futureAwareMPFChannelNumber),
                )


        self.config = config
        self.pos_encoding = None
        dmsg('self.L_x','self.L_y','self.N_prime')

    def prepare(self, x,energy,COM):

        BS=x.shape[0]

        xV = x.view(-1, self.N_prime, self.N_prime, 2)
        self.xvShape=xV.shape
        self.xShape=x.shape


        if energy is not None:
            energy = energy.view(-1, self.N_prime, self.N_prime, 1)



        tmpX,varianceX = getAvgDiff(rot=xV,energy=energy, step=1, axis=0, untilIndex=self.untilIndex,variance = self.config.use_variance)
        tmpY,varianceY = getAvgDiff(rot=xV,energy=energy, step=1, axis=1, untilIndex=self.untilIndex,variance = self.config.use_variance)

        tmpX = (torch.atan2(tmpX[:, 1], tmpX[:, 0] + 0.0000001) / (torch.pi * 2 / self.N_prime)).unsqueeze(-1)
        tmpY = (torch.atan2(tmpY[:, 1], tmpY[:, 0] + 0.0000001) / (torch.pi * 2 / self.N_prime)).unsqueeze(-1)

        tmpX = tmpX.view(-1, self.L_y, self.L_x,self.dimentionMultiplier).permute(0,3,1,2).contiguous()
        tmpY = tmpY.view(-1, self.L_y, self.L_x,self.dimentionMultiplier).permute(0,3,1,2).contiguous()

        if COM is not None:
            COMX = COM[:,:,0].view(-1,self.L_y,self.L_x,self.dimentionMultiplier).permute(0,3,1,2).contiguous()
            COMY = COM[:,:,1].view(-1,self.L_y,self.L_x,self.dimentionMultiplier).permute(0,3,1,2).contiguous()

        if self.config.use_variance:
            varianceX = varianceX.view(-1, self.L_y, self.L_x,self.dimentionMultiplier).permute(0,3,1,2).contiguous()
            varianceY = varianceY.view(-1, self.L_y, self.L_x,self.dimentionMultiplier).permute(0,3,1,2).contiguous()

        if energy is not None:
            energyTotal = energy.mean(dim=[1,2])
            energyTotal = energyTotal.view(-1, self.L_y, self.L_x,self.dimentionMultiplier).permute(0,3,1,2).contiguous()
        elif self.config.use_energy:
            energyTotal = torch.ones_like(tmpX)
        else:
            energyTotal = None

        vfBefore = [-tmpX.permute(0,2,3,1),
                     tmpY.permute(0,2,3,1),None,None,None]  

        if energyTotal is not None:
            vfBefore[2]=energyTotal.permute(0,2,3,1)

        if self.config.use_variance:
            vfBefore[3]=varianceX.permute(0,2,3,1)
            vfBefore[4]=varianceY.permute(0,2,3,1)

        toCat = [tmpX, tmpY]
        if self.config.use_variance:
            toCat+=[varianceX,varianceY]
        if self.config.useCOM:
            toCat+=[COMX,COMY]
        if energyTotal is not None:
            toCat+=[energyTotal]

        lInp = torch.cat(toCat, dim=1)
        
        return lInp,vfBefore

    def addToList(self,listToApp,lInp,length):
        if len(listToApp)==0:
            listToApp = [0.1 * torch.ones_like(lInp) for i in range(length-1)]
        else:
            listToApp.pop(0)

        listToApp.append(lInp)  
        return listToApp
        

    def forward(self,listToApp, aux,exportAux,epoch=1,isTest=False,log_vis=False):
        
        lInp = torch.cat(listToApp, dim=1)


        vfAfter = None

        if self.config.pos_encoding:
            if self.pos_encoding is None:
                self.pos_encoding = positionalencoding2d(self.pose_enc_level, lInp.shape[2], lInp.shape[3]).unsqueeze(
                    0).to(lInp.device).detach()

            lInp = torch.cat(
                (self.pos_encoding.expand(lInp.shape[0], self.pose_enc_level, lInp.shape[2], lInp.shape[3]),lInp), dim=1)

        if aux is not None:
            lInp = torch.cat((aux[...,None,None].expand(lInp.shape[0], self.config.futureAwareMPFChannelNumber, lInp.shape[2], lInp.shape[3]),lInp), dim=1)
        
        if not exportAux:
            lInp = self.cnn(lInp)

            lInp = lInp.view(-1,self.dimentionMultiplier,2,self.L_y, self.L_x)



            lInp = lInp.permute(0,3,4,1,2).contiguous()
            tmpX = lInp[:, :,:,:,0]
            tmpY = lInp[:, :,:,:,1]

            vfAfter = [-tmpX,
                            tmpY,None,None,None] 

            angAfterNorm = (tmpX.abs().mean() + tmpY.abs().mean())/2.

            tmpX = tmpX * (torch.pi * 2 / self.N_prime)
            tmpY = tmpY * (torch.pi * 2 / self.N_prime)
            ret = createPhaseDiff(tmpX.view(-1), tmpY.view(-1), self.xvShape)
            ret = ret.view(self.xShape)

            return ret, vfAfter, angAfterNorm
            
        else:
            res = self.auxModel(lInp)
            if self.config.futureAwareMPFContinuous:
                res =torch.tanh(res)
            else:
                if self.config.futureAwareMPFAlwaysSoft:
                    tau = max(self.config.futureAwareMPFtau*math.pow(self.config.futureAwareMPFtauSchedulRate,max(0,epoch)),self.config.futureAwareMPFtauSchedulToMin)
                    hard=False
                    if self.config.inference:
                        tau=self.config.futureAwareMPFtauSchedulToMin
                        hard=True
                    res = mysoftmax(res, tau=tau, hard=hard)
                    if log_vis:
                        wandblog({"tau":tau,"hard":int(False)}, commit=False)
                else:
                    res = torch.tanh(res)
                    if not isTest:
                        tau = max(self.config.futureAwareMPFtau*math.pow(self.config.futureAwareMPFtauSchedulRate,max(0,epoch-self.config.futureAwareMPFtauHardEpoch)),self.config.futureAwareMPFtauSchedulToMin)
                        hard = (epoch>self.config.futureAwareMPFtauHardEpoch)

                        res = mysoftmax(res, tau=tau, hard=hard)
                        if log_vis:
                            wandblog({"tau":tau,"hard":int(hard)}, commit=False)
                    else:
                        res = torch.nn.functional.one_hot(res.argmax(dim=-1),self.config.futureAwareMPFChannelNumber).float()

            return res

    

