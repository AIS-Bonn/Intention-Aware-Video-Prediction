import torch
import wandb
from lfdtn.LFT import compact_LFT, compact_iLFT
from asset.utils import getPhaseDiff, getPhaseAdd, clmp, listToTensor, showSeq, dimension, li, ui, show_phase_diff, \
    logMultiVideo, wandblog,dmsg,getAvgDiff,manyListToTensor,colorChannels,randColors,show_hist
from lfdtn.complex_ops import getEps
from lfdtn.window_func import  get_2D_Gaussian,get_ACGW
from torch.functional import F
from icecream import ic
import numpy as np
ic.configureOutput(includeContext=True)


M_transformList=None
auxHist=None

def predict(dataSet,aux, window,config,MRef_Out,M_transform,isIncremental,phase='train', log_vis=True,epoch=1,minib=1):
    global M_transformList,auxHist
    isTest = 'train' not in phase.lower()
    isTrain = not isTest
    eps = getEps()
    angAfterNorm = torch.tensor(0., requires_grad=isTrain)
    auxNorm = torch.tensor(0., requires_grad=isTrain)
    

    T=None
    energy= None
    BS,SQ,CS,HH,WW = dataSet.shape
    PRes=[]
    PResBeforeRef = []

    if isIncremental:
        if isTrain:
            raise Exception("Incremental prediction is Not suitable for training phase!")
        if log_vis:
            raise Exception("Incremental prediction is Not suitable for log_vis!")


    if log_vis:
        T_hist = []
        T_ref_hist = []
        WTs = []

    untilWhatIndex=config.sequence_seed
    createAux=False

    if config.futureAwareMPF=='Network' and (isTrain or config.futureAwareMPFNetwrokTestTime in ['Same','Round']):
        untilWhatIndex=config.sequence_length
        createAux=True


    if M_transformList is None or not isIncremental:
        M_transformList=[]
        isIncremental=False

    if createAux:
        M_transformAuxList=[]

    lastLFT=None
    curLFT =None
    for i in range(untilWhatIndex):
        isFuture = (i>=config.sequence_seed)
        if not isIncremental or i>=config.sequence_seed-2:
            curLFT = compact_LFT(dataSet[:,i]+ eps, window, config)
            if lastLFT is not None:
                Ttmp,Etmp = getPhaseDiff(curLFT[0],lastLFT[0] ,config.use_energy)
                lInp,visB = M_transform.prepare(Ttmp,Etmp,curLFT[1])
                if not isFuture:
                    M_transformList = M_transform.addToList(M_transformList,lInp,config.history_len)
                    energy=Etmp
                if createAux:
                    M_transformAuxList = M_transform.addToList(M_transformAuxList,lInp,config.futureAwareMPFHistory_len)
        
        if not isFuture:
            PRes.append(dataSet[:,i])
            if config.refine_output:
                PResBeforeRef.append(dataSet[:,i])
            if log_vis:
                WTs.append(torch.zeros_like(dataSet[:,i]))
                if i>0:
                    T_hist.append(visB)
                    if i<config.sequence_seed-1:
                        T_ref_hist.append([torch.zeros_like(k) if k is not None else None  for k in visB])
        lastLFT = curLFT
        


    if createAux:
        initAux = None if not config.share_trans_model_and_MPF else torch.zeros(BS,config.futureAwareMPFChannelNumber).to(dataSet.device)
        aux = M_transform(M_transformAuxList,aux=initAux,exportAux=True,epoch=epoch,isTest=isTest,log_vis=log_vis)
        if config.futureAwareMPFNetwrokTestTime=='Round':
            if config.futureAwareMPFContinuous:
                if len(config.futureAwareMPFRoundList)>0:
                    KN=torch.tensor(config.futureAwareMPFRoundList).to(aux.device)
                else:
                    KN=torch.arange(-config.futureAwareMPFRoundLimit,config.futureAwareMPFRoundLimit+0.00000001,config.futureAwareMPFRoundDecimal).to(aux.device)
                
                KNe=KN.expand(BS*config.futureAwareMPFChannelNumber,KN.shape[0])
                aux=KNe[0,(abs(aux.flatten()[:,None] - KNe)).argmin(dim=-1)].view(BS,config.futureAwareMPFChannelNumber)
            else:
                if len(config.futureAwareMPFRoundList)>0:
                    KN=torch.tensor(config.futureAwareMPFRoundList).long().to(aux.device)
                else:
                    KN=torch.tensor(range(0,config.futureAwareMPFChannelNumber)).long().to(aux.device)
                KNe=KN.expand(BS,KN.shape[0])
                aux=F.one_hot(KNe[0,(abs(aux.argmax(dim=1).flatten()[:,None] - KNe)).argmin(dim=-1)].long(), num_classes=config.futureAwareMPFChannelNumber).float()       
        if config.futureAwareMPFDropout>0:
            aux = aux*(torch.rand_like(aux)>config.futureAwareMPFDropout).float()
        if config.futureAwareMPFContinuous:
            auxNorm = (aux*aux).mean()
        else:
            auxNorm = (aux*aux*torch.arange(0,config.futureAwareMPFChannelNumber).to(aux.device)).mean()

        if auxHist is None or minib==1:
            auxHist = {"data":aux,"bin":'auto' if config.futureAwareMPFNetwrokTestTime!='Round' else len(KN)}
        else:
            auxHist["data"] =torch.cat([auxHist["data"],aux],dim=0)
    if log_vis and aux!=None and ((isTest and minib<10) or isTrain):
        print(("AUX "+(("KN "+str(KN.shape)) if config.futureAwareMPFNetwrokTestTime=='Round' else '')+
        " sum.round()="),aux.sum(dim=0).round().detach().tolist()," shape=",aux.shape, " [0]=",aux[0].detach().tolist())
    T,visA,angAfterNormTmp = M_transform(M_transformList,aux,exportAux=False,epoch=epoch,isTest=isTest)
    if log_vis:
        T_ref_hist.append(visA)
    angAfterNorm = angAfterNorm + angAfterNormTmp

    S_curr = dataSet[:,config.sequence_seed-1] 
    for i in range(config.sequence_length - config.sequence_seed):

        S_fft,lCOM = compact_LFT(S_curr + eps, window, config)
        ET = T

        NS_fft = getPhaseAdd(S_fft, ET)
        NS_, WT = compact_iLFT(NS_fft, window, ET, config,channel=S_curr.shape[1])
        if config.refine_output:
            PResBeforeRef.append(clmp(NS_, False))
            if config.refine_output_share_weight:
                refRes = MRef_Out(NS_.view(-1,1,HH,WW)).view(BS,-1,HH,WW)
            else:
                refRes = MRef_Out(NS_)

            if config.residual_refine_output:
                NS_=NS_+refRes
            else:
                NS_=refRes
            NS = clmp(NS_, False)
        else:
            NS = clmp(NS_, False)
        
        PRes.append(NS)


        if log_vis:
            WTs.append(WT)

        if config.enable_autoregressive_prediction:
            lLFT,lCOM = compact_LFT(NS + eps, window, config)
            T,energy = getPhaseDiff(lLFT, S_fft,config.use_energy)
        else:
            energy = None

        lInp,visB = M_transform.prepare(T,energy,lCOM)
        M_transformList = M_transform.addToList(M_transformList,lInp,config.history_len)
        T,visA,angAfterNormTmp = M_transform(M_transformList,aux,exportAux=False,epoch=epoch,isTest=isTest)
        angAfterNorm = angAfterNorm + angAfterNormTmp
        if log_vis:
            T_hist.append(visB)
            T_ref_hist.append(visA)


        S_curr = NS

    PRes = listToTensor(PRes)
    if config.refine_output:
        PResBeforeRef = listToTensor(PResBeforeRef)

    if (torch.isnan(PRes).any()):
        print(torch.pow((dataSet[:, :, 2:] - PRes[:, :, 2:]).cpu(), 2).mean().item())
        print(torch.isnan(dataSet - PRes).any())
        raise Exception("NAN Exception")



    if log_vis:
        with torch.no_grad():
            WTs = listToTensor(WTs)
            vis_image_string = phase+" predictions"
            vis_anim_string = phase+' animations'
            header = phase+': '
            setting = {'oneD': (dimension == 1), 'revert': True, 'dpi': 2.4, 'show': False,"vmin":0,"vmax":1}

            PResShow = colorChannels(PRes,2,config.input_channels==1).detach().cpu()
            dataSetShow = colorChannels(dataSet,2,config.input_channels==1).detach().cpu()
            L2lossShow = torch.pow((dataSetShow - PResShow), 2).detach().clamp(0, 1).detach().cpu()
            L1lossShow = (0.5 * (PResShow - dataSetShow) + 0.5).detach().clamp(0, 1).detach().cpu()
            WTsShow = colorChannels(WTs,2,True).clamp(0, 1).detach().cpu()
            if config.refine_output:
                PResBRefShow = colorChannels(PResBeforeRef,2,config.input_channels==1).clamp(0,1).detach().cpu()

            seqShowList=[PResShow[li:ui],
                        dataSetShow[li:ui], L1lossShow[li:ui],L2lossShow[li:ui], WTsShow[li:ui]]
            seqShowTxt ="PRes,dataSet,L1loss,L2loss,WTs"
            if config.refine_output:
                seqShowList+=[PResBRefShow[li:ui]]
                seqShowTxt+=",PredBeforeRef"
            pic = showSeq(False, -1, seqShowTxt, seqShowList,
                          **setting)



            show_phase_diff(pd_list=T_hist, gt=1-PResShow, config=config, title=header + "VF from Phase Diffs",clear_motion=False,gtOrig=PRes)
            
            
            show_phase_diff(pd_list=T_ref_hist, gt=1-PResShow, config=config, title=header + "VF after M_Transform",clear_motion=False,gtOrig=PRes)

            wandblog({vis_image_string: pic},commit=False)

 
            if auxHist is not None:
                dth=auxHist["data"].cpu().numpy()
                dis=False
                if not config.futureAwareMPFContinuous:
                    dth=np.argmax(dth, axis=1)
                    dis=True
                show_hist(dth,auxHist["bin"],dis,"aux")

            toGifList=[PResShow[li:ui],dataSetShow[li:ui], L1lossShow[li:ui]]
            toGifTxt = 'Prediction, GT, Diff'
            if config.refine_output:
                toGifList+=[PResBRefShow[li:ui]]
                toGifTxt+=",PredBeforeRef"
            logMultiVideo(toGifTxt,toGifList ,config.sequence_seed,
                          vis_anim_string=vis_anim_string)


    angAfterNorm = angAfterNorm / float(config.sequence_length - config.sequence_seed)
    return PRes,PResBeforeRef, angAfterNorm, auxNorm