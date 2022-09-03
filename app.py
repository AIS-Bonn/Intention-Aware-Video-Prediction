import sys
import os
import wandb
import kornia
import time
import ast
import random as rand
import numpy as np
from lfdtn.dataloaders import get_data_loaders
from lfdtn.train_step import predict
from lfdtn.transform_models import cellTransportRefine
from lfdtn.window_func import get_pascal_window, get_ACGW, get_2D_Gaussian
from asset.utils import generate_name, dmsg, CombinedLossDiscounted,wandblog,DenseNetLikeModel,niceL2S,setIfAugmentData,UNet
from past.builtins import execfile
from lfdtn.complex_ops import getEps
from colorama import Fore
from tqdm import tqdm
import torch
import click
from datetime import timedelta


execfile('lfdtn/helpers.py')

torch.utils.backcompat.broadcast_warning.enabled = True

print("Python Version:", sys.version)
print("PyTorch Version:", torch.__version__)
print("Cuda Version:", torch.version.cuda)
print("CUDNN Version:", torch.backends.cudnn.version())

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


deterministic = True
worker_init_fn = None
if deterministic:
    torch.backends.cudnn.deterministic = True
    randomSeed = 123
    torch.manual_seed(randomSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(randomSeed)
    rand.seed(randomSeed)
    np.random.seed(randomSeed)


    def worker_init_fn(worker_id):
        worker_id = worker_id + randomSeed
        torch.manual_seed(worker_id)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(worker_id)
        rand.seed(worker_id)
        np.random.seed(worker_id)

hyperparameter_defaults = dict(
    dryrun=False,
    inference=False,
    load_model='',
    limitDS=1.,
    epochs=5000,
    batch_size=8,
    sequence_length=10,
    sequence_seed=5,
    max_result_speed=6,
    stride=8,  
    window_size=15,
    window_type='ConfGaussian',
    lg_sigma_bias=0.1729,
    optimizer='AdamW',
    gain_update_lr=1,
    refine_lr = 0.001,
    refine_wd= 0.00001,
    refine_layer_cnt=5,
    refine_layer_cnt_a=6,
    refine_hidden_unit=16,
    refine_filters="33333",
    ref_non_lin = 'PReLU',
    M_transform_lr=0.001,
    M_transform_wd=0.000001,
    tran_hidden_unit=16,
    tran_filters="13333",
    untilIndex=12,
    history_len=5,
    tr_non_lin='PReLU',
    tr_bn=False,
    angleNormGain=0.00001,

    ds_subjects=[1,5,6,7,8],
    ds_sequences=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29], 
    ds_cameras = [0,1,2,3], 
    ds_joints =  ['Head','Root','LShoulder','RShoulder','LElbow','RElbow','LWrist','RWrist','LHip','RHip','LKnee','RKnee','LAnkle','RAnkle'],
    
    data_key='Skeleton',
    digitCount=2,
    res_x=64,
    res_y=64,
    max_loss_tol_general=0.2,
    max_loss_tol_index = 2,
    max_num_param_tol=40000,
    pos_encoding=True,
    use_variance = True,
    use_energy = True,
    useCOM=True,
    lr_scheduler = 'ReduceLROnPlateau',
    patience = 5,
    oneCycleMaxLRGain= 10,
    start_T_index = 2,
    tqdm=False,
    kill_no_improve = 3,
    validate_every=1,
    allways_refine= False,
    num_workers=12,
    hybridGain=0.7,
    augment_data=True,
    gainLR=1,
    excludeLoad="",
    refine_output=False,
    refine_output_share_weight=True,
    minimize_loss_before_refine_output=False,
    residual_refine_output=True,
    loss="L2PredAll",
    test_batch_size=-1,
    enable_autoregressive_prediction=False,
    cmd="",
    input_channels=1,
    stUpB=0,
    save_ds_and_exit=False,
    load_ds_from_saved=True,
    useGlobalLFT = False,
    RNDSeed=0,
    ArrowScale=2,
    showArrowInGOnCom=False,
    showRField=True,
    futureAwareMPF='No',
    futureAwareMPFChannelNumber=2,
    futureAwareMPFContinuous=True,
    futureAwareMPFtau=0.1,
    futureAwareMPFtauSchedulToMin=0.1,
    futureAwareMPFtauSchedulRate=0.9,
    futureAwareMPFtauHardEpoch=1,
    futureAwareMPFAlwaysSoft=False,
    futureAwareMPFL2=0.,
    futureAwareMPFDropout=0.,
    futureAwareMPFHistory_len=9,
    futureAwareMPFNetwrokTestTime='Same',
    futureAwareMPFZero=0,
    futureAwareMPFRoundDecimal=0.1,
    futureAwareMPFRoundLimit=1,
    futureAwareMPFRoundList=[],
    share_trans_model_and_MPF=False,
    alwaysSaveResult=False,
    multiGPU=False,
    pin_mem=True,
)


try:
    print("WANDB_CONFIG_PATHS = ", os.environ["WANDB_CONFIG_PATHS"])
except:
    pass

def mytqdm(x):
    return x

for a in sys.argv:
    if '--dryrun=True' in a:
        os.environ["WANDB_MODE"] = "dryrun"
    if ('--configs' in a and "=" in a) or '.yml' in a:
        try:
            try:
                v = a
                _, v = a[2:].split("=")
            except:
                pass
            if os.path.exists(v):
                v = str(os.getcwd()) + "/" + v
                os.environ["WANDB_CONFIG_PATHS"] = v
                print("Load configs from ", v)
        except Exception as e:
            print(e)
            pass


wandb.init(config=hyperparameter_defaults, project="ICANN2022") 
for k in wandb.config.keys():
    if '_constrained' in str(k):
        del wandb.config._items[k]


def myType(val):
    try:
        val = ast.literal_eval(val)
    except ValueError:
        pass
    return val


for a in sys.argv:
    if '--cmd=' in a[:6]:
        wandb.config.update({'cmd': str(a[6:])}, allow_val_change=True)
        continue
    if '--' in a[:2] and "=" in a:
        try:
            k, v = a[2:].split("=")
            v = myType(v)
            wandb.config.update({k: v}, allow_val_change=True)
        except Exception as e:
            pass
config = wandb.config
wandb.save('asset/*')
wandb.save('lfdtn/*')

if config.residual_refine_output and config.minimize_loss_before_refine_output:
    raise BaseException("config.residual_refine_output and config.minimize_loss_before_refine_output cannot both be True!")

if config.test_batch_size==-1:
    config.update({'test_batch_size': config.batch_size}, allow_val_change=True)




if config.start_T_index<1:
    config.update({'start_T_index': 1}, allow_val_change=True)

if config.share_trans_model_and_MPF:
    config.update({'futureAwareMPFHistory_len': config.history_len}, allow_val_change=True)

if "Skeleton" in config.data_key:
    config.update({'input_channels': 14}, allow_val_change=True)
elif config.data_key=="planets_3":
    config.update({'input_channels': 3}, allow_val_change=True)
else:
    config.update({'input_channels': 1}, allow_val_change=True)

if config.useGlobalLFT:
    config.update({'tr_bn': False}, allow_val_change=True)
    config.update({'window_type': 'Identity'}, allow_val_change=True)
    config.update({'useCOM': True}, allow_val_change=True)
    config.update({'max_result_speed': 0}, allow_val_change=True)
    config.update({'window_size': max(config.res_x,config.res_y)}, allow_val_change=True)
    config.update({'stride': max(config.res_x,config.res_y)}, allow_val_change=True)

if config.multiGPU:
    config.update({'pin_mem': False}, allow_val_change=True)
    config.update({'num_workers': 6}, allow_val_change=True)


config.tran_filter_sizes = [int(i) for i in list(str(config.tran_filters))]
config.refine_filter_size = [int(i) for i in list(str(config.refine_filters))]


for k in config.keys():
    if "_lr" in str(k) and config.gainLR!=1:
        print("update",k)
        v = config[k]
        wandb.config.update({k: v*config.gainLR}, allow_val_change=True)

config.model_name_constrained = generate_name()


config.res_x_constrained = config.res_x
config.res_y_constrained = config.res_y


config.window_padding_constrained = config.max_result_speed

config.image_pad_size_old_constrained = int(config.stride * ((config.window_size - 1) // config.stride))
config.num_windows_y_old_constrained = (
                                                   config.res_y_constrained + 2 * config.image_pad_size_old_constrained - config.window_size) // config.stride + 1
config.num_windows_x_old_constrained = (
                                                   config.res_x_constrained + 2 * config.image_pad_size_old_constrained - config.window_size) // config.stride + 1
config.num_windows_total_old_constrained = config.num_windows_x_old_constrained * config.num_windows_y_old_constrained

config.image_pad_size_constrained = int(
    config.stride * (((config.window_size + 2 * config.window_padding_constrained) - 1) // config.stride))
config.num_windows_y_constrained = (
                                               config.res_y_constrained + 2 * config.image_pad_size_constrained - config.window_size - 2 * config.window_padding_constrained) // config.stride + 1
config.num_windows_x_constrained = (
                                               config.res_x_constrained + 2 * config.image_pad_size_constrained - config.window_size - 2 * config.window_padding_constrained) // config.stride + 1
config.num_windows_total_constrained = config.num_windows_x_constrained * config.num_windows_y_constrained

if ((config.res_x_constrained - 1) % config.stride != 0) or ((config.res_y_constrained - 1) % config.stride != 0):
    print(Fore.RED +"Not recommended/compatible stride "+str(config.res_x_constrained)+" "+str(config.stride)+Fore.RESET)
    


avDev = torch.device("cpu")
cuda_devices = list()
if torch.cuda.is_available():
    cuda_devices = [0]
    avDev = torch.device("cuda:" + str(cuda_devices[0]))
    if (len(cuda_devices) > 0):
        torch.cuda.set_device(cuda_devices[0])
print("avDev:", avDev)
dmsg('os.environ["CUDA_VISIBLE_DEVICES"]')
if config.tqdm:
    mytqdm=tqdm


inference_phase = config.inference
is_sweep = wandb.run.sweep_id is not None

print("config:{")
pretty(config._items,hyperparameter_defaults)
print("}")
critBCE = torch.nn.BCELoss()
critL1 = torch.nn.L1Loss()
critL2 = torch.nn.MSELoss()
critSSIM = kornia.losses.SSIMLoss(window_size=9, reduction='mean')
critHybrid = CombinedLossDiscounted()


MRef_Out = None
M_transform = None

LG_Sigma = torch.tensor(config.lg_sigma_bias, requires_grad=True, device=avDev) 

paramN = []
minLR = []
wD = []

chennelC = 1 if config.refine_output_share_weight else config.input_channels

if config.refine_output:

    MRef_Out = DenseNetLikeModel(inputC=chennelC,outputC=chennelC, hiddenF=config.refine_hidden_unit,
                                 filterS=config.refine_filter_size,nonlin=config.ref_non_lin,lastNonLin=False,initWIdentity=not config.residual_refine_output,bn=False).to(avDev)
    paramN.append('MRef_Out')
    minLR.append(config.refine_lr)
    wD.append(config.refine_wd)



M_transform = cellTransportRefine(config).to(avDev)
paramN.append('M_transform')
minLR.append(config.M_transform_lr)
wD.append(config.M_transform_wd)




loadModels(config.load_model,config.excludeLoad)

paramList = []

max_lrs = []
for i, p in enumerate(paramN):
    par = eval(p)
    paramList.append({'params': [par] if type(par) is torch.Tensor else par.parameters(),
                      'lr': minLR[i],
                      'weight_decay': wD[i],
                      'name': p})
    max_lrs.append(minLR[i]*config.oneCycleMaxLRGain)

optimizer = eval('torch.optim.'+config.optimizer+'(paramList,eps=getEps())')


numParam = 0
for par in optimizer.param_groups:
    numParam += sum(l.numel() for l in par["params"] if l.requires_grad)

config.parameter_number_constrained = numParam
wandblog({"numParam": numParam})


for par in optimizer.param_groups:
    print(Fore.CYAN, par["name"], sum(l.numel() for l in par["params"] if l.requires_grad), Fore.RESET)
    for l in par["params"]:
        if l.requires_grad:
            print(Fore.MAGENTA, l.shape, "  =>", l.numel(), Fore.RESET)

print("Number of trainable params: ", Fore.RED + str(numParam) + Fore.RESET)
if is_sweep and numParam > config.max_num_param_tol:
    wandblog({"cstate": 'High Param', 'sweep_metric': 1.1},commit=True)
    print(Fore.RED, "TOO high #Params ", numParam, " > ", config.max_num_param_tol, Fore.RESET)
    sys.exit(0)

trainloader, validloader, testloader = get_data_loaders(config,key=config.data_key,
                                            size=(config.res_x_constrained, config.res_y_constrained),
                                            batch_size=config.batch_size,test_batch_size=config.test_batch_size, num_workers=config.num_workers, limit=config.limitDS,
                                            sequence_length=config.sequence_length)

if len(config.cmd)>1:
    exec(config.cmd)

print(Fore.MAGENTA,'Trainloader:',len(trainloader),'Validloader:',len(validloader),'Testloader:',len(testloader),Fore.RESET)

startOptimFromIndex = 0
lGains = [i * 1.2 for i in range(config.sequence_length + 1, startOptimFromIndex + 1, -1)]

lGains[0]=0.4
lGains[1]=0.8
lGains = [i / sum(lGains) for i in lGains]

print(lGains)

li = 0
ui = 1
t = 0  
bestFullL2Loss = 1e25
SHOWINTER = False

print(Fore.MAGENTA + ("Sweep!" if is_sweep else "Normal Run") + Fore.RESET)
if inference_phase:
    print(Fore.CYAN + "Inference Phase!" + Fore.RESET)
    bs = config.batch_size
    inferenceRes = []
    paintEvery = 3
    paintOncePerEpoch = False
    runs = 0 if config.epochs==0 else 1
else:
    print(Fore.GREEN + "Training Phase!" + Fore.RESET)
    bs = config.batch_size
    paintEvery = None
    paintOncePerEpoch = True
    runs = config.epochs

torch.set_grad_enabled(not inference_phase)

if not inference_phase:
    if config.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=config.patience, threshold=0.0001,
                                                        cooldown=0, verbose=True, min_lr=0.000001)
    elif config.lr_scheduler == 'OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lrs, total_steps=len(trainloader)*config.epochs)

    else:
        class dummyOpt():
            def step(self,inp=None):
                pass
            def get_last_lr(self):
                return [0.0]
            def get_lr(self):
                return [0.0]
        scheduler=dummyOpt()

if config.RNDSeed>0:
    randomSeed = config.RNDSeed
    torch.manual_seed(randomSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(randomSeed)
    rand.seed(randomSeed)
    np.random.seed(randomSeed)
last_improved=0


def calcLoss(netOut,target):
    if config.loss=="Hybrid":
        ll = 0
        l = torch.pow((target - netOut), 2)
        for i in range(config.sequence_length - startOptimFromIndex):
            ll += l[:, i, :, :, :].mean() * lGains[i]
        
        ll = (1-config.hybridGain)*ll + (config.hybridGain)*critSSIM(netOut.view(-1,1,netOut.shape[3],netOut.shape[4]),
            target.view(-1,1,target.shape[3],target.shape[4]))
    elif config.loss=="L2Pred":
        ll = critL2(netOut[:,config.sequence_seed:], target[:,config.sequence_seed:])
    elif config.loss=="L2PredAll":
        ll = critL2(netOut[:,0:], target[:,0:])
    else:
        upB = min(config.sequence_length,(t//2)+1+config.stUpB)
        lowB = 0
        ll = critL2(netOut[:,lowB:upB], target[:,lowB:upB])
    return ll


with torch.no_grad():

    if config.window_type == 'Pascal':
        window = get_pascal_window(config.window_size).to(avDev)
    elif config.window_type == 'ConfGaussian':
        window = get_ACGW(windowSize=config.window_size, sigma=LG_Sigma).detach()
    elif config.window_type == 'Identity':
        window = torch.ones(config.window_size, config.window_size).to(avDev)
    else:
        window = get_2D_Gaussian(resolution=config.window_size, sigma=LG_Sigma * config.window_size)[0, 0, :, :]


wandblog({"windowImg": wandb.Image(window, caption=config.window_type)})


while t < runs:
    if t == 1 and torch.backends.cudnn.benchmark == True:
        torch.cuda.empty_cache()
    wandbLog = {}

    if not inference_phase:
        start_time = time.time()
        phase = 'Training'
        print('Going through ',phase,' set at epoch', t , '...')
        train_c, train_ll,angAfterNorm_ll,angAfterNorm_t,auxNorm_ll,auxNorm_t = (0,0,0,0,0,0)

        setEvalTrain(True)
        setIfAugmentData(config.augment_data)
        if paintOncePerEpoch:
            paintEvery = rand.randint(1,len(trainloader))

        for mini_batch in mytqdm(trainloader):
            if type(mini_batch) == list  and len(mini_batch)==3:
                mini_batch,aux_batchC,aux_batchD = mini_batch
                aux_batch=aux_batchC if config.futureAwareMPFContinuous else aux_batchD

            futureAwareC=None
            if config.futureAwareMPF=='Zero':
                futureAwareC = torch.zeros(mini_batch.shape[0],config.futureAwareMPFChannelNumber).to(avDev)
            elif config.futureAwareMPF=='GT':
                futureAwareC = aux_batch.to(avDev)

            optimizer.zero_grad()
            train_c += 1
            data = mini_batch.to(avDev)
            

            if paintOncePerEpoch:
                show_images = train_c == paintEvery
            else:
                show_images = True if train_c % paintEvery == 0 else False

            show_images = show_images and not config.dryrun

            pred_frames,pred_frames_before_ref, angAfterNorm, auxNorm= predict(data,futureAwareC, window, config,MRef_Out, M_transform,isIncremental=False,phase=phase, log_vis=show_images,epoch=t,minib=train_c)

            if pred_frames is not False:
                netOut = pred_frames[:, startOptimFromIndex:]
                target = data[:, startOptimFromIndex:]
                angAfterNorm_t +=angAfterNorm.item()
                auxNorm_t +=auxNorm.item()
                angAfterNorm_loss = angAfterNorm * config.angleNormGain
                auxNorm_loss = auxNorm * config.futureAwareMPFL2

                ll = calcLoss(netOut,target)

                if config.minimize_loss_before_refine_output and config.refine_output and not config.residual_refine_output:
                    ll = (ll + calcLoss(pred_frames_before_ref,target))/2.
                
                ll = ll + angAfterNorm_loss + auxNorm_loss
                ll.backward()

                optimizer.step()
                if not inference_phase and config.lr_scheduler == 'OneCycleLR':
                    scheduler.step()

                train_ll += ll.item()
                angAfterNorm_ll += angAfterNorm_loss.item()
                auxNorm_ll += auxNorm_loss.item()
            else:
                print(Fore.RED + "NAN found!" + Fore.RESET)
                raise BaseException("NAN error!")

            wandblog(wandbLog, commit=(not paintOncePerEpoch and show_images))

        wandbLog["trainLoss"] = train_ll / train_c
        wandbLog["angAfterNormLoss"] = angAfterNorm_ll / train_c
        wandbLog["auxNormLoss"] = auxNorm_ll / train_c

        wandbLog["angAfterNorm"] = angAfterNorm_t / train_c
        wandbLog["auxNorm"] = auxNorm_t / train_c
        

        tshow = str(timedelta(seconds=time.time() - start_time)).split(".")
        tshow = [tshow[0],tshow[1][:2]] if len(tshow)==2 else tshow
        print('...done! ',Fore.LIGHTYELLOW_EX+".".join(tshow)+Fore.RESET)

    if t%config.validate_every>0 and not inference_phase and not is_sweep:
        print(Fore.LIGHTYELLOW_EX," ==> Skip validation",(config.validate_every-(t%config.validate_every)),"!...",Fore.RESET)
        wandblog(wandbLog, commit=True)
        t=t+1
        continue

    start_time = time.time()
    tPhase = ('Validation' if not inference_phase else 'Testing')
    tloader =  validloader if not inference_phase else testloader
    print('Going through ',tPhase,' set...')


    with torch.no_grad():
        setEvalTrain(False)
        setIfAugmentData(False)
        valid_c, bceFull, bceFullMin, L1FullNet, L2FullNet, ssimFull, ssimHybrid = (0, 0, 0, 0, 0, 0, 0)
        if paintOncePerEpoch:
            paintEvery = rand.randint(1, len(tloader))
        for mini_batch in mytqdm(tloader):
            valid_c += 1

            if type(mini_batch) == list  and len(mini_batch)==3:
                mini_batch,aux_batchC,aux_batchD = mini_batch
                aux_batch=aux_batchC if config.futureAwareMPFContinuous else aux_batchD
            
            futureAwareC=None
            if config.futureAwareMPFNetwrokTestTime!='Same' and config.futureAwareMPF!='No':
                if config.futureAwareMPFNetwrokTestTime=='Rand':
                    futureAwareC = torch.rand(mini_batch.shape[0],config.futureAwareMPFChannelNumber).to(avDev)*2-1
                    if not config.futureAwareMPFContinuous:
                        futureAwareC=torch.nn.functional.gumbel_softmax(futureAwareC, tau=1, hard=True)
                else:
                    futureAwareC = torch.zeros(mini_batch.shape[0],config.futureAwareMPFChannelNumber).to(avDev)
                    futureAwareC+=config.futureAwareMPFZero
                    if not config.futureAwareMPFContinuous:
                        futureAwareC[:,0]=1
            else:
                if config.futureAwareMPF=='Zero':
                    futureAwareC = torch.zeros(mini_batch.shape[0],config.futureAwareMPFChannelNumber).to(avDev)
                    futureAwareC+=config.futureAwareMPFZero
                elif config.futureAwareMPF=='GT':
                    futureAwareC = aux_batch.to(avDev)

            data = mini_batch.to(avDev)

            if paintOncePerEpoch:
                show_images = valid_c == paintEvery
            else:
                show_images = True if valid_c % paintEvery == 0 else False
            show_images = show_images and not config.dryrun

            pred_frames,_, _, _= predict(data,futureAwareC, window, config,MRef_Out,M_transform,isIncremental=False, phase=tPhase, log_vis=show_images,epoch=t,minib=valid_c)
            
            netOut = pred_frames[:,  config.sequence_seed:].clamp(0,1)
            target = data[:,  config.sequence_seed:].clamp(0,1)
            bceFull += critBCE(netOut, target)
            bceFullMin += critBCE(target, target)
            L1FullNet += critL1(netOut, target)
            L2FullNet += critL2(netOut, target)
            netOutSSIM = netOut.reshape(-1, 1, netOut.shape[3], netOut.shape[4])
            targetSSIM = target.reshape(-1, 1, target.shape[3], target.shape[4])
            ssimFull += critSSIM(netOutSSIM,targetSSIM)
            ssimHybrid += critHybrid(netOutSSIM,targetSSIM)

            wandblog(wandbLog, commit=(not paintOncePerEpoch and show_images))  
    if inference_phase:
        print("FPS: {:.3f}".format((valid_c*config.batch_size) / float(time.time() - start_time)))
    if not inference_phase and config.lr_scheduler == 'ReduceLROnPlateau':
        scheduler.step(L2FullNet.item() / valid_c)
    wandbLog["hybridSSIMLoss"] = ssimHybrid.item() / valid_c
    wandbLog["L1FullLoss"] = L1FullNet.item() / valid_c
    wandbLog["L2FullLoss"] = L2FullNet.item() / valid_c
    wandbLog["bceFullMin"] = bceFullMin.item() / valid_c
    wandbLog["bceFull"] = bceFull.item() / valid_c
    wandbLog["SSIMFull"] = ssimFull.item() / valid_c

    paramGain = 1 if (
                config.parameter_number_constrained < config.max_num_param_tol / 3.) else config.parameter_number_constrained / (
                config.max_num_param_tol / 3.)
    wandbLog["paramGain"] = paramGain
    wandbLog["sweep_metric"] = wandbLog["L2FullLoss"] * paramGain

    if (wandbLog["L2FullLoss"] < bestFullL2Loss or config.alwaysSaveResult) and not inference_phase:
        last_improved = t
        if not config.dryrun:
            nameM = config.data_key[:6]+"_"+"{:.7f}".format(wandbLog["L2FullLoss"]).replace(".","_")+"_"+wandb.run.project+"_"+wandb.run.name+"_"+config.model_name_constrained.replace("-","_")
        else:
            nameM = config.data_key[:6]+"_"+"{:.7f}".format(wandbLog["L2FullLoss"]).replace(".","_")+"_"+config.model_name_constrained.replace("-","_")
        nameM = nameM.replace("-","_").replace(".","_")
        print('Model improved:',Fore.GREEN+str(wandbLog["L2FullLoss"])+Fore.RESET,' Model saved! :', nameM)
        bestFullL2Loss = wandbLog["L2FullLoss"]
        saveModels(nameM)
        cFile=wandb.run.settings.sync_dir+'/files/config.yaml'
        if os.path.exists(cFile):
            open('savedModels/'+nameM+ ".yml", 'wb').write(open(cFile, 'rb').read())

    if is_sweep and config.kill_no_improve>=0 and (t-last_improved)>config.kill_no_improve:
        print(Fore.RED, "No improvement!", (t-last_improved),t,last_improved, Fore.RESET)
        wandblog(wandbLog,commit=True)
        sys.exit(0)

    if t>=config.max_loss_tol_index and is_sweep and wandbLog['sweep_metric'] > config.max_loss_tol_general:
        wandbLog["cstate"]= 'High Loss'
        print(Fore.RED, "Loss too high!", wandbLog['sweep_metric'], is_sweep, Fore.RESET)
        wandblog(wandbLog,commit=True)
        sys.exit(0)

    if inference_phase:
        inferenceRes.append(
            [bceFullMin.item() / valid_c, bceFull.item() / valid_c, L1FullNet.item() / valid_c
            , L2FullNet.item() / valid_c, ssimFull.item() / valid_c])


    t = t + 1
    url = "DRY"
    if not config.dryrun:
        url=click.style(wandb.run.get_url().replace('https://',""), underline=True, fg="blue")

    tshow = str(timedelta(seconds=time.time() - start_time)).split(".")
    tshow = [tshow[0],tshow[1][:2]] if len(tshow)==2 else tshow
    print('...done! ',Fore.LIGHTYELLOW_EX+".".join(tshow)+Fore.RESET,
        'Run:',url)
    wandblog(wandbLog, commit=True)

    if (inference_phase and t >= runs):
        inferenceRes = np.array(inferenceRes)
        print("BCEMin=", inferenceRes[:, 0].mean(), " BCE=", inferenceRes[:, 1].mean(), " L1=",
            inferenceRes[:, 2].mean(), " L2=", inferenceRes[:, 3].mean(), " SSIM=", inferenceRes[:, 4].mean())
        break
wandblog({"cstate": 'Done'},commit=config.epochs!=0)
print("Run completed!")
