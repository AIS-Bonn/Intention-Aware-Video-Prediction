# Intention-Aware-Video-Prediction

The code for the paper: "Intention-Aware Frequency Domain Transformer Networks for Video Prediction"

If you use the code for your research paper, please cite the following paper:
<p>
  <b>Hafez Farazi</b> and Sven Behnke:<br>
  <a href="http://www.ais.uni-bonn.de/papers/ICANN_2022_Farazi.pdf"><u>Intention-Aware Frequency Domain Transformer Networks for Video Prediction</u></a>&nbsp;<a href="http://www.ais.uni-bonn.de/papers/ICANN_2022_Farazi.pdf">[PDF]</a><br>
  Accepted for 31st International Conference on Artificial Neural Networks (ICANN), Bristol, UK, September 2022. <br><b></b><br>
</p>

BIB:

```
@Conference{Farazi2022_ICANN,
  Title                    = {Intention-Aware Frequency Domain Transformer Networks for Video Prediction},
  Author                   = {Farazi, Hafez and Behnke, Sven},
  Booktitle                = {International Conference on Artificial Neural Networks (ICANN)},
  Year                     = {2022},
  Address                  = {Bristol, UK}
}
```

## Dependencies
The code was tested with Ubuntu 20.04 and PyTorch 1.10

## Sample Result
GT | z=0 | z=1 | z=2 | z=3

PropMNIST_4D:

![](sample/MNIST_4D/File1.gif)
![](sample/MNIST_4D/File2.gif)
![](sample/MNIST_4D/File3.gif)
![](sample/MNIST_4D/File4.gif)
![](sample/MNIST_4D/File5.gif)

GT | z=-1 | z=-0.8 | z=-0.6 | z=-0.4 | z=-0.2 | z=0 | z=0.2 | z=0.4 | z=0.6 | z=0.8 | z=1 

PropMNIST_1C:

![](sample/MNIST_1C/File1.gif)
![](sample/MNIST_1C/File2.gif)
![](sample/MNIST_1C/File3.gif)
![](sample/MNIST_1C/File4.gif)
![](sample/MNIST_1C/File5.gif)

GT | z=-1 | z=-0.8 | z=-0.6 | z=-0.4 | z=-0.2 | z=0 | z=0.2 | z=0.4 | z=0.6 | z=0.8 | z=1 

PropMNIST_1C:

![](sample/Skeleton_1C/File1.gif)
![](sample/Skeleton_1C/File2.gif)
![](sample/Skeleton_1C/File3.gif)
![](sample/Skeleton_1C/File4.gif)
![](sample/Skeleton_1C/File5.gif)
![](sample/Skeleton_1C/File6.gif)
![](sample/Skeleton_1C/File7.gif)
![](sample/Skeleton_1C/File8.gif)
![](sample/Skeleton_1C/File9.gif)
![](sample/Skeleton_1C/File10.gif)
![](sample/Skeleton_1C/File11.gif)


## Run

```
python app.py --data_key=PropMMNIST --batch_size=500 --useGlobalLFT=True --res_x=64 --res_y=64 --inference=False --ArrowScale=2 --futureAwareMPFNetwrokTestTime=Same --futureAwareMPFDropout=0 --futureAwareMPFHistory_len=5 --futureAwareMPFChannelNumber=1 --futureAwareMPF=Network --sequence_length=6 --sequence_seed=3 --history_len=4 --digitCount=1 --futureAwareMPFContinuous=True --refine_output=True --M_transform_lr=0.007 --epochs=30 --refine_lr=0.0011
```
