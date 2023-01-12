# IMG-CAP

## Requirements (Our Main Enviroment)
+ Python 3.7
+ PyTorch 1.8.2
+ TorchVision 0.8.0
+ [cider](https://github.com/ruotianluo/cider)
+ [coco-caption](https://github.com/tylin/coco-caption)(Remember to follow initialization steps in coco-caption/README.md)
+ numpy
+ tqdm
+ gensim
+ matplotlib
+ yacs
+ lmdbdict
## Install
```bash
python -m pip install -e 
```
More detailed environment settings can be found on: https://github.com/ruotianluo/ImageCaptioning.pytorch

## Data preparing
### 1. Download:
[coco_pred_sg](https://drive.google.com/file/d/1gJl1aLn2GeN7J5sm-g9I43tA6gxR0DMC/view?usp=sharing), and unzip it in data/.
### 2. feature preparing:
The features of MSCOCO 2014 are extracted by previous works: [coco_bu_feats](https://github.com/ruotianluo/ImageCaptioning.pytorch) and [coco_swin_feats](https://github.com/232525/PureT).
You can also extract image features of MSCOCO 2014 using [resnet model](https://drive.google.com/open?id=0B7fNdx_jAqhtbVYzOURMdDNHSGM), and [swin transformer model](https://github.com/microsoft/Swin-Transformer)

## Training
The core code are given in the models/TSGMModel3.py
```bash
python train_tsg.py --cfg tsg_configs/tsgmt1.yml
```
