# TFSGC

The repository for our paper: **Transforming Visual Scene Graphs to Image Captions**

<a href='https://github.com/GaryJiajia/TSG'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='https://arxiv.org/pdf/2305.02177.pdf'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> 

<a href='https://aclanthology.org/2023.acl-long.694/'>The 61st Annual Meeting of the Association for Computational Linguistics (ACL2023)  main conference</a>

## Table of Contents
- [Installation](#installation)
- [Data preparing](#Data-preparing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Citing](#citing)

## Installation
```bash
git clone https://github.com/GaryJiajia/TSG.git
cd TSG
python -m pip install -e 
```
More detailed environment settings can be found on: https://github.com/ruotianluo/ImageCaptioning.pytorch

### Our Main Enviroment
+ Python 3.7
+ PyTorch 1.8.2
+ TorchVision 0.8.0
+ numpy
+ tqdm
+ gensim
+ matplotlib
+ yacs
+ lmdbdict
```bash
## Please check the version of torch/torchvision if it matches your own version
pip install -r requirements.txt 
```

+ [cider](https://github.com/ruotianluo/cider)
+ [coco-caption](https://github.com/tylin/coco-caption)(Remember to follow initialization steps in coco-caption/README.md)

## Data preparing
### 1. Download:
[coco_pred_sg](https://drive.google.com/file/d/1gJl1aLn2GeN7J5sm-g9I43tA6gxR0DMC/view?usp=sharing), and unzip it in `data/`.
### 2. feature preparing:
The features of MSCOCO 2014 are extracted by previous works: [coco_bu_feats](https://github.com/ruotianluo/ImageCaptioning.pytorch) and [coco_swin_feats](https://github.com/232525/PureT).
(You can also extract image features of MSCOCO 2014 using [resnet model](https://drive.google.com/open?id=0B7fNdx_jAqhtbVYzOURMdDNHSGM), and [swin transformer model](https://github.com/microsoft/Swin-Transformer))
Download them, unzip them in `data/`.
```bash
|-- TSG
|    |
|    |- cider
|    |- coco-caption
|    |- data
|        |
|        |- coco_pred_sg
|        |- coco_swin_feats
|        |- cocobu.json
|        |- cocobu_label.h5
|        |- ...
```
If you download all the swin feature files, you can use the following command to decompress in a Linux system: 
```bash
cat [compressed_file_name].* | tar -xzf -
## for example: (Don't forget the . after gz)
cat feats.tar.gz.* | tar -xzf - 
```
## Training
The core code are given in the `models/TSGMModel3.py`
You can use the configs in the `/tsg_configs` to start training:
```bash
python train_tsg.py --cfg tsg_configs/tsgmt1.yml
```
Args:
- `checkpoint_path`: The storage location for the checkpoints.
- `input_att_dir`: The path for cocobu_att/coco_swin_feats. When you use the coco_swin_feats for `input_att_dir`, you need to add the `att_feat_size` into the config.
- `batch_size`: batch size=14 can be adapted to a single RTX 3090GPU (24GB), 20 requires around 33GB. You can modify it to fit your device and modify the `structure_after`, `max_epochs` to achieve better training results.

## Evaluation
### Evaluate on Karpathy's test split
```bash
python eval_tsg.py --dump_images 0 --num_images 5000 --model tsgmt1/modeltsgmt10011.pth --infos_path tsgmt1/infos_tsgmt10011.pkl  --language_eval 1 --beam_size 5
```

### Evaluate on COCO test set

```bash
$ python eval_tsg.py --input_json data/cocotest.json --input_fc_dir data/cocotest_bu_fc --input_att_dir data/cocotest_bu_att --input_label_h5 none --num_images -1 --model model.pth --infos_path infos.pkl --language_eval 0  --beam_size 5
```

You can download the preprocessed file `cocotest.json`, `cocotest_bu_att` and `cocotest_bu_fc` from [link](https://drive.google.com/open?id=1eCdz62FAVCGogOuNhy87Nmlo5_I0sH2J) according to [ruotianluo/ImageCaptioning](https://github.com/ruotianluo/ImageCaptioning.pytorch/tree/master).

## Citing
If you found this repository useful, please consider citing:
```
@inproceedings{yang-etal-2023-transforming,
    title = "Transforming Visual Scene Graphs to Image Captions",
    author = "Yang, Xu and Peng, Jiawei and Wang, Zihua and Xu, Haiyang and Ye, Qinghao and Li, Chenliang and Huang, Songfang and Huang, Fei and Li, Zhangzikang and Zhang, Yu",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2023",
    publisher = "Association for Computational Linguistics",
    doi = "10.18653/v1/2023.acl-long.694",
    pages = "12427--12440",
}
```