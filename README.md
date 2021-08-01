# Structured Context Enhancement Network for Mouse Pose Estimation

This repository provides a PyTorch implementation of the paper [ Structured Context Enhancement Network for Mouse Pose Estimation](https://ieeexplore.ieee.org/abstract/document/9492104). 

## Demo

<img src="1-1.gif" width="40%">           <img src="1-2.gif" width="40%">

## Requirements
Tested with:
* PyTorch 1.4.0

* Torchvision 0.5.0

* Python 3.6.8

## Data Preparation
* Download the  data from [DeepLabCut Mouse Pose](https://zenodo.org/record/4008504#.YQaXwpNKjDJ) and [DeepPoseKit Animal Pose](https://github.com/jgraving/DeepPoseKit-Data/tree/master/datasets). Then put them under the data directory:
	-labeled-data\  
	  -mouse\  
              ...
	  -flyimage\  
              ...   
          -zebraimage\  
              ...
              
## Training

* Before running `train.py`, we need to compile Directionmax operation used in our paper, which is inspired by the corner pooling scheme in [CornerNet](https://github.com/princeton-vl/CornerNet).
```
`cd <CornerNet dir>/models/py_utils/_cpools/`
`python setup.py install --user`
```
* Then train the model
```
`python train.py`
```

## Ethical Proof
All experimental procedures were performed in accordance with the Guidance on the Operation of the Animals (Scientific Procedures) Act, 1986 (UK) and approved by the Queen’s University Belfast Animal.

## Citation
If you find this repository useful, please cite our paper:
```
@article{zhou2021structured,
  title={Structured Context Enhancement Network for Mouse Pose Estimation},
  author={Zhou, Feixiang and Jiang, Zheheng and Liu, Zhihua and Chen, Fang and Chen, Long and Tong, Lei and Yang, Zhile and Wang, Haikuan and Fei, Minrui and Li, Ling and others},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2021},
  publisher={IEEE}
}
```



