# MLSFNet
A Multi-scale Learnable Soft-Fusion Network for Breast Cancer Pathology Image Classification
## Installation 
We express our respect for their outstanding work. To prepare the environment, please follow the following instructions.<br>
<code>conda create --name openmmlab python=3.8 -y</code><br>
<code>conda activate openmmlab</code><br>
<code>conda install pytorch torchvision -c pytorch</code> <br> This command will automatically install the latest version PyTorch and cudatoolkit, please check whether they match your environment.<br>
<code>pip install -U openmim</code><br>
<code>mim install mmengine</code><br>
<code>mim install mmcv</code><br>
<code>mim install mmdet</code><br> 
<code>mim install mmpose</code><br>
<code>cd MLSFNet</code><br>
<code>pip install -v -e .</code><br>
## Datasets
The used datasets are provided in [BACH](https://drive.google.com/drive/folders/0B5peJ1MHnIWGd3pFbzMyTG5BSGs?resourcekey=0-hZ1jo5t1hIauRpYhYIvWYA) and [BRACS](https://github.com/kittenish/Frame-Transformer-Network). The train/test splits in both two datasets follow the official procedure. To prepare the data, you can refer to VideoMAE V2 for a general guideline.
## Model
We now provide the model weights in the following [link](https://pan.baidu.com/s/1nt9LOERcNLfcv-i3EdVPow?pwd=mlsf).
## Train
The model can be trained with the following command.<br>
<code>export CUBLAS_WORKSPACE_CONFIG=":4096:8"</code><br>
<code>CUDA_VISIBLE_DEVICES='0,1' bash tools/dist_test.sh configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip_8xb32-u8_kinetics400-rgb.py 2 --seed 220 --deterministic</code><br>
## Acknowledgement
This project is based on [MMPretrain](https://github.com/open-mmlab/mmpretrain). Thanks to the OpenMMLab team for their great work.


