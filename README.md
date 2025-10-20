# MPDA
Multi-Person Data Augmentation for 3d Human Pose Estimation

This repository is the implementation for:

> ["Overcoming single-person constraints in multi-person 3D human pose estimation: A data augmentation technique with occlusion-aware heatmaps"](https://link.springer.com/article/10.1007/s00371-025-04040-2)
>
> Sanghyeon Lee, Jong Taek Lee
>
> The Visual Computer
> 

## Training Pipeline
![overall](https://github.com/hyeon0819/MPDA/assets/153258272/e933f34e-43c2-4172-b0af-db4305850e50)


## Dataset
We follow [VoxelPose](https://github.com/microsoft/voxelpose-pytorch) to download the dataset.
### Shelf/Campus datasets
1. Download the datasets from [Shelf & Campus](http://campar.in.tum.de/Chair/MultiHumanPose) and extract them under `${POSE_ROOT}/data/Shelf` and `${POSE_ROOT}/data/CampusSeq1`, respectively.

2. We have processed the camera parameters to our formats and you can download them from this repository. They lie in `${POSE_ROOT}/data/Shelf/` and `${POSE_ROOT}/data/CampusSeq1/`, respectively.

3. Due to the limited and incomplete annotations of the two datasets, we don't train our model using this dataset. Instead, we directly use the 2D pose estimator trained on COCO, and use independent 3D human poses from the Panoptic dataset to train our 3D model. It lies in `${POSE_ROOT}/data/panoptic_training_pose.pkl.` See our paper for more details.

4. For testing, we first estimate 2D poses and generate 2D heatmaps for these two datasets in this repository. The predicted poses can also download from the repository. They lie in `${POSE_ROOT}/data/Shelf/` and `${POSE_ROOT}/data/CampusSeq1/`, respectively. You can also use the models trained on COCO dataset (like HigherHRNet) to generate 2D heatmaps directly.
   
### CMU Panoptic datasets
1. Download the dataset by following the instructions in [panoptic-toolbox](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox) and extract them under `${POSE_ROOT}/data/panoptic_toolbox/data`.
- You can only download those sequences you need. You can also just download a subset of camera views by specifying the number of views (HD_Video_Number) and changing the camera order in `./scripts/getData.sh`. The sequences and camera views used in our project can be obtained from our paper.
- Note that we only use HD videos, calibration data, and 3D Body Keypoint in the codes. You can comment out other irrelevant codes such as downloading 3D Face data in `./scripts/getData.sh`.

2. Download the pretrained backbone model from [pretrained backbone](https://onedrive.live.com/?id=93774C670BD4F835!1917&resid=93774C670BD4F835!1917&authkey=!AMf08ZItxtILRuU&cid=93774c670bd4f835) and place it here: `${POSE_ROOT}/models/pose_resnet50_panoptic.pth.tar` (ResNet-50 pretrained on COCO dataset and finetuned jointly on Panoptic dataset and MPII).

To download the organization of the data in this paper, run `sh scripts/getData_voxelpose.sh`.

## Visual Results
![qualitative1](https://github.com/hyeon0819/MPDA/assets/153258272/aa497e5d-2a2e-4f20-b64e-b3e47b565b1a)

## Citation
If you use our method in your research, please cite with:
`
@article{lee2025occlusion,
  title={Occlusion-aware heatmap generation for enhancing 3D human pose estimation in multi-person environments},
  author={Lee, Sanghyeon and Lee, Jong Taek},
  journal={The Visual Computer},
  pages={1--13},
  year={2025},
  publisher={Springer}
}
`
