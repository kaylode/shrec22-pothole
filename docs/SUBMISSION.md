# Submission

- Description for our team's submissions. We approach this task in two ways: fully-supervised and semi-supervised semantic segmentation.
- For each approach, we submit one run, for the last run is the ensemble of multiple models. 

## Training strategy
- We conduct many experiment with different augmentation and criterion settings. Finally we highlight some techniques that help us boost up the overall performance of our models. These settings are applied for all the runs we have submitted.

- Highlighted techniques:
    - Combination of Focal Tversky loss and Cross Entropy with Online Hard Example Mining (OHEM) as our objective function. This combined loss increase the precision and recall rate more than standard Cross Entropy loss.

    - Mosaic augmentation to blend multiple classes into an image. We observe that the dataset lacks of interaction between cracks and potholes (some images are full of cracks with no pothole or vice versa). This help introduce variety of possible situtations where both cracks and potholes present in the scene, which also help the model generalize better.

## **Run 1: Efficient Unet++**
- In this run, we simply adapt the traditional Unet++ with some modification. We reuse the pretrained EfficientNets on the ImageNet dataset as the new backbone and train the whole process with fully-annotated labels.


## **Run 2: Masked Soft Cross Pseudo Supervision**
- In this run, we observe that while run 1 gives overall good metric scores on the validation set, it performs worse when comes to out-of-distribution samples, such as frames from rgbd videos. We alleviate this by strengthening the model with unsupervised data or rather data "in the wild". We inherit ideas from the recent SOTA semi-supervised method: Cross Pseudo Supervision and apply with some critical improvements. Instead of using hardcoded pseudo labels, we soften them with softmax normalization and mask out the background channel, hence the name "Masked Soft CPS". The reason behind this will be discussed in the working note later. 

## **Run 3: Ensemble of models**
- In this run, we perform ensemble strategy to combine predictions from multiple models. This helps stablize and consolidate the masks prediction.  

## References

```
@incollection{zhou2018unet++,
  title={Unet++: A nested u-net architecture for medical image segmentation},
  author={Zhou, Zongwei and Rahman Siddiquee, Md Mahfuzur and Tajbakhsh, Nima and Liang, Jianming},
  booktitle={Deep learning in medical image analysis and multimodal learning for clinical decision support},
  pages={3--11},
  year={2018},
  publisher={Springer}
}
```

```
@misc{abraham2018novel,
      title={A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation}, 
      author={Nabila Abraham and Naimul Mefraz Khan},
      year={2018},
      eprint={1810.07842},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```
@inproceedings{chen2021-CPS,
  title={Semi-Supervised Semantic Segmentation with Cross Pseudo Supervision},
  author={Chen, Xiaokang and Yuan, Yuhui and Zeng, Gang and Wang, Jingdong},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```
```
@misc{filipiak2022ncps,
      title={n-CPS: Generalising Cross Pseudo Supervision to n Networks for Semi-Supervised Semantic Segmentation}, 
      author={Dominik Filipiak and Piotr Tempczyk and Marek Cygan},
      year={2022},
      eprint={2112.07528},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```