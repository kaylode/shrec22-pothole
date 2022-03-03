# Submission

- Description for our team's submissions. We approach this task in two ways: fully-supervised and semi-supervised semantic segmentation.
- For each approach, we submit one run, for the last run is the ensemble of multiple models. 

## Training strategy
- We conduct many experiment with different augmentation and criterion settings. Finally we highlight some techniques that help us boost up the overall performance of our models. These settings are applied for all the runs we have submitted.

- Highlighted techniques:
    - Combination of Focal Tversky loss and Cross Entropy with Online Hard Example Mining (OHEM) as our objective function. This combined loss increase the precision and recall rate more than standard Cross Entropy loss.

    <p align="center"> <img height="50" alt="screen" src="./figures/tversky.png"> <img height="50" alt="screen" src="./figures/focal_tversky.png"> <br> <strong>Focal Tversky loss</strong> </p>


    - Mosaic augmentation to blend multiple images into a single one. We observe that the dataset lacks of interaction between cracks and potholes (some images are full of cracks with no pothole or vice versa). This help introduce variety of possible situtations where both cracks and potholes present in the same scene, which also help the model generalize better.

    <p align="center"> <img height="250" alt="screen" src="./figures/mosaic.png"> <br> <strong>Mosaic augmenation merges 4 input images into one</strong></p>

## **Run 1: Efficient DeepLabV3+**
- In this run, we simply adapt the traditional DeepLabV3+ with some modification. We reuse the pretrained EfficientNets on the ImageNet dataset as the new backbone and train the whole process with fully-annotated labels.

## **Run 2: Masked Soft Cross Pseudo Supervision**
- In this run, we observe that while run 1 gives overall good metric scores on the validation set, it performs worse when comes to out-of-distribution samples, such as frames from rgbd videos. We alleviate this by strengthening the model with unsupervised data or rather data "in the wild". We inherit ideas from the recent SOTA semi-supervised method: Cross Pseudo Supervision (CPS) and apply with some critical improvements. Instead of using hardcoded pseudo labels, we soften them with softmax normalization and mask out the background channel, hence the name "Masked Soft CPS". The reason behind this will be discussed in the working note later. 

- CPS works by combining both the annotated and non-annotated data and train two neural networks simultaneously (DeepLabV3+ and Unet++ in our experiment). For the annotated samples, supervision loss is applied typically. For the non-annotated, the outputs from one model become the other's targets and are judged also by the supervision loss. The figures bellow illustrate the training pipeline. 


|  Supervised branch | Unsupervised branch |
| :----------------------------------------------------------: | :----------------------------------------------------------: | 
| <img height="150" alt="screen" src="./figures/supervised.png">  | <img height="150" alt="screen" src="./figures/unsupervised.png">  | 

## References

<details close> <summary><strong>Show more</strong></summary>

``` bibtex
@incollection{zhou2018unet++,
  title={Unet++: A nested u-net architecture for medical image segmentation},
  author={Zhou, Zongwei and Rahman Siddiquee, Md Mahfuzur and Tajbakhsh, Nima and Liang, Jianming},
  booktitle={Deep learning in medical image analysis and multimodal learning for clinical decision support},
  pages={3--11},
  year={2018},
  publisher={Springer}
}
```

``` bibtex
@inproceedings{chen2018encoder,
  title={Encoder-decoder with atrous separable convolution for semantic image segmentation},
  author={Chen, Liang-Chieh and Zhu, Yukun and Papandreou, George and Schroff, Florian and Adam, Hartwig},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={801--818},
  year={2018}
}
```

``` bibtex
@article{xie2021segformer,
  title={SegFormer: Simple and efficient design for semantic segmentation with transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

``` bibtex
@inproceedings{abraham2019novel,
  title={A novel focal tversky loss function with improved attention u-net for lesion segmentation},
  author={Abraham, Nabila and Khan, Naimul Mefraz},
  booktitle={2019 IEEE 16th international symposium on biomedical imaging (ISBI 2019)},
  pages={683--687},
  year={2019},
  organization={IEEE}
}
```

``` bibtex
@inproceedings{chen2021-CPS,
  title={Semi-Supervised Semantic Segmentation with Cross Pseudo Supervision},
  author={Chen, Xiaokang and Yuan, Yuhui and Zeng, Gang and Wang, Jingdong},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```
``` bibtex
@article{filipiak2021n,
  title={$ n $-CPS: Generalising Cross Pseudo Supervision to $ n $ networks for Semi-Supervised Semantic Segmentation},
  author={Filipiak, Dominik and Tempczyk, Piotr and Cygan, Marek},
  journal={arXiv preprint arXiv:2112.07528},
  year={2021}
}
```
</details>
