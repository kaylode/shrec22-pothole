# References

- We want to give many thanks for these open-source repositories, public datasets and researches which greatly contribute to our challenge approaches. 

## Code References
- https://github.com/kaylode/theseus
- https://github.com/charlesCXK/TorchSemiSeg
- https://github.com/qubvel/segmentation_models.pytorch
- https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch

## Paper References

``` bibtex
@misc{Yakubovskiy:2019,
  Author = {Pavel Yakubovskiy},
  Title = {Segmentation Models Pytorch},
  Year = {2020},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/qubvel/segmentation_models.pytorch}}
}
```

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

## Data References

Unlabelled data:
- https://public.roboflow.com/object-detection/pothole
- https://www.kaggle.com/atulyakumar98/pothole-detection-dataset
- https://www.kaggle.com/sachinpatel21/pothole-image-dataset

Labelled data:

crack500
``` bibtex
@inproceedings{zhang2016road, 
    title={Road crack detection using deep convolutional neural network}, 
    author={Zhang, Lei and Yang, Fan and Zhang, Yimin Daniel and Zhu, Ying Julie}, 
    booktitle={Image Processing (ICIP), 2016 IEEE International Conference on}, 
    pages={3708--3712}, 
    year={2016}, 
    organization={IEEE} 
}
```
``` bibtex
@article{yang2019feature, 
    title={Feature Pyramid and Hierarchical Boosting Network for Pavement Crack Detection}, 
    author={Yang, Fan and Zhang, Lei and Yu, Sijia and Prokhorov, Danil and Mei, Xue and Ling, Haibin}, 
    journal={IEEE Transactions on Intelligent Transportation Systems}, 
    year={2019}, publisher={IEEE} 
}
```

GAPs384
``` bibtex
@article{yang2019feature, 
    title={Feature Pyramid and Hierarchical Boosting Network for Pavement Crack Detection}, 
    author={Yang, Fan and Zhang, Lei and Yu, Sijia and Prokhorov, Danil and Mei, Xue and Ling, Haibin}, 
    journal={IEEE Transactions on Intelligent Transportation Systems}, 
    year={2019}, publisher={IEEE} 
}
```
``` bibtex
@inproceedings{eisenbach2017how, 
    title={How to Get Pavement Distress Detection Ready for Deep Learning? A Systematic Approach.}, 
    author={Eisenbach, Markus and Stricker, Ronny and Seichter, Daniel and Amende, Karl and Debes, 
    Klaus and Sesselmann, Maximilian and Ebersbach, Dirk and Stoeckert, Ulrike and Gross, Horst-Michael}, 
    booktitle={International Joint Conference on Neural Networks (IJCNN)}, 
    pages={2039--2047}, year={2017} 
}
```


EdmCrack600

``` bibtex
@article{MEI2020103018, 
    title = {Densely connected deep neural network considering connectivity of pixels for automatic crack detection}, 
    journal = {Automation in Construction}, 
    volume = {110}, pages = {103018}, 
    year = {2020}, issn = {0926-5805}, 
    doi = {https://doi.org/10.1016/j.autcon.2019.103018}, 
    url = {https://www.sciencedirect.com/science/article/pii/S0926580519307502}, 
    author = {Qipei Mei and Mustafa Gül and Md Riasat Azim}
}
```
``` bibtex
@article{MEI2020119397, 
    title = {A cost effective solution for pavement crack inspection using cameras and deep neural networks}, 
    journal = {Construction and Building Materials}, 
    volume = {256}, pages = {119397}, year = {2020}, 
    issn = {0950-0618}, doi = {https://doi.org/10.1016/j.conbuildmat.2020.119397}, 
    url = {https://www.sciencedirect.com/science/article/pii/S0950061820314021}, 
    author = {Qipei Mei and Mustafa Gül}
}
```
``` bibtex
@article{Mei2020, 
    author="Mei, Qipei and G{\"u}l, Mustafa and Shirzad-Ghaleroudkhani, Nima", 
    title="Towards smart cities: crowdsensing-based monitoring of transportation infrastructure using in-traffic vehicles", 
    journal="Journal of Civil Structural Health Monitoring", year="2020", 
    month="Sep", day="01", volume="10", number="4", pages="653--665", issn="2190-5479", 
    doi="10.1007/s13349-020-00411-6", 
    url="https://doi.org/10.1007/s13349-020-00411-6"
}
```

Pothole-600

``` bibtex
@InProceedings{10.1007/978-3-030-66823-5_17, 
    author="Fan, Rui and Wang, Hengli and Bocus, Mohammud J. and Liu, Ming", 
    editor="Bartoli, Adrien and Fusiello, Andrea", 
    title="We Learn Better Road Pothole Detection: From Attention Aggregation to Adversarial Domain Adaptation", 
    booktitle="Computer Vision -- ECCV 2020 Workshops", year="2020", 
    publisher="Springer International Publishing", address="Cham", 
    pages="285--300", isbn="978-3-030-66823-5"
}
``` 
``` bibtex
@article{8300645, 
    author={Fan, Rui and Ai, Xiao and Dahnoun, Naim}, 
    journal={IEEE Transactions on Image Processing},
    title={Road Surface 3D Reconstruction Based on Dense Subpixel Disparity Map Estimation}, 
    year={2018}, volume={27}, number={6}, pages={3025-3035}, doi={10.1109/TIP.2018.2808770}
}
```
``` bibtex
@article{8809907, 
    author={Fan, Rui and Ozgunalp, Umar and Hosking, Brett and Liu, Ming and Pitas, Ioannis}, 
    journal={IEEE Transactions on Image Processing}, 
    title={Pothole Detection Based on Disparity Transformation and Road Surface Modeling}, 
    year={2020}, volume={29}, number={}, pages={897-908}, doi={10.1109/TIP.2019.2933750}
}
```
``` bibtex
@article{8890001, 
    author={Fan, Rui and Liu, Ming}, 
    journal={IEEE Transactions on Intelligent Transportation Systems}, 
    title={Road Damage Detection Based on Unsupervised Disparity Map Segmentation}, 
    year={2020}, volume={21}, number={11}, pages={4906-4911}, doi={10.1109/TITS.2019.2947206}
}
```

Cracks and Potholes in Road Images Dataset
``` bibtex
@article{passos_cassaniga_fernandes_medeiro_comunello_2020, 
    title={Cracks and Potholes in Road Images Dataset}, 
    url={10.17632/t576ydh9v8.4}, 
    author={Passos, Bianka Tallita and Cassaniga, Mateus Junior and Fernandes, 
    Anita Maria da Rocha and Medeiro, Kátya Balvedi and Comunello, Eros}, 
    year={2020}
}
```