# Datasets

## Non-labelled dataset

- Link: [Google Drive](https://drive.google.com/file/d/18chlFiZkM0tDdiF7eY8Q-Q7koBlBWtWk/view?usp=sharing)
- We utilize non-annotated datasets for unsupervised training branch. These are images of cracks and potholes "in the wild" which will enhance the capabilities of model predicting on out-of-distribution samples.

<p align="center"> <img width="1200" alt="screen" src="docs/figures/meta.png"> </p>

- Moreover, the provided rgb-d videos from the organizers are cut into separated frames and used as unlabelled samples as well.

<p align="center"> <img width="1200" alt="screen" src="docs/figures/disp.png"> </p>


- Sources:
    - https://public.roboflow.com/object-detection/pothole
    - https://www.kaggle.com/atulyakumar98/pothole-detection-dataset
    - https://www.kaggle.com/sachinpatel21/pothole-image-dataset


## Pixel-wise annotated dataset (Pothole Mix dataset)

- Link: [Google Drive](https://drive.google.com/drive/folders/1LFMWnjBxITs_j0cmLky_awIgcn8eroaA?usp=sharing)

- We use annotated data which have been given to all the track's participants for fully-supervised training.

> This dataset for the semantic segmentation of potholes and cracks on the road surface was assembled from 5 other datasets already publicly available, plus a very small addition of segmented images on our part. To speed up the labeling operations, we started working with depth cameras to try to automate, to some extent, this extremely time-consuming phase. We hope to be able to release a new version of the Pothole Mix dataset soon with a more significant contribution from us.
>
> Below, you can find links to all the sources that constitute the Pothole Mix dataset and the citations of the papers that published those datasets.

<p align="center"> <img width="1200" alt="screen" src="docs/figures/datasets.png"> </p>


crack500
```
@inproceedings{zhang2016road, title={Road crack detection using deep convolutional neural network}, author={Zhang, Lei and Yang, Fan and Zhang, Yimin Daniel and Zhu, Ying Julie}, booktitle={Image Processing (ICIP), 2016 IEEE International Conference on}, pages={3708--3712}, year={2016}, organization={IEEE} }'
```
```
@article{yang2019feature, title={Feature Pyramid and Hierarchical Boosting Network for Pavement Crack Detection}, author={Yang, Fan and Zhang, Lei and Yu, Sijia and Prokhorov, Danil and Mei, Xue and Ling, Haibin}, journal={IEEE Transactions on Intelligent Transportation Systems}, year={2019}, publisher={IEEE} }
```

GAPs384
```
@article{yang2019feature, title={Feature Pyramid and Hierarchical Boosting Network for Pavement Crack Detection}, author={Yang, Fan and Zhang, Lei and Yu, Sijia and Prokhorov, Danil and Mei, Xue and Ling, Haibin}, journal={IEEE Transactions on Intelligent Transportation Systems}, year={2019}, publisher={IEEE} }
```
```
@inproceedings{eisenbach2017how, title={How to Get Pavement Distress Detection Ready for Deep Learning? A Systematic Approach.}, author={Eisenbach, Markus and Stricker, Ronny and Seichter, Daniel and Amende, Karl and Debes, Klaus and Sesselmann, Maximilian and Ebersbach, Dirk and Stoeckert, Ulrike and Gross, Horst-Michael}, booktitle={International Joint Conference on Neural Networks (IJCNN)}, pages={2039--2047}, year={2017} }
```


EdmCrack600

```
@article{MEI2020103018, title = {Densely connected deep neural network considering connectivity of pixels for automatic crack detection}, journal = {Automation in Construction}, volume = {110}, pages = {103018}, year = {2020}, issn = {0926-5805}, doi = {https://doi.org/10.1016/j.autcon.2019.103018}, url = {https://www.sciencedirect.com/science/article/pii/S0926580519307502}, author = {Qipei Mei and Mustafa Gül and Md Riasat Azim}}
```
```
@article{MEI2020119397, title = {A cost effective solution for pavement crack inspection using cameras and deep neural networks}, journal = {Construction and Building Materials}, volume = {256}, pages = {119397}, year = {2020}, issn = {0950-0618}, doi = {https://doi.org/10.1016/j.conbuildmat.2020.119397}, url = {https://www.sciencedirect.com/science/article/pii/S0950061820314021}, author = {Qipei Mei and Mustafa Gül}}
```
```
@article{Mei2020, author="Mei, Qipei and G{\"u}l, Mustafa and Shirzad-Ghaleroudkhani, Nima", title="Towards smart cities: crowdsensing-based monitoring of transportation infrastructure using in-traffic vehicles", journal="Journal of Civil Structural Health Monitoring", year="2020", month="Sep", day="01", volume="10", number="4", pages="653--665", issn="2190-5479", doi="10.1007/s13349-020-00411-6", url="https://doi.org/10.1007/s13349-020-00411-6"}
```

Pothole-600

```
@InProceedings{10.1007/978-3-030-66823-5_17, author="Fan, Rui and Wang, Hengli and Bocus, Mohammud J. and Liu, Ming", editor="Bartoli, Adrien and Fusiello, Andrea", title="We Learn Better Road Pothole Detection: From Attention Aggregation to Adversarial Domain Adaptation", booktitle="Computer Vision -- ECCV 2020 Workshops", year="2020", publisher="Springer International Publishing", address="Cham", pages="285--300", isbn="978-3-030-66823-5"}
@article{8300645, author={Fan, Rui and Ai, Xiao and Dahnoun, Naim}, journal={IEEE Transactions on Image Processing},title={Road Surface 3D Reconstruction Based on Dense Subpixel Disparity Map Estimation}, year={2018}, volume={27}, number={6}, pages={3025-3035}, doi={10.1109/TIP.2018.2808770}}
```
```
@article{8809907, author={Fan, Rui and Ozgunalp, Umar and Hosking, Brett and Liu, Ming and Pitas, Ioannis}, journal={IEEE Transactions on Image Processing}, title={Pothole Detection Based on Disparity Transformation and Road Surface Modeling}, year={2020}, volume={29}, number={}, pages={897-908}, doi={10.1109/TIP.2019.2933750}}
@article{8890001, author={Fan, Rui and Liu, Ming}, journal={IEEE Transactions on Intelligent Transportation Systems}, title={Road Damage Detection Based on Unsupervised Disparity Map Segmentation}, year={2020}, volume={21}, number={11}, pages={4906-4911}, doi={10.1109/TITS.2019.2947206}}
```

Cracks and Potholes in Road Images Dataset
```
@article{passos_cassaniga_fernandes_medeiro_comunello_2020, title={Cracks and Potholes in Road Images Dataset}, url={10.17632/t576ydh9v8.4}, author={Passos, Bianka Tallita and Cassaniga, Mateus Junior and Fernandes, Anita Maria da Rocha and Medeiro, Kátya Balvedi and Comunello, Eros}, year={2020}}
```

