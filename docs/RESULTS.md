# Results


## Quantitative results

- In our observation, these models have approximately the same performance on labelled validation dataset, since these samples have similar distribution with training data. The huge difference in performance can be seen in the "wild" samples which are mostly out-of-distribution data. 

Model name | Training set | Evaluation set | Dice Score | Precision | Recall | mIOU
--- | --- | --- | --- | --- | --- | ---
segformer-b0 | fully-labelled | fully-labelled | 0.68709 | 0.2803 | **0.4471** | 0.25781
unet++, efficientnet-b0 | fully-labelled | fully-labelled | 0.79572	| 0.38636 | 0.39616 | **0.31355**
deeplabv3+, efficientnet-b0 | fully-labelled | fully-labelled | **0.80709**	| 0.35663 | 0.37296 | 0.28735
cps (unet++ and deeplabv3+) | fully-labelled + unlabelled | fully-labelled | 0.75947 | 0.34868 | 0.39306 | 0.29248
softcps (unet++ and deeplabv3+) | fully-labelled + unlabelled | fully-labelled | 0.78204 | 0.34106 | 0.40687 | 0.28917
masked-soft-cps (unet++ and unet++) | fully-labelled + unlabelled | fully-labelled | 0.73429 | 0.34489 | 0.4032 | 0.28718
masked-soft-cps (unet++ and deeplabv3+) | fully-labelled + unlabelled | fully-labelled | 0.76348 | 0.32661 | **0.41668** | 0.28112

## Qualitative results

- On labelled validation data

Input image | Ground truth | Efficient Unet++ | Masked Soft CPS
| :------------: | :------------: | :------------: | :------------: |
| <img height="150" alt="screen" src="./figures/qualitative/labelled/gt/010.jpg">  | <img height="150" alt="screen" src="./figures/qualitative/labelled/gt/010_mask.png">  | <img height="150" alt="screen" src="./figures/qualitative/labelled/unetplusplus/010.jpg">  | <img height="150" alt="screen" src="./figures/qualitative/labelled/masked-soft-cps/010.jpg">  | 
| <img height="150" alt="screen" src="./figures/qualitative/labelled/gt/20160316_143445.jpg">  | <img height="150" alt="screen" src="./figures/qualitative/labelled/gt/20160316_143445_mask.png">  | <img height="150" alt="screen" src="./figures/qualitative/labelled/unetplusplus/20160316_143445.jpg">  | <img height="150" alt="screen" src="./figures/qualitative/labelled/masked-soft-cps/20160316_143445.jpg">  | 
| <img height="150" alt="screen" src="./figures/qualitative/labelled/gt/994110_RS_386_386RS191729_07315.jpg">  | <img height="150" alt="screen" src="./figures/qualitative/labelled/gt/994110_RS_386_386RS191729_07315_mask.png">  | <img height="150" alt="screen" src="./figures/qualitative/labelled/unetplusplus/994110_RS_386_386RS191729_07315.jpg">  | <img height="150" alt="screen" src="./figures/qualitative/labelled/masked-soft-cps/994110_RS_386_386RS191729_07315.jpg">  | 


- On unlabelled validation "in the wild" data

Input image | Efficient Unet++ | Efficient DeepLabV3+ | Masked Soft CPS
| :------------: | :------------: | :------------: | :------------: |
| <img width="230" alt="screen" src="./figures/qualitative/unlabelled/meta/raw/potholes325.png">  | <img width="230" alt="screen" src="./figures/qualitative/unlabelled/meta/unetplusplus/potholes325.png">  | <img width="230" alt="screen" src="./figures/qualitative/unlabelled/meta/deeplabv3plus/potholes325.png">  | <img width="230" alt="screen" src="./figures/qualitative/unlabelled/meta/masked-soft-cps/potholes325.png">  | 
| <img width="230" alt="screen" src="./figures/qualitative/unlabelled/meta/raw/potholes415.png">  | <img width="230" alt="screen" src="./figures/qualitative/unlabelled/meta/unetplusplus/potholes415.png">  | <img width="230" alt="screen" src="./figures/qualitative/unlabelled/meta/deeplabv3plus/potholes415.png">  | <img width="230" alt="screen" src="./figures/qualitative/unlabelled/meta/masked-soft-cps/potholes415.png">  | 
| <img width="230" alt="screen" src="./figures/qualitative/unlabelled/meta/raw/img-585_jpg.rf.5affd0b2859d074e9e52f8540e31ce8d.jpg">  | <img width="230" alt="screen" src="./figures/qualitative/unlabelled/meta/unetplusplus/img-585_jpg.rf.5affd0b2859d074e9e52f8540e31ce8d.jpg">  | <img width="230" alt="screen" src="./figures/qualitative/unlabelled/meta/deeplabv3plus/img-585_jpg.rf.5affd0b2859d074e9e52f8540e31ce8d.jpg">  | <img width="230" alt="screen" src="./figures/qualitative/unlabelled/meta/masked-soft-cps/img-585_jpg.rf.5affd0b2859d074e9e52f8540e31ce8d.jpg">  | 


- On unlabelled extracted frame from disparity camera

Input image | Efficient Unet++ | Efficient DeepLabV3+ | Masked Soft CPS
| :------------: | :------------: | :------------: | :------------: |
| <img height="150" alt="screen" src="./figures/qualitative/unlabelled/disp/raw/e-20220203-093527-color-c000098-00004.jpg">  | <img height="150" alt="screen" src="./figures/qualitative/unlabelled/disp/unetplusplus/e-20220203-093527-color-c000098-00004.jpg">  | <img height="150" alt="screen" src="./figures/qualitative/unlabelled/disp/deeplabv3plus/e-20220203-093527-color-c000098-00004.jpg">  | <img height="150" alt="screen" src="./figures/qualitative/unlabelled/disp/masked-soft-cps-dlunet/e-20220203-093527-color-c000098-00004.jpg">  | 
| <img height="150" alt="screen" src="./figures/qualitative/unlabelled/disp/raw/a-20220203-110420-color-c000253-00010.jpg">  | <img height="150" alt="screen" src="./figures/qualitative/unlabelled/disp/unetplusplus/a-20220203-110420-color-c000253-00010.jpg">  | <img height="150" alt="screen" src="./figures/qualitative/unlabelled/disp/deeplabv3plus/a-20220203-110420-color-c000253-00010.jpg">  | <img height="150" alt="screen" src="./figures/qualitative/unlabelled/disp/masked-soft-cps-dlunet/a-20220203-110420-color-c000253-00010.jpg">  | 
| <img height="150" alt="screen" src="./figures/qualitative/unlabelled/disp/raw/a-20220203-090653-color-c000432-00006.jpg">  | <img height="150" alt="screen" src="./figures/qualitative/unlabelled/disp/unetplusplus/a-20220203-090653-color-c000432-00006.jpg">  | <img height="150" alt="screen" src="./figures/qualitative/unlabelled/disp/deeplabv3plus/a-20220203-090653-color-c000432-00006.jpg">  | <img height="150" alt="screen" src="./figures/qualitative/unlabelled/disp/masked-soft-cps-dlunet/a-20220203-090653-color-c000432-00006.jpg">  | 