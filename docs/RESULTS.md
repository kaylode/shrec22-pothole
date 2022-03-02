# Results


## Quantitative results


Model name | Training set | Evaluation set | Dice Score | Precision | Recall | mIOU
--- | --- | --- | --- | --- | --- | ---
unet++, efficientnet-b0 | fully-labelled | fully-labelled | 0.79572	| **0.38636** | 0.39616 | **0.31355**
deeplabv3+, efficientnet-b0 | fully-labelled | fully-labelled | **0.80709**	| 0.35663 | 0.37296 | 0.28735
cps (unet++ and deeplabv3+) | fully-labelled + unlabelled | fully-labelled | 0.75947 | 0.34868 | 0.39306 | 0.29248
softcps (unet++ and deeplabv3+) | fully-labelled + unlabelled | fully-labelled | 0.78204 | 0.34106 | 0.40687 | 0.28917
masked-soft-cps (unet++ and unet++) | fully-labelled + unlabelled | fully-labelled | 0.73429 | 0.34489 | 0.4032 | 0.28718
masked-soft-cps (unet++ and deeplabv3+) | fully-labelled + unlabelled | fully-labelled | 0.76348 | 0.32661 | **0.41668** | 0.28112

## Qualitative results

- On labelled validation data



- On unlabelled validation "in the wild" data

<!-- a-20220203-110420-color-c000253-00010
a-20220203-090653-color-c000432-00006
e-20220203-093527-color-c000098-00004

potholes325.png
potholes415.png
img-585_jpg.rf.5affd0b2859d074e9e52f8540e31ce8d -->