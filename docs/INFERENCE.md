# Inferences

- This guide you on how to perform inference using our submited models for crack and pothole semantic segmentation on given images and videos. 

## Preparation
- Install the requirements using `pip install -e .` in the root folder

- Download the checkpoints and yaml config files and extract to the root folder. In each of these extracted folder, there will be 4 files `best.pth`, `test.yaml`, `test_video.yaml`, `transform.yaml`. Except for `3.ensembles` folder which doesn't have separate `.pth` file.

- The project structure should be as follow:
```
root
└───ckpt
│   └───1.unetplusplus-tversky
│   │   best.pth
│   │   test.yaml
│   │   test_video.yaml
│   │   transform.yaml
│   └───2.maskedsoftcps-doubleunets
│   │   ....
│   └───3.ensembles
│   │   ....
│   │
└───configs  
│   └───segmentation
│   └───cps
│   └───ensemble
|   ...
```

- After the execution is finished, the results will be saved into `outputs` folder

## **Run 1: Efficient Unet++**

- On images
``` bash
python configs/segmentation/infer.py \
        -c ckpt/1.unetplusplus-tversky/test.yaml \
        -o data.dataset.args.image_dir=$IMAGE_DIR
```

- On single video
``` bash
python configs/segmentation/infer_video.py \
        -c ckpt/1.unetplusplus-tversky/test_video.yaml \
        -o data.dataset.args.video_path=$VIDEO_PATH
```

## **Run 2: Masked Soft Cross Pseudo Supervision**

- On images
``` bash
python configs/cps/infer.py \
        -c ckpt/2.maskedsoftcps-doubleunets/test.yaml \
        -o data.dataset.args.image_dir=$IMAGE_DIR
```

- On single video
``` bash
python configs/cps/infer_video.py \
        -c ckpt/2.maskedsoftcps-doubleunets/test_video.yaml \
        -o data.dataset.args.video_path=$VIDEO_PATH
```

## **Run 3: Ensemble of models**

- On images
``` bash
python configs/ensemble/infer.py \
        -c ckpt/3.ensembles/test.yaml \
        -o data.dataset.args.image_dir=$IMAGE_DIR
```

- On single video
``` bash
python configs/ensemble/infer_video.py \
        -c ckpt/3.ensembles/test_video.yaml \
        -o data.dataset.args.video_path=$VIDEO_PATH
```

## Notebooks

- Example notebook for inference: [![Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GFSW_T0Mb57hzaPu4xu1u4iq9GGTGpYX?usp=sharing)