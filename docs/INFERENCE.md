# Inferences

- This guide you on how to perform inference using our submited models for crack and pothole semantic segmentation on given images and videos. 

## Preparation
- Install the requirements using `pip install -e .` in the root folder

- Download the checkpoints and yaml config files from [Google Drive](https://drive.google.com/file/d/1mWj4i9pMuC6_1UjrKg_QXBPhLEkuQz4U/view?usp=sharing) and extract to the root folder. In each of these extracted folder, there will be 4 files `best.pth`, `test.yaml`, `test_video.yaml`, `transform.yaml`.

- The project structure should be as follow:
```
root
└───ckpt
│   └───1.deeplabv3plus
│   │   best.pth
│   │   test.yaml
│   │   test_video.yaml
│   │   transform.yaml
│   └───2.maskedsoftcps-dlunet
│   │   ....
│   │
└───configs  
│   └───segmentation
│   └───cps
|   ...
```

- After the execution is finished, the results will be saved into `outputs` folder

## **Run 0: SegFormer**

- On images
``` bash
python configs/segmentation/infer.py \
        -c ckpt/0.segformer/test.yaml \
        -o data.dataset.args.image_dir=$IMAGE_DIR
```

- On single video
``` bash
python configs/segmentation/infer_video.py \
        -c ckpt/0.segformer/test_video.yaml \
        -o data.dataset.args.video_path=$VIDEO_PATH
```

## **Run 1: Efficient DeeplabV3+**

- On images
``` bash
python configs/segmentation/infer.py \
        -c ckpt/1.deeplabv3plus/test.yaml \
        -o data.dataset.args.image_dir=$IMAGE_DIR
```

- On single video
``` bash
python configs/segmentation/infer_video.py \
        -c ckpt/1.deeplabv3plus/test_video.yaml \
        -o data.dataset.args.video_path=$VIDEO_PATH
```

## **Run 2: Masked Soft Cross Pseudo Supervision**

- On images
``` bash
python configs/cps/infer.py \
        -c ckpt/2.maskedsoftcps-dlunet/test.yaml \
        -o data.dataset.args.image_dir=$IMAGE_DIR
```

- On single video
``` bash
python configs/cps/infer_video.py \
        -c ckpt/2.maskedsoftcps-dlunet/test_video.yaml \
        -o data.dataset.args.video_path=$VIDEO_PATH
```

## Notebooks

- Example notebook for inference: [![Notebook](https://colab.research.google.com/assets/colab-badge.svg)](./assets/[SHREC22]_HCMUS_Inference.ipynb)