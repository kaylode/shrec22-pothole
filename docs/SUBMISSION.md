# Submission

## Descriptions

Description for our team's submissions. We approach this task in two ways: fully-supervised and semi-supervised semantic segmentation.
For each approach, we submit one run, for the last run is the ensemble of multiple models. 

### **Run 1: Efficient Unet++**
...

### **Run 2: Masked Soft Cross Pseudo Supervision**
...

### **Run 3: Ensemble of models**
...

## Inferences

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
│   └───2.maskedsoftcps-dlunet
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

### **Run 1: Efficient Unet++**

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

### **Run 2: Masked Soft Cross Pseudo Supervision**

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

### **Run 3: Ensemble of models**

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