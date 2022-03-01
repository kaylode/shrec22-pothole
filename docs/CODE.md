# Reproducibility

## Installation

- Install requirements
```
pip install -e .
```

## Fully-supervised

- Modify configuration file: `configs/segmentation/pipeline.yaml`
- To train:
```
python  configs/segmentation/train.py \
        -c configs/segmentation/pipeline.yaml \
        -o global.save_dir=runs 
```

- To evaluate:
```
python  configs/segmentation/eval.py \
        -c configs/segmentation/pipeline.yaml \
        -o global.save_dir=runs \
        global.pretrained=<path to checkpoint>
```

- To inference on folder of images:
```
python  configs/segmentation/infer.py \
        -c configs/segmentation/test.yaml \
        -o global.save_dir=runs \
        global.weights=<path to checkpoint>
```

- To inference on a single video:
```
python  configs/segmentation/infer_video.py \
        -c configs/segmentation/test_video.yaml \
        -o global.save_dir=runs \
        global.weights=<path to checkpoint>
```

## Semi - supervised (Masked Soft CPS)

- Modify configuration file: `configs/cps/pipeline.yaml`
- To train:
```
python  configs/cps/train.py \
        -c configs/cps/pipeline.yaml \
        -o global.save_dir=runs \
```

- To evaluate:
```
python  configs/cps/eval.py \
        -c configs/cps/pipeline.yaml \
        -o global.save_dir=runs \
        global.pretrained=<path to checkpoint>
```

- To inference on folder of images:
```
python  configs/cps/infer.py \
        -c configs/cps/test.yaml \
        -o global.save_dir=runs \
        global.weights=<path to checkpoint>
```

- To inference on a single video:
```
python  configs/cps/infer_video.py \
        -c configs/cps/test_video.yaml \
        -o global.save_dir=runs \
        global.weights=<path to checkpoint>
```

## Reproduce reported results

- Comming soon