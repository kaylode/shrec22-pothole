# Training and Evaluation

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

## Resources
- Training and evaluation notebook: coming soon