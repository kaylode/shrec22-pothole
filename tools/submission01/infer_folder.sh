# ./tools/submission_1_video.sh data/a-20220203-085153-color-c000169.mp4 cpu 
CHECKPOINT='ckpt/unet/best.pth'
CONFIG_TRANSFORM='ckpt/unet/transform.yml'
SAVE_DIR='./output/submission01'
CONFIG_TEST='tools/submission01/test_images.yml'
CLS_TXT='tools/classes.txt'

mkdir -p $SAVE_DIR

python  configs/segmentation/infer.py \
        -c $CONFIG_TEST \
        -o global.save_dir=$SAVE_DIR \
        global.device=$2 \
        global.cfg_transform=$CONFIG_TRANSFORM \
        global.weights=$CHECKPOINT \
        data.dataset.args.txt_classnames=$CLS_TXT \
        data.dataset.args.image_dir=$1