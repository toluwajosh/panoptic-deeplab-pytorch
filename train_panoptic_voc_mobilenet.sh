python train_panoptic.py \
--backbone mobilenet_3stage \
--lr 0.007 \
--workers 4 \
--use-sbd \
--epochs 100 \
--batch-size 2 \
--checkname panoptic-deeplab-mobilenet \
--eval-interval 1 \
--dataset pascal

# --start_epoch 8 \
# --resume ./run/pascal/deeplab-mobilenet/experiment_6/checkpoint.pth.tar


# --resume deeplab-mobilenet.pth.tar
# --ft # must be used with --resume
# CUDA_VISIBLE_DEVICES=0,1,2,3
# --gpu-ids 0,1,2,3 \