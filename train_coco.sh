# CUDA_VISIBLE_DEVICES=0,1,2,3 
python train.py \
--backbone resnet --lr 0.01 \
--workers 4 \
--epochs 40 \
--batch-size 16 \
--checkname deeplab-resnet \
--eval-interval 1 \
--dataset coco
# --gpu-ids 0,1,2,3 \
