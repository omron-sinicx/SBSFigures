
CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune_chartqa.py \
    --data-path "ahmed-masry/chartqa_without_images"\
    --train-images 'dataset/chartqa/png/' \
    --valid-images 'dataset/chartqa/png/' \
    --max-epochs 20 \
    --batch-size 6 \
    --valid-batch-size 1 \
    --num-workers 12 \
    --lr 5e-5 \
    --gpus-num 4 \
    --check-val-every-n-epoch 10 \
    --warmup-steps 100 \
    --checkpoint-epochs 10 \
    --checkpoint_processor "omron-sinicx/sbsfigures-pretrain-donut"\
    --checkpoint-model  "omron-sinicx/sbsfigures-pretrain-donut" \
    --output-dir "output/finetune/chartqa/from_sbsfigures" \
