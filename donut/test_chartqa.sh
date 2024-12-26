
CUDA_VISIBLE_DEVICES=0 python test_chartqa_pix.py \
    --data-path "ahmed-masry/chartqa_without_images"\
    --test-images 'dataset/chartqa/png/' \
    --batch-size 1 \
    --num-workers 1\
    --checkpoint_processor "omron-sinicx/sbsfigures-chartqa-donut"\
    --checkpoint-model  "omron-sinicx/sbsfigures-chartqa-donut" \
    --output-dir "output/finetune/chartqa/from_sbsfigures/result" \