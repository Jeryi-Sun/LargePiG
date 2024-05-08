export CUDA_VISIBLE_DEVICES="4,5,6,7"

python query_mc_eval.py --model-name qwen1_5-7b --early-exit-layers 16,18,20,22,24,26,28,30,32 --data_path ./data/ --output_path ./video_query_result/qwen1_5_7b_pointer.json --num-gpus 4 --pointer --only_pointer --max_gpu_memory 8 --scale_value 500 --normalization
