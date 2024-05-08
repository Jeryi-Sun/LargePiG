export CUDA_VISIBLE_DEVICES="0,1,2,3"
python query_mc_eval.py --model-name qwen1_5-7b --data_path ./data/ --output_path ./video_query_result/qwen1_5_7b_baseline.json --num-gpus 4 --max_gpu_memory 8