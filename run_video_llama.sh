export CUDA_VISIBLE_DEVICES="4,5,6,7"
python query_mc_eval.py --model-name Llama2-Chinese-7b  --data_path ./data/ --output_path ./video_query_result/llama2_7b_baseline.json --num-gpus 4 --max_gpu_memory 8