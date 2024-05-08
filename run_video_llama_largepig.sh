export CUDA_VISIBLE_DEVICES="0,1,2,3"
python query_mc_eval.py --model-name Llama2-Chinese-7b --early-exit-layers 16,18,20,22,24,26,28,30,32 --data_path ./data/ --output_path ./video_query_result/llama2_7b.json --num-gpus 4 --pointer --only_pointer --max_gpu_memory 8
