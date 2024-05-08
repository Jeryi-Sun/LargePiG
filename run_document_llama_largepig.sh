export CUDA_VISIBLE_DEVICES="0,1,2,3"
python query_mc_eval.py --model-name llama2-7b --early-exit-layers 16,18,20,22,24,26,28,30,32 --data_path ./data/ --output_path ./document_query_result/llama2_7b.json --data_type document --num-gpus 4 --pointer --only_pointer --max_gpu_memory 8 --scale_value 500 --normalization
