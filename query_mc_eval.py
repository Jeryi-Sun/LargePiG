# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/metrics.py
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/utilities.py
# Ref: https://github.com/voidism/DoLa
import re
import os
import json
import random
import torch
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm, trange
import argparse
import pandas as pd

from largepig import largepig

transformers.logging.set_verbosity(40)

DEBUG = False

def load_json(file_path, is_gzip=False):
    # ['instruction', 'input', 'human_query', 'photo_id', 'good_query', 'bad_query']

    with open(file_path, 'r') as f:
        list_data = json.load(f)

    return list_data



def build_icl(icl_type):
    if args.data_type == 'video':
        with open(args.data_path+icl_type+".json", 'r') as f:
            input_list = json.load(f)
    elif args.data_type == 'document':
        with open(args.data_path+"/document/"+icl_type+".json", 'r') as f:
            input_list = json.load(f)
    # Concatenate demonstration examples ...
    demo_text = ""
    if args.icl_num != None:
        for i in range(args.icl_num):
            demo_text += input_list[i]+"\n\n"
    else:
        for i in range(len(input_list)):
            demo_text += input_list[i]+"\n\n"
    return demo_text


def build_prompt_and_answer(instruction, input, answer):
    if "qwen" in args.model_name:
        input_text_prompt = """<|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        {}
        {}<|im_end|>
        <|im_start|>assistant
        """.format(instruction, input[:1024])
    elif "Llama2-Chinese" in  args.model_name:
        input_text_prompt = """[INST] <<SYS>>
        You are a brilliant assistant. 你是一个乐于助人的助手。
        <</SYS>>
        {}
        {} [/INST]""".format(instruction, input[:1024])
    elif "llama2" in  args.model_name:
        input_text_prompt = """[INST] <<SYS>>
        You are a brilliant assistant.
        <</SYS>>
        {}
        {} [/INST]""".format(instruction, input[:1024])
    else:
        print("name error")
        exit()

    continue_text = " " + answer

    return input_text_prompt, continue_text


def build_prompt_and_answer_icl(instruction, input, answer, icl_type):

    icl = build_icl(icl_type)
    if "qwen" in args.model_name:
        input_text_prompt = """<|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        {}
        {}
        {}<|im_end|>
        <|im_start|>assistant
        """.format(instruction, icl, input[:1024])
        ref_prompt = """
        {} <|im_end|>
        <|im_start|>assistant""".format(input[:1024])
    elif "Llama2-Chinese" in  args.model_name:
        input_text_prompt = """[INST] <<SYS>>
        You are a brilliant assistant. 你是一个乐于助人的助手。
        <</SYS>>
        {}
        {}
        {} [/INST]""".format(instruction, icl, input[:1024])
        ref_prompt = """
        {} [/INST]""".format(input[:1024])
    elif "llama2" in  args.model_name:
        input_text_prompt = """[INST] <<SYS>>
        You are a brilliant assistant.
        <</SYS>>
        {}
        {}
        {} [/INST]""".format(instruction, icl, input[:1024])
        ref_prompt = """
        {} [/INST]""".format(input[:1024])
    else:
        print("name error")
        exit()



    continue_text = " " + answer

    return input_text_prompt, continue_text, ref_prompt


def MC_calcs(scores_true, scores_false, ref_true, ref_best):

    """Given model scores for true / false reference answers, calculates MC scores"""
    scores = {}
    scores['max'] = max(scores_true)
    scores['diff'] = max(scores_true) - max(scores_false)
    scores['scores-true'] = scores_true
    scores['scores-false'] = scores_false

    # compute MC1: 1vFalse -- best correct answer vs all false answers
    max_false = max(scores_false)
    if scores_true[ref_true.index(ref_best)] > max_false:
        scores['MC1'] = 1.0
    else:
        scores['MC1'] = 0.0

    # compute MC3: 1vFalse -- each correct answer vs all false answers
    max_false = max(scores_false)
    onevall = sum(np.array(scores_true) > max_false) / float(len(scores_true))
    scores['MC3'] = onevall

    # compute MC2: normalized probability mass for correct answers
    probs_true = np.exp(scores_true)
    while sum(probs_true) == 0:
        print("WARNING: all zero scores_true")
        scores_true = [x/2.0 for x in scores_true]
        probs_true = np.exp(scores_true)
    probs_false = np.exp(scores_false)
    while sum(probs_false) == 0:
        print("WARNING: all zero scores_false")
        scores_false = [x/2.0 for x in scores_false]
        probs_false = np.exp(scores_false)

    probs_true = probs_true / (sum(probs_true) + sum(probs_false))
    
    # check nan
    if np.isnan(sum(probs_true)):
        scores['MC2'] = 0.0
        print(f"WARNING: nan in probs_true: sum(probs_true)={sum(probs_true)}, sum(probs_false)={sum(probs_false)}")
    else:
        scores['MC2'] = sum(probs_true)

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max_gpu_memory", type=int, default=27)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    # parallel mode (split the dataset into multiple parts, inference by separate processes)
    parser.add_argument("--early-exit-layers", type=str, default="-1")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--total-shard", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--icl_type", type=str, default=None)
    parser.add_argument("--do-rating", action="store_true")
    parser.add_argument("--gpt3-config", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--relative_top", type=float, default=0.0)
    parser.add_argument("--relative_top_value", type=float, default=-1000.0)
    parser.add_argument("--scale_value", type=float, default=100.0)
    parser.add_argument("--pointer", action="store_true")
    parser.add_argument("--only_pointer", action="store_true")
    parser.add_argument("--lora_weights", type=str, default=None)
    parser.add_argument("--normalization", action="store_true")
    parser.add_argument("--icl_num", type=int, default=None)
    parser.add_argument("--hallucination_detect", action="store_true")
    parser.add_argument("--use_entropy", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--attention_layers", type=str, default=None)
    parser.add_argument("--data_type", type=str, default="video")

    
    args = parser.parse_args()
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device
    if args.data_type=='video':
        fp = args.data_path + "video_query.json"
    elif args.data_type=='document':
        fp = args.data_path + "document_query.json"



    list_data_dict = load_json(fp)

    if args.debug:
        list_data_dict = list_data_dict[:10]
    
    if args.parallel:
        chunk_size = len(list_data_dict) // args.total_shard
        list_data_dict = list_data_dict[args.shard_id * chunk_size: (args.shard_id + 1) * chunk_size]
    
    llm = largepig(model_name, device, num_gpus, args.max_gpu_memory, args.lora_weights)
    # stop_word_list = ["Q:"]
    # llm.set_stop_words(stop_word_list)
    early_exit_layers = [int(x) for x in args.early_exit_layers.split(',')]
    if len(early_exit_layers) == 1:
        print("MODE: naive decoding from the last layer", flush=True)
        mode = "baseline"
        mature_layer = None
        anchor_layer = None
        candidate_layers = None
    elif len(early_exit_layers) == 2:
        print(f"MODE: largepig-static decoding with mature layer: {early_exit_layers[1]} and premature layer: {early_exit_layers[0]}")
        mode = "early_exit_contrastive"
        mature_layer = early_exit_layers[1]
        anchor_layer = early_exit_layers[0]
        candidate_layers = None
    else:
        print(f"MODE: largepig decoding with mature layer: {early_exit_layers[-1]} and premature layers: {early_exit_layers[:-1]}")
        mode = "largepig"
        mature_layer = early_exit_layers[-1]
        anchor_layer = None
        candidate_layers = early_exit_layers[:-1]
        anchor_layer_dist = {l:0 for l in candidate_layers}
    answers = []
    result_step = []
    result_dict = {'total_mc1': 0.0, 'total_mc2': 0.0, 'total_mc3': 0.0, 'question': [], 'model_scores': []}
    with torch.no_grad():
        for sample in tqdm(list_data_dict):
            # reference answers
            # ['instruction', 'input', 'human_query', 'photo_id', 'good_query', 'bad_query']
            ref_best = sample['good_query'][0]
            ref_true = sample['good_query']
            ref_false = sample['bad_query']

            scores_true = []
            scores_false = []
            generate_kwargs = dict(max_new_tokens=args.max_new_tokens, repetition_penalty=args.repetition_penalty, mode=mode, mature_layer=mature_layer, 
                                   anchor_layer=anchor_layer, candidate_layers=candidate_layers, relative_top=args.relative_top, 
                                   relative_top_value=args.relative_top_value, post_softmax=True, pointer=args.pointer, only_pointer=args.only_pointer, 
                                   scale_value=args.scale_value, normalization=args.normalization, hallucination_detect=args.hallucination_detect, 
                                   use_entropy=args.use_entropy, alpha=args.alpha, attention_layers=args.attention_layers)

            for temp_ans in ref_true:
                # append the current answer choice to the prompt
                if args.icl_type == None:
                    prompt, answer = build_prompt_and_answer(sample['instruction'], sample['input'], temp_ans)
                    log_probs, c_dist = llm.lm_score(prompt, answer, **generate_kwargs)

                else:
                    prompt, answer, ref_prompt = build_prompt_and_answer_icl(sample['instruction'], sample['input'], temp_ans, args.icl_type)
                    log_probs, c_dist = llm.lm_score(prompt, answer, **generate_kwargs, ref_prompt=ref_prompt)


                scores_true.append(log_probs)

                if mode == "largepig":
                    for k, v in c_dist.items():
                        anchor_layer_dist[k] += v

            for temp_ans in ref_false:
                # append the current answer choice to the prompt
                if args.icl_type == None:
                    prompt, answer = build_prompt_and_answer(sample['instruction'], sample['input'], temp_ans)
                    log_probs, c_dist = llm.lm_score(prompt, answer, **generate_kwargs)

                else:
                    prompt, answer, ref_prompt = build_prompt_and_answer_icl(sample['instruction'], sample['input'], temp_ans, args.icl_type)
                    log_probs, c_dist = llm.lm_score(prompt, answer, **generate_kwargs, ref_prompt=ref_prompt)


                scores_false.append(log_probs)

                if mode == "largepig":
                    for k, v in c_dist.items():
                        anchor_layer_dist[k] += v

            scores = MC_calcs(scores_true, scores_false, ref_true, ref_best)
            # check nan in mc1/2/3
            if np.isnan(scores['MC1']) or np.isnan(scores['MC2']) or np.isnan(scores['MC3']):
                import ipdb; ipdb.set_trace()

            result_dict['model_scores'].append(scores)
            result_dict['question'].append(sample)
            # update total scores
            result_dict['total_mc1'] += scores['MC1']
            result_dict['total_mc2'] += scores['MC2']
            result_dict['total_mc3'] += scores['MC3']
            result_step.append([scores['MC1'], scores['MC2'], scores['MC3']])

            if DEBUG:
                print(f'Full input_text:\n{input_text}\n\n')
            # print(f'Question: {sample}\n\n'
            #     f'Model Scores: {scores}\n\n')
            # print(f'Avergaed MC1: {result_dict["total_mc1"]/len(result_dict["question"])}'
            #     f' MC2: {result_dict["total_mc2"]/len(result_dict["question"])}'
            #     f' MC3: {result_dict["total_mc3"]/len(result_dict["question"])}\n\n')


    if mode == "largepig" and args.debug:
        total_tokens = sum(anchor_layer_dist.values())
        if total_tokens > 0:
            for l in candidate_layers:
                print('Premature layer {0} was used {1} times, {2}%'.format(l, anchor_layer_dist[l], round(anchor_layer_dist[l] / total_tokens * 100, 2)))


    # Average the scores
    result_dict['total_mc1'] /= len(result_dict['question'])
    result_dict['total_mc2'] /= len(result_dict['question'])
    result_dict['total_mc3'] /= len(result_dict['question'])

    # Print the final scores, separated by ', '
    print(f'Final MC1/2/3: \n{result_dict["total_mc1"]}, {result_dict["total_mc2"]}, {result_dict["total_mc3"]}')

    # save results to a json file
    model_tag = model_name.split('/')[-1] if model_name[-1] != '/' else model_name.split('/')[-2]
    output_file = args.output_path if args.shard_id is None else (args.output_path+"_"+str(args.shard_id)+".json")
    with open(output_file, 'w') as f:
        json.dump(result_step, f, ensure_ascii=False)