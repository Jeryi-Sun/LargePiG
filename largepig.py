# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/metrics.py
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/utilities.py
# Ref: https://github.com/voidism/DoLa
import argparse
import time
import csv
import tqdm
import os
import json
from peft import PeftModel
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers.generation.stopping_criteria import StoppingCriteriaList, LLamaQaStoppingCriteria, MaxLengthCriteria
from torch.nn import NLLLoss
import argparse
import warnings
import pandas as pd
import numpy as np

class largepig:
    def __init__(self, model_name, device, num_gpus, max_gpu_memory=27, lora_weights=None):
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.stopping_criteria = None
        self.max_gpu_memory = max_gpu_memory
        self.model, self.tokenizer = self.load_model(model_name, lora_weights)

    def load_model(self, model_name, lora_weights=None):
        if self.device == "cuda":
            kwargs = {"torch_dtype": torch.float16, "offload_folder": f"{model_name}/offload"}
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                self.num_gpus = int(self.num_gpus)
                if self.num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: f"{self.max_gpu_memory}GiB" for i in range(self.num_gpus)},
                    })
        elif self.device == "cpu":
            kwargs = {}
        else:
            raise ValueError(f"Invalid device: {self.device}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, **kwargs)
        if lora_weights!=None:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
                device_map='auto'
            )

        if self.device == "cuda" and self.num_gpus == 1:
            model.cuda()
        
        return model, tokenizer

    def set_stop_words(self, stop_words):
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()
        list_stop_word_ids = []
        for stop_word in self.stop_words:
            stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
            list_stop_word_ids.append(stop_word_ids)
            print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)
        self.stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))



    def generate(self, input_text, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, mature_layer=None, anchor_layer=None, candidate_layers=[], mode='baseline', verbose=True, remove_stop_words=False, relative_top=0.1, pointer=False, only_pointer=False, 
               start_p=None, end_p=None, ref_prompt=None, scale_value=100, 
               normalization=False, hallucination_detect=False, use_entropy=False,
               alpha=0.5, attention_layers=None, **kwargs):
        with torch.no_grad():
            generation_config = self.model.generation_config
            if ref_prompt!=None:
                ref_prompt = self.tokenizer(ref_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
                
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            max_len = input_ids.shape[-1] + max_new_tokens
            generation_config.max_length = max_len
            
            if mode == 'baseline':
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                    output_scores=True, return_dict_in_generate=True, largepig_decoding=False,
                                    top_p=top_p, top_k=top_k, temperature=temperature, generation_config=generation_config, **kwargs)
            elif only_pointer:
                assert mature_layer is not None, "mature_layer must be specified"
                assert candidate_layers is not None, "candidate_layers must be specified"
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                        output_scores=True, return_dict_in_generate=True, largepig_decoding=True,
                                        top_p=top_p, top_k=top_k, temperature=temperature,  relative_top=relative_top, 
                                        mature_layer=mature_layer, anchor_layer=anchor_layer, candidate_layers=candidate_layers, 
                                        pointer=pointer, only_pointer=only_pointer,
                                        start_p=start_p, end_p=end_p, ref_prompt=ref_prompt, 
                                        scale_value=scale_value, normalization=normalization, 
                                        hallucination_detect=hallucination_detect, use_entropy=use_entropy,
                                        alpha=alpha, attention_layers=attention_layers, generation_config=generation_config,
                                        **kwargs)
                anchor_layer_dist = outputs.anchor_layer_dist
            elif mode == 'largepig' and not pointer:
                assert mature_layer is not None, "mature_layer must be specified"
                assert candidate_layers is not None, "candidate_layers must be specified"
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                        output_scores=True, return_dict_in_generate=True, largepig_decoding=True,
                                        top_p=top_p, top_k=top_k, temperature=temperature,  relative_top=relative_top, 
                                        mature_layer=mature_layer, anchor_layer=None, 
                                        candidate_layers=candidate_layers,generation_config=generation_config, **kwargs,)
                anchor_layer_dist = outputs.anchor_layer_dist
            elif mode == 'largepig' and pointer:
                assert mature_layer is not None, "mature_layer must be specified"
                assert candidate_layers is not None, "candidate_layers must be specified"
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                        output_scores=True, return_dict_in_generate=True, largepig_decoding=True,
                                        top_p=top_p, top_k=top_k, temperature=temperature,  relative_top=relative_top, 
                                        mature_layer=mature_layer, anchor_layer=None, candidate_layers=candidate_layers, 
                                        generation_config=generation_config, **kwargs,)
                anchor_layer_dist = outputs.anchor_layer_dist

            sequences, scores = outputs.sequences, outputs.scores

            # skip the tokens in the input prompt
            gen_sequences = sequences[:, input_ids.shape[-1]:][0, :]
            gen_arr = gen_sequences.cpu().numpy()

            output_str = self.tokenizer.decode(gen_sequences, skip_special_tokens=True)

            if verbose:
                print('MODEL OUTPUT: \n{0}'.format(output_str))

            if remove_stop_words:
                for stop_word in self.stop_words:
                    length_to_remove = len(stop_word)
                    if output_str[-length_to_remove:] == stop_word:
                        output_str = output_str[:-length_to_remove]
                output_str = output_str.strip()

        if self.device:
            torch.cuda.empty_cache()

        return output_str
    
    def entropy(self, p):
        return torch.sum(-torch.where(p > 0, p * p.log2(), p.new([0.0])), dim=-1)
    
    def calculate_hallucination_attention_mask(self, start_p, end_p, attentions, len_alpha=0.5):
        length = int((end_p-start_p)*len_alpha)
        shift_attentions = torch.mean(attentions[:, start_p+length:end_p, :length], dim=1)
        hallucination_mask = self.get_bottom_mask(shift_attentions, 0.1).float()
        ones_to_concat = torch.ones((attentions.shape[0], end_p-start_p-length), dtype=hallucination_mask.dtype, device=hallucination_mask.device)
        hallucination_mask_with_one = torch.cat((hallucination_mask, ones_to_concat), dim=1)
        return hallucination_mask_with_one
        
    
    def calculate_hallucination_uncen_mask(self, input_ids, words_logits, start_p, end_p, attentions, use_entropy, alpha=1.0, weight=0.5):
        threshold = np.log(self.model.config.vocab_size)+self.model.config.vocab_size if use_entropy else np.log(self.model.config.vocab_size)
        threshold *= alpha
        if end_p-start_p<8:
            mask = torch.ones((batch_size, end_p-start_p), dtype=input_ids.dtype, device=input_ids.device)
            return mask

        shift_logits = words_logits[:, start_p+6:end_p-1, :]
        shift_labels = input_ids[:, start_p+7:end_p]
        shift_attentions = attentions[:, start_p+6:end_p-1, start_p+6:end_p-1] 
        attentions_sum = shift_attentions.sum(dim=-1)
        # shift_attentions shape(bs, seq_len, seq_len) hc shape(bs, seq_len) attentions_sum shape(bs, seq_len)

        # Flatten the tokens
        loss_fct = NLLLoss(reduction="none")
        shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        log_prob = F.log_softmax(shift_logits, dim=-1)
        prob = torch.exp(log_prob)
        entropy = torch.exp2(self.entropy(prob))
        loss = loss_fct(torch.log(prob + 1e-7), shift_labels)
        hc = loss + (entropy if use_entropy else 0.)
        hc = hc.view(words_logits.size(0), -1)
        hc_his = torch.bmm(shift_attentions.to(hc.dtype), hc.unsqueeze(2)).squeeze(2)
        hc_final = weight*(hc_his + hc * (1 - attentions_sum)) + (1-weight)*hc
        # hc_final shape((bs, seq_len)
        hallucination_mask = self.get_top_mask(hc_final, 0.1, threshold=threshold).float()
        batch_size, seq_len_minus_one = hallucination_mask.shape
        # Create a tensor of ones with shape (batch_size, 1)
        ones_to_concat = torch.ones((batch_size, 7), dtype=hallucination_mask.dtype, device=hallucination_mask.device)
        # Concatenate the tensor of ones at the beginning of each sequence in the batch
        hallucination_mask_with_one = torch.cat((ones_to_concat, hallucination_mask), dim=1)
        return hallucination_mask_with_one



    def get_relative_top_filter(self, scores: torch.FloatTensor, relative_top: float = 0.1, min_tokens_to_keep: int = 1):
        scores_normalized = scores.log_softmax(dim=-1) 
        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
        min_thresh = sorted_logits[..., min_tokens_to_keep-1] 
        probs_max = torch.max(scores_normalized, dim=-1).values
        probs_thresh = probs_max + np.log(relative_top)
        probs_thresh = torch.min(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        return scores_normalized < probs_thresh
    
    def get_top_mask(self, scores: torch.FloatTensor, relative_top: float = 0.1, threshold: int = None):
        select_index = int(scores.shape[-1]*relative_top)
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        probs_thresh = sorted_logits[..., select_index] 
        probs_thresh = probs_thresh.cpu().item()
        if threshold != None:
            probs_thresh = max(threshold, probs_thresh)
        else:
            probs_thresh = probs_thresh
        return scores < probs_thresh
    
    def get_bottom_mask(self, scores: torch.FloatTensor, relative_top: float = 0.1, threshold: int = None):
        select_index = int(scores.shape[-1]*relative_top)
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        probs_thresh = sorted_logits[..., select_index] 
        probs_thresh = probs_thresh.cpu().item()
        if threshold != None:
            probs_thresh = min(threshold, probs_thresh)
        else:
            probs_thresh = probs_thresh
        return scores > probs_thresh

    def lm_score(self, input_text1, input_text2, pmi=False, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, mature_layer=None, anchor_layer=None, candidate_layers=[], mode='baseline', verbose=True, remove_stop_words=False, relative_top=0.1, relative_top_value=-1000.0, post_softmax=True, pointer=False, only_pointer=False, start_p=None, end_p=None, ref_prompt=None, scale_value=100, normalization=False, hallucination_detect=False, use_entropy=False, alpha=0.5, attention_layers=None, **kwargs):
        with torch.no_grad():
            input_text = input_text1 + input_text2
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            prefix_ids = self.tokenizer(input_text1, return_tensors="pt").input_ids.to(self.device)
            if ref_prompt!=None:
                ref_prompt_ids = self.tokenizer(ref_prompt, return_tensors="pt").input_ids.to(self.device)
            continue_ids = input_ids[0, prefix_ids.shape[-1]:]
            if mode == 'baseline':
                outputs = self.model(input_ids)[0].squeeze(0)
                outputs = outputs.log_softmax(-1)  # logits to log probs

                # skip tokens in the prompt -- we only care about the answer
                outputs = outputs[prefix_ids.shape[-1] - 1: -1, :]

                # get logprobs for each token in the answer
                log_probs = outputs[range(outputs.shape[0]), continue_ids].sum().item()
                
            elif mode == 'largepig-static':
                dict_outputs, outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    early_exit_layers=[anchor_layer, mature_layer],
                )

                assert anchor_layer is not None
                base_logits = dict_outputs[anchor_layer][0, prefix_ids.shape[-1] - 1: -1, :]
                final_logits = dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1: -1, :]
                final_logits = final_logits.log_softmax(dim=-1)
                base_logits = base_logits.log_softmax(dim=-1)
                diff_logits = final_logits - base_logits
                if post_softmax:
                    diff_logits = diff_logits.log_softmax(dim=-1)
                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(final_logits, relative_top)
                    diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)
                    
                log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()

            elif mode == 'largepig' and not pointer:
                anchor_layer_dist = {l:0 for l in candidate_layers}
                picked_logits = []
                result_dict = {}
                anchor_layers = []

                dict_outputs, outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    early_exit_layers=candidate_layers + [mature_layer],
                )

                for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):
                    # Pick the less like layer to contrast with
                    # 1. Stacking all anchor_layers into a new dimension
                    stacked_anchor_layers = torch.stack([dict_outputs[i][:, seq_i, :] for i in candidate_layers], dim=0)

                    # 2. Calculate the softmax values for mature_layer and all anchor_layers
                    softmax_mature_layer = F.softmax(dict_outputs[mature_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)
                    softmax_anchor_layers = F.softmax(stacked_anchor_layers, dim=-1)  # shape: (num_anchor_layers, batch_size, num_features)

                    # 3. Calculate M, the average distribution
                    M = 0.5 * (softmax_mature_layer[None, :, :] + softmax_anchor_layers)  # shape: (num_anchor_layers, batch_size, num_features)

                    # 4. Calculate log-softmax for the KL divergence
                    log_softmax_mature_layer = F.log_softmax(dict_outputs[mature_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)
                    log_softmax_anchor_layers = F.log_softmax(stacked_anchor_layers, dim=-1)  # shape: (num_anchor_layers, batch_size, num_features)

                    # 5. Calculate the KL divergences and then the JS divergences
                    kl1 = F.kl_div(log_softmax_mature_layer[None, :, :], M, reduction='none').mean(-1)  # shape: (num_anchor_layers, batch_size)
                    kl2 = F.kl_div(log_softmax_anchor_layers, M, reduction='none').mean(-1)  # shape: (num_anchor_layers, batch_size)
                    js_divs = 0.5 * (kl1 + kl2)  # shape: (num_anchor_layers, batch_size)

                    # 6. Reduce the batchmean
                    js_divs = js_divs.mean(-1)  # shape: (num_anchor_layers,)
                    anchor_layer = candidate_layers[int(js_divs.argmax().cpu().item())]
                    anchor_layer_dist[anchor_layer] += 1

                    anchor_layers.append(anchor_layer)

                base_logits = torch.zeros_like(dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1])
                for i, l in enumerate(anchor_layers):
                   base_logits[i] = dict_outputs[l][0, prefix_ids.shape[-1] - 1 + i]
                final_logits = dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1]
                final_logits = final_logits.log_softmax(dim=-1)
                base_logits = base_logits.log_softmax(dim=-1)
                diff_logits = final_logits - base_logits
                if post_softmax:
                    diff_logits = diff_logits.log_softmax(dim=-1) # 这里是先取softmax 再取 log，可以将我们的 copy 概率加到 softmax 里去



                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(final_logits, relative_top)
                    diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)
                
                log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()

            elif mode == 'largepig' and pointer:
                anchor_layer_dist = {l:0 for l in candidate_layers}
                picked_logits = []
                result_dict = {}
                anchor_layers = []
                p_cp_list = []
                pointer_distr_list = []

                dict_outputs, outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=True,
                    output_hidden_states=False,
                    early_exit_layers=candidate_layers + [mature_layer],
                )
                # 计算一下输入的 context 里面有没有 hallucination 词，如果有的话 copy 的时候把他们的 pointer weight 调小
                # input: input_ids, corr token vocab distribution
                # output: hallucination score for the input_ids or hallucination mask
                # outputs.attentions is a tuple, taking the last layer's attentions
                if attention_layers == None:
                    attentions = outputs.attentions[-1]  # shape: (batch_size, num_heads, sequence_length, sequence_length)
                else:
                    attention_layers = eval(attention_layers)
                    if type(attention_layers) ==str:
                        selected_attentions = [outputs.attentions[i] for i in attention_layers]
                        attentions = torch.mean(torch.stack(selected_attentions), dim=0)
                    elif type(attention_layers) == int:
                        attentions = outputs.attentions[attention_layers]
                    else:
                        print("attentions layers error!!!")
                        exit(-1)

                attentions = attentions.mean(dim=1)  # shape: (batch_size, sequence_length, sequence_length)
                

                if hallucination_detect:
                    if start_p != None and end_p != None:
                        hallucination_mask = self.calculate_hallucination_uncen_mask(input_ids, dict_outputs[mature_layer], start_p, end_p, attentions, use_entropy=use_entropy, alpha=alpha)


                    elif ref_prompt != None:
                        start_p = prefix_ids.shape[-1]-ref_prompt_ids.shape[-1]
                        end_p = prefix_ids.shape[-1]
                        hallucination_mask = self.calculate_hallucination_uncen_mask(input_ids, dict_outputs[mature_layer], start_p, end_p, attentions, use_entropy=use_entropy, alpha=alpha)

                    else:
                        start_p = 0
                        end_p = prefix_ids.shape[-1]
                        hallucination_mask = self.calculate_hallucination_uncen_mask(input_ids, dict_outputs[mature_layer], start_p, end_p, attentions, use_entropy=use_entropy, alpha=alpha)


                # Step 1: Average the attention across the number of heads

                for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):

                    # Step 2: Extract the non-zero values from the last row/column
                    # Determine the position of the last token based on the attention_mask
                    # We find the index of the last occurrence of the number 1 in each sequence of the batch

                    # Now we gather the attention scores for the last token of each sequence
                    pointer_scores = attentions[:, seq_i, :]  # shape: (batch_size, sequence_length)

                    # Step 3: Perform a softmax over the modified attention scores
                    # pointer_probs = nn.F.softmax(pointer_scores, dim=-1)  # shape: (batch_size, sequence_length)
                    if start_p != None and  end_p != None:
                        pointer_probs =  pointer_scores[:,start_p:end_p]
                        input_ids_cp = input_ids[:,start_p:end_p]
                    elif ref_prompt != None:
                        pointer_probs =  pointer_scores[:,prefix_ids.shape[-1]-ref_prompt_ids.shape[-1]:prefix_ids.shape[-1]]
                        input_ids_cp =  input_ids[:,prefix_ids.shape[-1]-ref_prompt_ids.shape[-1]:prefix_ids.shape[-1]]
                    else:
                        pointer_probs =  pointer_scores[:,:prefix_ids.shape[-1]]  # shape: (batch_size, prefix_sequence_length) todo: 截取这一步还需要操作一下只让模型关注文本内容
                        input_ids_cp = input_ids[:,:prefix_ids.shape[-1]] 

                    # Step 4: Limit the attention scores to the context and remove special tokens
                    # Create an extended attention mask that masks out special tokens
                    if self.model.config.pad_token_id is None:
                        special_tokens_mask = input_ids_cp.eq(self.model.config.bos_token_id) | input_ids_cp.eq(self.model.config.eos_token_id)
                    else:
                        special_tokens_mask = input_ids_cp.eq(self.model.config.pad_token_id) | input_ids_cp.eq(self.model.config.bos_token_id) | input_ids_cp.eq(self.model.config.eos_token_id)

                    # Combine the original attention mask with the special tokens mask
                    # If a token is a special token, it should be masked regardless of the original attention mask
                    special_tokens_mask = special_tokens_mask.to(pointer_probs.dtype)  # Convert to float to perform subtraction shape: (batch_size, sequence_length)

                    # Instead of using a bitwise AND (&), multiply the masks. This works because the masks contain 0 for 'False' and 1 for 'True'
                    combined_mask = (1 - special_tokens_mask) # shape: (batch_size, sequence_length)

                    # Use the combined mask to zero out attention scores for special tokens
                    pointer_probs = pointer_probs * combined_mask # shape: (batch_size, sequence_length)
                    if hallucination_detect:
                        hallucination_mask = hallucination_mask.to(dtype=pointer_probs.dtype)
                        pointer_probs = pointer_probs*hallucination_mask

                    # new feature: normalization 
                    if normalization:   
                        pointer_probs = pointer_probs + 1e-7
                        pointer_probs = pointer_probs/pointer_probs.sum(dim=-1, keepdim=True)

                    # Step 5: Spread the context id probabilities across the entire vocabulary
                    # This creates a distribution over the entire vocabulary for each element in the batch
                    vocab_size = self.model.config.vocab_size
                    batch_size, sequence_length = input_ids_cp.shape
                    pointer_vocab_distr = torch.zeros(batch_size, vocab_size, device=pointer_probs.device, dtype=pointer_probs.dtype)
                

                    # Scatter the probabilities into the full vocabulary distribution
                    # Note: 'input_ids_cp' here is assumed to be the context tokens that we are pointing to
                    pointer_vocab_distr.scatter_add_(1, input_ids_cp, pointer_probs)
                    pointer_distr_list.append(pointer_vocab_distr.squeeze())

                    # Pick the less like layer to contrast with
                    # 1. Stacking all anchor_layers into a new dimension
                    stacked_anchor_layers = torch.stack([dict_outputs[i][:, seq_i, :] for i in candidate_layers], dim=0)

                    # 2. Calculate the softmax values for mature_layer and all anchor_layers
                    softmax_mature_layer = F.softmax(dict_outputs[mature_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)
                    softmax_anchor_layers = F.softmax(stacked_anchor_layers, dim=-1)  # shape: (num_anchor_layers, batch_size, num_features)

                    # 3. Calculate M, the average distribution
                    M = 0.5 * (softmax_mature_layer[None, :, :] + softmax_anchor_layers)  # shape: (num_anchor_layers, batch_size, num_features)

                    # 4. Calculate log-softmax for the KL divergence
                    log_softmax_mature_layer = F.log_softmax(dict_outputs[mature_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)
                    log_softmax_anchor_layers = F.log_softmax(stacked_anchor_layers, dim=-1)  # shape: (num_anchor_layers, batch_size, num_features)

                    # 5. Calculate the KL divergences and then the JS divergences
                    kl1 = F.kl_div(log_softmax_mature_layer[None, :, :], M, reduction='none').mean(-1)  # shape: (num_anchor_layers, batch_size)
                    kl2 = F.kl_div(log_softmax_anchor_layers, M, reduction='none').mean(-1)  # shape: (num_anchor_layers, batch_size)
                    js_divs = 0.5 * (kl1 + kl2)  # shape: (num_anchor_layers, batch_size)
                    p_cp_list.append(js_divs.max(0)[0].squeeze())

                    # 6. Reduce the batchmean
                    js_divs = js_divs.mean(-1)  # shape: (num_anchor_layers,)
                    anchor_layer = candidate_layers[int(js_divs.argmax().cpu().item())]
                    anchor_layer_dist[anchor_layer] += 1

                    anchor_layers.append(anchor_layer)



                base_logits = torch.zeros_like(dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1])
                for i, l in enumerate(anchor_layers):
                   base_logits[i] = dict_outputs[l][0, prefix_ids.shape[-1] - 1 + i]
                if only_pointer:
                    final_logits = dict_outputs[self.model.config.num_hidden_layers][0, prefix_ids.shape[-1] - 1:-1]
                    final_logits = (final_logits).softmax(dim=-1)
                    pointer_distr = torch.stack(pointer_distr_list, dim=0) # shape (number_token, num_features)
                    p_cp_final = torch.clamp(torch.stack(p_cp_list, dim=0).unsqueeze(1)*scale_value, min=0, max=0.5)
                    diff_logits = torch.log((1-p_cp_final)*final_logits + p_cp_final*pointer_distr)
                else:
                    print("largepig and pointer!!!")
                    final_logits = dict_outputs[self.model.config.num_hidden_layers][0, prefix_ids.shape[-1] - 1:-1]
                    final_logits = final_logits.log_softmax(dim=-1)
                    base_logits = base_logits.log_softmax(dim=-1)
                    diff_logits = final_logits - base_logits
                    if post_softmax:
                        diff_logits = (diff_logits).softmax(dim=-1)
                        if pointer:
                            pointer_distr = torch.stack(pointer_distr_list, dim=0) # shape (number_token, num_features)
                            p_cp_final = torch.clamp(torch.stack(p_cp_list, dim=0).unsqueeze(1)*scale_value, min=0, max=0.5)
                            diff_logits = (1-p_cp_final)*diff_logits + p_cp_final*pointer_distr

                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(final_logits, relative_top)
                    diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)
                
                log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()

        return log_probs, (anchor_layer_dist if mode == 'largepig' else None)