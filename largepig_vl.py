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
from transformers.generation.stopping_criteria import StoppingCriteriaList, LLamaQaStoppingCriteria
from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration
from torch.nn import NLLLoss
import argparse
import warnings
import pandas as pd
import numpy as np
import faiss

class largepig:
    def __init__(self, model_name, device, num_gpus, max_gpu_memory=27, lora_weights=None):
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.stopping_criteria = None
        self.max_gpu_memory = max_gpu_memory
        self.model, self.processor = self.load_model(model_name, lora_weights)
        self.all_token_embeds = self.model.get_input_embeddings().weight
        dimension = self.all_token_embeds.shape[1]  # 获取向量的维度
        self.index = faiss.IndexFlatL2(dimension)  # 创建基于L2距离的FAISS索引
        self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 1, self.index)
        self.index.add(self.all_token_embeds.cpu().numpy().astype(np.float32))  # 向索引中添加向量
        
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
        
        model = LlavaForConditionalGeneration.from_pretrained(model_name, low_cpu_mem_usage=True, **kwargs)
        processor = AutoProcessor.from_pretrained(model_name)
        if lora_weights!=None:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
                device_map='auto'
            )

        if self.device == "cuda" and self.num_gpus == 1:
            model.cuda()
        
        return model, processor
    
    def generate(self, input_text, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, mature_layer=None, premature_layer=None, candidate_premature_layers=[], mode='baseline', verbose=True, remove_stop_words=False, relative_top=0.1, pointer=False, **kwargs):
        with torch.no_grad():

            input_ids = self.processor(input_text, return_tensors="pt").input_ids.to(self.device)
            max_len = input_ids.shape[-1] + max_new_tokens

            if mode == 'baseline':
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                    output_scores=True, return_dict_in_generate=True, largepig_decoding=False,
                                    top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, **kwargs)
            elif mode == 'largepig' and not pointer:
                assert mature_layer is not None, "mature_layer must be specified"
                assert candidate_premature_layers is not None, "candidate_premature_layers must be specified"
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                        output_scores=True, return_dict_in_generate=True, largepig_decoding=True,
                                        top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, relative_top=relative_top, 
                                        mature_layer=mature_layer, premature_layer=None, candidate_premature_layers=candidate_premature_layers, **kwargs,)
                premature_layer_dist = outputs.premature_layer_dist
            elif mode == 'largepig' and pointer:
                assert mature_layer is not None, "mature_layer must be specified"
                assert candidate_premature_layers is not None, "candidate_premature_layers must be specified"
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                        output_scores=True, return_dict_in_generate=True, largepig_decoding=True,
                                        top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, relative_top=relative_top, 
                                        mature_layer=mature_layer, premature_layer=None, candidate_premature_layers=candidate_premature_layers, **kwargs,)
                premature_layer_dist = outputs.premature_layer_dist

            sequences, scores = outputs.sequences, outputs.scores

            # skip the tokens in the input prompt
            gen_sequences = sequences[:, input_ids.shape[-1]:][0, :]
            gen_arr = gen_sequences.cpu().numpy()

            output_str = self.processor.decode(gen_sequences, skip_special_tokens=True)

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

        return output_str, (premature_layer_dist if mode == 'largepig' else None)
    
    def entropy(self, p):
        return torch.sum(-torch.where(p > 0, p * p.log2(), p.new([0.0])), dim=-1)

    def calculate_hallucination_mask(self, input_ids, words_logits, start_p, end_p, attentions, use_entropy, alpha=0.5, weight=0.5):
        # threshold = np.log(self.model.config.vocab_size)+self.model.config.vocab_size if use_entropy else np.log(self.model.config.vocab_size)
        # threshold *= alpha
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
        hallucination_mask = self.get_top_mask(hc_final, 0.1).float()
        batch_size, seq_len_minus_one = hallucination_mask.shape
        # Create a tensor of ones with shape (batch_size, 1)
        ones_to_concat = torch.ones((batch_size, 7), dtype=hallucination_mask.dtype, device=hallucination_mask.device)
        # Concatenate the tensor of ones at the beginning of each sequence in the batch
        hallucination_mask_with_one = torch.cat((ones_to_concat, hallucination_mask), dim=1)
        return hallucination_mask_with_one

    def is_english_token(self, idx):
        # 使用 tokenizer 将 idx 转换为 token 文本
        token = self.processor.convert_ids_to_tokens([idx])[0]
        # 检查 token 是否仅由英文字符和特殊字符 "▁" 组成
        return all(c.isalpha() or c == '▁' for c in token)

    def get_relative_top_filter(self, scores: torch.FloatTensor, relative_top: float = 0.1, min_tokens_to_keep: int = 1):
        scores_normalized = scores.log_softmax(dim=-1) 
        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
        min_thresh = sorted_logits[..., min_tokens_to_keep-1] 
        probs_max = torch.max(scores_normalized, dim=-1).values
        probs_thresh = probs_max + np.log(relative_top)
        probs_thresh = torch.min(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        return scores_normalized < probs_thresh
    
    def get_top_mask(self, scores: torch.FloatTensor, relative_top: float = 0.1, min_tokens_to_keep: int = 1):
        select_index = int(scores.shape[-1]*relative_top)
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        probs_thresh = sorted_logits[..., select_index] 
        probs_thresh = probs_thresh.unsqueeze(-1)
        return scores < probs_thresh

    def lm_score(self, figure, input_text1, input_text2, pmi=False, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, mature_layer=None, premature_layer=None, candidate_premature_layers=[], mode='baseline', verbose=True, remove_stop_words=False, relative_top=0.1, relative_top_value=-1000.0, post_softmax=True, pointer=False, only_pointer=False, start_p=None, end_p=None, ref_prompt=None, scale_value=100, normalization=False, hallucination_detect=False, use_entropy=False, alpha=0.5, debug_get_emb=False, **kwargs):
        with torch.no_grad():
            image = Image.open(figure)
            input_text = input_text1 + input_text2
            inputs = self.processor(text=input_text, images=image, return_tensors="pt").to(self.device)
            input_ids = self.model.get_merged_input_ids(inputs.input_ids, inputs.attention_mask) 

            prefix_inputs = self.processor(input_text1, images=image, return_tensors="pt").to(self.device)
            prefix_ids = self.model.get_merged_input_ids(prefix_inputs.input_ids, prefix_inputs.attention_mask) 
            
            if ref_prompt!=None:
                ref_prompt_ids = self.tokenizer(ref_prompt, return_tensors="pt").input_ids.to(self.device)
            continue_ids = input_ids[0, prefix_ids.shape[-1]:]
            if mode == 'baseline':
                outputs = self.model(**inputs).logits.squeeze(0)
                outputs = outputs.log_softmax(-1)  # logits to log probs
                
                # skip tokens in the prompt -- we only care about the answer
                outputs = outputs[prefix_ids.shape[-1] - 1: -1, :]


                # get logprobs for each token in the answer
                log_probs = outputs[range(outputs.shape[0]), continue_ids].sum().item()
                
            elif mode == 'largepig-static':
                dict_outputs, outputs = self.model(
                    **inputs,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    early_exit_layers=[premature_layer, mature_layer],
                )

                assert premature_layer is not None
                base_logits = dict_outputs[premature_layer][0, prefix_ids.shape[-1] - 1: -1, :]
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
                premature_layer_dist = {l:0 for l in candidate_premature_layers}
                picked_logits = []
                result_dict = {}
                premature_layers = []

                dict_outputs, outputs = self.model(
                    **inputs,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    early_exit_layers=candidate_premature_layers + [mature_layer],
                )

                for seq_i in range(prefix_ids.shape[-1] - 1, inputs.input_ids.shape[-1] - 1):
                    # Pick the less like layer to contrast with
                    # 1. Stacking all premature_layers into a new dimension
                    stacked_premature_layers = torch.stack([dict_outputs[i][:, seq_i, :] for i in candidate_premature_layers], dim=0)

                    # 2. Calculate the softmax values for mature_layer and all premature_layers
                    softmax_mature_layer = F.softmax(dict_outputs[mature_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)
                    softmax_premature_layers = F.softmax(stacked_premature_layers, dim=-1)  # shape: (num_premature_layers, batch_size, num_features)

                    # 3. Calculate M, the average distribution
                    M = 0.5 * (softmax_mature_layer[None, :, :] + softmax_premature_layers)  # shape: (num_premature_layers, batch_size, num_features)

                    # 4. Calculate log-softmax for the KL divergence
                    log_softmax_mature_layer = F.log_softmax(dict_outputs[mature_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)
                    log_softmax_premature_layers = F.log_softmax(stacked_premature_layers, dim=-1)  # shape: (num_premature_layers, batch_size, num_features)

                    # 5. Calculate the KL divergences and then the JS divergences
                    kl1 = F.kl_div(log_softmax_mature_layer[None, :, :], M, reduction='none').mean(-1)  # shape: (num_premature_layers, batch_size)
                    kl2 = F.kl_div(log_softmax_premature_layers, M, reduction='none').mean(-1)  # shape: (num_premature_layers, batch_size)
                    js_divs = 0.5 * (kl1 + kl2)  # shape: (num_premature_layers, batch_size)

                    # 6. Reduce the batchmean
                    js_divs = js_divs.mean(-1)  # shape: (num_premature_layers,)
                    premature_layer = candidate_premature_layers[int(js_divs.argmax().cpu().item())]
                    premature_layer_dist[premature_layer] += 1

                    premature_layers.append(premature_layer)

                base_logits = torch.zeros_like(dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1])
                for i, l in enumerate(premature_layers):
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
                premature_layer_dist = {l:0 for l in candidate_premature_layers}
                picked_logits = []
                result_dict = {}
                premature_layers = []
                p_cp_list = []
                pointer_distr_list = []

                dict_outputs, outputs, image_mask, inputs_embeds = self.model(
                    **inputs,
                    return_dict=True,
                    output_attentions=True,
                    output_hidden_states=False,
                    early_exit_layers=candidate_premature_layers + [mature_layer],
                )

                # 先加工一下 input_ids， 将其中的 img token 替换为 text token id
                # embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
                # inputs_embeds 中 image_mask 中 1 对应的位置，去算 embed_tokens中 cosine similarity 最大的 index，之后替换对应位置的input_ids 为 index
                # 处理图像token
                batch_size = prefix_ids.shape[0]
                seq_len = prefix_ids.shape[-1]

                # 初始化一个空列表来收集图像嵌入和它们的位置
                image_embeds_to_process = []
                image_positions = []

                # 收集需要计算相似度的嵌入和它们的位置
                for i in range(batch_size):
                    for j in range(seq_len):
                        if image_mask[i, j]:
                            image_embeds_to_process.append(inputs_embeds[i, j, :].cpu().numpy())  # FAISS使用numpy数组
                            image_positions.append((i, j))

                # 将收集到的嵌入转换为numpy数组
                image_embeds_to_process = np.array(image_embeds_to_process, dtype=np.float32)

                # 使用FAISS索引查找最相似的嵌入的索引
                D, I = self.index.search(image_embeds_to_process, 1)  # 搜索每个图像嵌入的最相似嵌入

                # 更新input_ids
                all_img_tokens = []
                for pos, idx in zip(image_positions, I[:, 0]):
                    i, j = pos
                    # 检查当前 token 是否为英文或包含 "▁"，如果不是，则将 idx 设置为 -1
                    
                    idx = idx if self.is_english_token(idx) else self.model.config.pad_token_id
                    input_ids[i, j] = idx

                    if debug_get_emb:
                        all_img_tokens.append(idx)

                if debug_get_emb:
                    return all_img_tokens
                # 计算一下输入的 context 里面有没有 hallucination 词，如果有的话 copy 的时候把他们的 pointer weight 调小
                # input: input_ids, corr token vocab distribution
                # output: hallucination score for the input_ids or hallucination mask
                # outputs.attentions is a tuple, taking the last layer's attentions
                attentions = outputs.attentions[-1]  # shape: (batch_size, num_heads, sequence_length, sequence_length)
                attentions = attentions.mean(dim=1)  # shape: (batch_size, sequence_length, sequence_length)
                

                if hallucination_detect:
                    if start_p != None and end_p != None:
                        hallucination_mask = self.calculate_hallucination_mask(input_ids, dict_outputs[mature_layer], start_p, end_p, attentions, use_entropy=use_entropy, alpha=alpha)


                    elif ref_prompt != None:
                        start_p = prefix_ids.shape[-1]-ref_prompt_ids.shape[-1]
                        end_p = prefix_ids.shape[-1]
                        hallucination_mask = self.calculate_hallucination_mask(input_ids, dict_outputs[mature_layer], start_p, end_p, attentions, use_entropy=use_entropy, alpha=alpha)

                    else:
                        start_p = 0
                        end_p = prefix_ids.shape[-1]
                        hallucination_mask = self.calculate_hallucination_mask(input_ids, dict_outputs[mature_layer], start_p, end_p, attentions, use_entropy=use_entropy, alpha=alpha)


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
                    no_none_tokens = []
                    for token in [self.model.config.pad_token_id, self.model.config.bos_token_id, self.model.config.eos_token_id]:
                        if token is None:
                            continue
                        else:
                            no_none_tokens.append(token)
                    if len(no_none_tokens) == 3:
                        special_tokens_mask = input_ids_cp.eq(no_none_tokens[0]) | input_ids_cp.eq(no_none_tokens[1]) | input_ids_cp.eq(no_none_tokens[2])
                    elif len(no_none_tokens) == 2:
                        special_tokens_mask = input_ids_cp.eq(no_none_tokens[0]) | input_ids_cp.eq(no_none_tokens[1])
                    else:
                        special_tokens_mask = input_ids_cp.eq(no_none_tokens[0])

                    # Combine the original attention mask with the special tokens mask
                    # If a token is a special token, it should be masked regardless of the original attention mask
                    special_tokens_mask = special_tokens_mask.to(pointer_probs.dtype)  # Convert to float to perform subtraction shape: (batch_size, sequence_length)

                    # Instead of using a bitwise AND (&), multiply the masks. This works because the masks contain 0 for 'False' and 1 for 'True'
                    combined_mask = (1 - special_tokens_mask) # shape: (batch_size, sequence_length)

                    # Use the combined mask to zero out attention scores for special tokens
                    pointer_probs = pointer_probs * combined_mask # shape: (batch_size, sequence_length)

                    # 调整 token 的 weight
                    pointer_probs = image_mask * torch.where(image_mask == 1, torch.tensor(0.00001, device=image_mask.device), torch.tensor(1.0, device=image_mask.device))

                    if hallucination_detect:
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
                    # 1. Stacking all premature_layers into a new dimension
                    stacked_premature_layers = torch.stack([dict_outputs[i][:, seq_i, :] for i in candidate_premature_layers], dim=0)

                    # 2. Calculate the softmax values for mature_layer and all premature_layers
                    softmax_mature_layer = F.softmax(dict_outputs[mature_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)
                    softmax_premature_layers = F.softmax(stacked_premature_layers, dim=-1)  # shape: (num_premature_layers, batch_size, num_features)

                    # 3. Calculate M, the average distribution
                    M = 0.5 * (softmax_mature_layer[None, :, :] + softmax_premature_layers)  # shape: (num_premature_layers, batch_size, num_features)

                    # 4. Calculate log-softmax for the KL divergence
                    log_softmax_mature_layer = F.log_softmax(dict_outputs[mature_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)
                    log_softmax_premature_layers = F.log_softmax(stacked_premature_layers, dim=-1)  # shape: (num_premature_layers, batch_size, num_features)

                    # 5. Calculate the KL divergences and then the JS divergences
                    kl1 = F.kl_div(log_softmax_mature_layer[None, :, :], M, reduction='none').mean(-1)  # shape: (num_premature_layers, batch_size)
                    kl2 = F.kl_div(log_softmax_premature_layers, M, reduction='none').mean(-1)  # shape: (num_premature_layers, batch_size)
                    js_divs = 0.5 * (kl1 + kl2)  # shape: (num_premature_layers, batch_size)
                    p_cp_list.append(js_divs.max(0)[0].squeeze())

                    # 6. Reduce the batchmean
                    js_divs = js_divs.mean(-1)  # shape: (num_premature_layers,)
                    premature_layer = candidate_premature_layers[int(js_divs.argmax().cpu().item())]
                    premature_layer_dist[premature_layer] += 1

                    premature_layers.append(premature_layer)



                base_logits = torch.zeros_like(dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1])
                for i, l in enumerate(premature_layers):
                   base_logits[i] = dict_outputs[l][0, prefix_ids.shape[-1] - 1 + i]
                if only_pointer:
                    final_logits = dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1]
                    final_logits = (final_logits).softmax(dim=-1)
                    pointer_distr = torch.stack(pointer_distr_list, dim=0) # shape (number_token, num_features)
                    p_cp_final = torch.clamp(torch.stack(p_cp_list, dim=0).unsqueeze(1)*scale_value, min=0, max=0.5)
                    weight = 1.0 #[(1-p_cp_final), 1]
                    diff_logits = torch.log(weight*final_logits + p_cp_final*pointer_distr)
                else:
                    print("largepig and pointer!!!")
                    final_logits = dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1]
                    final_logits = final_logits.log_softmax(dim=-1)
                    base_logits = base_logits.log_softmax(dim=-1)
                    diff_logits = final_logits - base_logits
                    if post_softmax:
                        diff_logits = (diff_logits).softmax(dim=-1)
                        if pointer:
                            pointer_distr = torch.stack(pointer_distr_list, dim=0) # shape (number_token, num_features)
                            p_cp_final = torch.clamp(torch.stack(p_cp_list, dim=0).unsqueeze(1)*scale_value, min=0, max=0.5)
                            weight = (1-p_cp_final)
                            diff_logits = weight*diff_logits + p_cp_final*pointer_distr

                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(final_logits, relative_top)
                    diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)
                
                log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()

        return log_probs, (premature_layer_dist if mode == 'largepig' else None)