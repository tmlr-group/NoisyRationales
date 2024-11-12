import numpy as np
from .code.utils.mask import mask_sentence, mask_forbidden_index
import copy
from .code.old_code.denoiser import chatgpt_cli
import re

class SimpleArgs:
    def __init__(self,  denoise_method, mask_word, sparse_mask_rate):
        self.denoise_method = denoise_method
        self.sparse_mask_rate = sparse_mask_rate
        self.mask_word = mask_word

class SelfDenoise:
    def __init__(self, n_reason) -> None:
        self.args = SimpleArgs(denoise_method="chatgpt_single_by_model", mask_word ="###", sparse_mask_rate=0.1)
        self.mask_token ="<mask>"
        self.n_reason = n_reason
    
    def wr_log(self, obj):
        print(obj)
        log_file = self.log
        log_file.write(str(obj) + "\n")
        log_file.flush()
    
    def certify(self, case_batch, model, log_file, **kwargs):
        self.log = log_file
        args = self.args
        chatgpt_cli.set_mask_word(args.mask_word)
        all_cases = []
        for case in case_batch:
            shots = case["in-context"]
            if len(shots) >0 :
                new_cases_pass = 0
                while(new_cases_pass == 0):   
                    if "system-prompt" in case:
                        system_prompt = case["system-prompt"]
                    else:
                        system_prompt = None
                    qa_prompts = []
                    for shot in shots:
                        for i in range(self.n_reason):
                            answer = copy.deepcopy(shot[1])                
                            modifiable_a = answer[0:answer.rfind(".")+1]
                            tail_a = answer[answer.rfind(".")+1:]
                            tmp_sentence = mask_sentence(modifiable_a, args.sparse_mask_rate, self.mask_token, 1, False, random_probs=None)
                            answer = tmp_sentence[0] + tail_a
                            qa_prompts.append(f"Question: {shot[0].replace(self.mask_token, args.mask_word)}\n" + f"Answer: {answer.replace(self.mask_token, args.mask_word)}")
                    sentences_list, rephrase_case_batch = chatgpt_cli.get_batch_response_by_model(system_prompt, qa_prompts, model, self.n_reason)
                    n_shot_list = []
                    for shot, sentences in zip(shots, sentences_list):
                        n_shot = []
                        for sentence in sentences:
                            new_shot = []
                            user_text_match = re.search(r'[Uu]ser:(.*?)\n', sentence)
                            assistant_text_match = re.search(r'[Aa]ssistant:(.*)', sentence)
                            user_text = user_text_match.group(1) if user_text_match else shot[0]
                            assistant_text = assistant_text_match.group(1) if assistant_text_match else sentence
                            new_shot.append(user_text)
                            new_shot.append(assistant_text)
                            n_shot.append(new_shot)
                        n_shot_list.append(n_shot)
                    new_cases_pass = 1
                    cases = []
                    for context in zip(*n_shot_list):
                        new_case = copy.deepcopy(case)
                        new_case['in-context'] = list(context)
                        if model.model.startswith("gpt"):
                            tokens = model.compute_prompt_token_by_case(new_case) 
                            if tokens >= model.max_prompt_tokens:
                                new_cases_pass = 0
                                break
                            cases.append(new_case)
                all_cases += cases
            else:
                all_cases.append(case)
            
        model.query_case_batch(cases = all_cases, temperature = 1, n = 1)
        return all_cases
          