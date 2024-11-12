

import json
import os
import re
from llm_model.my_gpt.my_gpt import my_gpt

def backtrace(case_batch, model):
    for case in case_batch:
        error_position_list = case["first_error_position_list"]
        shots = case["in-context"]
        rephrase_messages_list = []
        for error_sentence_index, shot in zip(error_position_list, shots):
            question = shot[0]
            answer = shot[1]
            sentences = answer.split(".")
            if error_sentence_index != -1:
                error_sentence = sentences[error_sentence_index]
                messages = [{'role': "user", 'content': "I will give you one question and one answer. \nQustion:{} \nAnswer: {} \nThere are some mistakes in this Answer. I give you the first mistake: {}. \nplease remove these mistakes and only provide me the clean version of answer to this question. ".format(question, answer, error_sentence)}]
                rephrase_messages_list.append(messages)
        model.query_messages_batch(rephrase_messages_list)
        for shot, messages in zip(shots, rephrase_messages_list):
            response = messages[-1][0]["content"]
            shot[1] = response
    return case_batch
    
            