import json
import os
import copy
import random
import re
import time


class CDCoT:
    def __init__(self, n_rephrase, temperature_rephrase, topp_rephrase, m_select,
                 c_reason, temp_reason, topp_reason, model, use_clean_shot, once_process=False):
        self.n_rephrase = n_rephrase
        self.temperature_rephrase = temperature_rephrase
        self.topp_rephrase = topp_rephrase
        self.m_select = m_select
        self.c_reason = c_reason
        self.temp_reason = temp_reason
        self.topp_reason = topp_reason
        self.model = model
        self.use_clean_shot = use_clean_shot
        self.clean_shot = []
        self.once_process = once_process

    def CD_CoT_with_ICL_list(self, case, ICL_list):
        n_case = []
        for ICL in ICL_list:
            new_case = copy.deepcopy(case)
            new_case['in-context'] = ICL[1:]
            n_case.append(new_case)
        self.model.query_n_case(n_case, self.c_reason, self.temp_reason, self.topp_reason)
        return n_case

    def _select_ICL_shot(self, ICL_correct_process, data_processor):
        select_shot_list = []
        for shot_correct_process in ICL_correct_process:
            select_shot = []
            question = shot_correct_process["question"]
            corrected_responses = shot_correct_process["corrected_responses"]
            not_none_corrected_responses = []
            for corrected_response in corrected_responses:
                if corrected_response is not None:
                    new_shot = [question, corrected_response]
                    select_shot.append(new_shot)
                    not_none_corrected_responses.append(corrected_response)
            if len(select_shot) != 0:
                select_shot_list.append(select_shot)
            else:
                # ablation
                new_shot = [question, random.sample(not_none_corrected_responses, 1)[0]]
                select_shot.append(new_shot)
                select_shot_list.append(select_shot)
        return select_shot_list

    def _comprise_ICL(self, select_shot_list, clean_shot, m):
        ICL_list = []
        for i in range(m):
            if self.use_clean_shot:
                ICL = [clean_shot]
            else:
                ICL = []
            for j in range(len(select_shot_list)):
                random.seed(time.time())
                selected_shot = random.sample(select_shot_list[j], 1)[0]
                ICL.append(selected_shot)
            ICL_list.append(ICL)
        return ICL_list

    def rephrase_icl_shots(self, case, dataset_name, dataset_processor):
        if dataset_name == "math":
            expr = "47+58"
        elif dataset_name == "symbolic":
            expr = ["walk around right twice after run opposite left",
                    ["I_TURN_LEFT", "I_TURN_LEFT", "I_RUN", "I_TURN_RIGHT", "I_WALK", "I_TURN_RIGHT", "I_WALK",
                     "I_TURN_RIGHT", "I_WALK", "I_TURN_RIGHT", "I_WALK", "I_TURN_RIGHT", "I_WALK", "I_TURN_RIGHT",
                     "I_WALK", "I_TURN_RIGHT", "I_WALK", "I_TURN_RIGHT", "I_WALK"]]
        elif dataset_name == "commonsense":
            expr = dataset_processor.get_demos(1)[0]
        else:
            raise ValueError("dataset type {} not support rephrase".format(dataset_name))
        temperature_rephrase = self.temperature_rephrase
        topp_rephrase = self.topp_rephrase
        n_rephrase = self.n_rephrase
        contrastive_queries = []
        in_context = case["in-context"]
        clean_shot = []
        noisy_ICL_correct_object = []
        for shot in in_context:
            noisy_shot_correct_object = dict()
            noisy_shot_correct_object["question"] = shot[0]
            noisy_shot_correct_object["noisy response"] = shot[1]
            
            contrastive_case = dict()
            if dataset_name == "symbolic":
                contrastive_question = dataset_processor.get_sys_prompt()
            elif dataset_name == "blocksworld":
                contrastive_question = dataset_processor.get_system_prompt()
            else:
                contrastive_question = ""
            contrastive_question += "The following are two examples for the same type of task: " \
                                    "the first example has correct explanation and correct answer, " \
                                    "and the second example has distracted explanation and correct answer. " \
                                    "Please follow the first example and give me the correct explanation and " \
                                    "answer for the second example, which should be logically consistent " \
                                    "with the first one."
            
            contrastive_question += "\nFirst Example:\n"
            if dataset_name == "blocksworld":
                clean_shot = expr
            else:
                if len(clean_shot) == 0:
                    clean_shot.append(dataset_processor.get_question(expr))
                if len(clean_shot) == 1:
                    clean_shot.append(dataset_processor.get_correct_answer(expr))
            contrastive_question += "Question:"
            contrastive_question += clean_shot[0]
            contrastive_question += "\nExplanation:"
            contrastive_question += clean_shot[1]
            if dataset_name == "symbolic":
                label = dataset_processor.match_answer(clean_shot[1])  # only symbolic
                contrastive_question += f"\nAnswer: {label}."
            contrastive_question += f"\nSecond Example:\nQuestion: {shot[0]}"
            contrastive_question += f"\nExplanation: {shot[1]}"
            if dataset_name == "symbolic":
                label = dataset_processor.match_answer(shot[1])  # only symbolic
                contrastive_question += f"\nAnswer: {label}."
            contrastive_question += "\nYou must respond in the format of \"correct version is: {the correct " \
                                    "explanation and answer}."
            contrastive_question += "Don't offer anything else."
            contrastive_case["question"] = contrastive_question
            contrastive_queries.append(contrastive_case)
            noisy_shot_correct_object["correct_prompt"] = contrastive_question
            noisy_ICL_correct_object.append(noisy_shot_correct_object)
        self.model.query_case_batch(contrastive_queries, temperature_rephrase, n_rephrase, topp_rephrase)
        shot_query_list = list(zip(in_context, contrastive_queries))
        for i in range(len(shot_query_list)):
            shot, query = shot_query_list[i]
            noisy_shot_correct_object = noisy_ICL_correct_object[i]
            noisy_shot_correct_object["corrected_responses"] = []
            responses = query["messages"][-1]
            for response in responses:
                content = response['content']
                match = re.search(r'[Cc]orrect [Vv]ersion.*?:([\s\S]*)', content)
                if match:
                    answer = match.group(1)
                    noisy_shot_correct_object["corrected_responses"].append(answer)
                else:
                    noisy_shot_correct_object["corrected_responses"].append(None)
        self.clean_shot = clean_shot
        return noisy_ICL_correct_object, self.clean_shot

    def CD_CoT(self, case, ICL_correct_process, reason_ICL_list, data_processor):
        select_shot_list = self._select_ICL_shot(ICL_correct_process, data_processor)
        ICL_list = self._comprise_ICL(select_shot_list, self.clean_shot, self.m_select)
        reason_ICL_list.append(ICL_list)
        n_case = []
        for ICL in ICL_list:
            new_case = copy.deepcopy(case)
            new_case['in-context'] = ICL
            n_case.append(new_case)
        self.model.query_n_case(n_case, self.c_reason, self.temp_reason, self.topp_reason)
        return select_shot_list, reason_ICL_list, n_case
