# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
from llm_model.my_gpt.my_gpt import my_gpt
import copy

class SelfPolish:
    def __init__(self, model:my_gpt=None, temp=1):
        self.final_choose="last_one"
        self.max_times = 2
        self.model=model
        self.temp = temp
        pass



    def generate_one_new_answer_query(self, original_answer, question, system_propmt =None):
        if system_propmt == None:
            prompt_input_to_generate_new_question = "Question: {}\nOriginal: {}\nNew:".format(question, original_answer)
            messages = [{"role": "system", "content": "Please rewrite new versions of the original answer to be more understandable and more relevant to the question. Don't omit any useful information, especially the numbers, and please maintain their original meaning when polysemous words appear."},
                        {"role": "user", "content": prompt_input_to_generate_new_question}]
        else:
            prompt_input_to_generate_new_question = "Please rewrite new versions of the original answer to be more understandable and more relevant to the question. Don't omit any useful information, especially the numbers, and please maintain their original meaning when polysemous words appear.\n\nQuestion: {}\nOriginal: {}\nNew:".format(question, original_answer)
            messages = [{"role": "system", "content": system_propmt},
                        {"role": "user", "content": prompt_input_to_generate_new_question}]
        return messages

    def polish_batch(self, case_batch):
        times = 0
        while True:
            if times >= self.max_times:
                # print("More than {} times!".format(self.max_times))
                break
            times += 1
            generate_new_answer_messages_batch = []
            
            for case in case_batch:
                shots = case["in-context"]
                for shot in shots:
                    question = shot[0]
                    original_answer = shot[1]
                    system_prompt = case["system-prompt"] if "system-prompt" in case else None
                    generate_new_answer_messages = self.generate_one_new_answer_query(original_answer, question, system_prompt)
                    generate_new_answer_messages_batch.append(generate_new_answer_messages)
                
            self.model.query_messages_batch(generate_new_answer_messages_batch, temperature=self.temp)
            
            messages_index = 0
            for case in case_batch:
                shots = copy.deepcopy(case["in-context"])
                for i, messages in enumerate(generate_new_answer_messages_batch[messages_index:messages_index + len(shots)]):
                    response = messages[-1][0]["content"]
                    shots[i][1] = response
                messages_index += len(shots)
                case["in-context"] = shots
        return case_batch


if __name__ == '__main__':
    pass
