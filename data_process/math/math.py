import numpy as np
import random
import json
import re
import math
import os
import time

from ..processor_base import data_processor 

class math(data_processor):
    """
    Processor of a base caculation dataset.
    
    Attributes:
        config: A dictionary containing all the attribute values. If provided, values in the config dictionary will be used to overwrite the following parameters.
            base: The base number for calculations. Default is 9.
        
    """
    def __init__(self,  n_shots=0, n_noisy_shots=0, noise_type="irrelevant", noise_semantic_related = 0, noise_ratio = 0.5, noise_distribution = "fixed", prefix_context = False, config: dict = None, subtask="base-9") -> None:
        super().__init__(n_shots, n_noisy_shots, noise_type, noise_ratio, noise_distribution, prefix_context)
        self._noise_semantic_related = noise_semantic_related
        assert self._noise_semantic_related <= 2
        if config is not None:
            self.base = int(config["subtask"].split("base-")[1])
        else:
            self.base = int(subtask.split("base-")[1])
        self.irrelevant_index = 0
        self.file_path = self.get_file_path()
    
    def _get_label(self, expr):
        base = self.base
        lhs, rhs = expr.split("+")
        lhs_base10 = int(lhs, base)
        rhs_base10 = int(rhs, base)
        sum_base10 = lhs_base10 + rhs_base10
        return np.base_repr(sum_base10, base)

    def get_label(self, raw_data):
        expr = raw_data.split("\t")[0]
        return self._get_label(expr)
    
    def get_correct_answer(self, raw_data, generate_info = None):
        expr = raw_data.split("\t")[0]
        if generate_info!=None:
            generate_info["total_thought"] = 8
            generate_info["noise_thought"] = 0
            generate_info["sentences_with_noise"] = [0] * 9
        digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        base = self.base
        lhs, rhs = expr.split("+")
        lt, lo = lhs  # tens, ones
        rt, ro = rhs
        ones_sum = self._get_label(f"{lo}+{ro}")
        carry_over = len(ones_sum) > 1 
        tens_sum_wo_carry = self._get_label(f"{lt}+{rt}")
        if carry_over:
            ones_carry_digit = 1
            assert ones_sum[0] == "1"
            tens_sum_w_carry = self._get_label(f"{tens_sum_wo_carry}+1")
        else:
            ones_carry_digit = 0
            tens_sum_w_carry = tens_sum_wo_carry
        assert self._get_label(expr) == tens_sum_w_carry + ones_sum[-1:]
        tens_carry_over = len(tens_sum_w_carry) > 1
        tens_carry_digit = 1 if tens_carry_over else 0
        
        explaination = f"Since we're in base-{base}, that exceeds the maximum value of {digits[base-1]} for a single digit. " if carry_over == 1 else f"Since we're in base-{base}, that doesn't exceed the maximum value of {digits[base-1]} for a single digit. "
        
        #In base-{base} where the digits are \"{digits[:base]}\".
        ret = f"In base-{base}, the digits are \"{digits[:base]}\". We have {lo} + {ro} = {int(lo, base) + int(ro, base)} in base-10. "+ explaination + f"{int(lo, base) + int(ro, base)} mod {base} = {ones_sum[-1]}, so the digit is {ones_sum[-1]} and the carry is {ones_carry_digit}. We have {lt} + {rt} + {ones_carry_digit} = {int(lt, base) + int(rt, base) + ones_carry_digit} in base 10. {int(lt, base) + int(rt, base) + ones_carry_digit} mod {base} = {tens_sum_w_carry[-1]}, so the digit is {tens_sum_w_carry[-1]} and the carry is {tens_carry_digit}. A leading digit is {tens_carry_digit}. So the answer is {self._get_label(expr)}. Answer:\\box{{{self._get_label(expr)}}}"
        return ret
    
    def _generate_noise_distribution_list(self, n_thought, noise_ratio, noise_distribution):
        noise_distribution_list = [0] * n_thought
        if noise_distribution == "fixed":
            noise_count = round(n_thought * noise_ratio)
            noise_positions = random.sample(range(n_thought), noise_count)
            for pos in noise_positions:
                noise_distribution_list[pos] = 1
        elif noise_distribution == "random":
            for pos in range(len(noise_distribution_list)):
                if random.random() < noise_ratio:
                    noise_distribution_list[pos] = 1
        elif noise_distribution == "n_thought":
            assert noise_ratio <= n_thought
            noise_positions = random.sample(range(n_thought), noise_ratio)
            for pos in noise_positions:
                noise_distribution_list[pos] = 1
        else:
            raise ValueError(f"noise_distribution {noise_distribution} not supported")
        self.noise_pos = 0
        return noise_distribution_list
        
    def _should_add_noise(self, noise_distribution_list):
        if_noise = noise_distribution_list[self.noise_pos]
        self.noise_pos += 1
        return if_noise

    def get_generation_config(self, noise_distribution_list, generate_info):
        generate_info["total_thought"] = len(noise_distribution_list) + noise_distribution_list.count(1)
        generate_info["noise_thought"] = noise_distribution_list.count(1)
        generate_info["sentences_with_noise"] = []
        for if_noise in noise_distribution_list:
            generate_info["sentences_with_noise"].append(0)
            if if_noise:
                generate_info["sentences_with_noise"].append(1)
        generate_info["sentences_with_noise"].append(0)

    def get_irrelevant_answer(self, raw_data, noise_ratio, generate_info = None):
        expr = raw_data.split("\t")[0]
        digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        semantic_related = self._noise_semantic_related
        base = self.base
        lhs, rhs = expr.split("+")
        lt, lo = lhs  # tens, ones
        rt, ro = rhs
        ones_sum = self._get_label(f"{lo}+{ro}")
        carry_over = len(ones_sum) > 1
        tens_sum_wo_carry = self._get_label(f"{lt}+{rt}")
        
        noise_distribution_list = self._generate_noise_distribution_list(n_thought=8, noise_ratio=noise_ratio, noise_distribution=self.noise_distribution)
        if generate_info!=None:
            self.get_generation_config(noise_distribution_list, generate_info)
        
        if carry_over:
            ones_carry_digit = 1
            assert ones_sum[0] == "1"
            tens_sum_w_carry = self._get_label(f"{tens_sum_wo_carry}+1")
        else:
            ones_carry_digit = 0
            tens_sum_w_carry = tens_sum_wo_carry
        assert self._get_label(expr) == tens_sum_w_carry + ones_sum[-1:]
        tens_carry_over = len(tens_sum_w_carry) > 1
        tens_carry_digit = 1 if tens_carry_over else 0
        explaination = f"Since we're in base-{base}, that exceeds the maximum value of {digits[base-1]} for a single digit. " if carry_over == 1 else f"Since we're in base-{base}, that doesn't exceed the maximum value of {digits[base-1]} for a single digit. "
        
        selected_noise_set = set()
    
        
        ret = f"In base-{base}, the digits are \"{digits[:base]}\". "
        if self._should_add_noise(noise_distribution_list):
            if semantic_related == 0:
                fact = self._random_choose_fact(base, selected_noise_set)    
                ret += f"{fact}. "
            if semantic_related == 1:
                fact = self._random_choose_semantic1_position_noise(0)    
                ret += f"{fact}. "
            if semantic_related == 2:
                fact = self._random_choose_semantic2_position_noise(0)    
                ret += f"{fact}. "
        
        ret += f" We have {lo} + {ro} = {int(lo, base) + int(ro, base)} in base-10. "
        if self._should_add_noise(noise_distribution_list):
            number = int(lo, base) + int(ro, base)
            if semantic_related == 0:
                fact = self._random_choose_fact(number, selected_noise_set)    
                ret += f"{fact}. "
            if semantic_related == 1:
                fact = self._random_choose_semantic1_number_noise(number, selected_noise_set)    
                ret += f"{fact}. "
            if semantic_related == 2:
                fact = self._random_choose_semantic2_position_noise(1, lo, ro)    
                ret += f"{fact}. "
         
        ret += explaination
        if self._should_add_noise(noise_distribution_list):
            number = int(digits[base-1], base)
            if semantic_related == 0:
                fact = self._random_choose_fact(number, selected_noise_set)    
                ret += f"{fact}. " 
            if semantic_related == 1:
                fact = self._random_choose_semantic1_position_noise(1)    
                ret += f"{fact}. "
            if semantic_related == 2:
                fact = self._random_choose_semantic2_position_noise(2)    
                ret += f"{fact}. "
        
        ret += f"{int(lo, base) + int(ro, base)} mod {base} = {ones_sum[-1]}, so the digit is {ones_sum[-1]} and the carry is {ones_carry_digit}. "
        if self._should_add_noise(noise_distribution_list):
            number = int(ones_sum[-1], base)
            if semantic_related == 0:
                fact = self._random_choose_fact(number, selected_noise_set)    
                ret += f"{fact}. "
            if semantic_related == 1:
                fact = self._random_choose_semantic1_position_noise(2)    
                ret += f"{fact}. "
            if semantic_related == 2:
                fact = self._random_choose_semantic2_position_noise(3)    
                ret += f"{fact}. "
            
        ret += f"We have {lt} + {rt} + {ones_carry_digit} = {int(lt, base) + int(rt, base) + ones_carry_digit} in base 10. " 
        number = int(lt, base) + int(rt, base) + ones_carry_digit
        if self._should_add_noise(noise_distribution_list):
            if semantic_related == 0:
                fact = self._random_choose_fact(number, selected_noise_set)    
                ret += f"{fact}. "
            if semantic_related == 1:
                fact = self._random_choose_semantic1_number_noise(number, selected_noise_set)    
                ret += f"{fact}. "
            if semantic_related == 2:
                fact = self._random_choose_semantic2_position_noise(4, int(lt, base), int(rt, base), ones_carry_digit)    
                ret += f"{fact}. "
        
        ret += f"{int(lt, base) + int(rt, base) + ones_carry_digit} mod {base} = {tens_sum_w_carry[-1]}, so the digit is {tens_sum_w_carry[-1]} and the carry is {tens_carry_digit}. "
        if self._should_add_noise(noise_distribution_list):
            number = int(tens_sum_w_carry[-1], base)
            if semantic_related == 0:
                fact = self._random_choose_fact(number, selected_noise_set)    
                ret += f"{fact}. "
            if semantic_related == 1:
                fact = self._random_choose_semantic1_position_noise(3, int(lt, base) + int(rt, base) + ones_carry_digit, tens_sum_w_carry[-1], tens_carry_digit)    
                ret += f"{fact}. "
            if semantic_related == 2:
                fact = self._random_choose_semantic2_position_noise(5)    
                ret += f"{fact}. "
                
        ret += f"A leading digit is {tens_carry_digit}. "
        if self._should_add_noise(noise_distribution_list):
            number = tens_carry_digit
            if semantic_related == 0:
                fact = self._random_choose_fact(number, selected_noise_set)    
                ret += f"{fact}. "
            if semantic_related == 1:
                fact = self._random_choose_semantic1_position_noise(4)    
                ret += f"{fact}. "
        if semantic_related == 2:
            fact = self._random_choose_semantic2_position_noise(6)    
            ret += f"{fact}. "
        
        ret += f"So the answer is {self._get_label(expr)}. "
        if self._should_add_noise(noise_distribution_list):
            number = int(self._get_label(expr)[-1], base)
            if semantic_related == 0:
                fact = self._random_choose_fact(number, selected_noise_set)    
                ret += f"{fact}. "
            if semantic_related == 1:
                fact = self._random_choose_semantic1_number_noise(number, selected_noise_set)    
                ret += f"{fact}. "
        if semantic_related == 2:
            fact = self._random_choose_semantic2_position_noise(7, result=self._get_label(expr))    
            ret += f"{fact}. "
        
        ret += f"Answer:\\box{{{self._get_label(expr)}}}"
        return ret
            
    def _get_random_number(self):
        randomnum = 0
        while randomnum == 0:
            randomnum = random.randrange(1, 9, 1)
        return randomnum
    
    def get_inaccurate_answer(self, raw_data, noise_ratio, generate_info = None):
        expr = raw_data.split("\t")[0]
        digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        base = self.base
        lhs, rhs = expr.split("+")
        lt, lo = lhs  # tens, ones
        rt, ro = rhs
        ones_sum = self._get_label(f"{lo}+{ro}")
        carry_over = len(ones_sum) > 1
        tens_sum_wo_carry = self._get_label(f"{lt}+{rt}")
        
        noise_distribution_list = self._generate_noise_distribution_list(n_thought=8, noise_ratio=noise_ratio, noise_distribution=self.noise_distribution)
        if generate_info!=None:
            self.get_generation_config(noise_distribution_list, generate_info)
        
        if carry_over:
            ones_carry_digit = 1
            assert ones_sum[0] == "1"
            tens_sum_w_carry = self._get_label(f"{tens_sum_wo_carry}+1")
        else:
            ones_carry_digit = 0
            tens_sum_w_carry = tens_sum_wo_carry
        assert self._get_label(expr) == tens_sum_w_carry + ones_sum[-1:]
        tens_carry_over = len(tens_sum_w_carry) > 1
        tens_carry_digit = 1 if tens_carry_over else 0
        explaination = f"Since we're in base-{base}, that exceeds the maximum value of {digits[base-1]} for a single digit. " if carry_over == 1 else f"Since we're in base-{base}, that doesn't exceed the maximum value of {digits[base-1]} for a single digit. "
    
        
        ret = f"In base-{base}, the digits are \"{digits[:base]}\". "
        if self._should_add_noise(noise_distribution_list):
            fact = ""
            randomnum = 0
            while randomnum == 0: 
                randomnum = random.randrange(1, 9, 1)
            fact += f"{base} + {randomnum} = {base + randomnum}"
            ret += f"{fact}. "
        
        ret += f" We have {lo} + {ro} = {int(lo, base) + int(ro, base)} in base-10. "
        if self._should_add_noise(noise_distribution_list):
            fact = ""
            randomnum = self._get_random_number()
            fact += "{} + {} = {}".format(int(lo, base) + int(ro, base), randomnum, int(lo, base) + int(ro, base) + randomnum)
            ret += f"{fact}. "
         
        ret += explaination 
        if self._should_add_noise(noise_distribution_list):
            fact = ""
            number = int(digits[base-1], base)
            randomnum = self._get_random_number()
            fact += "{} + {} = {}".format(number, randomnum, number + randomnum)
            ret += f"{fact}. "
        
        ret += f"{int(lo, base) + int(ro, base)} mod {base} = {ones_sum[-1]}, so the digit is {ones_sum[-1]} and the carry is {ones_carry_digit}. "
        if self._should_add_noise(noise_distribution_list):
            fact = ""
            fact += f"{ones_sum[-1]} + {base} = {int(ones_sum[-1], base) + base}"
            ret += f"{fact}. "
        
        ret += f"We have {lt} + {rt} + {ones_carry_digit} = {int(lt, base) + int(rt, base) + ones_carry_digit} in base 10. " 
        if self._should_add_noise(noise_distribution_list):
            fact = ""
            randomnum = self._get_random_number()
            fact += f"{int(lt, base) + int(rt, base) + ones_carry_digit} + {randomnum} = {int(lt, base) + int(rt, base) + ones_carry_digit + randomnum}"
            ret += f"{fact}. "

        ret += f"{int(lt, base) + int(rt, base) + ones_carry_digit} mod {base} = {tens_sum_w_carry[-1]}, so the digit is {tens_sum_w_carry[-1]} and the carry is {tens_carry_digit}. "
        if self._should_add_noise(noise_distribution_list):
            fact = ""
            fact += f"{tens_sum_w_carry[-1]} + {base} = {int(tens_sum_w_carry[-1], base) + base}"
            ret += f"{fact}. "
        
        ret += f"A leading digit is {tens_carry_digit}. "    
        if self._should_add_noise(noise_distribution_list):
            fact = ""
            fact += f"{tens_carry_digit} + {base} = {tens_carry_digit + base}"
            ret += f"{fact}. "
        
        ret += f"So the answer is {self._get_label(expr)}. "
        if self._should_add_noise(noise_distribution_list):
            fact = ""
            number = int(self._get_label(expr)[-1], base)
            fact += f"{number} + {base} = {number + base}"
            ret += f"{fact}. "
        
        ret += f"Answer:\\box{{{self._get_label(expr)}}}. "
        return ret

    
    def _random_choose_semantic1_number_noise(self, number, selected_noise_set:set):
        facts = self.semantic1_noise[f"{number}"]
        while(1):
            random_index = random.randrange(0, len(facts))
            selected_fact = facts[random_index]
            fact_index = f"{number}-{random_index}"
            if fact_index not in selected_noise_set:
                selected_noise_set.add(fact_index)
                # print(fact_index)
                break
        # selected_fact = selected_fact[0] + selected_fact[1:]
        if selected_fact[-1] == ".":
            selected_fact = selected_fact[:-1] 
        return selected_fact
    
    def _random_choose_semantic1_position_noise(self, position_index, mod1 = 0, mod_result=0, carry=0):
        facts = self.semantic1_noise[f"position_{position_index}"]
       
        random_index = random.randrange(0, len(facts))
        selected_fact = facts[random_index]
        # selected_fact = selected_fact[0] + selected_fact[1:]
        if selected_fact[-1] == ".":
            selected_fact = selected_fact[:-1] 
            
        replaced_fact = selected_fact.replace("[mod1]", str(mod1))
        replaced_fact = replaced_fact.replace("[mod_result]", str(mod_result))
        replaced_fact = replaced_fact.replace("[carry]", str(carry))
        return replaced_fact
        
    
    def _random_choose_semantic2_position_noise(self, position_index, add1=0, add2=0, add3=0, result=0):
        facts = self.semantic2_noise[f"position_{position_index}"]
       
        random_index = random.randrange(0, len(facts))
        selected_fact = facts[random_index]
        # selected_fact = selected_fact[0] + selected_fact[1:]
        if selected_fact[-1] == ".":
            selected_fact = selected_fact[:-1] 
            
        replaced_fact = selected_fact.replace("[add1]", str(add1))
        replaced_fact = replaced_fact.replace("[add2]", str(add2))
        replaced_fact = replaced_fact.replace("[add3]", str(add3))
        replaced_fact = replaced_fact.replace("[result]", str(result))
        return replaced_fact
        
    
    def _random_choose_fact(self, number, selected_noise_set:set):
        facts = self.noise_data[number]["facts"]
        while(1):
            random_index = random.randrange(0, len(facts))
            selected_fact = facts[random_index]
            fact_index = f"{number}-{random_index}"
            if fact_index not in selected_noise_set:
                selected_noise_set.add(fact_index)
                # print(fact_index)
                break
        # selected_fact = selected_fact[0] + selected_fact[1:]
        if selected_fact[-1] == ".":
            selected_fact = selected_fact[:-1] 
        return selected_fact
        
        
    def get_question(self, raw_data):
        expr = raw_data.split("\t")[0]
        base = self.base
        digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if self.if_in_context:
            return f"In base-{base}, what is {expr}? Please reason it step by step. End the response with the result in \"Answer:\\boxed{{result}}\"."
        else:
            return f"You are a mathematician. Assuming that all numbers are in base-{base} where the digits are \"{digits[:base]}\", what is {expr}? Please reason it step by step. End the response with the result in \"Answer:\\boxed{{result}}\"."
        
    def get_demos(self, num, raw_data):
        demos = raw_data.split("\t")[1]
        return demos.split(",")[:num]
    
    def load_data(self):
        noise_file = "./data/math/noise/factsOfNumber.json"
        semantic1_noise_file = "./data/math/noise/semanticRalatedFactsOfNumber.json"
        semantic2_noise_file = "./data/math/noise/taskRelatedNoise.json"
        data_file = "./data/math/icl/base{}.txt".format(self.base)
        dataset = [line.strip() for line in open(data_file)]
        with open(noise_file, encoding="utf-8") as f:
            self.noise_data = json.load(f)["noise_info"]
        self.semantic1_noise = dict()
        with open(semantic1_noise_file, encoding="utf-8") as f:
            noise_info = json.load(f)["noise_info"]
            for number_facts in noise_info:
                self.semantic1_noise[number_facts["number"]] = number_facts["facts"]
        self.semantic2_noise = dict()        
        with open(semantic2_noise_file, encoding="utf-8") as f:
            noise_info = json.load(f)["noise_info"]
            for number_facts in noise_info:
                self.semantic2_noise[number_facts["number"]] = number_facts["facts"]
        return dataset
    
    @staticmethod
    def match_answer(answer_str):
        match = re.search(r'[Aa]nswer:\s*\n?.*?(-?[\d]+)(?!.*[\d])', answer_str)
        if match:
            answer = match.group(1)
        else:
            match = re.search(r'[Aa]nswer is \n?.*?(-?[\d]+)(?!.*[\d])', answer_str)
            if match:
                answer = match.group(1)
            else:
                answer = None
        if answer!=None:
            if float(answer) == int(float(answer)):
                answer = str(int(float(answer)))
        return answer

    @staticmethod
    def get_file_path():
        return os.path.join("data", "math")
        