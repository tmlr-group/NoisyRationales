import random
import json
import pandas as pd
import zipfile
import os
import re
import ast
import copy
from collections import deque
import math
from ..processor_base import data_processor 

class commonsense(data_processor):
    def __init__(self, n_shots=0, n_noisy_shots=0, noise_type="irrelevant", noise_ratio = 0.5, noise_distribution = "fixed", prefix_context =False, config: dict = None, subtask = "sym", hop = 3, trainset=5, testset = 5.3) -> None:
        super().__init__(n_shots, n_noisy_shots, noise_type, noise_ratio, noise_distribution, prefix_context)
        if config is not None:
            self.trainset = config["train_set"] if "train_set" in config else trainset
            self.subtask = config["subtask"]
            self.hop = config["hop"] if "hop" in config else hop
        else:
            self.trainset = trainset
            self.testset = testset
            self.subtask = subtask
            self.hop = hop
        if subtask == "sym":
            assert self.trainset >= 5
        elif subtask == "story":
            assert self.trainset < 5
        else:
            raise ValueError(f"reasoning type not support {subtask}")
        self.not_support_relation_reason = []
        self.replace_num = 0
        self.error_reason_num = 0
        self.file_path = os.path.join("data", "commonsense")
        self.unzip_data()
        self.init_noise_data()
        self.init_relation_dict()
        return
    
    def init_relation_dict(self):
        with open(os.path.join(self.file_path,  "relation_dict.json"), "r") as f:
            kv_list = json.load(f)
        self.relation_dict = dict()
        for kv in kv_list:
            key = kv["key"]
            key = (key[0], key[1])
            value = kv["value"]
            self.relation_dict[key] = value
        return
        
    
    def get_config(self):
        config = dict()
        config["subtask"] = self.subtask
        config["hop"] = self.hop
        return config
    
    def init_noise_data(self):
        with open(os.path.join(self.file_path, "noise", "noise_relation_facts.json"), "r") as f:
            noise_facts = json.load(f)
        self.noise_relation_facts = dict()
        for relation_facts in noise_facts:
            relation = relation_facts["relation"]
            facts = relation_facts["facts"] 
            self.noise_relation_facts[relation] = facts
            
        with open(os.path.join(self.file_path, "noise", "noise_facts.json"), "r") as f:
            noise_facts = json.load(f)
        self.noise_facts = dict()
        for relation_facts in noise_facts:
            relation = relation_facts["relation"]
            facts = relation_facts["facts"] 
            self.noise_facts[relation] = facts
    
    def unzip_data(self):
        zip_files = ["data_7c5b0e70.zip", "data_06b8f2a1.zip", "data_523348e6.zip", "data_d83ecc3e.zip", "data_db9b8f04.zip"]
        unzip_path = os.path.join(self.file_path, "unzip_data")
        if not os.path.exists(unzip_path):
            os.makedirs(unzip_path)
        for index, zip_file in enumerate(zip_files):
            with zipfile.ZipFile(os.path.join(self.file_path, zip_file), 'r') as zip_ref:
                unzip_file_path = os.path.join(unzip_path, str(index + 1))
                if not os.path.exists(unzip_file_path):
                    os.makedirs(unzip_file_path)
                zip_ref.extractall(unzip_file_path)
    
    def shuffle_dataset(self, step2_set, step3_set):
        set_list = []
        for i in range(max(len(step2_set), len(step3_set))):
                if i < len(step2_set):
                    set_list.append(step2_set.iloc[i])
                if i < len(step3_set):
                    set_list.append(step3_set.iloc[i])
        shuffled_set = pd.concat(set_list, axis=1).T.reset_index(drop=True)
        return shuffled_set 
    
    def _to_dict(self, df_row):
        columns_to_extract = ['edge_types', 'query', "target", "proof_state"]
        shot = {col: df_row[col] for col in columns_to_extract}
        return shot
        
    def _to_list(self, df):
        columns_to_extract = ['edge_types', 'query', "target", "proof_state"]
        shots = df.loc[:, columns_to_extract].to_dict(orient='records')
        return shots
    
    def generate_json(self, dataset, type):
        data_iter = dataset.iterrows()
        processed_path = os.path.join(self.file_path, "processed")
        if not os.path.exists(processed_path):
            os.makedirs(processed_path)
        if self.subtask == "sym":
            file_name = f"{self.subtask}_{self.hop}hop"
            if type == 1:
                file_name += "_demos.json"
                raw_data_IC_list = []
                for _, raw_data in data_iter:
                    demos = self.get_demos(num=10)
                    raw_data_IC = dict()
                    raw_data_IC["test"] = self._to_dict(raw_data)
                    raw_data_IC["demos"] = self._to_list(demos)
                    raw_data_IC_list.append(raw_data_IC)
                file_path = os.path.join(processed_path, file_name)
                with open(file_path, 'w') as json_file:
                    json.dump(raw_data_IC_list, json_file, indent=4)
            elif type == 2:
                cases = []
                if self.n_shots > 0:
                    file_name += f"_{self.n_shots}clean"
                if self.n_noisy_shots > 0:
                    file_name += f"_{self.n_shots}{self.noise_type}_{self.noise_ratio}_{self.noise_distribution}"
                file_name += ".json"
                file_path = os.path.join(processed_path, file_name)
                for data in dataset:
                    case = self.get_case(data)
                    cases.append(case)
                with open(file_path, 'w') as json_file:
                    json.dumps(cases, json_file, indent=4)
            
    
    def load_data(self):
        unzip_path = os.path.join(self.file_path, "unzip_data", str(self.trainset))
        if self.subtask !=  "sym":
            file_name = f"{self.trainset}.2,{self.trainset}.3_train.csv"
            raw_dataset = pd.read_csv(os.path.join(unzip_path, file_name))
            self.relation_list = list(set(raw_dataset["target"]))
            
            step3_set = raw_dataset[raw_dataset['query_edge'] == "(0, 3)"]
            step2_set = raw_dataset[raw_dataset['query_edge'] == "(0, 2)"]

            train_num = int(len(step3_set) * 0.5)
            step3_train = step3_set.iloc[:train_num]
            step3_test = step3_set.iloc[train_num:]

            train_num = int(len(step2_set) * 0.5)
            step2_train = step2_set.iloc[:train_num]
            step2_test = step2_set.iloc[train_num:]
            
            testset = self.shuffle_dataset(step2_test, step3_test)
            self.trainset = self.shuffle_dataset(step2_train, step3_train)
        else:
            file_name = f"1.2,1.3,1.4_train.csv"
            raw_dataset = pd.read_csv(os.path.join(unzip_path, file_name))
            self.relation_list = list(set(raw_dataset["target"]))
            
            step2_set = raw_dataset[raw_dataset['query_edge'] == "(0, 2)"]
            step3_set = raw_dataset[raw_dataset['query_edge'] == "(0, 3)"]
            step4_set = raw_dataset[raw_dataset['query_edge'] == "(0, 4)"]
            
            if self.hop == 3:
                dataset = step3_set
            elif self.hop == 4:
                dataset = step4_set
            elif self.hop == 2:
                dataset = step2_set
            else:
                raise ValueError(f"hop {self.hop} not support")
            
            mask = dataset['edge_types'].apply(lambda x: len(ast.literal_eval(x)) == 3)
            dataset = dataset[mask]
            dataset = dataset.sample(frac=1).reset_index(drop=True)
            unnamed_cols = [col for col in dataset.columns if col.startswith('Unnamed')]
            if unnamed_cols:
                dataset = dataset.drop(unnamed_cols, axis=1)
            dataset_processed = []
            data_iter = dataset.iterrows()
            for count, raw_data in data_iter:
                dataset_processed.append(raw_data)
            self.trainset = dataset
            testset = dataset_processed        
        return testset
    
    def get_question(self, raw_data):
        question = ""
        if self.subtask != "sym":
            story = raw_data["story"]
            question += f"Story:{story}\n"
            query = ast.literal_eval(raw_data["query"])
            head_name = query[0]
            tail_name = query[1]
            question += f"Question: Given the relationships described in the story information, please infer { tail_name } is { head_name }'s what. "
        else:
            relation_path = ast.literal_eval(raw_data["edge_types"])
            query = ast.literal_eval(raw_data["query"])
            head_name = query[0]
            tail_name = query[1]
            relation_str = "'s ".join(relation_path)
            question += f"In a family tree, if {tail_name} is {head_name}'s {relation_str}. \nQuestion: {tail_name} is {head_name}'s what? "
        question += "Please reason it step by step, and provide a single word answer describing the relationship. End the response in the format  \"Answer: {{relation}}\"\n"
        return question
    
    def get_label(self, raw_data):
        return raw_data["target"]
        
    def get_random_fact(self, relation, selected_set):
        facts = self.noise_facts[relation]
        while(1):
            random_index = random.randrange(0, len(facts))
            selected = f"{relation}_{random_index}"
            if selected not in selected_set:
                fact = facts[random_index]
                selected_set.add(selected)
                break
        return fact
    
    def get_random_relation_fact(self, relation, selected_set):
        facts = self.noise_relation_facts[relation]
        while(1):
            random_index = random.randrange(0, len(facts))
            selected = f"{relation}_{random_index}"
            fact = facts[random_index] + " "
            if selected not in selected_set:
                selected_set.add(selected)
                break
            go_on_random = 0
            for i in range(len(facts)):
                key = f"{relation}_{i}"
                if key not in selected_set:
                    go_on_random = 1
            if go_on_random == 0:
                break
        return fact
    
    def _search_relation_in_path(self, relation_path, r1, r2):
        search_elements = (r1, r2)
        path_index = -1
        for path_i in range(len(relation_path) - 1):
            if (relation_path[path_i], relation_path[path_i+1]) == search_elements:
                path_index = path_i
                break
        return path_index
        
    def get_random_relation(self, original_relation = None):
        relation_num = len(self.relation_list)
        while 1:
            random_index = random.randrange(0, relation_num)
            new_relation = self.relation_list[random_index]
            if new_relation != original_relation:
                break
        return new_relation
    
    def create_proof_chain(self, relation_path, proof_chain = None):
        if proof_chain == None:
            proof_chain = []
        if len(relation_path) <= 1:
            return proof_chain
        for index in range(len(relation_path) - 1):
            r1 = relation_path[index]
            r2 = relation_path[index + 1]
            
            if (r1, r2) not in self.relation_dict:
                self.not_support_relation_reason.append((r1, r2))
                continue
            else:
                r_mix = self.relation_dict[(r1, r2)]
                try_path = copy.deepcopy(relation_path)
                del try_path[index:index+2]
                try_path.insert(index, r_mix)
                if self.create_proof_chain(try_path, proof_chain) != None:
                    proof_chain.append([r1, r2, r_mix])
                    return proof_chain
        return None
            
    def _prepare_noise_distribution_iteration_state(self, n_thought, noise_ratio, noise_distribution):
        noise_distribution_list = [0] * n_thought
        if noise_distribution == "fixed":
            noise_thoughts = n_thought * noise_ratio
            integer_part = int(noise_thoughts)
            decimal_part = noise_thoughts - integer_part
            if decimal_part == 0.5:
                noise_count = math.ceil(n_thought * noise_ratio)
            else:
                noise_count = round(n_thought * noise_ratio)
            noise_positions = random.sample(range(n_thought), noise_count)
            for pos in noise_positions:
                noise_distribution_list[pos] = 1
        else:
            for pos in range(len(noise_distribution_list)):
                if random.random() < noise_ratio:
                    noise_distribution_list[pos] = 1 
        return [noise_distribution_list, 0]
    
    def _should_add_noise(self, noise_distribution_state):
        if noise_distribution_state == None:
            return 0
        distribution_list = noise_distribution_state[0]
        pos = noise_distribution_state[1]
        if_noise = distribution_list[pos]
        noise_distribution_state[1] += 1
        return if_noise
                    

    def get_random_inaccurate_thought(self, r1):
        r2 = self.get_random_relation(r1)
        
        if (r1, r2) not in self.relation_dict:
            err_mix = self.get_random_relation(None)
        else:
            r_mix = self.relation_dict[(r1, r2)]
            err_mix =  self.get_random_relation(r_mix)
        inaccurate_thought = f"We have {r1}'s {r2} is {err_mix}. "
        return inaccurate_thought
    
    def get_symbolic_relation_reason(self, raw_data, ir_noise_distrib_state=None, mn_noise_distrib_state=None):
        proofs =  ast.literal_eval(raw_data["proof_state"])
        relation_path = ast.literal_eval(raw_data["edge_types"])
        relation_path_str = ", ".join(relation_path)
        relation_desciption = "'s ".join(relation_path)
        
        query = ast.literal_eval(raw_data["query"])
        head_name = query[0]
        tail_name = query[1]
        
        n_ir_pos = 0
        n_mn_pos = 0
        selected_noise_set = set()
        
        answer = ""
        answer += f"{tail_name} is {head_name}'s {relation_desciption}, so the relations path is {relation_path_str}. "
        n_mn_pos+=1
        
        if self._should_add_noise(mn_noise_distrib_state):
            noise_fact = self.get_random_inaccurate_thought(relation_path[-1])
            answer += noise_fact
            
        n_ir_pos += 1
        if self._should_add_noise(ir_noise_distrib_state):
            noise_fact = self.get_random_relation_fact("family relation", selected_noise_set)
            answer += noise_fact
        
        noise_type = self.noise_type      
        proof_chain = []
        for proof in reversed(proofs):
            for conclusion, reasons in proof.items():
                proof_chain.append([reasons[0][1], reasons[1][1], conclusion[1]])
        
        
        new_proof_chain = copy.deepcopy(proof_chain)
        
        reasoning_relation_path = copy.deepcopy(relation_path)
        r_mix = None
        for proof in new_proof_chain:
            r1 = proof[0]
            r2 = proof[1]
            r_mix = proof[2]
            index = self._search_relation_in_path(reasoning_relation_path, r1, r2)
            del reasoning_relation_path[index:index+2]
            reasoning_relation_path.insert(index, r_mix)
            
            answer += f"For {r1}'s {r2}, we have {r1}'s {r2} is {r_mix}. "
            n_mn_pos+=1
            if self._should_add_noise(mn_noise_distrib_state):
                noise_fact = self.get_random_inaccurate_thought(r2)
                answer += noise_fact
            
            n_ir_pos += 1
            if self._should_add_noise(ir_noise_distrib_state):
                noise_fact = self.get_random_relation_fact(r2, selected_noise_set)
                answer += noise_fact
            
            relation_str = ", ".join(reasoning_relation_path)
            answer += f"So the relations path are reduced to {relation_str}. "
            
            n_mn_pos+=1
            if self._should_add_noise(mn_noise_distrib_state):
                noise_fact = self.get_random_inaccurate_thought(r_mix)
                answer += noise_fact
                    
            n_ir_pos += 1
            if self._should_add_noise(ir_noise_distrib_state):
                noise_fact = self.get_random_relation_fact(r_mix, selected_noise_set)
                answer += noise_fact
                
        answer += f"Therefore, Answer: {r_mix}. \n"
        return answer, n_ir_pos, n_mn_pos
    
    def find_sentence_containing_strings(self, text, name1, name2):
        sentences = text.split('.')
        for sentence in sentences:
            if name1 in sentence and name2 in sentence:
                return sentence
        return None
    
    def get_generation_config(self, noise_distribution_state, generate_info):
        noise_distribution_list = noise_distribution_state[0]
        generate_info["total_thought"] = len(noise_distribution_list) + noise_distribution_list.count(1)
        generate_info["noise_thought"] = noise_distribution_list.count(1)
        generate_info["sentences_with_noise"] = []
        for if_noise in noise_distribution_list:
            generate_info["sentences_with_noise"].append(0)
            if if_noise:
                generate_info["sentences_with_noise"].append(1)
        generate_info["sentences_with_noise"].append(0)
    
    def get_correct_answer(self, raw_data, generate_info=None):
        return self.get_answer(raw_data, generate_info=generate_info)
    
    def get_irrelevant_answer(self, raw_data, noise_ratio, generate_info):
        return self.get_answer(raw_data, "irrelevant", noise_ratio, generate_info)

    def get_inaccurate_answer(self, raw_data, noise_ratio, generate_info):
        return self.get_answer(raw_data, "inaccurate", noise_ratio, generate_info)
        
    def get_answer(self, raw_data, noise_type=None, noise_ratio=0, generate_info = None):
        answer = ""
        _, n_ir_pos, n_mn_pos = self.get_symbolic_relation_reason(raw_data)
        ir_noise_p = 0
        mn_noise_p = 0
        if noise_type == "irrelevant":
            ir_noise_p = noise_ratio
        elif noise_type == "inaccurate":
            mn_noise_p = noise_ratio
        ir_noise_distrib_state = self._prepare_noise_distribution_iteration_state(n_ir_pos, ir_noise_p, self.noise_distribution) 
        mn_noise_distrib_state = self._prepare_noise_distribution_iteration_state(n_mn_pos, mn_noise_p, self.noise_distribution)
        if generate_info is not None:          
            if(noise_type == "irrelevant"):
                self.get_generation_config(ir_noise_distrib_state, generate_info)
            elif(noise_type == "inaccurate"):
                self.get_generation_config(mn_noise_distrib_state, generate_info)
            else:
                self.get_generation_config(ir_noise_distrib_state, generate_info)                 
        answer, _, _ = self.get_symbolic_relation_reason(raw_data, ir_noise_distrib_state, mn_noise_distrib_state)
        return answer
    
    def get_demos(self, num, expr=None, index_list = None):
        if expr is not None:
            assert len(self.trainset) > num - 1
            expr_edge_types = expr["edge_types"]
            mask = self.trainset["edge_types"] == expr_edge_types
            trainset = self.trainset[~mask]
            demos = trainset.sample(n=num)
        else:
            demos = self.trainset.sample(n=num)
        if index_list is not None:
            index_list.extend(demos.index.tolist())
        demo_list = []
        data_iter = demos.iterrows()
        for count, raw_data in data_iter:
            demo_list.append(raw_data)
        return demo_list
    
    def get_demos_by_index_list(self, num, index_list):
        demos = self.trainset.loc[index_list[:num]]
        demo_list = []
        data_iter = demos.iterrows()
        for count, raw_data in data_iter:
            demo_list.append(raw_data)
        return demo_list
    
    
    @staticmethod
    def match_answer(answer_str):
        match = re.search(r'[Aa]nswer:.*?([A-Za-z\-]+)', answer_str)
        if match:
            return match.group(1).lower()
        else:
            return None
            
            
        
        