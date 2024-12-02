import yaml
import os
from typing import List, Optional
import json
import re
import pickle
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_process.math import math
from data_process.commonsense import commonsense 
from data_process.symbolic import symbolic
import pandas as pd
import nltk
import random
import time
from datetime import datetime
import copy
import string
import argparse


class generate_test:
    def __init__(self, args) -> None:
        self.config = args
        self.args = args
        self._dataset_name = args["dataset"]
        generate_config = args["raw_dataset_options"]
        self._if_noise = generate_config["if_noise"] if "if_noise" in generate_config else False
        if self._if_noise:
            self._n_shots = 0
            self._n_noisy_shots = 10
            self._noise_type = generate_config["noise_type"]
            self._noise_ratio = generate_config["noise_ratio"]
            self._noise_distribution = generate_config["noise_distribution"]
        else:
            self._n_shots = 10
            self._n_noisy_shots = 0
            self._noise_type = None
            self._noise_ratio = 0
            self._noise_distribution = None
        self._prefix_context = generate_config["prefix_context"] if "prefix_context" in generate_config else False
        random.seed(time.time())
        self._init_dataset()
        self._case_list = []
        return
    
    def _init_dataset(self):
        processor_config = self.config[self._dataset_name] if self._dataset_name in config else None
        if self._dataset_name == "math":
            self._dataset_processor = math.math(n_shots=self._n_shots,
                                                          n_noisy_shots=self._n_noisy_shots,
                                                          noise_type=self._noise_type, noise_ratio=self._noise_ratio, noise_distribution=self._noise_distribution,
                                                          prefix_context=self._prefix_context, config=processor_config)
        elif self._dataset_name == "commonsense":
            self._dataset_processor = commonsense.commonsense(n_shots=self._n_shots,
                                                                      n_noisy_shots=self._n_noisy_shots,
                                                                      noise_type=self._noise_type,
                                                                      noise_ratio=self._noise_ratio,
                                                                      prefix_context=self._prefix_context,
                                                                      config=processor_config)
            self._dataset_config = self._dataset_processor.get_config()
        elif self._dataset_name == "symbolic":
            self._dataset_processor = symbolic.symbolic(n_shots=self._n_shots, n_noisy_shots=self._n_noisy_shots, noise_type=self._noise_type,  noise_ratio=self._noise_ratio, noise_distribution=self._noise_distribution, prefix_context=self._prefix_context, config = processor_config)
        else:
            raise ValueError("Unsupported dataset {}".format(self._dataset_name))
        self._dataset = self._dataset_processor.load_data()
        
    def _question_insert(self, raw_data, ICL_index_list = None):
        if ICL_index_list is None:
            processed_case = self._dataset_processor.get_case(raw_data, if_generate_info=True)
        else:
            processed_case = self._dataset_processor.get_case(raw_data, if_generate_info=True, ICL_index_list = ICL_index_list)
        self._case_list.append(processed_case)
    
    def generate_shot_index(self, set_file_path=None):
        if self._dataset_name == "math":
            raise ValueError("The dataset for binary calculation inherently comes with ICL demos, so there is no need to generate indices. ")
        if set_file_path is None:
            file_path = self.get_shot_index_file_path()
        
        case_list = []
        if isinstance(self._dataset, pd.DataFrame):
            data_iter = self._dataset.iterrows()
        else:
            data_iter = enumerate(self._dataset)
        for current_index, data in data_iter:
            case = dict()
            other_indices = []
            if self._dataset_name == "symbolic":
                self._dataset_processor.get_demos(10, index_list=other_indices)
            else:
                self._dataset_processor.get_demos(10, expr=data, index_list=other_indices)
            case["index"] = current_index
            case["ICL_shots_index"] = other_indices
            case_list.append(case)
        with open(file_path, "w") as f:
            json.dump(case_list, f)

    def get_shot_index_file_path(self):
        file_dir = os.path.join(self._dataset_processor.file_path, "processed")
        if self._dataset_name != "commonsense":
            subtask = self.args[self._dataset_name]["subtask"]
            file_dir = os.path.join(file_dir, subtask)
        file_name = "10_shot_ICL_index.json"
        file_path = os.path.join(file_dir, file_name)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        return file_path

    def generate_dataset(self, set_file_path=None):
        dataset = dict()
        
        dataset_config = dict()
        dataset_config["dataset"] = self._dataset_name
        if self._dataset_name != "commonsense":
            dataset_config["subtask"] = self.config[self._dataset_name]["subtask"]
            
        dataset_content = []
        if isinstance(self._dataset, pd.DataFrame):
            data_iter = self._dataset.iterrows()
        else:
            data_iter = enumerate(self._dataset)
            
        if self._dataset_name == "math":
            for count, raw_data in data_iter:
                self._question_insert(raw_data)
        else:
            ICL_index_file_path = self.get_shot_index_file_path()
            with open(ICL_index_file_path, "r") as f:
                ICL_index_lists = json.load(f)
            for (count, raw_data), ICL_index_list in zip(data_iter, ICL_index_lists):
                self._question_insert(raw_data, ICL_index_list["ICL_shots_index"])
        cases = self._case_list
        
        dataset_config["question_num"] = len(cases)
        dataset_config["if_noise"] = self._if_noise
        dataset_config["noise_type"] = self._noise_type
        dataset_config["noise_ratio"] = self._noise_ratio
        dataset_config["noise_distribution"] = self._noise_distribution
        dataset_config["n_max_shots"] = self._n_noisy_shots + self._n_shots
        
        n_thoughts = 0 
        n_noisy_thoughts = 0
        for case in cases:
            case_store = dict()
            case_store["question"] = case["question"]
            case_store["label"] = case["label"]
            shots = case["in-context"]
            CoT_demos = []
            for shot in shots:
                demo = dict()
                demo["question"] = shot[0]
                demo["answer"] = shot[1]
                demo["n_total_thought"] = shot[2]["total_thought"]
                n_thoughts += demo["n_total_thought"]
                demo["n_noise_thought"] = shot[2]["noise_thought"]
                n_noisy_thoughts += demo["n_noise_thought"]
                demo["sentences_with_noise"] = ','.join(map(str, shot[2]["sentences_with_noise"]))
                CoT_demos.append(demo)
            case_store["CoT_demos"] = CoT_demos
            dataset_content.append(case_store)
        
        dataset_config["avg_demo_thought"] = n_thoughts / (dataset_config["n_max_shots"] * dataset_config["question_num"])
        dataset_config["avg_demo_noisy_thought"] = n_noisy_thoughts / (dataset_config["n_max_shots"] * dataset_config["question_num"])
            
        dataset["config"] = dataset_config
        if "system-prompt" in cases[0]:
            dataset["system-prompt"] = cases[0]["system-prompt"]
        dataset["content"] = dataset_content    
        
        file_dir = os.path.join(self._dataset_processor.file_path, "processed")
        if self._dataset_name != "commonsense":
            subtask = self.args[self._dataset_name]["subtask"]
            file_dir = os.path.join(file_dir, subtask)
        if set_file_path is None:
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            if self._if_noise == False:
                file_name = "clean.json"
            else:
                file_name = f"{self._noise_type}"
                if self._noise_ratio == 0.3: 
                    file_name += "_easy"
                elif self._noise_ratio == 0.5: 
                    file_name += "_medium"
                elif self._noise_ratio == 0.8: 
                    file_name += "_hard"
                else:
                    file_name += "_{}".format(self._noise_ratio)
                file_name += f"_{self._noise_distribution}.json"
            file_path = os.path.join(file_dir, file_name)
        with open(file_path, "w") as f:
            json.dump(dataset, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='./generate_processed_dataset/generate_dataset.yml', help='Path to the config file')
    args = parser.parse_args()
    config_path = args.config
    with open(config_path, 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)
    test = generate_test(args=config)
    test.generate_dataset()