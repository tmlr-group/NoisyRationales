import os
import pickle
import sys
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
from typing import (Generic, NamedTuple, Optional, Protocol, Tuple, TypeVar,
                    Union, runtime_checkable)


class data_processor(ABC):
    def __init__(self, n_shots=0, n_noisy_shots=0, noise_type="irrelevant", noise_ratio = 0.5, noise_distribution = "fixed", prefix_context =False, **kwargs) -> None:
        """
        Processor basic class.
        
        Attributes:
            n_shots: The number of normal shots. Default is 0.
            n_noisy_shots: The number of noisy shots. Default is 0.
            noise_type: The type of noisy, can be "inaccurate" or "irrelevant". Default is "irrelevant".
            noise_ratio: The ratio of noise. Each thought has a chance.
            noise_distribution: The method to fill the noise. ( fixed noise num in one shot or random num in one shot )
        """
        self._init_noise_setting(n_shots, n_noisy_shots, noise_type, noise_ratio, noise_distribution)
        self.prefix_context = prefix_context
        
    
    def _init_noise_setting(self, n_shots=0, n_noisy_shots=0, noise_type="irrelevant", noise_ratio = 0.5, noise_distribution = "fixed"):
        self.n_shots = n_shots
        self.n_noisy_shots = n_noisy_shots
        self.total_shot = self.n_shots + self.n_noisy_shots 
        if self.total_shot > 0:
            self.if_in_context = True
        else:
            self.if_in_context = False
        
        if self.n_noisy_shots > 0:
            self.noise_ratio = noise_ratio
            self.noise_distribution = noise_distribution
            assert noise_distribution == "fixed" or noise_distribution == "random" or noise_distribution == "n_thought"
        else:
            self.noise_ratio = 0
            self.noise_distribution = None
            self._noise_semantic_related = 0
        self.noise_type = noise_type 
    
    
    @abstractmethod
    def load_data(self):
        ...
    
    @abstractmethod
    def get_question(self, raw_data):
        ...
    
    @abstractmethod
    def get_label(self, raw_data):
        ...
    
    @abstractmethod
    def get_correct_answer(self, raw_data, generate_info=None, **kwargs):
        ...
    
    @abstractmethod
    def get_irrelevant_answer(self, raw_data, noise_ratio, generate_info=None, **kwargs):
        ...
    
    @abstractmethod
    def get_inaccurate_answer(self, raw_data, noise_ratio, generate_info=None, **kwargs):
        ...
    
    @abstractmethod 
    def get_demos(self, num, raw_data = None, **kwargs):
        ...
   
    def get_case(self, raw_data, if_generate_info=False, ICL_index_list=None):
        n_shots = self.n_shots
        total_shots = self.n_shots + self.n_noisy_shots
        n_noisy_shot = self.n_noisy_shots
        case = dict()
        shots = []
        if total_shots > 0:    
            if ICL_index_list is None:
                demos = self.get_demos(total_shots, raw_data)
            else:
                demos = self.get_demos_by_index_list(num=total_shots, index_list= ICL_index_list)
            normal_demos = demos[:n_shots]
            assert len(normal_demos) == self.n_shots
            for demo in normal_demos:
                if if_generate_info:
                    generate_info = dict()
                else:
                    generate_info = None
                shot_q = self.get_question(demo)
                shot_a =  self.get_correct_answer(demo, generate_info)
                if not if_generate_info:
                    shots.append([shot_q, shot_a])
                else:
                    shots.append([shot_q, shot_a, generate_info])   
            if self.n_noisy_shots > 0:    
                noisy_shots = []
                noisy_demos = demos[n_shots:n_shots + n_noisy_shot]
                for demo in noisy_demos:
                    if if_generate_info:
                        generate_info = dict()
                    else:
                        generate_info = None
                    shot_q = self.get_question(demo)
                    if self.noise_type == "inaccurate":
                        shot_a =  self.get_inaccurate_answer(demo, self.noise_ratio, generate_info)
                    elif self.noise_type == "irrelevant":
                        shot_a =  self.get_irrelevant_answer(demo, self.noise_ratio, generate_info)
                    else:
                        raise ValueError(f"noisy type not support:{self.noise_type}")
                    if not if_generate_info:
                        noisy_shots.append([shot_q, shot_a])
                    else:
                        noisy_shots.append([shot_q, shot_a, generate_info])
                shots = shots + noisy_shots
                # random.shuffle(shots)  
            case["in-context"] = shots
        question = self.get_question(raw_data)
        real_answer = self.get_label(raw_data)
        case["question"] = question
        case["label"] = real_answer 
        case["answer"] = self.get_correct_answer(raw_data)
        return case
    
    @abstractmethod
    def match_answer(answer_str):
        ...
