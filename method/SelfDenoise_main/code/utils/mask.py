import numpy as np
from typing import List, Dict

def mask_forbidden_index(sentence: str, forbidden_words: List[str]) -> List[int]:
    sentence_in_list = sentence.split()
    forbidden_indexes = []
    for index, word in enumerate(sentence_in_list):
        if word in forbidden_words:
            forbidden_indexes.append(index)
    if len(forbidden_indexes) == 0:
        return None
    else:
        return forbidden_indexes

def sampling_index_loop_nums(length: int,
                             mask_numbers: int, 
                             nums: int, 
                             sampling_probs: List[float] = None) -> List[int]:
    if sampling_probs is not None:
        assert length == len(sampling_probs)
        if sum(sampling_probs) != 1.0:
            sampling_probs = sampling_probs / sum(sampling_probs)
    mask_indexes = []
    for _ in range(nums):
        mask_indexes.append(np.random.choice(list(range(length)), mask_numbers, replace=False, p=sampling_probs).tolist()) 
    return mask_indexes


def mask_sentence(sentence: str, 
                  rate: float, 
                  token: str, 
                  nums: int = 1, 
                  return_indexes: bool = False, 
                  forbidden: List[int] = None,
                  random_probs: List[float] = None, 
                  min_keep: int = 2) -> List[str]:
    # str --> List[str]
    sentence_in_list = sentence.split()
    length = len(sentence_in_list)

    mask_numbers = round(length * rate)
    if length - mask_numbers < min_keep:
        mask_numbers = length - min_keep if length - min_keep >= 0 else 0

    mask_indexes = sampling_index_loop_nums(length, mask_numbers, nums, random_probs)
    tmp_sentences = []
    for indexes in mask_indexes:
        tmp_sentence = mask_sentence_by_indexes(sentence_in_list, indexes, token, forbidden)
        tmp_sentences.append(tmp_sentence)
    if return_indexes:
        return tmp_sentences, mask_indexes
    else:
        return tmp_sentences


def mask_sentence_by_indexes(sentence: List[str], indexes: np.ndarray, token: str, forbidden: List[str]=None) -> str:
    tmp_sentence = sentence.copy()
    for index in indexes:
        tmp_sentence[index] = token
    if forbidden is not None:
        for index in forbidden:
            tmp_sentence[index] = sentence[index]
    return ' '.join(tmp_sentence)