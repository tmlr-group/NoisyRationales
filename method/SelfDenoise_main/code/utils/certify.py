import math
import scipy
import numpy as np
from typing import Tuple, List
# import statsmodels.stats.proportion

def predict(scores: np.ndarray, alpha: float) -> int:
    label_num = scores.shape[1]
    preds = np.argmax(scores, axis=-1)
    votes = np.bincount(preds, minlength=label_num)
    top_two_idx = np.argsort(votes)[-2:][::-1]
    top_one_count, top_two_count = votes[top_two_idx[0]], votes[top_two_idx[1]]
    tests = scipy.stats.binom_test(top_one_count,top_one_count+top_two_count, .5)
    if tests <= alpha:
        pred = top_two_idx[0]
        return pred
    else:
        return -1

def lc_bound(k: int, n:int, alpha:float) -> Tuple[float]:
	return statsmodels.stats.proportion.proportion_confint(k, n, alpha=2*alpha, method="beta")


def delta(size: int, keep: int, radius: int) -> float:
    return math.factorial(size-radius) * math.factorial(size-keep)/ (math.factorial(size) * math.factorial(size-keep-radius))

#((size-r)! / (keep! * (size-r-keep!))) / (size!/ (keep! * (size-keep)!))
#((size-r)! (size-keep)!/ ( size! * size-r-keep!)) 
# score - λ + λ * (size- r) choose keep) /(size choose keep) > 0.5
# (0.5 + λ - score ) / λ < ((size- r) choose keep)/ (size choose keep)
# when λ = 1, ===> 1.5 - score < ((size- r) choose keep)/ (size choose keep)
def population_radius_for_majority(bound:float, size:int, keep:int, lambda_value: float = 1.0) -> int:
    radius = 0
    lhs = (0.5 + lambda_value - bound ) / lambda_value
    while True:
        try:
            rhs = delta(size, keep, radius)
            if lhs >= rhs:
                break
            else:
                radius += 1
        except:
            break
    return radius

def population_radius_for_majority_by_estimating_lambda(bound:float, 
                                                        size:int, 
                                                        keep:int, 
                                                        preds: np.ndarray, 
                                                        ablation_indexes:List[List[int]],
                                                        label_num: int,
                                                        target: int, 
                                                        samplers: int=200) -> int:
    radius = 0
    while True:
        if radius == 0:
            lhs = 1.5 - bound
        else:
            lambda_value = population_lambda(preds, ablation_indexes, size, radius, label_num, target, samplers)
            lhs = (0.5 + lambda_value - bound) / lambda_value
        rhs = delta(size, keep, radius)
        if lhs >= rhs:
            break
        else:
            radius += 1
    return radius

def population_lambda(preds: np.ndarray, 
                     ablation_indexes:List[List[int]] , 
                     size: int, 
                     radius: int, 
                     label_num: int,
                     target: int, 
                     samplers: int=200) -> float:
    sampler_indexes = [np.random.choice(list(range(size)), size=(radius, )).tolist() for _ in range(samplers)]
    take_pred_array = []
    for sampler_index in sampler_indexes:
        take_indics = []
        for index, ablation_index in enumerate(ablation_indexes):
            if any(x in ablation_index for x in sampler_index):
                take_indics.append(index)
        take_result=np.take(preds, take_indics, axis=0)
        take_pred_array.append(take_result)
    take_pred_array = np.concatenate(take_pred_array, axis=0)
    return np.bincount(take_pred_array, minlength=label_num)[target] / take_pred_array.shape[0] 