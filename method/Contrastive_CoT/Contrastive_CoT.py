import copy

def Contrastive_CoT(postive_QAL, case_batch, model, dataprocessor, n_reason):
    new_cases = []
    
    for case in case_batch:
        new_case = dict()
        prompt = "Here is a good Explanation:\n"
        prompt += f"Question:{postive_QAL[0]}\n"
        prompt += f"Explanation: {postive_QAL[1]}\n"
        prompt += f"Answer: {postive_QAL[2]}\n"
        
        
        assert case["in-context"] != None
        n_shot = len(case["in-context"])
        prompt += f"Here is {n_shot} wrong explanations:\n"
        shots = case["in-context"]
        for shot in shots:
            prompt += f"Question:{shot[0]}\n"
            prompt += f"Wrong explanation: {shot[1]}\n"
            label = dataprocessor.match_answer(shot[1])
            prompt += f"Answer: {label}\n"
        prompt += "Give me the good explanation and answer of Question:" + case["question"]
        new_case = copy.deepcopy(case)    
        new_case["question"] = prompt
        new_case["in-context"] = []
        new_cases.append(new_case)
    model.query_case_batch(new_cases, n = n_reason)
    return new_cases