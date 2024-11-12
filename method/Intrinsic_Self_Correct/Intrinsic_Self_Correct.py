import copy
# from llm_model.my_gpt.my_gpt import my_gpt

def Intrinsic_Self_Correct(case_batch, model, dataset_name, n_reason, answer_match_func = None, method = "ISC"):
    model.query_case_batch(case_batch, n = 1)
    
    assert method in ["ISC", "SCO"]
    
    # query answer
    if method == "SCO":
        max_times = 2
        cases_complete = [answer_match_func(case["messages"][-1][0]["content"]) == case["label"] for case in case_batch]
    else:
        cases_complete = [0] * len(case_batch)
        max_times = 1

    for _ in range(max_times):
        prompt1 = "Review your previous answer and find problems with your answer."
        prompt2 = "Based on the problems you found, improve vour answer. Please reiterate your answer, with your final answer "
        if dataset_name == "base_math":
            prompt2 += "in the format of \"Answer:\\boxed{{result}}\""
        elif dataset_name == "symbolic":
            prompt2 += "in the format of \"So, final answer is OUT: <action sequence>\""
        elif dataset_name == "commonsense":
            prompt2 += "in the format of \"Answer: {{relation}}\""
        else:
            prompt2 += "a single numerical number, in the form \\boxed{{answer}}"
        
        messages_list = []
        
        for i, case in enumerate(case_batch):
            responses = case["messages"][-1]
            if cases_complete[i] == 0:
                # add last query answer as in-context
                messages = copy.deepcopy(case["messages"][:-1])
                messages.append(responses[0])
                # add Review prompt1
                messages.append({'role': "user", 'content': prompt1})
                messages_list.append(messages)

        model.query_messages_batch(messages_list, n = 1)
        
        for messages in messages_list:
            messages[-1] = messages[-1][0]
            # add rephrase prompt2
            messages.append({'role': "user", 'content': prompt2})
            
       
        model.query_messages_batch(messages_list, n = n_reason)
        
        index = 0
        
        # write back to original case_batch
        for i, case in enumerate(case_batch):
            if cases_complete[i] == 0:
                case["messages"][-1] = messages_list[index][-1]
                label = case["label"]
                if method == "SCO":
                    if answer_match_func(messages_list[index][-1][0]["content"]) == label:
                        cases_complete[i] = 1
                        case["messages"][-1] = [messages_list[index][-1][0]] * n_reason
                index += 1
                            
        if method == "SCO":
            if all(complete == 1 for complete in  cases_complete):
                break
    for case in case_batch:
        if len(case["messages"][-1]) < n_reason:
            case["messages"][-1] = [case["messages"][-1][0]] * n_reason
    return case_batch