import os
import yaml
from ..multiple_key import init_api_key_handling
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import concurrent
import time


class my_gemini:
    def __init__(self, model='gemini-pro', config: dict = None) -> None:
        if model == 'gemini-pro':
            self.model = genai.GenerativeModel(model_name = "gemini-pro")
        else:
            raise ValueError(f"{model} is not supported")
        with open('gemini_key.yml', 'r') as f:
            gemini_config = yaml.safe_load(f)
        os.environ['https_proxy'] = 'http://127.0.0.1:7890'
        os.environ['http_proxy'] = 'http://127.0.0.1:7890'

        
        if isinstance(gemini_config["key"], list):   
            key_list = gemini_config["key"]
            genai.configure(api_key=init_api_key_handling(key_list, "gemini_apikey_manager.json") )
        else:
            genai.configure(api_key=gemini_config["key"])
    
        self.tokens = 0
    
    def token_cost(self, response):
        self.tokens
        return
    
    def generate_content(self, prompt_str, temperature=1, n=1, top_p=1):
        try:
            generation_config = GenerationConfig(
                candidate_count=1,  # So far, Only one candidate can be specified (Gemini)
                temperature=temperature,
                top_p=top_p
            )
            
            responses = []
            for _ in range(n):
                response = self.model.generate_content(prompt_str, generation_config=generation_config)
                responses.append(response.text)
                time.sleep(1)
            return (True, f''), responses
        except Exception as err:
            print(err)
            time.sleep(1)
            return (False, f'Gemini API Err: {err}'), err
        

    def query_case(self, case, temperature=1, n=1, top_p=1):
        if "get_prompt" not in case:
            prompt = ""
            if "system-prompt" in case:
                prompt += case["system-prompt"] + "\n"
            if "in-context" in case:
                shots = case["in-context"]
                for shot in shots:
                    prompt += f"user: {shot[0]}\n"
                    prompt += f"model: {shot[1]}\n"
            prompt += "user: {}\n".format(case["question"])
        else:
            prompt = case["get_prompt"](case)
        retval, responses = self.generate_content(prompt_str=prompt, temperature=temperature, n=n, top_p=top_p)
        
        response_content = []
        if retval[0]:
            for response in responses:
                response_content.append({"role":"assistant", "content": response})
            messages = [{"role":"user", "content":prompt}, response_content]  # the gemini format is "model" and "parts". This aims to use unique format in our program
            case["messages"] = messages
        else:
            messages = []
        return retval, messages

    def _query_and_append(self, single_query, temperature=1, n=1, top_p=1):
        err_count = 0
        while True:
            if isinstance(single_query, dict):  # case
                retval, messages = self.query_case(single_query, temperature, n, top_p)
            else:  # messages
                prompt = ""
                messages = single_query
                for message in messages[:-1]:
                    if message["role"] == "system":
                        prompt += message["content"] + "\n"
                    elif messages["role"] == "user":
                        prompt += "user: {}\n".format(message["content"])
                    else:
                        prompt += "model: {}\n".format(message["content"])
                prompt += "user: {}\n".format(messages[-1]["content"])
                retval, responses = self.generate_content(prompt, temperature, n, top_p)
                if retval[0]:
                    response_content = []
                    for response in responses:
                        response_content.append({"role":"assistant", "content": response})
                    messages.append(response_content)
            if retval[0]:
                return
            if err_count > 30:
                return
        
    def query_case_batch(self, cases, temperature=1, n=1, top_p=1):
        for case in cases:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_case = {executor.submit(self._query_and_append, case, temperature, n, top_p)}
                for future in concurrent.futures.as_completed(future_to_case):
                    future.result()
        return
    
    def query_messages_batch(self, messages_batch, temperature=1, n=1, top_p=1):
        
        for messages in messages_batch:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_case = {executor.submit(self._query_and_append, messages, temperature, n, top_p)}
                for future in concurrent.futures.as_completed(future_to_case):
                    future.result()
        return
    
    def query_n_case(self, n_case, c_reason, temperature=1, top_p=1):
        for i in range(len(n_case)):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_case = {executor.submit(self._query_and_append, n_case[i], temperature, c_reason[i], top_p)}
                for future in concurrent.futures.as_completed(future_to_case):
                    future.result()
        return
