from openai import OpenAI 
import requests
import yaml
import concurrent.futures
import time
import tiktoken
import os
import importlib.metadata
from ..multiple_key import init_api_key_handling

class my_zhipu:
    def __init__(self, model='glm-4-long', config: dict = None, api="openai", prefix_context=True) -> None:
        if config != None:
            api = config["api"] if "api" in config else api
        self.api = api
        self.model = model
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.embedding_tokens = 0
        self.total_tokens = 0
        self.max_prompt_tokens = 4096
        self.max_response_tokens = 1000
        self.prefix_context = prefix_context
       
        key = os.getenv('OPENAI_API_KEY')
        if ":" in key:
            key_list = key.split(":")
            self.key = init_api_key_handling(key_list)
        else:
            self.key = key

        self.client = OpenAI(
            api_key=self.key,
            base_url="https://open.bigmodel.cn/api/paas/v4/"
        ) 

        pass

    def chat(self, single_chat):
        messages = []
        messages.append({'role': "user", 'content': single_chat})
        retval, error = self.query(messages)
        if retval:
            return messages[-1][0]["content"]
        else:
            return f"error:{error}"

    def query(self, messages, temperature=1, n=1, top_p=1):
        try:
            completions = []
            for _ in range(n):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=self.max_response_tokens
                )
                self.completion_tokens += response.usage.completion_tokens
                self.prompt_tokens += response.usage.prompt_tokens
                self.total_tokens += response.usage.total_tokens
                for choice in response.choices:
                    message = choice.message
                    completion = dict()
                    completion['role'] = message.role
                    completion['content'] = message.content
                    completions.append(completion)
            messages.append(completions)
            return (True, ''), messages
        except Exception as err:
            print(err)
            return (False, f'OpenAI API Err: {err}'), messages
            
    def compute_cost(self):
        return "input tokens:{}, output tokens:{}, embedding tokens:{}, total tokens:{}".format(
            self.prompt_tokens, self.completion_tokens, self.embedding_tokens, self.total_tokens)

    def query_case(self, case, temperature, n, top_p):
        messages = []
        if "system-prompt" in case:
            system_prompt = case["system-prompt"]
            messages.append({'role': "system", 'content': system_prompt})
        if self.prefix_context == False:
            if "in-context" in case:
                IC_list = case["in-context"]
                for shot in IC_list:
                    shot_q = shot[0]
                    shot_a = shot[1]
                    messages.append({'role': "user", 'content': shot_q})
                    messages.append({'role': "assistant", 'content': shot_a})
            question = case["question"]
        else:
            question = ""
            if "in-context" in case:
                IC_list = case["in-context"]
                for shot in IC_list:
                    shot_q = shot[0]
                    shot_a = shot[1]
                    question += f"user: {shot_q}\n"
                    question += f"assistant: {shot_a}\n"
            question += "user: {}".format(case["question"]) 
        messages.append({'role': "user", 'content': question})
        case["messages"] = messages
        return self.query(messages, temperature, n, top_p)

    def _query_and_append(self, single_query, temperature, n, top_p):
        err_count = 0
        while True:
            if isinstance(single_query, dict):  # case
                retval, messages = self.query_case(single_query, temperature, n, top_p)
            else:  # messages
                retval, messages = self.query(single_query, temperature, n, top_p)
            if retval[0]:
                return
            if "This model's maximum context length is 4097 tokens. However, your messages resulted" in retval[1]:
                messages.append([{'role': "assistant", 'content': f"error:{retval}"}])
                break
            err_count += 1
            if err_count == 10:
                messages.append([{'role': "assistant", 'content': f"error:{retval}"}])
                break
            time.sleep(1)

    def query_case_batch(self, cases, temperature=1, n=1, top_p=1):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_case = {executor.submit(self._query_and_append, case, temperature, n, top_p): case for case in
                              cases}
            for future in concurrent.futures.as_completed(future_to_case):
                future.result()
        return

    def query_n_case(self, n_case, c_reason, temperature=1, top_p=1):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_case = {executor.submit(self._query_and_append, n_case[i], temperature, c_reason[i], top_p): n_case[i]
                              for i in range(len(n_case))}
            for future in concurrent.futures.as_completed(future_to_case):
                future.result()
        return

    def query_messages_batch(self, messages_batch, temperature=1, n=1, top_p=1):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_case = {executor.submit(self._query_and_append, messages, temperature, n, top_p): messages for
                              messages in messages_batch}
            for future in concurrent.futures.as_completed(future_to_case):
                future.result()
        return