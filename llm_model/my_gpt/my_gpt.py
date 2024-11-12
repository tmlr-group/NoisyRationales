import openai
import requests
import yaml
import concurrent.futures
import time
import tiktoken
import os
import importlib.metadata
# from packaging import version
from ..multiple_key import init_api_key_handling

def version_tuple(v):
    return tuple(map(int, (v.split("."))))

def lower_versions(version1, version2):
    return version_tuple(version1) < version_tuple(version2)

class my_gpt:
    def __init__(self, model='gpt-3.5-turbo-0125', config: dict = None, api="openai", prefix_context=False) -> None:
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
        if api == 'openai':
            openai_version = importlib.metadata.version('openai')
            new_version = '1.0.0'
            
            key = os.getenv('OPENAI_API_KEY')
            if ":" in key:
                key_list = key.split(":")
                self.key = init_api_key_handling(key_list)
            else:
                self.key = key
                
            if "OPENAI_API_BASE" in os.environ:
                self.api_base = os.getenv('OPENAI_API_BASE')
            else:
                self.api_base = "https://api.openai.com"
            
            if lower_versions(openai_version, new_version):
                self.api_version = "old"
                openai.api_base = self.api_base
                openai.api_key = self.key
            else:
                from openai import OpenAI
                self.api_version = "new"
                if "OPENAI_API_BASE" in os.environ:
                    openai.api_base = os.getenv('OPENAI_API_BASE')
                self.client = OpenAI(
                    base_url=self.api_base,
                    api_key=self.key
                )
        else:
            raise "Api not support: {}".format(api)
        pass

    def chat(self, single_chat):
        messages = []
        messages.append({'role': "user", 'content': single_chat})
        retval, error = self.query(messages)
        if retval:
            return messages[-1][0]["content"]
        else:
            return f"error:{error}"

    def get_embedding(self, text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        text = text.replace("\\", "")
        response = openai.Embedding.create(
            input=text,
            model=model
        )
        embedding = response['data'][0]['embedding']
        self.embedding_tokens += response['usage']['prompt_tokens']
        return embedding

    def query(self, messages, temperature=1, n=1, top_p=1):
        if self.api == 'openai':
            try:
                if self.api_version == "old":
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        n=n,
                        top_p=top_p,
                        max_tokens=self.max_response_tokens
                    )
                    self.completion_tokens += response["usage"]["completion_tokens"]
                    self.prompt_tokens += response["usage"]["prompt_tokens"]
                    self.total_tokens += response["usage"]["total_tokens"]
                    completions = []
                    for choice in response['choices']:
                        message = choice['message']
                        completion = dict()
                        completion['role'] = message['role']
                        completion['content'] = message['content']
                        completions.append(completion)
                    messages.append(completions)
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        n=n,
                        top_p=top_p,
                        max_tokens=self.max_response_tokens
                    )
                    self.completion_tokens += response.usage.completion_tokens
                    self.prompt_tokens += response.usage.prompt_tokens
                    self.total_tokens += response.usage.total_tokens
                    completions = []
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
        else:
            payload = {'messages': messages}
            response = requests.post(self.url, json=payload, headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                messages.append([data['choices'][0]["message"]])
                return (True, ''), messages
            else:
                return (False, f'Error: {response}'), messages

    def compute_cost(self):
        if "gpt-3.5" in self.model:
            input_price = 0.015
            output_price = 0.02
            embedding_price = 0.0001
        elif "gpt-4" in self.model:
            input_price = 0.16
            output_price = 0.48
            embedding_price = 0.0001
        else:
            input_price = 0.015
            output_price = 0.02
            embedding_price = 0.0001
        rate = 1
        cost = float(self.prompt_tokens) / 1000 * input_price * rate + \
               float(self.completion_tokens) / 1000 * output_price * rate + \
               float(self.embedding_tokens) / 1000 * embedding_price * rate
        return "input tokens:{}, output tokens:{}, embedding tokens:{}, total tokens:{}, total cost:{:.2f}".format(
            self.prompt_tokens, self.completion_tokens, self.embedding_tokens, self.total_tokens, cost)

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

    def compute_prompt_token_by_case(self, case):
        messages = []
        if "system-prompt" in case:
            system_prompt = case["system-prompt"]
            messages.append({'role': "system", 'content': system_prompt})
        if "in-context" in case:
            IC_list = case["in-context"]
            for shot in IC_list:
                shot_q = shot[0]
                shot_a = shot[1]
                messages.append({'role': "user", 'content': shot_q})
                messages.append({'role': "assistant", 'content': shot_a})
        question = case["question"]
        messages.append({'role': "user", 'content': question})
        return self.num_tokens_from_messages(messages)

    def num_tokens_from_messages(self, messages):
        """Return the number of tokens used by a list of messages."""
        model = self.model
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
        }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif "gpt-3.5-turbo" in model:
            return self.num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
        elif "gpt-4" in model:
            return self.num_tokens_from_messages(messages, model="gpt-4-0613")
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens
