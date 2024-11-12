import os  
import requests  
import openai 
import json
import openai
import asyncio
from typing import Any
import time
data_folder = "/mnt/data/zhenzhang/dir1/ranmask/alpaca/con"
world_size = 2

class chatgpt_denoiser:
    def __init__(self):
        self.prompt = """Fill the masked positions indicated <mask> in the given sentence to make it natural and coherent. Each <mask> should be replace with only one word. The sentence fragment provides enough context to determine what words could fill in the masks. The returned sentence should be of the same length with the given sententence. Give the answer directly. The given sentence is: """
    def set_mask_word(self,mask_word):
        self.prompt = self.prompt.replace("<mask>",mask_word)
    def get_single_response(self,sentence):
        while True:
            try:
                chat_completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": self.prompt+sentence},
                    ],
                    )
                break
            except Exception as e:
                print(e)
                continue
        result = ''
        for choice in chat_completion.choices:
            result += choice.message.content
        return result
    
    def get_batch_response_by_model(self, system_prompt, sentences, model, n_reasoning):
        while True:
            try:
                
                for sentence in sentences:
                    case_batch = []
                    case = dict()
                    if system_prompt is not None:
                        case["system-prompt"] = system_prompt
                    else:
                        case["system-prompt"] = "You are a helpful assistant."
                    case["question"] = self.prompt+sentence
                    model.query_case_batch([case], n=1)
                    case_batch.append(case)
                break
            except Exception as e:
                print(e)
                continue

        cases_batch = [case_batch[i:i + n_reasoning] for i in range(0, len(case_batch), n_reasoning)]
        
        new_sentences_list = []
        for cases in cases_batch:
            new_sentences = []
            for case in cases:
                response = case["messages"][-1][0]
                result = response["content"]
                new_sentences.append(result)
            new_sentences_list.append(new_sentences)
        return new_sentences_list, cases_batch
    
    def get_batch_response(self, message_list,batch_size=5):
        response_list = []
        start = 0
        while True:
            if start >= len(message_list):
                break
            message_list_batch = message_list[start:start+batch_size]
            while True:
                try:
                    predictions = asyncio.run(
                        self.dispatch_openai_requests(
                            messages_list=message_list_batch
                        )
                    )
                    response_list.extend([x['choices'][0]['message']['content'] for x in predictions])
                    break
                except Exception as e:
                    print(e)
                    time.sleep(5)
                    continue
            start += batch_size
                
        return response_list

    async def dispatch_openai_requests(
        self,
        messages_list: list[str]
    ) -> list[str]:
        """Dispatches requests to OpenAI API asynchronously.
        
        Args:
            messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        Returns:
            List of responses from OpenAI API.
        """
        async_responses = [
            openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": self.prompt+x},
                    ],
            )
            for x in messages_list
        ]
        return await asyncio.gather(*async_responses)

chatgpt_cli = chatgpt_denoiser()


def denoise_instance(instances, args=None, model=None):
    chatgpt_cli.set_mask_word(args.mask_word)
    
    
    if args.denoise_method == 'None':
        for instance in instances:
            print(instance.text_a)
        return
    
    elif args.denoise_method == 'chatgpt_single_by_model': # we add it for baseline test
        for instance in instances:
            text_a = instance.text_a
            response = chatgpt_cli.get_single_response_by_model(text_a, model)
            print(text_a)
            print(response)
            return response
    elif args.denoise_method == 'chatgpt_single':
        for instance in instances:
            text_a = instance.text_a
            response = chatgpt_cli.get_single_response(text_a)
            print(text_a)
            print(response)
            instance.text_a = response

            text_b = instance.text_b
            if text_b is not None:
                instance.text_b = chatgpt_cli.get_single_response(text_b)

    elif args.denoise_method == 'chatgpt_batch':
        text_a_list = []
        text_b_list = []
        for instance in instances:
            text_a_list.append(instance.text_a)
            text_b_list.append(instance.text_b)
        text_a_response_list = chatgpt_cli.get_batch_response(text_a_list)
        if text_b_list[0] is not None:
            text_b_response_list = chatgpt_cli.get_batch_response(text_b_list)

        for text_a_response, instance in zip(text_a_response_list, instances):
            instance.text_a = text_a_response
        if text_b_list[0] is not None:
            for text_b_response, instance in zip(text_b_response_list, instances):
                instance.text_b = text_b_response
        

    elif args.denoise_method == 'alpaca':
        
        alpaca_instruction = f"Replace each \"{args.mask_word}\" in the provided sentence with a suitable word to make it natural and coherent. Only one word should be used to replace each \"{args.mask_word}\". The returned sentence should be of the same length as the given sentence. Provide the answer directly."
        text_a_list = []
        text_b_list = []
        for instance in instances:
            text_a_list.append(instance.text_a)
            text_b_list.append(instance.text_b)
        # ======a======
        # write data
        with open(os.path.join(data_folder,'data.jsonl'), 'w') as f:
            for item in text_a_list:
                f.write(json.dumps({"input":item,"instruction":alpaca_instruction}) + '\n')
        # request for return
        for i in range(world_size):
            with open(os.path.join(data_folder,f'request_{i}'), 'w'):
                pass
        # wait for processing
        for i in range(world_size):
            while True:
                if os.path.exists(os.path.join(data_folder,f'finished_{i}')):
                    os.remove(os.path.join(data_folder,f'finished_{i}'))
                    break
        # read denoised data
    
        with open(os.path.join(data_folder,'return.jsonl'), 'r') as f:
            for line, instance in zip(f, instances):
                output = json.loads(line)["output"]
                print(instance.text_a)
                print(output)
                instance.text_a = output
        
    else:
        raise RuntimeError

        

if __name__ == "__main__":
    import os  
    import requests  
    import openai 



    # 给requests库设置全局代理  
    PROXY = '127.0.0.1:7890'  
    os.environ['HTTP_PROXY'] = os.environ['http_proxy'] = PROXY  
    os.environ['HTTPS_PROXY'] = os.environ['https_proxy'] = PROXY  
    os.environ['NO_PROXY'] = os.environ['no_proxy'] = '127.0.0.1,localhost,.local' 
 


    openai.api_key = ""
    chat_completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the best star wars movie?"},
    ],
    )
    print(chat_completion)
    result = ''
    for choice in chat_completion.choices:
        result += choice.message.content

    print(result)
