#-*- encoding:utf-8 -*-
import torch

import transformers
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from transformers.generation.utils import GenerationConfig
import time
import requests
# from modelscope import Model
# from modelscope.models.nlp.llama2 import Llama2Tokenizer, Llama2Config
import modelscope
import zhipuai
import json

class Aquila2Model:
    def __init__(self, model_path):
        self.tokenizer = modelscope.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # self.device = torch.device("cuda:0")
        self.model = modelscope.AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()
        # self.model.to(self.device)
    def infer(self, prompt):
        tokens = self.tokenizer.encode_plus(prompt)['input_ids']
        tokens = torch.tensor(tokens)[None,]#.cuda()
        stop_tokens = ["###", "[UNK]", "</s>"]
        with torch.no_grad():
            out = self.model.generate(tokens, do_sample=True, max_length=512, eos_token_id=100007, bad_words_ids=[[self.tokenizer.encode(token)[0] for token in stop_tokens]])[0]
            
            out = self.tokenizer.decode(out.cpu().numpy().tolist())
        # from predict import predict
        # out = predict(self.model, prompt, tokenizer=self.tokenizer, max_gen_len=200, top_p=0.9,
        #       seed=123, topk=15, temperature=1.0, sft=True)
        return out

class BaichuanModel:
    def __init__(self, model_path):
        from modelscope.utils.constant import Tasks
        from modelscope.pipelines import pipeline
        self.text_generation_zh  = pipeline(task=Tasks.text_generation, model=model_path, device_map='auto',model_revision='v1.0.7')
        self.text_generation_zh._model_prepare = True
    def infer(self, prompt):
        result_zh = self.text_generation_zh(prompt, min_length=10, max_length=512, num_beams=3,temperature=0.8,do_sample=False, early_stopping=True,top_k=50,top_p=0.8, repetition_penalty=1.2, length_penalty=1.2, no_repeat_ngram_size=6)
        return result_zh['text']
class Baichuan2Model:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
        self.model.generation_config = GenerationConfig.from_pretrained(model_path)
    def infer(self, prompt):
        messages = []
        messages.append({"role": "user", "content": prompt})
        response = self.model.chat(self.tokenizer, messages)
        return response

class ChatglmModel_v1:
    def __init__(self, model_path):
        from modelscope.utils.constant import Tasks
        from modelscope.pipelines import pipeline
        self.pipe = pipeline(task=Tasks.chat, model=model_path, model_revision='v1.0.3')
    def infer(self, prompt):
        inputs = {'text':prompt, 'history': []}
        result = self.pipe(inputs)
        return result['response']
class ChatglmModel:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
        self.model = self.model.eval()

    def infer(self, prompt):
        response, history = self.model.chat(self.tokenizer, prompt, history=[])
        return response
class ChatglmModel_base:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
        self.model = self.model.eval()

    def infer(self, prompt):
        response, history = self.model.chat(self.tokenizer, prompt, history=[])
        import pdb
        pdb.set_trace()
        return response['content']
class QwenModel:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()
        self.model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
    def infer(self, prompt):
        print("chating, prompt is :",prompt)
        response, history = self.model.chat(self.tokenizer, prompt, history=None)
        return response
class QwenModelBase:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()
        self.model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
    def infer(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors='pt')
        inputs = inputs.to(self.model.device)
        pred = self.model.generate(**inputs)
        response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
        return response
class QwenModelBase_ms:
    def __init__(self, model_path):
        self.tokenizer = modelscope.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = modelscope.AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()
        self.model.generation_config = modelscope.GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
    def infer(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors='pt')
        inputs = inputs.to(self.model.device)
        pred = self.model.generate(**inputs)
        response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
        return response
class InternlmModel:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)#.cuda()
        self.model = self.model.eval()
    def infer(self, prompt):
        response, history = self.model.chat(self.tokenizer, prompt, history=[])
        return response


class Llama2Model:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pipeline = transformers.pipeline("text-generation", model=model_path, torch_dtype=torch.float16, device_map="auto")
    def infer(self, prompt):
        response = self.pipeline(prompt,do_sample=True, top_k=10, num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id,max_length=4096)
        response = response[0]['generated_text']
        return response
class Llama2Model_ms:
    def __init__(self, model_path):
        from modelscope import Model
        from modelscope.models.nlp.llama2 import Llama2Tokenizer
        self.tokenizer = Llama2Tokenizer.from_pretrained(model_path)
        self.model = Model.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map='auto')

        
    def infer(self, prompt):
        system = 'you are a helpful assistant!'
        inputs = {'text': prompt, 'system': system, 'max_length': 4096}
        output = self.model.chat(inputs, self.tokenizer)
        return output['response']


class ZiyaModel:
    def __init__(self, model_path):
        self.model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    def infer(self, prompt):
        inputs = '<human>:' + prompt.strip() + '\n<bot>:'
            
        input_ids = self.tokenizer(inputs, return_tensors="pt").input_ids.to("cuda")
        generate_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=1024, 
                    do_sample = True, 
                    top_p = 0.85, 
                    temperature = 1.0, 
                    repetition_penalty=1., 
                    eos_token_id=2, 
                    bos_token_id=1, 
                    pad_token_id=0)
        response = self.tokenizer.batch_decode(generate_ids)[0]
        return response
class QimingModel:
    def __init__(self, model_path):
        self.model_url = model_path
    def infer(self, prompt):
        while True:
            try:
                resp = requests.post(self.model_url, json={'session_id': 1, 'question': prompt})
                resp.raise_for_status()
                break
            except Exception as e:
                print(e)
                time.sleep(5)
        print(resp.json()['text'][0].split('<bot>:')[-1])
        return resp.json()['text'][0].split('<bot>:')[-1]
class TelechatModel:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
                                                 torch_dtype=torch.float16)
        self.model.to("cuda:0")
        self.generate_config = GenerationConfig.from_pretrained(model_path)
        self.model.eval()
    def infer(self, prompt):
        response, history = self.model.chat(tokenizer = self.tokenizer, question=prompt, history=[], generation_config=self.generate_config,
                                 stream=False)
        return response

class ApiModel:
    def __init__(self, model_path):
        self.model_url = model_path
    def infer(self, prompt):
        while True:
            try:
                resp = requests.post(self.model_url, json={'session_id': 1, 'query': prompt})
                resp.raise_for_status()
                break
            except Exception as e:
                print(e)
                time.sleep(2)
        output = resp.json()['output']
        return output
class YiModel:
    def __init__(self, model_path):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    def infer(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        max_length = 256
        outputs = self.model.generate(
            inputs.input_ids.cuda(),
            max_length=max_length,
            eos_token_id=self.tokenizer.eos_token_id 
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt)+1:]
class MistralModel:
    def __init__(self, model_path):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    def infer(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        max_length = 256
        outputs = self.model.generate(
            inputs.input_ids.cuda(),
            max_length=max_length,
            eos_token_id=self.tokenizer.eos_token_id 
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
class BluelmModel:
    def __init__(self, model_path):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:0", torch_dtype=torch.bfloat16, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    def infer(self, prompt):
        self.model = self.model.eval()
        inputs = self.tokenizer("[|Human|]:"+prompt+"[|AI|]:", return_tensors="pt")
        inputs = inputs.to("cuda:0")
        
        outputs = self.model.generate(**inputs, max_new_tokens=64, repetition_penalty=1.1)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
class ZhipuModel:
    def __init__(self):

        zhipuai.api_key = "ff89d746bd272591eb462a50edac19d1.WyvZFQmyQKH8yINs"

    def infer(self, prompt):
        #response, history = self.model.chat(self.tokenizer, prompt, history=[])
        response = zhipuai.model_api.invoke(
                        model="chatglm_turbo",
                        prompt=[
                            {"role": "user", "content": prompt},])
        print(response)
        return str(response['data']['choices']) if response['success'] else ""
class CloseaiModel:
    def __init__(self, model_path):
        
        # OPENAI_API_KEY = "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"
        # OPENAI_API_KEY = "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"
        self.model_name = model_path
        self.OPENAI_API_KEY = "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"
        self.proxy = "127.0.0.1:58591"
        self.ENDPOINT = "https://api.openai.com/v1/chat/completions"             
        self.proxies = {
        'http': f'http://{self.proxy}',
        'https': f'http://{self.proxy}',
        }
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.OPENAI_API_KEY}",
        }
    def infer(self, prompt):
        
        data = {
        "messages": prompt,
        "model":self.model_name,
        "max_tokens": 1000,
        "temperature": 0.5,
        "top_p": 1,
        "n": 1
        }
        response = requests.post(self.ENDPOINT, headers=self.headers, json=data, proxies=self.proxies).json()
        return response

class QwenModel_72b_int4:
    def __init__(self, model_path):

        #AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        #self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()
        #self.model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = modelscope.AutoTokenizer.from_pretrained(model_path, revision='master', trust_remote_code=True)        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, revision='master',
            device_map="auto",
            trust_remote_code=True
        ).eval()
    def infer(self, prompt):
        print("chating, prompt is :",prompt)
        response, history = self.model.chat(self.tokenizer, prompt, history=None)
        return response

class Baichuan2TurboModel:
    def __init__(self, model_path):
        self.model_name = model_path
        self.API_KEY = "sk-07d3731cd1ee1c85ac94da303a26d486"
        self.ENDPOINT = "https://api.baichuan-ai.com/v1/chat/completions" 
      
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.API_KEY}",
        }
    def infer(self, prompt):
        data = {
        "messages": [{"role":'user',"content":prompt}],
        "model":self.model_name,
        }
        response = requests.post(self.ENDPOINT, headers=self.headers, json=data).json()
       
        return response['choices'][0]['message']['content'] if response else ""

class ErnieModel:
    def __init__(self, model_path):
        self.model_name = model_path
        if self.model_name=='ERNIE-Bot':
            apitype='completions'
        elif self.model_name == 'ERNIE-Bot_4_0':
            apitype='completions_pro'
        self.url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/"+apitype+"?access_token=" + self.get_access_token()
    def get_access_token(self):
        """
        使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
        """
            
        # url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id= &client_secret=9KvtQvsTohU3tIDrz9DK2kbtsESHQ3qV"
        url = "https://aip.baidubce.com/oauth/2.0/token?client_id={}&client_secret={}&grant_type=client_credentials".format('hzs8AuBurinroPg6knvlAgKt','9KvtQvsTohU3tIDrz9DK2kbtsESHQ3qV')
        
        payload = json.dumps("")
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        response = requests.request("POST", url, headers=headers, data=payload)
        return response.json().get("access_token")
    def infer(self,prompt):
        payload = json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })
        headers = {
            'Content-Type': 'application/json'
        }
        self.url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token=" + self.get_access_token()
        response = requests.request("POST", self.url, headers=headers, data=payload).json()
        # print(response.text)
        return response["result"]

def get_model(model_type):
    try:
        if model_type == 'aquila2-34b':  # ok
            model_path = "/data/xxxmian/llms/Aquila2-34B"
            model = Aquila2Model(model_path)
        elif model_type == 'aquila2-34b-chat':  # ok
            model_path = "/data/xxxmian/llms/AquilaChat2-34B"
            model = Aquila2Model(model_path)

        elif model_type == 'baichuan-7b':  # ok
            model_path = "/data/xxxmian/llms/baichuan-7B"
            model = BaichuanModel(model_path)
        elif model_type == 'baichuan-13b':  # ok
            model_path = "/data/xxxmian/llms/Baichuan-13B-Base"
            model = BaichuanModel(model_path)
        elif model_type == 'baichuan-13b-chat':  # ok
            model_path = "/data/xxxmian/llms/Baichuan-13B-Chat"
            model = BaichuanModel(model_path)
        elif model_type == 'baichuan2-7b':  # ok
            model_path = "/data/xxxmian/llms/Baichuan2-7B-Base"
            model = Baichuan2Model(model_path)
        elif model_type == 'baichuan2-7b-chat':  # ok
            model_path = "/data/xxxmian/llms/Baichuan2-7B-Chat"
            model = Baichuan2Model(model_path)
        elif model_type == 'baichuan2-13b-base':  # ok
            model_path = "/data/xxxmian/llms/Baichuan2-13B-Base"
            model = Baichuan2Model(model_path)
        elif model_type == 'baichuan2-13b-chat':  # ok
            model_path = "/data/xxxmian/llms/Baichuan2-13B-Chat"
            model = Baichuan2Model(model_path)
        elif model_type =='baichuan2-turbo':
            model = Baichuan2TurboModel('Baichuan2-Turbo')
        elif model_type =='baichuan2-53b':
            model = Baichuan2TurboModel('Baichuan2-53B')

        elif model_type == 'chatglm-6b-chat':  # ok
            model_path = "/data/xxxmian/llms/ChatGLM-6B"
            model = ChatglmModel_v1(model_path)
        elif model_type == 'chatglm2-6b-chat':  # ok
            model_path = "/data/xxxmian/llms/chatglm2-6b"
            model = ChatglmModel(model_path)
        elif model_type == 'chatglm3-6b':  # wrong! unstable response,sometimes dict format, somtimes none str!
            model_path = "/data/xxxmian/llms/chatglm3-6b-base"
            model = ChatglmModel_base(model_path)
        elif model_type == 'chatglm3-6b-chat':  # ok
            model_path = "/data/xxxmian/llms/chatglm3-6b"
            model = ChatglmModel(model_path)
        elif model_type == 'chatglm_turbo':
            model=ZhipuModel()
        
        elif model_type == 'internlm-7b-chat': # havn't deployed
            model_path = "/data/xxxmian/llms/internlm-chat-7b"
            model = InternlmModel(model_path)
        elif model_type == 'internlm-20b':  # ok
            model_path = "/data/xxxmian/llms/internlm-20b"
            model = InternlmModel(model_path)
        elif model_type == 'internlm-20b-chat':  # ok
            model_path = "/data/xxxmian/llms/internlm-chat-20b"
            model = InternlmModel(model_path)

        elif model_type == 'llama2-7b-chat':  # ok
            model_path = "/data/xxxmian/llms/Llama-2-7b-chat-ms"
            model = Llama2Model_ms(model_path)
        elif model_type == 'llama2-7b-base':  # ok
            model_path = "/data/xxxmian/llms/Llama-2-7b-ms"
            model = Llama2Model_ms(model_path)
        elif model_type == 'llama2-13b-chat':  # ok
            model_path = "/data/xxxmian/llms/Llama-2-13b-chat-ms"
            model = Llama2Model_ms(model_path)
        elif model_type == 'llama2-13b':  # ok
            model_path = "/data/xxxmian/llms/Llama-2-13b-ms"
            model = Llama2Model_ms(model_path)
        elif model_type == 'llama2-70b-chat':
            model_path = "/data/xxxmian/llms/Llama-2-70b-chat-ms"
            model = Llama2Model(model_path)
        elif model_type == 'llama2-70b':
            model_path = "/data/xxxmian/llms/Llama-2-70b-ms"
            model = Llama2Model_ms(model_path)
        
        elif model_type == 'qwen-7b-chat':  # ok
            model_path = "/data/xxxmian/llms/Qwen-7B-Chat"
            model = QwenModel(model_path)
        elif model_type == 'qwen-7b':  # ok
            model_path = "/data/xxxmian/llms/Qwen-7B"
            model = QwenModelBase(model_path)
        elif model_type == 'qwen-14b-chat':  # ok
            model_path = "/data/xxxmian/llms/Qwen-14B-Chat"
            model = QwenModel(model_path)
        elif model_type == 'qwen-14b':  # ok
            model_path = "/data/xxxmian/llms/Qwen-14B"
            model = QwenModelBase(model_path)
        elif model_type == 'qwen-72b-chat':  # ok
            model_path = "/data/xxxmian/llms/Qwen-72B-Chat"
            model = QwenModel(model_path)
        elif model_type == 'qwen-72b-chat-int4':
            model_path = '/data/fsj/llms/Qwen-72B-Chat-Int4'
            model = QwenModel(model_path)
        elif model_type == 'Qwen-72B-Int4':
            model_path = "/data/fsj/llms/Qwen-72B-Chat-Int4"
            model = QwenModel_72b_int4(model_path)

        elif model_type == 'yi-6b':  # ok
            model_path = "/data/xxxmian/llms/Yi-6B"
            model = YiModel(model_path)
        elif model_type == 'yi-34b':  # ok
            model_path = "/data/xxxmian/llms/Yi-34B"
            model = YiModel(model_path)
        elif model_type == 'yi-34b-chat':  # ok
            model_path = "/data/xxxmian/llms/Yi-34B-Chat"
            model = YiModel(model_path)

        elif model_type == 'mistral-7b':  # ok
            model_path = "/data/xxxmian/llms/Mistral-7B"
            model = MistralModel(model_path)

        elif model_type == 'ziya2-13b-chat':  # ok
            model_path = "/data/xxxmian/llms/Ziya2-13B-Chat"
            model = ZiyaModel(model_path)
        elif model_type == 'ziya2-13b-base':  # ok
            model_path = "/data/xxxmian/llms/Ziya2-13B-Base"
            model = ZiyaModel(model_path)

        elif model_type == 'bluelm-7b-chat':
            model_path = "/data/xxxmian/llms/BlueLM-7B-Chat"
            model = BluelmModel(model_path)
        elif model_type == 'bluelm-7b-base':
            model_path = "/data/xxxmian/llms/BlueLM-7B-Base"
            model = BluelmModel(model_path)

        elif model_type == 'qiming':  # ok
            api_host='180.110.205.49'
            api_port=9104
            model_path = f'http://{api_host}:{api_port}/v1/start_chat'
            model = QimingModel(model_path)
        elif model_type == 'telechat-7b':
            model_path = '/data/xxxmian/llms/Telechat-7B'
            model = TelechatModel(model_path)

        elif model_type.startswith('gpt'):
            # gpt-3.5-turbo,gpt-4,gpt-4-0314,gpt-4-32k,gpt-4-32k-0314
            model = CloseaiModel(model_type)
        elif model_type == 'ernie':
            model = ErnieModel('ERNIE-Bot')
        elif model_type.startswith('http'):
            model = ApiModel(model_type)
    
        return model
    
    except Exception as e:
        print(e)
        print(model_type, "Model not loaded, something went wrong!")
        return None
