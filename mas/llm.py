# import os
# import requests
# import re
# import json

# from typing import (
#     Protocol, 
#     Literal,  
#     Optional, 
#     List,
# )
# from openai import OpenAI
# from dataclasses import dataclass
# from abc import ABC, abstractmethod

# from .utils import load_config


# # model configs
# CONFIG: dict = load_config("configs/configs.yaml")
# LLM_CONFIG: dict = CONFIG.get("llm_config", {})
# MAX_TOKEN = LLM_CONFIG.get("max_token", 512)  
# TEMPERATURE = LLM_CONFIG.get("temperature", 0.1)
# NUM_COMPS = LLM_CONFIG.get("num_comps", 1)

# URL = os.environ["OPENAI_API_BASE"]
# KEY = os.environ["OPENAI_API_KEY"]
# print('# api url: ', URL)
# print('# api key: ', KEY)


# completion_tokens, prompt_tokens = 0, 0

# @dataclass(frozen=True)
# class Message:
#     role: Literal["system", "user", "assistant"]
#     content: str

# class LLMCallable(Protocol):

#     def __call__(
#         self,
#         messages: List[Message],
#         temperature: float = TEMPERATURE,
#         max_tokens: int = MAX_TOKEN,
#         stop_strs: Optional[List[str]] = None,
#         num_comps: int = NUM_COMPS
#     ) -> str:
#         pass

# class LLM(ABC):
    
#     def __init__(self, model_name: str):
#         self.model_name: str = model_name

#     @abstractmethod
#     def __call__(
#         self,
#         messages: List[Message],
#         temperature: float = TEMPERATURE,
#         max_tokens: int = MAX_TOKEN,
#         stop_strs: Optional[List[str]] = None,
#         num_comps: int = NUM_COMPS
#     ) -> str:
#         pass

# class GPTChat(LLM):

#     def __init__(self, model_name: str):
#         super().__init__(model_name=model_name)
#         self.client = OpenAI(
#             base_url=URL,
#             api_key=KEY
#         )

#     def __call__(
#         self,
#         messages: List[Message],
#         temperature: float = TEMPERATURE,
#         max_tokens: int = MAX_TOKEN,
#         stop_strs: Optional[List[str]] = None,
#         num_comps: int = NUM_COMPS
#     ) -> str:
#         import time
#         global prompt_tokens, completion_tokens
        
#         messages = [{"role": msg.role, "content": msg.content} for msg in messages]

#         max_retries = 5  
#         wait_time = 1 

#         for attempt in range(max_retries):
#             try:
#                 response = self.client.chat.completions.create(
#                     model=self.model_name,  
#                     messages=messages,
#                     max_tokens=max_tokens,
#                     temperature=temperature,
#                     n=num_comps,
#                     stop=stop_strs
#                 )

#                 answer = response.choices[0].message.content
#                 prompt_tokens += response.usage.prompt_tokens
#                 completion_tokens += response.usage.completion_tokens
                
#                 if answer is None:
#                     print("Error: LLM returned None")
#                     continue
#                 return answer  

#             except Exception as e:
#                 error_message = str(e)
#                 if "rate limit" in error_message.lower() or "429" in error_message:
#                     time.sleep(wait_time)
#                 else:
#                     print(f"Error during API call: {error_message}")
#                     break 

#         return "" 

# class OllamaChat(LLM):
#     def __init__(self, model_name: str, host: str = "http://localhost:6162"):
#         super().__init__(model_name)
#         self.base_url = f"{host}/api/chat"

#     def __call__(
#         self,
#         messages: List[Message],
#         temperature: float = 0.7,
#         max_tokens: int = 512,
#         stop_strs: Optional[List[str]] = None,
#         num_comps: int = 1
#     ) -> str:
#         headers = {"Content-Type": "application/json"}
#         data = {
#             "model": self.model_name,
#             "messages": [{"role": m.role, "content": m.content} for m in messages],
#             "options": {"temperature": temperature},
#             "stream": False,
#         }

#         max_retries = 5
#         for _ in range(max_retries):
#             try:
#                 resp = requests.post(self.base_url, headers=headers, data=json.dumps(data), timeout=30)
#                 if resp.status_code != 200:
#                     print(f"Error {resp.status_code}: {resp.text}")
#                     return ""
#                 result = resp.json()
#                 content = result.get("message", {}).get("content", "")
#                 content_remove_think = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
#                 return content_remove_think
#             except Exception as e:
#                 print("Request failed:", e)
#         return ""

# def get_price():
#     global completion_tokens, prompt_tokens
#     return completion_tokens, prompt_tokens, completion_tokens*60/1000000+prompt_tokens*30/1000000

import os

from typing import (
    Protocol, 
    Literal,  
    Optional, 
    List,
)
from openai import OpenAI
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .utils import load_config


# model configs
CONFIG: dict = load_config("configs/configs.yaml")
LLM_CONFIG: dict = CONFIG.get("llm_config", {})
MAX_TOKEN = LLM_CONFIG.get("max_token", 512)  
TEMPERATURE = LLM_CONFIG.get("temperature", 0.1)
NUM_COMPS = LLM_CONFIG.get("num_comps", 1)
USE_OLLAMA = LLM_CONFIG.get("use_ollama", True)

URL = os.environ["OPENAI_API_BASE"]
KEY = os.environ["OPENAI_API_KEY"]
print('# api url: ', URL)
print('# api key: ', KEY)


completion_tokens, prompt_tokens = 0, 0

@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str

class LLMCallable(Protocol):

    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:
        pass

class LLM(ABC):
    
    def __init__(self, model_name: str):
        self.model_name: str = model_name

    @abstractmethod
    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:
        pass

# class GPTChat(LLM):

#     def __init__(self, model_name: str):
#         super().__init__(model_name=model_name)
#         self._model_name = model_name
#         print(f"# init Using LLM model: {self._model_name}")
#         self.client = OpenAI(
#             base_url=URL,
#             api_key=KEY
#         )

#     def __call__(
#         self,
#         messages: List[Message],
#         temperature: float = TEMPERATURE,
#         max_tokens: int = MAX_TOKEN,
#         stop_strs: Optional[List[str]] = None,
#         num_comps: int = NUM_COMPS
#     ) -> str:
#         import time
#         global prompt_tokens, completion_tokens
        
#         messages = [{"role": msg.role, "content": msg.content} for msg in messages]

#         max_retries = 5  
#         wait_time = 1 

#         extra_body = {}

#         if self._model_name == "Qwen3-8B":
#             extra_body = {
#                 "chat_template_kwargs": {
#                     "enable_thinking": False
#                 }
#             }

#         for attempt in range(max_retries):
#             try:
#                 if self._model_name == "gpt-5-mini":
#                     response = self.client.chat.completions.create(
#                     model=self.model_name,  
#                     messages=messages,
#                     max_tokens=max_tokens,
#                     temperature=temperature,
#                     n=num_comps,
#                     # stop=stop_strs,
#                     extra_body=extra_body
#                 )
#                 else:
#                     response = self.client.chat.completions.create(
#                         model=self.model_name,  
#                         messages=messages,
#                         max_tokens=max_tokens,
#                         temperature=temperature,
#                         n=num_comps,
#                         stop=stop_strs,
#                         extra_body=extra_body
#                     )

#                 answer = response.choices[0].message.content
#                 prompt_tokens += response.usage.prompt_tokens
#                 completion_tokens += response.usage.completion_tokens
#                 if self._model_name == "gpt-5-mini":
#                     answer = answer.strip().split("\n")[0].strip()
                
#                 if answer is None:
#                     print("Error: LLM returned None")
#                     continue
#                 return answer  

#             except Exception as e:
#                 error_message = str(e)
#                 if "rate limit" in error_message.lower() or "429" in error_message:
#                     time.sleep(wait_time)
#                 else:
#                     print(f"Error during API call: {error_message}")
#                     break 

#         return "" 


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_qwen_model = None
_qwen_tokenizer = None
QWEN_MODEL_PATH = "/data/models/qwen3-8b/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"

def load_qwen_once():
    global _qwen_tokenizer, _qwen_model
    if _qwen_model is None:
        print("# Loading local Qwen3-8B model ...")
        _qwen_tokenizer = AutoTokenizer.from_pretrained(
            QWEN_MODEL_PATH,
            trust_remote_code=True
        )
        _qwen_model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_PATH,
            dtype=torch.float16,   # 不量化
            device_map="auto",
            trust_remote_code=True
        )
        _qwen_model.eval()
        print("# Qwen3-8B loaded.")

class GPTChat(LLM):

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        self._model_name = model_name
        print(f"# init Using LLM model: {self._model_name}")

        # 只有 API 模型才初始化 client
        if USE_OLLAMA:
            self.client = OpenAI(
                base_url=URL,
                api_key=KEY
            )
        else:
            # 本地模型：提前确保已加载
            load_qwen_once()

    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:

        import time
        global prompt_tokens, completion_tokens

        # ======== 不使用ollama本地推理路径 ========
        if not USE_OLLAMA:
            load_qwen_once()

            # Message -> Qwen chat format
            chat_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]

            # Qwen 官方推荐方式
            prompt = _qwen_tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )

            inputs = _qwen_tokenizer(
                prompt,
                return_tensors="pt"
            ).to(_qwen_model.device)

            with torch.no_grad():
                outputs = _qwen_model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    do_sample=(temperature > 0),
                    temperature=temperature if temperature > 0 else None
                )

            output_text = _qwen_tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )

            # 统计 token（近似）
            prompt_tokens += inputs["input_ids"].numel()
            completion_tokens += outputs[0].numel() - inputs["input_ids"].numel()

            return output_text.strip()

        # ======== 使用ollama：仍然走 OpenAI API ========
        messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        max_retries = 5
        wait_time = 1

        extra_body = {}

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=num_comps,
                    stop=stop_strs,
                    extra_body=extra_body
                )

                answer = response.choices[0].message.content
                prompt_tokens += response.usage.prompt_tokens
                completion_tokens += response.usage.completion_tokens

                if answer is None:
                    print("Error: LLM returned None")
                    continue

                return answer.strip()

            except Exception as e:
                error_message = str(e)
                if "rate limit" in error_message.lower() or "429" in error_message:
                    time.sleep(wait_time)
                else:
                    print(f"Error during API call: {error_message}")
                    break

        return ""


def get_price():
    global completion_tokens, prompt_tokens
    return completion_tokens, prompt_tokens, completion_tokens*60/1000000+prompt_tokens*30/1000000