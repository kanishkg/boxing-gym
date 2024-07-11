from crfm import crfmChatLLM
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import json
import anthropic
import time
import re
import openai

class LMExperimenter:
    def __init__(self, model_name, temperature=0.0, max_tokens=256):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system = None
        self.messages = []
        self.all_messages = []
        if "openai" in model_name:
            self.llm = crfmChatLLM(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        elif "gpt-4o" in model_name:
            self.llm = openai.OpenAI()
        elif "claude" in model_name:
            self.llm = anthropic.Anthropic()

    def set_system_message(self, message):
        self.all_messages.append(f"role:system, messaage:{message}")
        if "openai" in self.model_name:
            self.system = message
            self.messages.append(SystemMessage(content=message))
        elif "gpt-4o" in self.model_name:
            self.system = message
            self.messages.append({"role": "system", "content": [{"type": "text", "text": message}]})
        elif "claude" in self.model_name:
            self.system = message
        
    def add_message(self, message, role='user'):
        # if role == "assistant":
        #     # remove everything in thought tags
        #     message = re.sub(r'<thought>.*?</thought>', '', message)

        self.all_messages.append(f"role:{role}, messaage:{message}")
        if "openai" in self.model_name:
            if role == "user":
                self.messages.append(HumanMessage(content=message))
            else:
                self.messages.append(AIMessage(content=message))
        elif "gpt-4o" in self.model_name:
            self.messages.append(
                {
                    "role": role,
                    "content": [
                        {
                            "type": "text",
                            "text": message
                        }
                    ]
                })
        elif "claude" in self.model_name:
            self.messages.append(
                {
                    "role": role,
                    "content": [
                        {
                            "type": "text",
                            "text": message 
                        }
                    ]
                })

    def prompt_llm(self, request_prompt):
        self.add_message(request_prompt)
        if "openai" in self.model_name:
            full_response = self.llm.generate([self.messages], stop=["Q:"]).generations[0][0].text
        elif "gpt-4o" in self.model_name:
            full_response = self.llm.chat.completions.create(model=self.model_name, messages=self.messages, max_tokens=self.max_tokens, temperature=self.temperature)#.content[0].text
            full_response = full_response.choices[0].message.content
        elif "claude" in self.model_name:
            full_response = self.llm.messages.create(model=self.model_name, system=self.system, messages=self.messages, max_tokens=self.max_tokens, temperature=self.temperature).content[0].text
            # time.sleep(1)
        self.add_message(full_response, 'assistant')
        return full_response

    def parse_response(self, response, is_observation):
        if is_observation:
            pattern = r'<observe>(.*?)</observe>'   
        else:
            pattern = r'<answer>(.*?)</answer>'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return None

        
    def prompt_llm_and_parse(self, request_prompt, is_observation, max_tries=4):
        used_retries = 0
        for i in range(max_tries):
            full_response = self.prompt_llm(request_prompt)
            # print(full_response)
            response = self.parse_response(full_response, is_observation)
            # print(f"parsed response: {response}")
            if response is not None:
                # check if response has numbers
                numbers = re.findall(r'[0-9]+', response)
                if len(numbers) == 0:
                    response = None
            if response == None or "done" in response:
                # TODO: make this better
                if is_observation:
                    request_prompt = "Please stick to the specified format and respond using <observe> tags. Continue making observations even if you think you have an accurate estimate. Your previous response was not valid."
                else:
                    request_prompt = "Please stick to the specified format and respond using <answer> tags. Make assumptions and provide your best guess. Your previous response was not valid."
                used_retries += 1
            else:
                break
        if used_retries == max_tries:
            ValueError("Failed to get valid response")
        return response, used_retries
    
    def generate_predictions(self, request_prompt):
        request_prompt += "\nAnswer in the following format:\n<answer>your answer</answer>."
        prediction, used_retries = self.prompt_llm_and_parse(request_prompt, False)
        self.messages = self.messages[:-2 * (used_retries+1)]  # Remove the last 2 messages
        return prediction
    
    def generate_actions(self, experiment_results=None):
        if experiment_results is None:
            follow_up_prompt = f"Think about where to observe next. Articulate your strategy for choosing measurements in <thought>.\nProvide a new measurement point in the format:\n<thought>your thought</thought>\n<observe> your observation</observe>\nMake an observation now."
        else:
            follow_up_prompt = f"Result: {experiment_results}\nThink about where to observe next. Articulate your strategy for choosing measurements in <thought>.\nProvide a new measurement point in the format:\nThought: <thought>\n<observe> your observation (remember the type of inputs accepted)</observe>"
        observe, used_retries = self.prompt_llm_and_parse(follow_up_prompt, True)
        return observe 
    
    def print_log(self):
        for entry in self.messages:
            print("Message Type:", type(entry).__name__)
            print("Content:", entry.content)
            print("------")