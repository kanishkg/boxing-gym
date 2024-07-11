import pymc as pm
import numpy as np
from crfm import crfmChatLLM
from langchain.schema import HumanMessage, SystemMessage
import re
import anthropic

class SocialCognitionEnv:
    def __init__(self, model_name="openai/gpt-4-1106-preview"):
        self.model_name = model_name
        if "openai" in model_name:
            self.llm = crfmChatLLM(model_name=model_name, temperature=0.7, max_tokens=200)
        elif "claude" in model_name:
            self.llm = anthropic.Anthropic()
        self.system = """You are an observer trying to understand the beliefs and goals of an agent named Sally based on her actions and the outcomes of those actions.
Sally is interacting with a vending machine that has two buttons (a and b) and dispenses either a bagel or a cookie.
You know the true probabilities of the vending machine dispensing a bagel or a cookie for each button, but you don't know if Sally knows these probabilities.
Your task is to infer Sally's beliefs about the vending machine probabilities and her goal (whether she wants a bagel or a cookie) based on her observed actions and the outcomes.
You will be provided with Sally's observed action and the outcome, and you should respond with your inferences about her beliefs and goal.
"""
        self.model = self._build_model()
        self.reset()

    def _build_model(self):
        with pm.Model() as model:
            sally_belief = pm.Uniform('sally_belief', lower=0, upper=1, shape=2)
            goal = pm.Categorical('goal', p=[0.5, 0.5])
        return model

    def reset(self):
        with self.model:
            self.sally_belief = pm.sample_prior_predictive(samples=1)['sally_belief'][0]
            self.goal = pm.sample_prior_predictive(samples=1)['goal'][0]
        self.buttons_to_bagel_probs = {'a': 0.9, 'b': 0.1}

    def true_vending_machine(self, action):
        return np.random.choice(['bagel', 'cookie'], p=[self.buttons_to_bagel_probs[action], 1 - self.buttons_to_bagel_probs[action]])

    def sally_vending_machine(self, action):
        return np.random.choice(['bagel', 'cookie'], p=[self.sally_belief[0] if action == 'a' else self.sally_belief[1], 1 - (self.sally_belief[0] if action == 'a' else self.sally_belief[1])])

    def choose_action(self):
        if self.goal == 0:
            goal = 'bagel'
        else:
            goal = 'cookie'
        if self.sally_vending_machine('a') == goal:
            return 'a'
        else:
            return 'b'

    def step(self):
        action = self.choose_action()
        outcome = self.true_vending_machine(action)

        query = f"""Sally has taken the action '{action}' and observed the outcome '{outcome}'.
Based on this, what do you infer about:
1. Sally's beliefs about the probabilities of the vending machine dispensing a bagel or a cookie for each button (a and b)?
2. Sally's goal (whether she wanted a bagel or a cookie)?
Provide your response in the following format:
Sally's beliefs:
Button a bagel probability: {{button_a_bagel_prob}}
Button b bagel probability: {{button_b_bagel_prob}}
Sally's goal: {{goal}}
"""
        if "openai" in self.model_name:
            user_message = HumanMessage(content=query)
            system_message = SystemMessage(content=self.system)
            response = self.llm.generate([[system_message, user_message]], stop=["Q:"]).generations[0][0].text
        else:
            response = self.llm.generate([self.system, query]).get("message", "")

        return action, outcome, response

if __name__ == "__main__":
    env = SocialCognitionEnv()
    action, outcome, result = env.step()
    print(f"Action: {action}")
    print(f"Outcome: {outcome}")
    print(f"Inference: {result}")