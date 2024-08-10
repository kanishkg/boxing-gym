from typing import (
    Any,
    Dict,
    List, 
)

from abc import ABC, abstractmethod
 # The cost per token for each model input.
MODEL_COST_PER_INPUT = {
    'gpt-4': 3e-05,
    'gpt-4o': 3e-05,
    "gpt-4o-2024-05-13": 3e-05,
    "gpt-4-1106-vision-preview": 0.03/1000,
    'gpt-3.5-turbo-0125': 3e-05,
    "gpt-4-turbo-2024-04-09": 0.01/1000,
}
# The cost per token for each model output.
MODEL_COST_PER_OUTPUT = {
    'gpt-4': 6e-05,
    "gpt-4-1106-vision-preview": 3e-05,
    'gpt-3.5-turbo-0125': 3e-05,
    "gpt-4-turbo-2024-04-09": 0.03/1000,
    "gpt-4o-2024-05-13": 3e-05,
}

class BaseAgent(ABC):
    """
    Base agent class.
    """
    def __init__(
        self, 
        llm: Any,
        model_id: str, 
    ) -> None:
        """Initializes a chat model used for code or feedback models.

        Args:
            llm: The LLM Chat model. Currently either a CRFM or Azure Chat Model.
            model_id: The unique identifier of the model
        """
        self.llm = llm
        self.model_id = model_id
    
    def calc_cost(
        self, 
        response
    ) -> float:
        """
        Calculates the cost of a response from the openai API. Taken from https://github.com/princeton-nlp/SWE-bench/blob/main/inference/run_api.py

        Args:
        response (openai.ChatCompletion): The response from the API.

        Returns:
        float: The cost of the response.
        """
        model_name = response.model
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = (
            MODEL_COST_PER_INPUT[model_name] * input_tokens
            + MODEL_COST_PER_OUTPUT[model_name] * output_tokens
        )
        return cost
        
    @abstractmethod
    def get_prompt(self) -> List[Dict[str, str]]:
        """
        Get the prompt fed into the model. 
        """
        raise NotImplementedError

    @abstractmethod
    def get_response(self) -> str:
        """
        Get the response from the model. 
        """
        raise NotImplementedError

    @abstractmethod 
    def run(self) -> Dict[str, Any]:
        """
        Run the agent.
        """
        raise NotImplementedError