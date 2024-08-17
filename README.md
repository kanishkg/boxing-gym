# BoxingGym
-------
## Overview
BoxingGym is a benchmarking framework designed to evaluate the capabilities of language-based agents in experimental design and model discovery. The framework consists of several simulated environments where agents can perform experiments, propose models, and refine them based on collected data.

## Getting Started with Installation
To install BoxingGym, clone the repository and install the dependencies:

```bash
git clone https://github.com/kanishkg/boxing-gym.git
cd boxing_gym
pip install -e .
```

You should now be able to import the BoxingGym package in your Python environment.
```python
import boxing_gym
```

## Interacting with an Environment
Environments in BoxingGym simulate models of the world across different domains. You can interact with an environment using predefined methods to conduct experiments, collect data, and test hypotheses.

Example code to interact with an environment(see `run_experiment.py` for a complete example):
    
```python
from boxing_gym.envs import SomeEnvironment

env = SomeEnvironment()
env.reset()
action = env.sample_random_input()
observation = env.step(action)
```


## Interacting with an Agent
Agents in BoxingGym can perform experimental design, run simulations, and propose models. The framework includes pre-built agents like Box's Apprentice and the LLM Agent.

Example pseudo-code to interact with an agent (see `run_experiment.py` for a complete example):
```python
from boxing_gym.agents import LLMAgent
from boxing_gym.envs.some_env import SomeEnv, SomeGoal

env = SomeEnv()
goal = SomeGoal(env, include_prior=True)
agent = LLMAgent()
agent.set_goal(goal_description)
observation = env.reset()
action = agent.act(observation)
next_observation = env.step(action)
```

## Creating a New Environment 
Environment in BoxingGym define the simulated world model and the interactions an agent can have with it. To create a new environment, subclass the Environment class and implement the necessary methods:

```python

class CustomEnvironment:
    def __init__(self, param1, param2, param3):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3
        self.reset()
        self.env_name = "custom_environment"

    def reset(self):
        # Initialize or reset the environment to a starting state
        # sample params for the world model
        self.model_params = sample...
        self.data = []

    def get_system_message(self, include_prior=True, goal=None):
        # Add a system message to the environment
    
    def step(self, action):
        # Process the input_value to produce an output
        result = # pass input through model to get the next observation
        self.data.append(result)  # Store the result if necessary
        return result
    
    def validate_input(self, action):
        # validate the input action
        # return error message if invalid so that the agent can correct it
        return action
    
    def sample_random_input(self):
        # Sample a random valid action for the environment

    def run_experiment(self, action):
        validated_input = self.validate_input(action)
        if isinstance(validated_input, str):
            return validated_input, False
        result = self.step(validated_input)
        return result, True
```

## Creating a New Goal
Goals in BoxingGym define the objectives for an agent within an environment. To create a new goal, subclass the Goal class and implement the necessary methods:

```python
from boxing_gym.goals import Goal

class MyCustomGoal(Goal):
    def __init__(self, env):
        super().__init__(env)  # Initialize with environment
        self.eval_points = []  # Store evaluation points
        self.eval_pointer = 0  # Pointer for evaluation
        # other initialization code
    
    def get_system_message(self, include_prior):
        # Generate goal description based on prior knowledge
        goal_description = "Your goal is to "# goal description
        return self.env.get_system_message(include_prior, goal_description)
    
    def get_goal_eval_question(self, include_prior):
        # Generate or retrieve a question for evaluation
        if self.eval_pointer >= len(self.eval_points):
            x = ...  # Generate new input
            y = self.env.step(x) # Get output
            self.eval_points.append((time, infected_num)) # Store evaluation point
        else:
            x, y = self.eval_points[self.eval_pointer] # Retrieve evaluation point
        
        self.eval_pointer += 1
        question = f"What y for {x}"
        return question, y
    
    def evaluate_predictions(self, predictions, measurements):
        # Evaluate the predictions made by the agent
        return mse, std
    
    def expected_information_gain(self, query_point, num_outer_samples=1000, num_inner_samples=10):
        # Calculate EIG for a new query point using a Bayesian approach
        # See existing environments for more details on how to implement this method
```

## Creating a New Agent
An agent in BoxingGym is an entity that interacts with environments to perform experiments, propose models, and refine them based on collected data. To create a new agent, subclass the Agent class and implement the necessary methods:

```python
class MyCustomAgent:
    def __init__(self, param1, param2, param3):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3
        self.reset()
        self.agent_name = "custom_agent"
    
    def generate_actions(self, observation):
        # Generate an action based on the past result 
        return action
    
    def generate_predictions(self, query):
        # Generate a prediction based on the current query 
        return prediction
```

## Running Experiments
To run an experiment in BoxingGym, you can use the `run_experiment.py` script. The script allows you to specify the environment, agent, and goal for the experiment. You can also configure the experiment using Hydra configuration files.

Example configuration file for running an experiment:

example.yaml
```yaml
  - _self_
  - llms: openai # LLM config to use
  - exp: oed # Experiment type discovery or oed (Optimal Experiment Design)
  - envs: custom_env # Environments to use
include_prior: true # Include prior knowledge in the goal description
```

openai.yaml
```yaml
  - model_name: "gpt-4o"
  - temperature: 0.0
  - max_tokens: 512
```

oed.yaml
```yaml
  - num_experiments: [0, 1, 3, 5, 7, 10]
  - experiment_type: "oed"
```

custom_env.yaml
```yaml
  - env_name: "custom_environment"
  - goal_name: "custom_goal"
  - num_evals: 10
  - env_params:
    - param1: 1
    - param2: 2
    - param3: 3
```

## Directory Structure
- Environments: `src/boxing_gym/envs/`
- Agents: `src/boxing_gym/agents/`
    - LLM Agent: `src/boxing_gym/agents/agent.py`
    - Box's Apprentice: Model Criticism LLM in `src/boxing_gym/agents/llm.py`
- Configurations for experiments: We use hydra for configs, `conf/`
- Running experiments: `run_experiment.py`
- Scripts for running experiments: `scripts/`
- Analysis of experiments: `analysis/`

## Existing Agents
### Box's Apprentice
Box's Apprentice is an agent that combines language model capabilities with statistical modeling to perform experimental design and model discovery. It can build explicit generative models to improve predictions.
Box's Apprentice: Model Criticism LLM in `src/boxing_gym/agents/llm.py`
### LLM Agent
The LLM Agent is a language-based agent that interacts with environments purely through natural language. It is capable of proposing and testing scientific theories but relies on its language processing abilities.
LLM Agent: `src/boxing_gym/agents/agent.py`


## Contributing
We welcome contributions to BoxingGym, especially for new environments and agents. If you want to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a new Pull Request.
7. For major changes, please open an issue first to discuss what you would like to change.