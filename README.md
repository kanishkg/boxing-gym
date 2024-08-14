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
Environments in BoxingGym simulate scientific models across different domains. You can interact with an environment using predefined methods to conduct experiments, collect data, and test hypotheses.

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

## Creating a New Goal
Goals in BoxingGym define the objectives for an agent within an environment. To create a new goal, subclass the Goal class and implement the necessary methods:

```python
from boxing_gym.goals import Goal

class MyCustomGoal(Goal):
    def __init__(self, env):
        super().__init__(env)  # Initialize with environment
        self.eval_points = []  # Store evaluation points
        self.eval_pointer = 0  # Pointer for evaluation
        self.update_param_cache = {}  # Cache for parameter updates
    
    def get_system_message(self, include_prior):
        # Generate goal description based on prior knowledge
        goal_description = "Predict output based on observations." if include_prior else "Predict environment's integer output."
        return self.env.get_system_message(include_prior, goal_description)
    
    def get_goal_eval_question(self, include_prior):
        # Generate or retrieve a question for evaluation
        if self.eval_pointer >= len(self.eval_points):
            x = ...  # Generate new input
            y = self.env.step(x) # Get output
            self.eval_points.append((time, infected_num)) # Store evaluation point
        else:
            time, infected_num = self.eval_points[self.eval_pointer] # Retrieve evaluation point
        
        self.eval_pointer += 1
        question = f"What is the number of infected individuals at time {time}?" if include_prior else f"What is the output of the environment at input {time}?"
        return question + " Respond with a positive integer.", infected_num
    
    def evaluate_predictions(self, predictions, measurements):
        # Evaluate predictions using Mean Squared Error (MSE)
        parsed_predictions = [int(float(pred)) for pred in predictions]
        mse = np.mean((np.array(parsed_predictions) - np.array(measurements)) ** 2)
        std = np.std((np.array(parsed_predictions) - np.array(measurements)) ** 2)
        return mse, std
    
    def expected_information_gain(self, query_point, num_outer_samples=1000, num_inner_samples=10):
        # Calculate EIG for a new query point using a Bayesian approach
        xi = query_point
        existing_data = self.env.observation_data
        
        if tuple(existing_data) not in self.update_param_cache:
            with pm.Model() as model:
                theta = pm.TruncatedNormal('theta', mu=self.env.mu, sigma=self.env.sigma, lower=self.env.lower_bound, upper=self.env.upper_bound)
                pm.Binomial('infected_num', n=self.env.N, p=1 - pm.math.exp(-theta * xi), observed=[d[1] for d in existing_data])
                trace = pm.sample(num_outer_samples * num_inner_samples + num_outer_samples, tune=1000, return_inferencedata=False)
            
            thetas = trace['theta']
            updated_theta_samples = thetas[:num_outer_samples]
            self.update_param_cache[tuple(existing_data)] = (updated_theta_samples, thetas[num_outer_samples:])
        else:
            updated_theta_samples, thetas = self.update_param_cache[tuple(existing_data)]
        
        log_likelihoods = []
        for n, outer_theta in enumerate(updated_theta_samples):
            eta = 1 - np.exp(-outer_theta * xi)
            sampled_infected_num = np.random.binomial(self.env.N, eta)
            log_liks = [np.log(eta) if i < sampled_infected_num else np.log(1 - eta) for i in range(self.env.N)]
            log_likelihood = np.mean(log_liks)
            
            marginal_log_likelihoods = []
            for inner_theta in thetas[n*num_inner_samples:(n+1)*num_inner_samples]:
                eta = 1 - np.exp(-inner_theta * xi)
                marginal_log_liks = [np.log(eta) if i < sampled_infected_num else np.log(1 - eta) for i in range(self.env.N)]
                marginal_log_likelihoods.append(np.mean(marginal_log_liks))
            
            log_marginal_likelihood = np.max(marginal_log_likelihoods) + np.log(np.mean(np.exp(np.array(marginal_log_likelihoods) - np.max(marginal_log_likelihoods))))
            log_likelihoods.append(log_likelihood - log_marginal_likelihood)
        
        return np.mean(log_likelihoods)
    
    def get_norm_factors(self):
        # Calculate normalization factors (mean and std of errors)
        outs = [self.env.step(self.env.sample_random_input()) for _ in range(10000)]
        mean_err, std_err = np.mean(outs), np.std(outs)
        return mean_err, std_err

```

## Creating a New Agent

## Metrics
BoxingGym provides a set of metrics to evaluate the performance of agents in different environments. These metrics include:
- **Expected Information Gain (EIG)**: Measures the expected reduction in uncertainty after collecting new data.
- **Mean Squared Error (MSE)**: Measures the average squared difference between predictions and measurements.
- **Communication Error**: ...

## Directory Structure
- Environments: `src/boxing_gym/envs/`
- Agents: `src/boxing_gym/agents/`
- Configurations for experiments: `configs/`
- Running experiments: `run_experiment.py`
- Scripts for running experiments: `scripts/`
- Analysis of experiments: `analysis/`

## Existing Environments
BoxingGym comes with a set of pre-built environments, each representing different scientific domains:

- **Location Finding**: Predict and locate signal sources in an n-dimensional space.
- **Hyperbolic Temporal Discounting**: Understand participant behavior in reward delay scenarios.
- **Predator-Prey Dynamics**: Simulate and predict population dynamics over time.
- **Item Response Theory (IRT)**: Model student responses based on question difficulty and student ability.
And more...

## Existing Agents
### Box's Apprentice
Box's Apprentice is an agent that combines language model capabilities with statistical modeling to perform experimental design and model discovery. It can build explicit generative models to improve predictions.

See ...

### LLM Agent
The LLM Agent is a language-based agent that interacts with environments purely through natural language. It is capable of proposing and testing scientific theories but relies on its language processing abilities.

See ...

## Contributing
We welcome contributions to BoxingGym, especially for new environments and agents. If you want to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a new Pull Request.
7. For major changes, please open an issue first to discuss what you would like to change.