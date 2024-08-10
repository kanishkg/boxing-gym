import numpy as np
from scipy.integrate import odeint

from .goal import Goal
from ..agents.box_loop_helper import construct_dataframe

class DirectGoal(Goal):
    def __init__(self, env):
        super().__init__(env)
        self.eval_points = []
        self.eval_pointer = 0
        self.update_param_cache = {}
        self.norm_mu, self.norm_sigma = (8.327445247142364, 17.548285564117467)
    
    def get_system_message(self, include_prior):
        if include_prior:
            goal_description = f"""
Your goal is to be able to reliably predict the populations of predators and prey (two positive integers) at any given (positive float) time. Conduct experiments to learn more about the environment.
"""
        else:
            goal_description = f"""
Your goal is to be able to reliably predict the response of the environment (a tuple of two positive integers) to a given input (positive float). Conduct experiments to learn more about the environment.
"""
        description = self.env.generate_system_message(include_prior, goal_description)
        return description
    
    def get_goal_eval_question(self, include_prior):
        if self.eval_pointer+1 > len(self.eval_points):
            time = np.round((np.random.beta(2, 5) * 4), 1)
            counts = self.env.step(time)
            self.eval_points.append((time, counts))
        else:
            time, counts = self.eval_points[self.eval_pointer]
        self.eval_pointer += 1

        if include_prior:
            question = f"""Predict the population counts of prey and predators at time: {time}. 
When asked to answer a question about the environment, respond in the format specified below. Make assumptions about the environment and provide your best guess. Answer in the format: [num of prey, num of predator].
Here is an example.
<thought> your thought </thought>
<answer>[2, 3]</answer>"""

        else:
            question = f"""
            Predict the response (tuple of two positive integers) to the following input: {time}.
When asked to answer a question about the environment, respond in the format specified below. Make assumptions about the environment and provide your best guess.
Here is an example.
<thought> your thought </thought>
<answer>[2, 3]</answer>"""
        return question, counts

    def evaluate_predictions(self, predictions, measurements):
        assert len(predictions) == len(measurements), f"Predictions and measurements must have the same length. {len(predictions)}, {len(measurements)}"
        predictions = [eval(x) for x in predictions]
        abs_error = np.abs(np.array(predictions) - np.array(measurements))
        mean_abs = np.mean(abs_error)
        std_abs = np.std(abs_error)
        return mean_abs, std_abs
    
    def get_norm_factors(self):
        N = 100000
        errs = []
        measurements = []
        import logging
        import tqdm
        logger = logging.getLogger("pymc")
        logger.setLevel(logging.WARNING)
        for i in tqdm.tqdm(range(N)):
            if i % 10 == 0:
                self.env.reset()
            inp = self.env.sample_random_input()
            out = self.env.step(inp)
            if out[0] > 1000 or out[1] > 1000:
                continue
            measurements.append(out)

        mu1 = np.mean([x[0] for x in measurements])
        mu2 = np.mean([x[1] for x in measurements])
        print(mu1,mu2)
        pred = [f"[{mu1},{mu2}]"] * len(measurements)
        err_mean, err_std = self.evaluate_predictions(pred, measurements)
        return err_mean, err_std

    def expected_information_gain(self, query_point, num_outer_samples=1000, num_inner_samples=10):
        raise NotImplementedError("Expected Information Gain not implemented for this goal.")
        # t_query = query_point
        # existing_data = self.env.observation_data
        # if tuple(existing_data) not in self.update_param_cache:
        #     print("Updating parameters cache")
        #     with pm.Model() as model:
        #         alpha = pm.Normal("alphai", mu=self.env.alpha_mu, sigma=0.01)
        #         beta = pm.Normal("betai", mu=self.env.beta_mu, sigma=0.01)
        #         gamma = pm.Normal("gammai", mu=self.env.gamma_mu, sigma=0.04)
        #         delta = pm.Normal("deltai", mu=self.env.delta_mu, sigma=0.001)
        #         # Define the likelihood function
        #         def likelihood(ti, obs_prey, obs_pred):
        #             params = [alpha, beta, gamma, delta]
        #             pop_init = [self.env.prey_init, self.env.predator_init]
        #             solution = odeint(self.env.lotka_volterra_system, pop_init, [0, ti], args=tuple(params))
        #             prey_count = pm.Normal("prey_count", mu=solution[-1][0], sigma=0.1, observed=obs_prey)
        #             predator_count = pm.Normal("predator_count", mu=solution[-1][1], sigma=0.1, observed=obs_pred)
        #             return prey_count, predator_count

        #         if len(existing_data) > 0:
        #             tis = pm.Data("tis", np.array([x[0] for x in existing_data]))
        #             preyc = pm.Data("preyc", np.array([x[1][0] for x in existing_data]))
        #             predc = pm.Data("predc", np.array([x[1][1] for x in existing_data]))
        #             likelihood(tis, preyc, predc)
        #         trace = pm.sample(num_outer_samples*num_inner_samples+num_outer_samples, tune=1000, return_inferencedata=False, chains=2, cores=2, target_accept=0.95)
        #         alphas = trace["alphai"]
        #         betas = trace["betai"]
        #         gammas = trace["gammai"]
        #         deltas = trace["deltai"]
        #         shuffling = zip(alphas, betas, gammas, deltas)
        #         random.shuffle(shuffling)
        #         alphas, betas, gammas, deltas = zip(*shuffling)
        #         self.update_param_cache[tuple(existing_data)] = (alphas, betas, gammas, deltas)
        # else:
        #     alphas, betas, gammas, deltas = self.update_param_cache[tuple(existing_data)]
        
        # outer_alphas = alphas[:num_outer_samples]
        # outer_betas = betas[:num_outer_samples]
        # outer_gammas = gammas[:num_outer_samples]
        # outer_deltas = deltas[:num_outer_samples]
        # log_likelihoods = []
        # for n, (o_alpha, o_beta, o_gamma, o_delta) in enumerate(zip(outer_alphas, outer_betas, outer_gammas, outer_deltas)):
        #     with pm.Model() as inner_model:
        #         pop_init = [self.env.prey_init, self.env.predator_init]
        #         solution = odeint(self.env.lotka_volterra_system, pop_init, [0, t_query], args=(o_alpha, o_beta, o_gamma, o_delta))
        #         pred_prey = solution[-1][0]
        #         pred_predator = solution[-1][1]
        #         prob_prey = norm.pdf(solution[-1][0], loc=solution[-1][1], scale=0.1)
        #         prob_predator = norm.pdf(solution[-1][0], loc=solution[-1][1], scale=0.1)
        #         log_likelihood = np.log(prob_prey+0.0001) + np.log(prob_predator+0.0001)

        #         inner_alphas = alphas[num_outer_samples+n*num_inner_samples:num_outer_samples+(n+1)*num_inner_samples]
        #         inner_betas = betas[num_outer_samples+n*num_inner_samples:num_outer_samples+(n+1)*num_inner_samples]
        #         inner_gammas = gammas[num_outer_samples+n*num_inner_samples:num_outer_samples+(n+1)*num_inner_samples]
        #         inner_deltas = deltas[num_outer_samples+n*num_inner_samples:num_outer_samples+(n+1)*num_inner_samples]
        #         marginal_log_likelihoods = []
        #         for i, (i_alpha, i_beta, i_gamma, i_delta) in enumerate(zip(inner_alphas, inner_betas, inner_gammas, inner_deltas)):
        #             pop_init = [self.env.prey_init, self.env.predator_init]
        #             solution = odeint(self.env.lotka_volterra_system, pop_init, [0, t_query], args=(i_alpha, i_beta, i_gamma, i_delta))
        #             prob_prey = norm.pdf(prob_prey, loc=solution[-1][1], scale=0.1)
        #             prob_predator = norm.pdf(prob_predator, loc=solution[-1][1], scale=0.1)
        #             log_prob = np.log(prob_prey+0.0001) + np.log(prob_predator+0.0001)
        #             marginal_log_likelihoods.append(log_prob)
        #         max_log = np.max(marginal_log_likelihoods)    
        #         log_marginal_likelihood = max_log + np.log(np.mean(np.exp(np.array(marginal_log_likelihoods) - max_log)))
        #         log_likelihoods.append(log_likelihood - log_marginal_likelihood)
        # eig_value = np.mean(log_likelihoods)
        # return eig_value

class DirectGoalNaive(DirectGoal):
    def __init__(self, env):
        super().__init__(env)
        self.eval_points = []
        self.eval_pointer = 0
    
    def get_system_message(self, include_prior):
        goal_description = "Your goal is to conduct experiments and explain the environment to the user so that they can achieve their goal."

        if include_prior:
            goal_description = f"""
The goal of the user is to be able to reliably predict the population counts (tuple of two nonnegative floats) at any given time.
"""
        else:
            goal_description = f"""
The goal of the user is to be able to reliably predict a tuple of two nonnegative floats (output) for any given float (input).
"""
        description = self.env.generate_system_message(include_prior, goal_description)
        return description
    
    def get_naive_system_message(self, include_prior):
        if include_prior:
            goal_description = f"""
Your goal is to be able to reliably predict the population counts of prey and predator (tuple of two positive integers) at any given time.
"""
        else:
            goal_description = f"""
Your goal is to be able to reliably predict a tuple of two positive integers (output) for any given float (input).
"""
        format_instructions = f"""
You will be provided with a set of inputs to this environment and will be tasked with predicting the output for each input.
You must give number values. You may also think before providing your predictions.
You must respond with a single number! 
If you do not do so, I will be penalized one million dollars and it will be a complete disaster.
Here is an example:
<thought>your thought</thought>
<answer>1</answer>"""
        description = goal_description + '\n' + format_instructions
        description += "Here is what you know about the environment:\n"
        return description    
    
    def get_comm_prompt(self, include_prior, com_limit=300, use_ppl=False, str_prob_prog=None, params_summary_str=None):    
        extra_prompt = ""
        if include_prior:
            extra_prompt = f"""
"""
        description = f"""
Assume that the user has no prior knowledge, and will not be able to run any experiments before making predictions. 
They will make predictions based solely on your explanation, so provide as much detail as possible. You cannot provide your own experiments or observations.
Limit your explanation to {com_limit} words
{extra_prompt}
"""
        if use_ppl:
            description += f"To make your explanation clearer and more informative, look at the statistical model (written in pymc) designed by a colleague for the experimental data and the inferred parameters. \n"
            description += f"Here is the statistical model. \n {str_prob_prog} \n"
            description += f"Here are the inferred params. \n {params_summary_str} \n"
            description += f"Don't literally describe the model verbatim but use it to conceptually motivate your explanation."
            description += f"The agent will not be able to use the model explicitly but having a conceptual understanding will be beneficial."
        return description
    
class LotkaVolterra:
    def __init__(self, 
                prey_init=40, 
                predator_init=9, 
                alpha=0.1, 
                beta=0.02, 
                gamma=0.4, 
                delta=0.01,
                lower_limit = 0,
                upper_limit = 50):
        self.prey_init = prey_init
        self.predator_init = predator_init
        self.alpha_mu = alpha
        self.beta_mu = beta
        self.gamma_mu = gamma
        self.delta_mu = delta
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.reset()
        
        self.env_name = "lotka_volterra"
    
    # TO-DO: para to be determined
    def reset(self):
        self.alpha = np.random.normal(self.alpha_mu, 0.01)
        self.beta = np.random.normal(self.beta_mu, 0.01)
        self.gamma = np.random.normal(self.gamma_mu, 0.04)
        self.delta = np.random.normal(self.delta_mu, 0.001)
        self.observation_data = []


    def generate_system_message(self, include_prior, goal_description):
        if include_prior:
            PRIOR = f"""You are observing the populations of predators and prey at different times.
Make observations by specifying a single time you want to observe with a float number. The environment will return the population counts of prey and predators successively.
The time values are between {self.lower_limit} and {self.upper_limit}.
"""
            message= f"""
{goal_description}
{PRIOR}

Here is an example:
<thought> your thought </thought>
<observe>2</observe>
When asked to answer a question about the environment, respond in the format specified in the question.
<thought> your thought </thought>
<answer> your answer </answer>
"""
        else:
            NO_PRIOR = f"""You may observe the value of the function one input value at a time. Make observations by specifying a single value you want to observe with a float number.
The environment will return two nonnegative floating numbers as response of the function at that input.
The input values are between {self.lower_limit} and {self.upper_limit}.
"""
            message = f"""
{goal_description}
{NO_PRIOR}

Here is an example:
<thought> your thought </thought>
<observe>2</observe>
When asked to answer a question about the environment, respond in the format specified in the question.
Example:
<thought> your thought </thought>
<answer> your answer </answer>

"""
        return message

    @staticmethod
    def lotka_volterra_system(y, t, alpha, beta, gamma, delta):
        prey, predator = y
        dprey_dt = alpha * prey - beta * prey * predator
        dpredator_dt = delta * prey * predator - gamma * predator
        return [dprey_dt, dpredator_dt]

    def sample_random_input(self):
        ti = np.random.uniform(self.lower_limit, self.upper_limit)
        return ti

    def step(self, time):
        params = [self.alpha, self.beta, self.gamma, self.delta]
        populations = [self.prey_init, self.predator_init]

        # Use scipy.integrate.odeint to solve the differential equations at the specified time
        solution = odeint(
            self.lotka_volterra_system, populations, [0, time], args=tuple(params)
        )

        # Round the solution to the nearest integer and convert to integer type for realistic population counts
        rounded_solution = np.round(solution[-1]).astype(int)  # Take the last value (at the specified time)
        rounded_solution = rounded_solution.clip(min=0)  # Ensure that the populations are nonnegative
        rounded_solution = rounded_solution.tolist()
        return rounded_solution

    def validate_input(self, input_string):
        # Remove the opening and closing brackets
        try:
            query = float(input_string.strip())
        except:
            return "Input must be a float."
        if query < self.lower_limit or query > self.upper_limit:
            return f"Input must be between {self.lower_limit} and {self.upper_limit}."
        # obs_times = [float(x) for x in self.observation_data]
        # max_time = max(obs_times) if len(obs_times) > 0 else self.lower_limit      
        # if query <= max_time:
        #     return "Input must be greater than the previous observation."
        # Split the string by commas and remove any leading/trailing whitespace
        return query

    def run_experiment(self, input_string):
        time = self.validate_input(input_string)
        if isinstance(time, str):
            return time, False
        result = self.step(time)
        self.observation_data.append((time, tuple(result)))
        return result, True

    def get_data(self):
        return self.observation_data

    def get_df(self):
        '''
            Construct dataframe used for Box's Loop
        '''
        self.df = construct_dataframe(self)
    
    def get_description(self):
        if self.include_prior:
            return f"""
The environment consists of the populations of predators and prey (two nonnegative floats) at any given (real number) time.
"""
        else:
            return f""""
The environment returns a response (tuple of two real numbers) to a scalar input.
"""

    def describe_data_columns(self):
        return self.format_column_description()

    def get_ordered_column_names(self):
        if self.include_prior:
            return ["t", "prey", "predator"]
        else:
            return ["t", "y1", "y2"]
    
    def get_ordered_features(self):
        return self.get_ordered_column_names()[:-1] 

    def format_column_description(self):
        if self.include_prior:
            return (f"The observations are: \n -prey: prey population -predator: predator population \n"
                    f"The input values are \n -t: time"
                    f"Use the values of the input values to help you model the observations. ")
        else:
            return ""

if __name__ == "__main__":
    env = LotkaVolterra()
    goal = DirectGoal(env)
    print(goal.get_norm_factors())
    # input_str = "2"
    # print(env.run_experiment(input_str))
    # # test eig
    # input_tis = [0.1, 0.2, 0.3, 0.4, 0.5]
    # for t in input_tis:
    #     print(goal.expected_information_gain(t))
