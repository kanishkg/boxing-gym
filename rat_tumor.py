import numpy as np
from scipy.stats import norm, halfnorm
import json
from goal import Goal
import matplotlib.pyplot as plt

# citation https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/GLM-hierarchical-binomial-model.html

# RAT TUMORS 
# - The solution to modeling the relationship between tR and nR is using a Beta probability distribution 
#       ( theta_i = Beta(a, b); tR = (theta_i, nR) )
# - The scientist can access data by specifying nR, and is returned tR.
# - The scientist is trying to find a relationship between 
#   the tR (rats with tumor in the observed population) and nR (total rats in the osberved population). 
# - The novice agent is trying to estimate tR given nR.
# - We check the accuracy of a novice agent's hypothesized tR using the solution probability distribution
#   by finding the P(novice agent's hypothesized probability) in the solution distribution


# RUNNING ENVIRONMENT
# run the cfrm key thing; then run "python run_experiment.py"
# run python rat_tumor_197.py to see if everything works
# check duogongs for example for run_experiment function
# keep num experiments small


# DONE - Aditi
PRIOR = """You are observing the relationship between the total number of lab rats in a population, and the number who develop endometrial stromal polyps."""
NO_PRIOR = """You are observing a relationship between two positive integer values (integer, integer).""" 

class DirectGoal(Goal):
    def __init__(self, env):
        super().__init__(env)
        self.eval_points = []
        self.eval_pointer = 0
        self.norm_mu, self.norm_sigma = (-0.253343, 158.18241292989669)

    # DONE - Aditi
    def get_system_message(self, include_prior):
        if include_prior:
            goal_description = "Your goal is to be able to predict the number of lab rats that develop endometrial stromal polyps given the population size. Conduct experiments to learn about the environment."
        else:
            goal_description = "Your goal is to be able to predict an integer given an integer. Conduct experiments to learn about the environment."
        description = self.env.generate_system_message(include_prior, goal_description)
        return description
    
    # DONE - Aditi
    def get_goal_eval_question(self, include_prior):
        # for different num_rat values, use self.step and get number of rats with the tumor
        if self.eval_pointer+1 > len(self.eval_points):
            nR = np.random.uniform(0, 300) # number of rats total
            tR = self.env.calculate_values(nR) # the probability that the rat develops esp
            self.eval_points.append((nR, tR))
        else:
            (nR, tR) = self.eval_points[self.eval_pointer]
        
        self.eval_pointer += 1
        
        if include_prior:
            question = f"Given the total rats in the population nR = {nR}, how many rats will develop the tumor tR? You are observing the infection rate."
        else:
            question = f"If the input is {nR}, what is the output? You are observing a ratio between the input and output."
        question += " Respond using an integer greater than or equal to 0."
        return question, tR 
    
    # DONE - Mohammed
    def evaluate_predictions(self, predictions, measurements):
        assert len(predictions) == len(measurements), "Predictions and measurements must have the same length."
        parsed_predictions = []
        for p in predictions:
            # print("p", p)
            p = p.strip()
            p = int(float(p))
            parsed_predictions.append(p)
            # print("parsed", p)
        abs_errors = []
        for i in range(len(predictions)):
            if parsed_predictions[i] is not None:
                abs_error = (parsed_predictions[i] - measurements[i]) 
                abs_errors.append(abs_error)
        mse = np.mean(abs_errors)
        std_dev = np.std(abs_errors)
        return (mse, std_dev)
    
    def get_norm_factors(self):
        N = 1000000    
        errs = []
        measurements = []
        
        for i in range(N):
            if i % 10 == 0:
                self.env.reset()
            inp = self.env.sample_random_input()
            # print("inp",inp)
            out = self.env.step(inp)
            # print("out", out)
            measurements.append(out)
        mu = np.mean(measurements)
        pred = [str(mu)] * N
        err_mean, err_std = self.evaluate_predictions(pred, measurements)
        return err_mean, err_std
        
class DirectGoalNaive(DirectGoal):
    def __init__(self, env):
        super().__init__(env)
    
    # DONE - Aditi
    def get_system_message(self, include_prior):
        goal_description = "Your goal is to conduct experiments and explain the environment to the user."

        if include_prior:
            goal_description += "The goal of the user is to be able to predict the integer number of lab rats that develop endometrial stromal polyps given the populations size, and this can be modeled using an infection rate."
        else:
            goal_description += "The goal of the user is to be able to predict the integer number response of the environment to a given integer using a ratio that relates the integers to each other."
        description = self.env.generate_system_message(include_prior, goal_description)
        return description
    
    # DONE - Aditi
    def get_naive_system_message(self, include_prior):
        if include_prior:
            goal_description = "Your goal is to be able to predict the integer number of lab rats that develop endometrial stromal polyps given the populations size."
        else:
            goal_description = "Your goal is to be able to predict the integer number response of the environment to a given integer."
        format_instructions = """You will be provided with a set of inputs to this environment and will be tasked with predicting the output for each input.
            Format your answers using a JSON list of values. You must give integer number values. You may also think before providing your predictions.
            Here is an example:
            <thought>your thought</thought>
            <answer>1</answer>"""
        description = goal_description + '\n' + format_instructions
        description += "Here is what you know about the environment:\n" 
        return description    
    
    # DONE - Aditi 
    def get_comm_prompt(self, include_prior, com_limit=300):
        description = f"""Assume that the user has no prior knowledge, and will not be able to run any experiments before making predictions. 
            They will make predictions based solely on your explanation, so provide as much detail as possible.
            Limit your explanation to {com_limit} words."""
        return description

class RatTumor(Goal):
    def __init__(self, env):
        super().__init__(env)
    
    # DONE - Aditi
    def get_system_message(self, include_prior):
        if not include_prior:
            raise ValueError("RatTumor requires prior knowledge.")
        goal_description = """Your goal is to predict the number of rats tR that develops endometrial stromal polyps. 
            Remember, the given data points are non-deterministic.
            """
        description = self.env.generate_system_message(include_prior, goal_description)
        return description

    # DONE - Aditi
    def get_goal_eval_question(self, include_prior):
        assert include_prior, "RatTumor requires prior knowledge."
        (k, _) = self.env.truth
        question = "What is the number of rats tR that will develop endometrial stromal polyps?" 
        question += " Respond with the integer value for tR."
        return question, k
    
    # DONE - Mohammed
    def evaluate_predictions(self, predictions, measurements):
        assert len(predictions) == len(measurements), "Predictions and measurements must have the same length."
        parsed_predictions = []
        for p in predictions:
            try: 
                parsed_predictions.append(int(p))
            except ValueError:
                print("Prediction not Parsed")
                parsed_predictions.append(None)
        absolute_errors = []
        for i in range(len(predictions)):
            if parsed_predictions[i] is not None:
                absolute_error = abs(parsed_predictions[i] - measurements[i])
                absolute_errors.append(absolute_error)
        mae = np.mean(absolute_errors)
        std_dev = np.std(absolute_errors)
        return (mae, std_dev)

class RatTumorModel:
    # DONE - Aditi, Mohammed
    def __init__(self, alpha=0.15, beta=0.15, max_num_rats=100):
        self.env_name = "rat_tumor_model"
        self.max_num_rats = max_num_rats
        self.alpha = alpha
        self.beta = beta
        self.reset()
        
    # DONE - Mohammed, Aditi
    def reset(self): 
        self.theta = np.random.beta(self.alpha, self.beta) # 

    # DONE - Aditi
    def generate_system_message(self, include_prior=True, goal=None):
        assert goal is not None
        if include_prior:
            message = f"""{PRIOR}
                {goal}
                Make observations by specifying the positive integer size nR of the population you would like to observe, 
                where nR is less than or equal to {self.max_num_rats}. 
                The environment will return an integer tR, representing the number of rats with the tumor in that population.
                Please specify the integer parameter in a list [nR] where the integer.

                Here is an example:
                <thought> your thought </thought>
                <observe>[42]</observe>
                When asked to answer a question about the environment, respond in the format specified in the question.
                <thought> your thought(integer)</thought>
                <answer>your answer(float)</answer>
                """
        else:
            message = f"""{NO_PRIOR}
                {goal}
                Make observations by specifying where you want to observe in list of 1 positive integer which is 
                less than or equal to {self.max_num_rats}.
                The observation returns an integer containing information about the environment.

                Here is an example:
                <thought> your thought </thought>
                <observe>[42]</observe>
                When asked to answer a question about the environment, respond in the format specified in the question.
                <thought> your thought(integer)</thought>
                <answer>your answer(float)</answer>
                """
        return message

    # DONE - Aditi, Mohammed
    # use step from the num_rats and get the (num rats, tumor rats) tuple out of the distribution function
    def calculate_values(self, num_rats):
        num_tumors = self.step(num_rats) 
        return (num_rats, num_tumors)
    
    def sample_random_input(self):
        return np.random.randint(0, self.max_num_rats)

    # DONE - Mohammed 
    def step(self, num_rats):
        num_tumors = np.random.binomial(num_rats, self.theta)
        return int(num_tumors)
    
    # DONE - Mohammed
    def validate_input(self, input_string): # take in a number of total rats:
        try:
            total_rats = int(input_string)
        except ValueError:
            return ("Error: Input must be an integer.", False)
        
        if total_rats < 0:
            return ("Error: Number of rats is not >=0.", False)
        if total_rats > self.max_num_rats: 
            return (f"Error: Number of rats should be <= {self.max_num_rats}.", False)
        
        return (total_rats, True)

    # DONE - Mohammed
    def run_experiment(self, input_string):
        validated_input = self.validate_input(input_string)
        if not validated_input[1]: 
            return validated_input[0], False   # Returning the error if the second element of the tuple is False.
        total_rats = validated_input[0]
        return self.step(total_rats), True

# DONE - Mohammed
if __name__ == "__main__":
    # Test RatTumorModel
    input_list = ["50", "100", "200", "400", "800"]
    output_list = []
    env = RatTumorModel(10, 10, 1000) #alpha, beta, max_population
    goal = DirectGoal(env)
    print(goal.get_norm_factors())
    # for input_string in input_list:
    #     output, _ = model.run_experiment(input_string)
    #     output_list.append(output) 

    # input_list = [int(x) for x in input_list]
    # plt.figure(figsize=(10, 5))
    # plt.plot(input_list, output_list, marker='o', linestyle='-', color='b')
    # plt.title('Number of Tumors vs. Total Rats')
    # plt.xlabel('Total Number of Rats')
    # plt.ylabel('Number of Tumors Predicted')
    # plt.grid(True)
    # plt.xticks(input_list)  # Ensure all input values appear as ticks on the x-axis
    # plt.show()
