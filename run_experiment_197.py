import hydra
import pymc
import random
import numpy as np
from agent import LMExperimenter
from omegaconf import DictConfig, OmegaConf
import location_finding
import hyperbolic_temporal_discount
import death_process
import irt
import survival_analysis
import peregrines
import dugongs
import lotka_volterra
import tqdm
import moral_machines
import emotion

MAX_TRIES = 3
def iterative_experiment(goal, scientist, num_experiments, num_evals, include_prior, naive_agent=None, com_limit=None):
    results = []

    if 0 in num_experiments:
        final_results = "You cannot make observations now. Make assumptions and provide you best guess to the following query."
        if naive_agent is not None:
            result = evaluate_naive_explanation(final_results, goal, scientist, naive_agent, num_evals, include_prior, com_limit)
        else:
            result = evaluate(final_results, goal, scientist, num_evals, include_prior)
        results.append(result)

    observations = None
    for i in tqdm.tqdm(range(num_experiments[-1])):                
        if i+1 in num_experiments:
            final_results = f"The final results are {experiment_results}."
            if naive_agent is not None:
                result = evaluate_naive_explanation(final_results, goal, scientist, naive_agent, num_evals, include_prior, com_limit)
            else:
                result = evaluate(final_results, goal, scientist, num_evals, include_prior)
            results.append(result)
        success = False
        observations = scientist.generate_actions(observations)
        experiment_results, success = goal.env.run_experiment(observations)
        tries = 1
        while not success and tries < MAX_TRIES:
            observe, _ = scientist.prompt_llm_and_parse(experiment_results, True)
            experiment_results, success = goal.env.run_experiment(observe)
            if not success:
                tries += 1
    return results

def evaluate(final_results, goal, scientist, num_evals, include_prior):
    predictions, gts = [], []
    print(f"running {num_evals} evals")
    goal.eval_pointer = 0 # reset pointer, some goals have a static eval set
    for i in tqdm.tqdm(range(num_evals)):
        question, gt = goal.get_goal_eval_question(include_prior)
        question = final_results + '\n' + question
        # TODO check scientist does not save goal question
        prediction = scientist.generate_predictions(question)
        gts.append(gt)
        print(f"prediction: {prediction}, gt: {gt}")
        predictions.append(prediction)

    return goal.evaluate_predictions(predictions, gts)

def evaluate_naive_explanation(final_results, goal, scientist, naive_agent, num_evals, include_prior, com_limit):
    request_prompt = goal.get_comm_prompt(com_limit=com_limit, include_prior=include_prior)
    explanation = scientist.prompt_llm(request_prompt)
    print(f"explanation: {explanation}")
    naive_system_message = goal.get_naive_system_message(include_prior)
    naive_system_message += explanation
    naive_agent.set_system_message(naive_system_message)
    return evaluate(final_results, goal, naive_agent, num_evals, include_prior)

@hydra.main(config_path="conf", config_name="location_finding_direct")
def main(config: DictConfig):
    seed = config.seed
    print(f"seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)

    model_name = config.model_name
    temperature = config.temperature
    max_tokens = config.max_tokens
    num_experiments = config.num_experiments
    env_params = config.env_params
    experiment_type = config.experiment_type
    include_prior = config.include_prior
    num_evals = config.num_evals
    env_name = config.env_name
    goal_name = config.goal_name
    com_limit = config.com_limit

    nametoenv = {    
        "location_finding": location_finding.Signal,
        "hyperbolic_temporal_discount": hyperbolic_temporal_discount.TemporalDiscount,
        "death_process": death_process.DeathProcess,
        "irt": irt.IRT,
        "survival": survival_analysis.SurvivalAnalysis,
        "dugongs": dugongs.Dugongs,
        "peregrines": peregrines.Peregrines,
        "morals": moral_machines.MoralMachine,
        "emotion": emotion.EmotionFromOutcome,
        "lotka_volterra": lotka_volterra.LotkaVolterra
    }
    nameenvtogoal = {
        ("hyperbolic_temporal_discount", "direct"): hyperbolic_temporal_discount.DirectGoal,
        ("hyperbolic_temporal_discount", "discount"): hyperbolic_temporal_discount.DiscountGoal,
        ("hyperbolic_temporal_discount", "direct_discovery"): hyperbolic_temporal_discount.DirectGoalNaive,
        ("location_finding", "direct"): location_finding.DirectGoal,
        ("location_finding", "source"): location_finding.SourceGoal,
        ("location_finding", "direct_discovery"): location_finding.DirectGoalNaive,
        ("death_process", "direct"): death_process.DirectDeath,
        ("death_process", "direct_discovery"): death_process.DirectDeathNaive,
        ("death_process", "infection"): death_process.InfectionRate,
        ("irt", "direct"): irt.DirectCorrectness,
        ("irt", "direct_discovery"): irt.DirectCorrectnessNaive,
        ("irt", "best_student"): irt.BestStudent,
        ("irt", "difficult_question"): irt.DifficultQuestion,
        ("irt", "discriminate_question"): irt.DiscriminatingQuestion,
        ("survival", "direct"): survival_analysis.DirectGoal,
        ("survival", "direct_discovery"): survival_analysis.DirectGoalNaive,
        ("dugongs", "direct"): dugongs.DirectGoal,
        ("dugongs", "direct_discovery"): dugongs.DirectGoalNaive,
        ("peregrines", "direct"): peregrines.DirectGoal,
        ("peregrines", "direct_discovery"): peregrines.DirectGoalNaive,
        ("emotion", "direct"): emotion.DirectEmotionPrediction,
        ("emotion", "direct_discovery"): emotion.DirectEmotionNaive,
        ("morals", "direct"): moral_machines.DirectPrediction,
        ("morals", "direct_discovery"): moral_machines.DirectPredictionNaive,
        ("lotka_volterra", "direct"): lotka_volterra.DirectGoal,
        ("lotka_volterra", "direct_discovery"): lotka_volterra.DirectGoalNaive,
    }

    env = nametoenv[env_name](**env_params)
    goal = nameenvtogoal[(env_name, goal_name)](env)

    scientist_agent = LMExperimenter(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
    naive_agent = None
    if experiment_type == "discovery":
        naive_agent = LMExperimenter(model_name=model_name, temperature=temperature, max_tokens=max_tokens)

    system_message = goal.get_system_message(include_prior)
    scientist_agent.set_system_message(system_message)
    
    print(f"running {num_experiments} experiments")
    final_results = iterative_experiment(goal, scientist_agent, num_experiments, num_evals, include_prior, naive_agent, com_limit)

    for i in range(len(num_experiments)):
        print(f"{num_experiments[i]}: {final_results[i]}")

if __name__ == "__main__":
    main()