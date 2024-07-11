import tqdm
import random
import numpy as np
import hydra
from omegaconf import DictConfig
import json

import location_finding
import hyperbolic_temporal_discount
import death_process
import irt
import survival_analysis
import peregrines
import dugongs
import lotka_volterra
import moral_machines
import emotion


@hydra.main(version_base=None, config_path="conf", config_name="config_eig")
def main(config: DictConfig):
    seed = config.seed
    print(f"seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)

    model_name = config.llms.model_name
    temperature = config.llms.temperature
    max_tokens = config.llms.max_tokens
    num_experiments = config.exp.num_experiments
    env_params = config.envs.env_params
    experiment_type = config.exp.experiment_type
    include_prior = config.include_prior
    num_evals = config.envs.num_evals
    env_name = config.envs.env_name
    goal_name = config.envs.goal_name
    num_random = config.num_random
    redo = config.redo
    box = config.box

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

    if box:
        model_name = f"{model_name}-boxloop"
    res_dir = f"results/{env_name}"
    res_filename = f"{goal_name}_{model_name}_{experiment_type}_{include_prior}_{seed}.json"

    with open(f"{res_dir}/{res_filename}", "r") as f:
        data = json.load(f)
    

    queries = data['data']['queries']
    successes = data['data']['successes']
    if not redo:
        eigs = data['data']['eigs']
    else:
        eigs = []

    eigs_regret = []
    eigs_max = []
    inputs_max = []
    for i, o in tqdm.tqdm(enumerate(queries)):
        if not successes[i]:
            continue
        eigs_random = []
        inputs = []
        for n in range(num_random):
            random_input = env.sample_random_input()
            eig = goal.expected_information_gain(random_input)
            inputs.append(random_input)
            eigs_random.append(eig)
        if redo:
            input_query = env.validate_input(o)
            eig = goal.expected_information_gain(input_query)
            eigs.append(eig)
        else:
            eig = eigs[i]
        max_eig = max(eigs_random)
        max_input = inputs[eigs_random.index(max_eig)]
        eigs_max.append(max_eig)
        inputs_max.append(max_input)
        eigs_regret.append(max_eig - eig)
        print(f"eig: {eig}, max_eig: {max_eig}, regret: {max_eig - eig}")
        # make obs
        _ = goal.env.run_experiment(o)
    
    
    # convert all elements of list to float to be json serializable
    eigs_regret = [float(x) for x in eigs_regret]
    eigs_max = [float(x) for x in eigs_max]
    eigs = [float(x) for x in eigs]
    # convert inputs to list if they are np arrays or tuples
    inputs_max = [x.tolist() if type(x) == np.ndarray else x for x in inputs_max]
    # check if inputs are tuples, and the elements are np arrays
    new_inputs_max = []
    for x in inputs_max:
        if type(x) == tuple:
            new_x = [y.tolist() if type(y) == np.ndarray else y for y in x]
            new_inputs_max.append(tuple(new_x))
        else:
            new_inputs_max.append(x)
    
    store_dict = {
            "eigs_regret": eigs_regret,
            "eigs_max": eigs_max,
            "inputs_max": new_inputs_max,
            "eigs": eigs
    }
    with open(f"{res_dir}/regret_{res_filename}", 'w') as f:
        json.dump(store_dict, f, indent=4)

if __name__ == "__main__":
    main()