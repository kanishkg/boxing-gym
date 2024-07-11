SEEDS=(1 2 3 4 5)
for seed in "${SEEDS[@]}"
    # OED
    python run_experiment.py seed=$seed llms=openai include_prior=true exp=oed envs=rat_tumor_direct
    python run_experiment.py seed=$seed llms=openai include_prior=false exp=oed envs=rat_tumor_direct
    # Discovery
    python run_experiment.py seed=$seed llms=openai include_prior=true exp=discovery envs=location_finding_direct_discovery
    python run_experiment.py seed=$seed llms=openai include_prior=false exp=discovery envs=location_finding_direct_discovery

