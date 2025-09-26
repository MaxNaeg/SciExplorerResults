
# replace by your path to sciexplorer
SCIEXPLORER_PATH="/Users/mnaegel/Documents/sciexplorer_framework/sciexplorer"
# replace by folder where you want to save results
BASE_RESULT_FOLDER="/Users/mnaegel/Documents/sciexplorer_framework/agentic_exploration_of_physics_models/quantum_dynamics_final_runs"
import sys
sys.path.append(BASE_RESULT_FOLDER)
import os
import multiprocessing

import matplotlib
matplotlib.use('Agg')

import physics_run_quantumspins as run
from functools import partial

import sciexplorer.model_utils
sciexplorer.model_utils.TIMEOUT=120



EXAMPLES_PATH=os.path.join(SCIEXPLORER_PATH, 'examples')

# replace by your API key and model name
llm=dict( api_key= '', # replace by your API key
base_url= None,
model_name= 'gpt-5-2025-08-07' )



tasks=[
    "DYNAMICS_ARBITRARY_THREE_SPINS",
    "DYNAMICS_ARBITRARY_THREE_SPINS_TWO_SPINS_ACCESS",
    "DYNAMICS_ARBITRARY_THREE_SPINS_TWO_SPINS_ACCESS_VAR_FIELD",
    "DYNAMICS_HEISENBERG",
    "DYNAMICS_HEISENBERG_VAR_FIELD",
    "DYNAMICS_HEISENBERG_VAR_FIELD_TWO_SPINS_ACCESS",
    "DYNAMICS_TFIM_VAR_COUPLING",
    "DYNAMICS_TFIM_VAR_COUPLING_TWO_SPINS_ACCESS",
]

repeats = 3


def one_run(i, path_results, task):
    print(f"Running repeat {i+1} of {repeats}...")
    try:
        result=run.dynamics_model_discovery(llm,EXAMPLES_PATH,path_results, task)
    except Exception as e:
        print(f"An error occurred during the run {i}: {e}")
        return 0
    return 1

def main():
    for task in tasks:
        path_results = os.path.join(BASE_RESULT_FOLDER, task)
        os.makedirs(path_results, exist_ok=True)    

        one_run_fixed = partial(one_run, path_results=path_results, task=task)

        with multiprocessing.Pool(processes=repeats) as pool:
            results = pool.map(one_run_fixed, range(repeats))

    os.system("say 'experiments complete'")

if __name__ == "__main__":
    main()