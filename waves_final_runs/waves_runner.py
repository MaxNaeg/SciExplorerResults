# replace by your path to sciexplorer
SCIEXPLORER_PATH="/Users/mnaegel/Documents/sciexplorer_framework/sciexplorer"
# replace by folder where you want to save results
BASE_RESULT_FOLDER="/Users/mnaegel/Documents/sciexplorer_framework/agentic_exploration_of_physics_models/quantum_gs_final_runs"
import sys
sys.path.append(BASE_RESULT_FOLDER)
import os
import multiprocessing

import matplotlib
matplotlib.use('Agg')

import physics_run_waves as run
from functools import partial

import sciexplorer.model_utils
sciexplorer.model_utils.TIMEOUT=120



EXAMPLES_PATH=os.path.join(SCIEXPLORER_PATH, 'examples')

# replace by your API key and model name
llm=dict(api_key= '',  # replace by your API key
base_url= None,
model_name= 'gpt-5-2025-08-07' )



tasks=["LIN_SEQ","LIN_SEQ_NNN","LIN_SEQ_POTENTIAL","LIN_SEQ_PERIODIC",
       "NONLIN_SEQ","NONLIN_SEQ_POTENTIAL","SIMPLE_GL","COMPLEX_GL",
      "NONLIN_SEQ_PHI6", "NONLIN_SEQ_NNN", "COMPLEX_GL_NNN", "SIN_POTENTIAL_RELAXATION", 
      ]

repeats = 3

def one_run(i, path_results, task):
    print(f"Running repeat {i+1} of {repeats}...")
    try:
        result=run.field_dynamics(llm,EXAMPLES_PATH,path_results, "model_discovery", task)
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