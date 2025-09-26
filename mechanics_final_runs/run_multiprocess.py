import os
import argparse
import multiprocessing

from sciexplorer import runner
from sciexplorer.analyze_conv_utils.analyze_results import analyze_folder




# change this to your local sciexplorer folder, needed to use the example simulators, tasks, and tools
github_repo_path = '/Users/mnaegel/Documents/sciexplorer_framework/sciexplorer'
# path where results will be stored
res_base_folder = '/Users/mnaegel/Documents/sciexplorer_framework/agentic_exploration_of_physics_models/mechanics_final_runs'

# Experiment to explore
default_simulator = 'HiddenNBody2DGravity'
# Task to perform
default_task = 'model_discovery_hidden'

add_to_agent_name = 'chat'


default_repeats = 5 # number of runs to perform
default_max_steps = 60 #30 # maximum number of steps the agent can take in one run
max_tools = 3 * default_max_steps # maximum number of tools the agent can use in one run
default_n_initial_exps = 0 # number of initial experiments to run and plot before the conversation

# insert your API key here, or pass it as command line argument --api_key
default_api_key= '' # replace by your API key
default_base_url= None # Can be set to use non-Ooen AI models
default_model_name= 'gpt-5-2025-08-07' # Model name

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run experiments')
parser.add_argument('--task', type=str, default=default_task, 
                    help='Task to run (e.g. model_discovery)')
parser.add_argument('--simulator', type=str, default=default_simulator, 
                    help='Simulator to use (e.g., MysteryPendulumInit, MysteryDoubleWellInit, ParametricOsci, CoupledOscillatorInit, DoublePendulumInit, ParticleIn2DGravity, ParticleIn2DGravity_Circular)')
parser.add_argument('--repeats', type=int, default=default_repeats,
                    help='Number of runs to perform')
parser.add_argument('--max_steps', type=int, default=default_max_steps,
                    help='Maximum number of steps the agent can take in one run')
parser.add_argument('--n_initial_exps', type=int, default=default_n_initial_exps,
                    help='Number of initial experiments to run and plot before the conversation starts')
parser.add_argument('--api_key', type=str, default=default_api_key,
                    help='API key for the model')
parser.add_argument('--base_url', type=str, default=default_base_url,
                    help='Base URL for the model API')
parser.add_argument('--model_name', type=str, default=default_model_name,
                    help='Model name to use')

args = parser.parse_args()
task = args.task
simulator = args.simulator
repeats = args.repeats
max_steps = args.max_steps
n_initial_exps = args.n_initial_exps
api_key = args.api_key
base_url = args.base_url
model_name = args.model_name

result_folder = os.path.join(res_base_folder, f"{task}/{simulator}") # save results in production folder
os.makedirs(result_folder, exist_ok=True)


dt = 0.001 # time step seen by the agent
solver_steps_per_timestep = 10 # simulator uses a smaller time step internally for better accuracy

# Sensible parameters for the simulators
simulators_initials = {
    # Duffing oscillator
    'MysteryDoubleWellInit': dict(a = 1.1321, b = -0.8123, gamma=0.043, 
                                  dt=dt, solver_steps_per_timestep=solver_steps_per_timestep,),
    'MysteryPendulumInit': dict(alpha = 1.712, omega = 1.0, gamma=0.043, 
                                dt=dt, solver_steps_per_timestep=solver_steps_per_timestep,),
    'MysteryAsymmetricDoubleWellInit': dict(a = 1.1321, b = 0.8123, c=0.1, gamma=0.043, 
                                  dt=dt, solver_steps_per_timestep=solver_steps_per_timestep,),
    'VelocityPosCoupling': dict(a = 1.7, b = 0.4, 
                                    dt=dt, solver_steps_per_timestep=solver_steps_per_timestep,),
    'Arbitrary1dPot': dict(potential = "lambda x: 0.5 * x**2 + 0.8 * jnp.sin(6*x)",
                           min_random=-3., max_random=3.,
                           dt=dt, solver_steps_per_timestep=solver_steps_per_timestep,),
    'DrivenOsci': dict(omega = 1.523, gamma = 0.6, epsilon = 1.7123, drive_freq = 1.551, 
                       dt=dt, solver_steps_per_timestep=solver_steps_per_timestep,),
    'ParametricOsci': dict(omega = 1.523, gamma = 0.3, epsilon = 1.7123, drive_freq = 1.551, T=20, 
                           dt=dt, solver_steps_per_timestep=solver_steps_per_timestep,),

    'DoublePendulumInit': dict(l1 = 1.712, l2 = 0.851, gamma=0.143, 
                               dt=dt, solver_steps_per_timestep=solver_steps_per_timestep,),
    'CoupledOscillatorInit': dict(k1 = 1.712, k2 = 0.851, kc = 0.15, gamma=0.043,
                                  dt=dt, solver_steps_per_timestep=solver_steps_per_timestep,),
    'Arbitrary2dPot' : dict(potential='lambda X: 0.3 * (X[0]**2 + X[1]**2) + 0.8 * jnp.sin(6*jnp.sqrt(X[0]**2 + X[1]**2))',
                            min_random=-3., max_random=3., give_2d_info=True, 
                            dt=dt, solver_steps_per_timestep=solver_steps_per_timestep,),
    'ParticleIn2DGravity': dict(give_2d_info=True, mass = 2.3, xp0 = -0.7, xp1 = 0.2, 
                                dt=dt, solver_steps_per_timestep=solver_steps_per_timestep,),
    'MexicanHat': dict(give_2d_info=True, a = 2.0, b = 0.7, gamma=0.2, 
                        dt=dt, solver_steps_per_timestep=solver_steps_per_timestep,),
    'TwoParticlesIn2DGravity': dict(mass1=8.123, mass2=0.781,
                                   dt=dt, solver_steps_per_timestep=solver_steps_per_timestep,),
    'ThreeParticlesIn2DGravity': dict(mass1=1.3, mass2=9.0, mass3=0.2,
                                     dt=dt, solver_steps_per_timestep=solver_steps_per_timestep,),

    'ThreeCoupledOscillatorInit': dict(k1=1.0, k2=1.5, k3=0.5, kc12=0.2, kc13=0.3, kc23=0.4, gamma=0.162,
                                       dt=dt, solver_steps_per_timestep=solver_steps_per_timestep,),
    
    'HiddenOscillators': dict(N=2, ks = [1.1, 1.54], kcs = [1.2, 0.], 
                              init_pos=[0.4,], init_vel=[-0.2,], hide_N=True, 
                              dt=dt, solver_steps_per_timestep=solver_steps_per_timestep,),

    'NBodyPairwisePotential': dict(N=10, potential_str = 'lambda r: - 0.8 * jnp.exp(-1.3*r)', dt=dt, solver_steps_per_timestep=solver_steps_per_timestep,),

    # 'HiddenOscillators': dict(N=3, 
    #                     ks = jax.random.uniform(jax.random.PRNGKey(3), shape=(3,), minval=0.5, maxval=2.0).tolist(), 
    #                     kcs = jax.random.uniform(jax.random.PRNGKey(2), shape=(3,), minval=0.1, maxval=2.0).tolist(), 
    #                     init_pos=jax.random.uniform(jax.random.PRNGKey(1), shape=(3-1,), minval=-1.0, maxval=1.0).tolist(), 
    #                     init_vel=jax.random.uniform(jax.random.PRNGKey(0), shape=(3-1,), minval=-1.0, maxval=1.0).tolist(),
    #                     hide_N=True, dt=dt, solver_steps_per_timestep=solver_steps_per_timestep, T = 100),

    'HiddenNBody2DGravity': dict(N=2, masses = [0.7, 1.8,], hidden_init = [0.5, 0.5, -0.5, 0.], hide_N=True, 
                                 dt=dt, solver_steps_per_timestep=solver_steps_per_timestep,),
    }

simulator_init=simulators_initials[simulator] 
tool_use_path = os.path.join(github_repo_path, 'examples')

metric_key = 'R2' # quality metric used by reult tool, used for automatic evaluation once experiments finiish
plot_tool="plot_single_1d" # function in analyzewr.py used to plot the initial experiments
random_exp_func='observe_random_evolution' # function used to generate initial experiments, should be function of simulator object, only used if default_n_initial_exps > 0
system_prompts_file="prompts/system_prompts.md" # file containing the system prompts, as subpath of github_repo_path
system_prompt = 'reduced_gpt5_opt'  # name of system prompt in system_prompts_file
intermediate_prompts_file='prompts/intermediate_prompts.md' # file containing the intermediate prompts, as subpath of github_repo_path
intermediate_prompt= 'no_separate_tools_ask_answer'#'no_separate_tools'#'very_short'#'default' # intermediate prompt key
tasks_file="tasks/tasks.md" # file containing the tasks for the agent, as subpath of github_repo_path
simulators_file="simulators/equations_of_motion" # file containing the simulators, as subpath of github_repo_path
simulator_tools=['observe_evolution', 'observe_multiple_evolutions'] # tools that the experimental simulator provides to the agent
analyzer_file="analysis/analysis_eom" # file containing the analyzer tools, as subpath of github_repo_path
analyzer_tools="coding" # analysis toolbox specified in analyzer_file
save_to_file_agent=os.path.join(result_folder, add_to_agent_name)
separate_tool_calls = False # whether to force separate tool calls and verbal responses, or allow the agent to combine them in one step

# for tool agent
run_kwargs = dict(
            final_response_key = "", # stop exploration if agent uses this key in the response, not used
            allow_multiple_tool_calls = True, # allow the agent to call multiple tools in one step
            keep_only_one_image = False, # keep only one image in the response, for limited models, not used
            max_tries_for_correct_tool_call = 5, # maximum number of tries to get a correct tool call from the agent
            max_steps = max_steps, max_tools = max_tools # maximum number of steps and tools the agent can use in one run
            )


def one_run(i):
    print(f"Running repeat {i+1} of {repeats}...")
    try:
        import copy
        simulator_init_run = copy.deepcopy(simulator_init)

        result,messages,model,simulator_object,_fields = runner.run(
            base_dir=tool_use_path,
            tasks_file=tasks_file,
            task=task,
            simulators_file=simulators_file,
            simulator=simulator,
            simulator_init=simulator_init_run,
            simulator_tools=simulator_tools,
            analyzer_file=analyzer_file,
            analyzer_tools=analyzer_tools,
            system_prompts_file=system_prompts_file,
            system_prompt=system_prompt,
            intermediate_prompts_file=intermediate_prompts_file,
            intermediate_prompt=intermediate_prompt,
            save_to_file=save_to_file_agent,
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            run_kwargs=run_kwargs,
            random_exp_func=random_exp_func,
            plot_tool=plot_tool,
            n_exps_to_run=n_initial_exps,
            use_tools=True,  # Use tools in the agent run
            summarize_at_end=True,  # Summarize the results at the end of the run
            replace_message_function=None,
            separate_tool_calls=separate_tool_calls,
            response_api = True, # if True, use the response API instead of chat completions
            reasoning_level = "high", # can be 'low', 'medium', 'high'
            verbosity = "high", # can be 'low', 'medium', 'high'
            reasoning_summary = "auto", # provide summary of internal reasoning process
            timeout=30, # timeout for each tool call
        )
        return 1
    except Exception as e:
        print(f"An error occurred during the run {i}: {e}")
        return 0

if __name__ == "__main__":
    r = 0
    i = 0
    max_parallel = 5 # maximum number of parallel processes, adjust based on your system and rate limits
    max_tries = 2 # maximum number of iterations to start new processes if not all runs are completed
    max_processes = min([max_parallel, repeats])
    while r < repeats and i < max_tries:
        print(f"Starting run ...")
        n_processes = max_processes - r  # Number of parallel processes to run
        with multiprocessing.Pool(processes=n_processes) as p:
            results = p.map(one_run, range(n_processes))
        r = sum(results) + r  # Update the count of completed runs
        i += 1
        print(f"ONE ITERATION DONE, r={r}, i={i}." )

 
    analyze_folder(saved_path=result_folder,
                title=task + ' ' + simulator,
                metric_key=metric_key,
                save_fig=True,
                contained_in_name=[add_to_agent_name,],)

    os.system("say 'experiments complete'")
