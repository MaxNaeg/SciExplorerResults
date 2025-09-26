import sciexplorer.runner as runner
import jax.numpy as jnp

system_prompt="reduced_gpt5_opt"
intermediate_prompt="no_separate_tools_ask_answer"
max_steps=100
max_tools=200

def fields_run(llm,PATH,PATH_RESULTS,task,init,
                     analyzer_tools,select_system_prompt=None):    
    if select_system_prompt is None:
        selected_system_prompt=system_prompt
    else:
        selected_system_prompt=select_system_prompt
    
    return runner.run(
    base_dir=PATH,
    tasks_file="tasks/tasks.md",
    task=task,
    simulators_file="simulators/wave_sim_v2",
    simulator="WaveSimulator",
    simulator_init=init,
    simulator_tools="field",
    analyzer_file="analysis/wave_analysis_tools",
    analyzer_tools=analyzer_tools,
    system_prompts_file="prompts/system_prompts.md",
    system_prompt=selected_system_prompt,
    intermediate_prompts_file="prompts/intermediate_prompts.md",
    intermediate_prompt=intermediate_prompt,
    save_to_file=PATH_RESULTS+"/run",
    summarize_at_end=True,
    api_key=llm['api_key'],
    base_url=llm['base_url'],
    model_name=llm['model_name'],
    run_kwargs = dict(
                final_response_key = "",
                allow_multiple_tool_calls = True,
                keep_only_one_image = False,
                max_tries_for_correct_tool_call = 2,
                max_steps = max_steps, max_tools = max_tools),
    separate_tool_calls = False,
    response_api = True,
    reasoning_level = "high",
    verbosity = "high",
    reasoning_summary = "auto"
    )

def field_dynamics(llm,PATH,PATH_RESULTS,task,rhs_choice,select_system_prompt=None):
    init=dict(x_max=5, N_x=100, t_max=20., N_t=200, rhs_choice=rhs_choice, 
                        rhs_code=None, description_type="FIELD")
    return fields_run(llm,PATH,PATH_RESULTS,task,init,"field",
                            select_system_prompt=select_system_prompt)

