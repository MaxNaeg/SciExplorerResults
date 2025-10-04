# Agentic Exploration of Physics Models

This notebook contains the results of the paper 'Agentic Exploration of Physics Models'.
To run the Python files and notebooks in this repository, you should first clone and install the [SciExplorer](https://github.com/MaxNaeg/SciExplorer.git) package.

## WARNING

The tools provided to the agent enable automatic execution of LLM generated Python code. While we did not observe the agent acting malicious in our experiments, these tools should best be run in a safe environment.

## Contents

- ablation_final_runs: contains results and run file
- mechanics_final_runs: contains results and run file
- quantum_dynamics_final_runs: contains results and run file
- quantum_gs_final_runs: contains results and run file
- wave_final_runs: contains results and run files

- summaries: contains the summaroes of example conversations included in the supplement
- wave_comp_true_exp: contains plots of true wave evolution and the agent's prediction for all experiments

The following notebooks are included to analyze the results:

- bar_plot_mechanics, bar_plot_waves, quantum_final_plots, and plot_time create the bar plots comntained in the paper
- ask_for_summary shows how to use an LLM to automatically generate the summarioes included in the supplement
- example_print_run shows how to print a complete exploration of the agent
- pick_better_conversation shows how to ask the agent to pick the best of multiple independet explorations of a system.
- print_final_Hamiltonians shows the true and predicted Hamiltonians for all experiments
- print_wave_results prints the true and predicted propagators for all experiments




