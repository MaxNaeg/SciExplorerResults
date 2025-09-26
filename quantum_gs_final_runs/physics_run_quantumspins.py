import sciexplorer.runner as runner
import jax.numpy as jnp

system_prompt="reduced_gpt5_opt"
intermediate_prompt="no_separate_tools_ask_answer"
max_steps=100
max_tools=200

models={
"topological_Ising_field_A_topocoupling_B":
dict(hamiltonian_code=
"""
H=0*Sz[0]
for j in range(N-2):
    H+=B * (Sz[j] @ Sx[j+1] @ Sz[j+2])
for j in range(N-1):
    H-=Sz[j] @ Sz[j+1]
for j in range(N):
    H-=A * Sx[j]
""", N=10, ground_state_experiments_with_params_variable_N=True, 
additional_description="""The system has two tuneable parameters, B and A, that you can set. You must also
set the integer number N of spins in the experimental system.""",
params_dict= {'A': (-1., 1.), 'B': (-1., 1.)}, 
),

"topological_Ising_field_A_topocoupling_B_fixed_N":
dict(hamiltonian_code=
"""
H=0*Sz[0]
for j in range(N-2):
    H+=B * (Sz[j] @ Sx[j+1] @ Sz[j+2])
for j in range(N-1):
    H-=Sz[j] @ Sz[j+1]
for j in range(N):
    H-=A * Sx[j]
""", N=10, ground_state_experiments_with_params=True, 
additional_description="""The system has two tuneable parameters, B and A, that you must set before running an experiment.""",
params_dict= {'A': (-1., 1.), 'B': (-1., 1.)},),
    
"topological_Ising_topocoupling_A":
dict(hamiltonian_code=
"""
H=0*Sz[0]
for j in range(N-2):
    H+=A * (Sz[j] @ Sx[j+1] @ Sz[j+2])
for j in range(N-1):
    H-=Sz[j] @ Sz[j+1]
for j in range(N):
    H-=Sx[j]
""", N=10, ground_state_experiments_with_params_variable_N=True, 
additional_description="""The system has one tuneable parameter A that you can set. You must also
set the integer number N of spins in the experimental system.""",
params_dict= {'A': (-1., 1.)}
),

"topological_Ising_topocoupling_A_fixed_N":
dict(hamiltonian_code=
"""
H=0*Sz[0]
N=10
for j in range(N-2):
    H+=A * (Sz[j] @ Sx[j+1] @ Sz[j+2])
for j in range(N-1):
    H-=Sz[j] @ Sz[j+1]
for j in range(N):
    H-=Sx[j]
""", N=10, ground_state_experiments_with_params=True, 
additional_description="""The system has one tuneable parameter A that you must set.""",
params_dict= {'A': (-1., 1.)},),

"transverse_Ising_coupling_A":
dict(hamiltonian_code=
"""
H=0*Sz[0]
for j in range(N-1):
    H+=A*Sz[j] @ Sz[j+1]
for j in range(N):
    H-=Sx[j]
""", N=10, ground_state_experiments_with_params_variable_N=True, 
additional_description="""The system has one tuneable parameter A that you can set. You must also
set the integer number N of spins in the experimental system.""",
params_dict= {'A': (-1., 1.)}
),

"TFIM_coupling_A_fixed_N":
dict(hamiltonian_code=
"""
N=10
H=0*Sz[0]
for j in range(N-1):
    H+=A*Sz[j] @ Sz[j+1]
for j in range(N):
    H-=0.6*Sx[j]
""", N=10, ground_state_experiments_with_params=True, 
additional_description="""The system has one tuneable parameter A that you can set.""",
params_dict= {'A': (-1., 1.)},),

"TFIM_coupling_A_variable_N":
dict(hamiltonian_code=
"""
H=0*Sz[0]
for j in range(N-1):
    H+=A*Sz[j] @ Sz[j+1]
for j in range(N):
    H-=0.6*Sx[j]
""", N=10, ground_state_experiments_with_params_variable_N=True, 
additional_description="""The system has one tuneable parameter A that you can set. You must also
set the integer number N of spins in the experimental system.""",
params_dict= {'A': (-1., 1.)},),

"TFIM_coupling_A_field_B":
dict(hamiltonian_code=
"""
N=10
H=0*Sz[0]
for j in range(N-1):
    H+=A*Sz[j] @ Sz[j+1]
for j in range(N):
    H-=2*B*Sx[j]
""", N=10, ground_state_experiments_with_params=True, 
additional_description="""The system has two tuneable parameters A and B that you must set before running an experiment.""",
params_dict= {'A': (-1., 1.), 'B': (-1., 1.)}),

"TFIM_coupling_A_x_field_B_fixed_y_field":
dict(hamiltonian_code=
"""
N=10
H=0*Sz[0]
for j in range(N-1):
    H+=A*Sz[j] @ Sz[j+1]
for j in range(N):
    H-=2*B*Sx[j]+Sy[j]
""", N=10, ground_state_experiments_with_params=True, 
additional_description="""The system has two tuneable parameters A and B that you must set before running an experiment.""",
params_dict= {'A': (-1., 1.), 'B': (-1., 1.)}),
    
"Heisenberg_field_A":
dict(hamiltonian_code=
"""
H=0*Sz[0]
for j in range(N-1):
    H+=0.5*(Sx[j] @ Sx[j+1] + Sy[j] @ Sy[j+1] + Sz[j] @ Sz[j+1])
for j in range(N):
    H-=A*Sx[j]
""", N=10, ground_state_experiments_with_params_variable_N=True, 
additional_description="""The system has one tuneable parameter A that you can set. You must also
set the integer number N of spins in the experimental system.""",
params_dict= {'A': (-1., 1.)}),

"2D_lattice":
dict(hamiltonian_code=
"""
M=3
N=M**2

def idx(jx,jy,M):
    return (jx+M)%M + ((jy+M)%M)*M

H=0*Sz[0]
for jx in range(M):
    for jy in range(M):
        for dx,dy in [(-1,0),(+1,0),(0,+1),(0,-1)]:
            H+=0.5*(Sx[idx(jx,jy,M)] @ Sx[idx(jx+dx,jy+dy,M)] + Sy[idx(jx,jy,M)] @ Sy[idx(jx+dx,jy+dy,M)] + Sz[idx(jx,jy,M)] @ Sz[idx(jx+dx,jy+dy,M)])
for j in range(N):
    H-=2*Sx[j]
""", N=9, ground_state_experiments=True, 
additional_description="""The system is a 2D system of spins on a 3x3 grid, with spin index j=jx+jy*M, for M=3."""),

"2D_lattice_Heisenberg_param":
dict(hamiltonian_code=
"""
M=3
N=M**2

def idx(jx,jy,M):
    return (jx+M)%M + ((jy+M)%M)*M

H=0*Sz[0]
for jx in range(M):
    for jy in range(M):
        for dx,dy in [(-1,0),(+1,0),(0,+1),(0,-1)]:
            H+=A*(Sx[idx(jx,jy,M)] @ Sx[idx(jx+dx,jy+dy,M)] + Sy[idx(jx,jy,M)] @ Sy[idx(jx+dx,jy+dy,M)] + Sz[idx(jx,jy,M)] @ Sz[idx(jx+dx,jy+dy,M)])
for j in range(N):
    H-=2*Sx[j]
""", N=9, ground_state_experiments_with_params=True, 
additional_description="""The system is a 2D system of spins on a 3x3 grid, with spin index j=jx+jy*M, for M=3. There is one tuneable parameter A, which you have to set before running an experiment.""",
params_dict= {'A': (-1., 1.)}),

"2D_lattice_Heisenberg_two_params":
dict(hamiltonian_code=
"""
M=3
N=M**2

def idx(jx,jy,M):
    return (jx+M)%M + ((jy+M)%M)*M

H=0*Sz[0]
for jx in range(M):
    for jy in range(M):
        for dx,dy in [(-1,0),(+1,0),(0,+1),(0,-1)]:
            H+=A*(Sx[idx(jx,jy,M)] @ Sx[idx(jx+dx,jy+dy,M)] + Sy[idx(jx,jy,M)] @ Sy[idx(jx+dx,jy+dy,M)] + Sz[idx(jx,jy,M)] @ Sz[idx(jx+dx,jy+dy,M)])
for j in range(N):
    H-=2*B*Sx[j]
""", N=9, ground_state_experiments_with_params=True, 
additional_description="""The system is a 2D system of spins on a 3x3 grid, with spin index j=jx+jy*M, for M=3. There are two tuneable parameters A and B, which you have to set before running an experiment.""",
params_dict= {'A': (-1., 1.), 'B': (-1., 1.)}),

"TFIM":
dict(hamiltonian_code=
"""
H=0*Sz[0]
N=10
for j in range(N-1):
    H+=1.5*Sz[j] @ Sz[j+1]
for j in range(N):
    H-=0.6*Sx[j]
""", N=10, ground_state_experiments=True, 
additional_description=""),

"topological_Ising":
dict(hamiltonian_code=
"""
N=10
H=0*Sz[0]
for j in range(N-2):
    H+=0.5 * (Sz[j] @ Sx[j+1] @ Sz[j+2])
for j in range(N-1):
    H-=Sz[j] @ Sz[j+1]
for j in range(N):
    H-=0.3*Sx[j]
""", N=10, ground_state_experiments=True, 
additional_description=""),

"dynamics_Heisenberg":
dict(hamiltonian_code=
"""
N=10
H=0*Sz[0]
for j in range(N-1):
    H+=0.5*(Sx[j] @ Sx[j+1] + Sy[j] @ Sy[j+1] + Sz[j] @ Sz[j+1])
for j in range(N):
    H-=1.5*Sx[j]
""", N=10, 
additional_description=""),

"dynamics_Heisenberg_field_A":
dict(hamiltonian_code=
"""
N=10
H=0*Sz[0]
for j in range(N-1):
    H+=0.5*(Sx[j] @ Sx[j+1] + Sy[j] @ Sy[j+1] + Sz[j] @ Sz[j+1])
for j in range(N):
    H-=A*Sx[j]
""", N=10, dynamics_with_params=True,
additional_description="""The system has one tuneable parameter A that you can set.""",
params_dict= {'A': (-1., 1.)}),

"dynamics_Heisenberg_2spins_observed_and_controllable_field_A":
dict(hamiltonian_code=
"""
N=10
H=0*Sz[0]
for j in range(N-1):
    H+=0.5*(Sx[j] @ Sx[j+1] + Sy[j] @ Sy[j+1] + Sz[j] @ Sz[j+1])
for j in range(N):
    H-=A*Sx[j]
""", N=10, observed=[0,1], default_Bloch_vectors=jnp.array(10*[[0,0,-1.0]]), dynamics_with_params=True,
additional_description="""The system has one tuneable parameter A that you can set.""",
params_dict= {'A': (-1., 1.)}),


"TFIM_TWO_VAR_PARAMS_VAR_N":
dict(hamiltonian_code=
"""
H=0*Sz[0]
for j in range(N-1):
    H+=A*Sz[j] @ Sz[j+1]
for j in range(N):
    H-=2*B*Sx[j]+Sy[j]
""", N=10, ground_state_experiments_with_params_variable_N=True, 
additional_description="""The system has two tuneable parameters A and B that you must set before running an experiment. You must also set the integer number N of spins in the experimental system.""",
params_dict= {'A': (-1., 1.), 'B': (-1., 1.)}),

# SELECTED FOR PRODUCTION RUNS (including statistics)
"GROUND_TFIM":
dict(hamiltonian_code=
"""
H=0*Sz[0]
N=10
for j in range(N-1):
    H+=1.5*Sz[j] @ Sz[j+1]
for j in range(N):
    H-=0.6*Sx[j]
""", N=10, ground_state_experiments=True, 
additional_description=""),

"GROUND_TFIM_VAR_COUPLING":
dict(hamiltonian_code=
"""
N=10
H=0*Sz[0]
for j in range(N-1):
    H+=A*Sz[j] @ Sz[j+1]
for j in range(N):
    H-=0.6*Sx[j]
""", N=10, ground_state_experiments_with_params=True, 
additional_description="""The system has one tuneable parameter A that you must set.""",
params_dict= {'A': (-1., 1.)},),

"GROUND_TFIM_VAR_COUPLING_VAR_N":
dict(hamiltonian_code=
"""
H=0*Sz[0]
for j in range(N-1):
    H+=A*Sz[j] @ Sz[j+1]
for j in range(N):
    H-=0.6*Sx[j]
""", N=10, ground_state_experiments_with_params_variable_N=True, 
additional_description="""The system has one tuneable parameter A that you must set. You must also
set the integer number N of spins in the experimental system.""",
params_dict= {'A': (-1., 1.)},),

"GROUND_2D_HEISENBERG":
dict(hamiltonian_code=
"""
M=3
N=M**2

def idx(jx,jy,M):
    return (jx+M)%M + ((jy+M)%M)*M

H=0*Sz[0]
for jx in range(M):
    for jy in range(M):
        for dx,dy in [(-1,0),(+1,0),(0,+1),(0,-1)]:
            H+=0.5*(Sx[idx(jx,jy,M)] @ Sx[idx(jx+dx,jy+dy,M)] + Sy[idx(jx,jy,M)] @ Sy[idx(jx+dx,jy+dy,M)] + Sz[idx(jx,jy,M)] @ Sz[idx(jx+dx,jy+dy,M)])
for j in range(N):
    H-=2*Sx[j]
""", N=9, ground_state_experiments=True, 
additional_description="""The system is a 2D system of spins on a 3x3 grid, with spin index j=jx+jy*M, for M=3.""",),

"GROUND_2D_HEISENBERG_VAR":
dict(hamiltonian_code=
"""
M=3
N=M**2

def idx(jx,jy,M):
    return (jx+M)%M + ((jy+M)%M)*M

H=0*Sz[0]
for jx in range(M):
    for jy in range(M):
        for dx,dy in [(-1,0),(+1,0),(0,+1),(0,-1)]:
            H+=A*(Sx[idx(jx,jy,M)] @ Sx[idx(jx+dx,jy+dy,M)] + Sy[idx(jx,jy,M)] @ Sy[idx(jx+dx,jy+dy,M)] + Sz[idx(jx,jy,M)] @ Sz[idx(jx+dx,jy+dy,M)])
for j in range(N):
    H-=2*Sx[j]
""", N=9, ground_state_experiments_with_params=True, 
additional_description="""The system is a 2D system of spins on a 3x3 grid, with spin index j=jx+jy*M, for M=3. There is one tuneable parameter A, which you have to set before running an experiment.""",
params_dict= {'A': (-1., 1.)}),

"GROUND_2D_HEISENBERG_TWO_VAR":
dict(hamiltonian_code=
"""
M=3
N=M**2

def idx(jx,jy,M):
    return (jx+M)%M + ((jy+M)%M)*M

H=0*Sz[0]
for jx in range(M):
    for jy in range(M):
        for dx,dy in [(-1,0),(+1,0),(0,+1),(0,-1)]:
            H+=A*(Sx[idx(jx,jy,M)] @ Sx[idx(jx+dx,jy+dy,M)] + Sy[idx(jx,jy,M)] @ Sy[idx(jx+dx,jy+dy,M)] + Sz[idx(jx,jy,M)] @ Sz[idx(jx+dx,jy+dy,M)])
for j in range(N):
    H-=2*B*Sx[j]
""", N=9, ground_state_experiments_with_params=True, 
additional_description="""The system is a 2D system of spins on a 3x3 grid, with spin index j=jx+jy*M, for M=3. There are two tuneable parameters A and B, which you have to set before running an experiment.""",
params_dict= {'A': (-1., 1.), 'B': (-1., 1.)}),

"GROUND_TOPO_ISING":
dict(hamiltonian_code=
"""
N=10
H=0*Sz[0]
for j in range(N-2):
    H+=0.5 * (Sz[j] @ Sx[j+1] @ Sz[j+2])
for j in range(N-1):
    H-=Sz[j] @ Sz[j+1]
for j in range(N):
    H-=0.3*Sx[j]
""", N=10, ground_state_experiments=True, 
additional_description=""),

"GROUND_TOPO_ISING_VAR_COUPLING":
dict(hamiltonian_code=
"""
H=0*Sz[0]
N=10
for j in range(N-2):
    H+=A * (Sz[j] @ Sx[j+1] @ Sz[j+2])
for j in range(N-1):
    H-=Sz[j] @ Sz[j+1]
for j in range(N):
    H-=Sx[j]
""", N=10, ground_state_experiments_with_params=True, 
additional_description="""The system has one tuneable parameter A that you must set.""",
params_dict= {'A': (-1., 1.)}),

"GROUND_TOPO_ISING_VAR_COUPLING_VAR_N":
dict(hamiltonian_code=
"""
H=0*Sz[0]
for j in range(N-2):
    H+=A * (Sz[j] @ Sx[j+1] @ Sz[j+2])
for j in range(N-1):
    H-=Sz[j] @ Sz[j+1]
for j in range(N):
    H-=Sx[j]
""", N=10, ground_state_experiments_with_params_variable_N=True, 
additional_description="""The system has one tuneable parameter A that you must set. You must also
set the integer number N of spins in the experimental system.""",
params_dict= {'A': (-1., 1.)}),


"GROUND_ARBITRARY":
dict(hamiltonian_code=
"""
H=0*Sz[0]
N=10
for j in range(N-1):
    H+=1.5*(Sx[j] @ Sz[j+1])
    H-=0.7*(Sy[j] @ Sx[j+1])
for j in range(N):
    H-=0.6*Sx[j]
    H+=0.4*Sy[j]
""", N=10, ground_state_experiments=True, 
additional_description=""),

# ADDITIONS FOR PHASE DIAGRAM EXPLORATION

"GROUND_TOPO_ISING_VAR_TOPO_VAR_ZZ_VAR_N":
dict(hamiltonian_code=
"""
H=0*Sz[0]
for j in range(N-2):
    H+= A * (Sz[j] @ Sx[j+1] @ Sz[j+2])
for j in range(N-1):
    H+= B * (Sz[j] @ Sz[j+1])
for j in range(N):
    H-= Sx[j]
""", N=10, ground_state_experiments_with_params_variable_N=True, 
additional_description="""The system has two tuneable parameters A and B that you must set. You must also
set the integer number N of spins in the experimental system.""",
params_dict= {'A': (-2., 2.), 'B': (-2.,2.)}),


"GROUND_XZ_TWO_VARS_N":
dict(hamiltonian_code=
"""
H=0*Sz[0]
for j in range(N-1):
    H+= A * (Sx[j] @ Sx[j+1])
for j in range(N-1):
    H+= B * (Sz[j] @ Sz[j+1])
for j in range(N):
    H-= Sy[j]
""", N=10, ground_state_experiments_with_params_variable_N=True, 
additional_description="""The system has two tuneable parameters A and B that you must set. You must also
set the integer number N of spins in the experimental system.""",
params_dict= {'A': (-2., 2.), 'B': (-2.,2.)}),

    
# DYNAMICS

"DYNAMICS_HEISENBERG":
dict(hamiltonian_code=
"""
N=10
H=0*Sz[0]
for j in range(N-1):
    H+=0.5*(Sx[j] @ Sx[j+1] + Sy[j] @ Sy[j+1] + Sz[j] @ Sz[j+1])
for j in range(N):
    H-=1.5*Sx[j]
""", N=10, 
additional_description=""),

"DYNAMICS_HEISENBERG_VAR_FIELD":
dict(hamiltonian_code=
"""
N=10
H=0*Sz[0]
for j in range(N-1):
    H+=0.5*(Sx[j] @ Sx[j+1] + Sy[j] @ Sy[j+1] + Sz[j] @ Sz[j+1])
for j in range(N):
    H-=A*Sx[j]
""", N=10, dynamics_with_params=True,
additional_description="""The system has one tuneable parameter A that you must set.""",
params_dict= {'A': (-1., 1.)}),

"DYNAMICS_ARBITRARY_THREE_SPINS":
dict(hamiltonian_code=
"""
H=Sx[0] @ Sz[1] + 0.5*(Sy[0] @ Sx[2]) - 0.7 * Sy[1] + 0.3 * Sy[2] - 0.8 * Sx[1] @ Sy[2]
""", N=3, 
additional_description=""),

"DYNAMICS_ARBITRARY_THREE_SPINS_TWO_SPINS_ACCESS":
dict(hamiltonian_code=
"""
H=Sx[0] @ Sz[1] + 0.5*(Sy[0] @ Sx[2]) - 0.7 * Sy[1] + 0.3 * Sy[2] - 0.8 * Sx[1] @ Sy[2]
""", N=3, observed=[0,1], default_Bloch_vectors=jnp.array(3*[[0,0,-1.0]]), 
additional_description=""),

"DYNAMICS_ARBITRARY_THREE_SPINS_TWO_SPINS_ACCESS_VAR_FIELD":
dict(hamiltonian_code=
"""
H=Sx[0] @ Sz[1] + 0.5*(Sy[0] @ Sx[2]) - A * Sy[1] + 0.3 * Sy[2] - 0.8 * Sx[1] @ Sy[2]
""", N=3, observed=[0,1], default_Bloch_vectors=jnp.array(3*[[0,0,-1.0]]), dynamics_with_params=True,
additional_description="The system has one tuneable parameter A that you must set.",
params_dict= {'A': (-1., 1.)}),
    
"DYNAMICS_HEISENBERG_VAR_FIELD_TWO_SPINS_ACCESS":
dict(hamiltonian_code=
"""
N=10
H=0*Sz[0]
for j in range(N-1):
    H+=0.5*(Sx[j] @ Sx[j+1] + Sy[j] @ Sy[j+1] + Sz[j] @ Sz[j+1])
for j in range(N):
    H-=A*Sx[j]
""", N=10, observed=[0,1], default_Bloch_vectors=jnp.array(10*[[0,0,-1.0]]), dynamics_with_params=True,
additional_description="""The system has one tuneable parameter A that you must set.""",
params_dict= {'A': (-1., 1.)}),

"DYNAMICS_TFIM_VAR_COUPLING":
dict(hamiltonian_code=
"""
N=10
H=0*Sz[0]
for j in range(N-1):
    H+=A*Sz[j] @ Sz[j+1]
for j in range(N):
    H-=Sx[j]
""", N=10, dynamics_with_params=True, 
additional_description="""The system has one tuneable parameter A that you must set.""",
params_dict= {'A': (-1., 1.)}),

"DYNAMICS_TFIM_VAR_COUPLING_TWO_SPINS_ACCESS":
dict(hamiltonian_code=
"""
N=10
H=0*Sz[0]
for j in range(N-1):
    H+=A*Sz[j] @ Sz[j+1]
for j in range(N):
    H-=Sx[j]
""", N=10, observed=[0,1], default_Bloch_vectors=jnp.array(10*[[0,0,-1.0]]), dynamics_with_params=True, 
additional_description="""The system has one tuneable parameter A that you must set.""",
params_dict= {'A': (-1., 1.)}),
}

def quantum_spin_run(llm,PATH,PATH_RESULTS,task,init,
                     analyzer_tools,select_system_prompt=None):    
    if select_system_prompt is None:
        selected_system_prompt=system_prompt
    else:
        selected_system_prompt=select_system_prompt
    
    return runner.run(
    base_dir=PATH,
    tasks_file="tasks/tasks.md",
    task=task,
    simulators_file="simulators/quantum_spins",
    simulator="QuantumSpinSystem",
    simulator_init=init,
    simulator_tools="default",
    analyzer_file="analysis/analysis_quantum_spins",
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

def ground_state(llm,PATH,PATH_RESULTS,task,init,select_system_prompt=None):
    return quantum_spin_run(llm,PATH,PATH_RESULTS,task,init,"groundstate",
                            select_system_prompt=select_system_prompt)
                            

def dynamics(llm,PATH,PATH_RESULTS,task,init,select_system_prompt=None):
    return quantum_spin_run(llm,PATH,PATH_RESULTS,task,init,"default",
                           select_system_prompt=select_system_prompt)

def dynamics_model_discovery(llm,PATH,PATH_RESULTS,choice,select_system_prompt=None):
    return dynamics(llm,PATH,PATH_RESULTS,"model_discovery_Hamiltonian",
                                            models[choice],
                    select_system_prompt=select_system_prompt)
    
def ground_state_model_discovery(llm,PATH,PATH_RESULTS,choice,select_system_prompt=None):
    return ground_state(llm,PATH,PATH_RESULTS,"model_discovery_Hamiltonian",
                                            models[choice],
                    select_system_prompt=select_system_prompt)

def ground_state_phase_diagram(llm,PATH,PATH_RESULTS,choice,select_system_prompt=None):
    return ground_state(llm,PATH,PATH_RESULTS,"phase_diagram",
                                            models[choice],
                       select_system_prompt=select_system_prompt)
