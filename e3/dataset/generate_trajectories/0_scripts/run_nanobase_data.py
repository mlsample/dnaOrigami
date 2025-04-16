from ipy_oxdna.oxdna_simulation import SimulationManager, Simulation, Observable
from ipy_oxdna.generate_replicas import ReplicaGroup
import oxpy
import os
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import scipy.stats as stats

path = os.path.abspath('/scratch/matthew/project_files/dnaOrigami/e3/dataset/generate_trajectories/nanobase_data')
systems = [file_dir for file_dir in os.listdir(path) if os.path.isdir(f'{path}/{file_dir}')]

### Load 0_relax file_info
file_dir_list = [f'{path}/{system_name}' for system_name in systems]
sim_dir_list = [f'{path}/{system_name}/0_relax'  for system_name in systems]
n_replicas = 3
replica_generator = ReplicaGroup()
replica_info = [(systemname, file_dir, sim_dir) for systemname, file_dir, sim_dir in zip(systems, file_dir_list, sim_dir_list)]
replica_generator.multisystem_replica(
    replica_info,
    n_replicas, 
)
relax_sim_list = replica_generator.sim_list
relax_parameters_list = [
    {
    'max_backbone_force': '50',
    'max_backbone_force_far': '100',
    'steps': '5e2'
    } for _ in range(len(relax_sim_list))
]

### Load 1_relax file_info
file_dir_list = [sim.sim_dir for sim in relax_sim_list]
sim_dir_list = [f'{sim.file_dir}/1_relax/{sim.sim_dir.name}' for sim in relax_sim_list]
relax_1_sim_list = [Simulation(file_dir, sim_dir) for file_dir, sim_dir in zip(file_dir_list, sim_dir_list)]
relax_parameters_list = [
    {
    'max_backbone_force': '50000',
    'max_backbone_force_far': '100000',
    'steps': '5e2'
    } for _ in range(len(relax_1_sim_list))
]


### Load 0_eq file_info
file_dir_list = [sim.sim_dir for sim in relax_1_sim_list]
sim_dir_list = [f'{sim.sim_dir.parent.parent}/0_eq/{sim.sim_dir.name}' for sim in relax_1_sim_list]
eq_sim_list = [Simulation(file_dir, sim_dir) for file_dir, sim_dir in zip(file_dir_list, sim_dir_list)]
eq_parameters_list = [
    {
    'interaction_type':'DNA2',
    'T':'25C',
    'salt_concentration':'1',
    'steps':f'1e5',
    'print_energy_every': f'1e5',
    'print_conf_interval':f'1e5',
    "dt": "0.003",
    "max_density_multiplier":'10',
    'max_backbone_force': '50',
    'max_backbone_force_far': '100',
    } for _ in range(len(eq_sim_list))
]

### Load 1_eq file_info
file_dir_list = [sim.sim_dir for sim in eq_sim_list]
sim_dir_list = [f'{sim.sim_dir.parent.parent}/1_eq/{sim.sim_dir.name}' for sim in eq_sim_list]
eq_sim_list = [Simulation(file_dir, sim_dir) for file_dir, sim_dir in zip(file_dir_list, sim_dir_list)]
eq_parameters_list = [
    {
    'interaction_type':'DNA2',
    'T':'25C',
    'salt_concentration':'1',
    'steps':f'5e5',
    'print_energy_every': f'1e5',
    'print_conf_interval':f'1e5',
    "dt": "0.003",
    "max_density_multiplier":'10',
    'max_backbone_force': '500',
    'max_backbone_force_far': '1000',
    } for _ in range(len(eq_sim_list))
]

### Load 2_eq file_info
file_dir_list = [sim.sim_dir for sim in eq_sim_list]
sim_dir_list = [f'{sim.sim_dir.parent.parent}/2_eq/{sim.sim_dir.name}' for sim in eq_sim_list]
eq_sim_list = [Simulation(file_dir, sim_dir) for file_dir, sim_dir in zip(file_dir_list, sim_dir_list)]
eq_parameters_list = [
    {
    'interaction_type':'DNA2',
    'T':'25C',
    'salt_concentration':'1',
    'steps':f'1e8',
    'print_energy_every': f'5e5',
    'print_conf_interval':f'5e5',
    "dt": "0.003",
    "max_density_multiplier":'10',
    # 'max_backbone_force': '500',
    # 'max_backbone_force_far': '1000',
    } for _ in range(len(eq_sim_list))
]


### Run eq simulations
eq_sim_manager = SimulationManager(sleep_time=5)
for idx, sim in enumerate(eq_sim_list):
    sim.build(clean_build='force')
    sim.input.swap_default_input('cuda_MD')
    sim.input_file(eq_parameters_list[idx])
    sim.make_sequence_dependant()
    eq_sim_manager.queue_sim(sim)
    
eq_sim_manager.worker_manager()