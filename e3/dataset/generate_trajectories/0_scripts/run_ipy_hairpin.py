path = os.path.abspath('/scratch/matthew/ipy_oxDNA/ipy_oxdna_examples/ssmRNA/coding_embedded')

conditions = ['20C_1M', '37C_012M']
systems = [sys for sys in os.listdir(path) if os.path.isdir(f'{path}/{sys}') for _ in conditions]

file_dir_list = [f'{path}/{system_name}/{conditions[idx % 2]}/eq_0' for idx, system_name in enumerate(systems)]
sim_dir_list  = [f'{path}/{system_name}/{conditions[idx % 2]}/prod' for idx, system_name in enumerate(systems)]


n_replicas = 1
replica_generator = GenerateReplicas()

replica_generator.multisystem_replica(
    systems,
    n_replicas, 
    file_dir_list,
    sim_dir_list
)

prod_sim_list = replica_generator.sim_list

prod_steps = 1e9
temperatures = ['20C', '37C']
molarites = ['1', '0.12']


prod_parameters_list = [
    {
    'interaction_type':'RNA2',
    'mismatch_repulsion': '1',
    
    'T':f'{temperatures[idx % 2]}',
    'salt_concentration':f'{molarites[idx % 2]}',
    
    'steps':f'{prod_steps}',
    'print_energy_every': f'1e5',
    'print_conf_interval':f'1e5',

    "dt": "0.003",
    
    } for idx in range(len(prod_sim_list))
]