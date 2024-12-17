import oxpy
import os
import numpy as np

def compute_all_energies(sim):
    energies = []
    os.chdir(sim.sim_dir)
    with oxpy.Context():
        
        def compute_single_conf_energies():
            n_particles = backend.config_info().N()
            e_txt = backend.config_info().get_observable_by_id("my_obs").get_output_string(backend.config_info().current_step).strip().split('\n')
            names = ['backbone', 'bonded_excluded_volume', 'stacking', 'nonbonded_excluded_volume', 'hydrogen_bonding',
                              'cross_stacking', 'coaxial_stacking', 'debye_huckel', 'total']
            energies = np.zeros((n_particles , n_particles, len(names)))
            for e in e_txt:
                if not e[0] == '#':
                    energy = e.split(' ')
                    p_0_idx = int(energy[0])
                    p_1_idx = int(energy[1])
                    l = np.array(list(map(float, energy[2:])))
                    
                    energies[p_0_idx, p_1_idx] = l
                    
            return energies
        
        inp = oxpy.InputFile()
        inp["analysis_data_output_1"] = '{ \n name = stdout \n print_every = 1e10 \n col_1 = { \n id = my_obs \n type = pair_energy \n } \n }'
        inp.init_from_filename(sim.sim_files.input.as_posix())
        backend = oxpy.analysis.AnalysisBackend(inp)


        while backend.read_next_configuration():        
            energies.append(compute_single_conf_energies())
        del backend    
        
    return np.array(energies)