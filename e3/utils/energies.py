import oxpy
import os


def compute_all_energies(filepath, topology_filepath):
    parent_dir = os.path.dirname(filepath)
    energies = []
    stacking_energies = []
    os.chdir(parent_dir)
    with oxpy.Context():
        
        def compute_single_conf_energies(n_particles):
            e_txt = backend.config_info().get_observable_by_id("my_obs").get_output_string(backend.config_info().current_step).strip().split('\n')
            names = ['backbone', 'bonded_excluded_volume', 'stacking', 'nonbonded_excluded_volume', 'hydrogen_bonding',
                              'cross_stacking', 'coaxial_stacking', 'debye_huckel', 'total']
            energies = np.zeros((len(names)))
            for e in e_txt:
                if not e[0] == '#':
                    e = e.split(' ')
                    p = int(e[0])
                    q = int(e[1])
                    
            return energies
        
        inp = oxpy.InputFile()
        inp["analysis_data_output_1"] = '{ \n name = stdout \n print_every = 1e10 \n col_1 = { \n id = my_obs \n type = pair_energy \n } \n }'
        inp.init_from_filename(sim.sim_files.input.as_posix())
        backend = oxpy.analysis.AnalysisBackend(inp)
        while backend.read_next_configuration():        
            # print(backend.config_info().current_step)
            energies.append(compute_single_conf_energies(indexes))
        del backend    
        
    return energies