import os
import numpy as np
from oxDNA_analysis_tools.UTILS.RyeReader import describe, get_confs, inbox
from ipy_oxdna.oxdna_simulation import Simulation
from utils.energies import compute_all_energies

def parse_dna_origami_data(filepath, topology_filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"file {filepath} does not exist.")
    if not os.path.exists(topology_filepath):
        raise FileNotFoundError(f"file {topology_filepath} does not exist.")

    #Parse the trajectory file
    top_info, traj_info = describe(topology_filepath, filepath)
    n_confs = traj_info.nconfs
    n_bases = top_info.nbases
    start_conf = 0
    # print(traj_info.incl_v)
    confs = get_confs(top_info, traj_info, start_conf, n_confs)
    confs = [inbox(conf, center=True) for conf in confs]
        
    trajectory_data = np.zeros((n_confs, n_bases, 15))
    with open(filepath, 'r') as f:
        
        p_idx = 0
        traj_idx = 0
        for line in f:
            
            stripped_line = line.strip()
            if stripped_line.startswith(('E', 't', 'b')):
                continue
            
            values = [float(x) for x in stripped_line.split()[:15]]
            trajectory_data[traj_idx, p_idx, :] = np.array(values)
            p_idx += 1
            
            if p_idx == n_bases:
                p_idx = 0
                traj_idx += 1
    
    pos = [conf.positions for conf in confs]
    pos = np.array(pos)
    trajectory_data[:, :, :3] = pos
    
    # Parse the topology file
    with open(topology_filepath, 'r') as f:
        topology_lines = f.readlines()

    # Extract number of nucleotides and strands from the first line
    first_line = topology_lines[0].strip()
    num_nucleotides, num_strands = map(int, first_line.split())

    topology_data = []
    base_encoding = {'A': 0, 'T': 3, 'C': 2, 'G': 1}
    for line in topology_lines[1:]:  # collect data from the second line
        stripped_line = line.strip()
        parts = stripped_line.split()
        if len(parts) >= 4:
            strand_index = int(parts[0])
            base = base_encoding[parts[1]]
            neighbor_3 = int(parts[2])
            neighbor_5 = int(parts[3])
            topology_data.append((strand_index, base, neighbor_3, neighbor_5))

    topology_data = np.array([topology_data for _ in range(n_confs)])
    
    parent_file_path = os.path.dirname(filepath)
    sim = Simulation(parent_file_path)
    energy_data = compute_all_energies(sim)
    # energy_data = np.zeros((n_confs, n_particles, 9))
    # print(energy_data.shape)
    # Combine the data
    combined_data = np.concatenate((trajectory_data, np.array(topology_data)), axis=2)

    return np.array(combined_data), energy_data

# Example usage
# combined_data, num_nucleotides, num_strands = parse_dna_origami_data('path/to/trajectory.dat', 'path/to/topology.dat')