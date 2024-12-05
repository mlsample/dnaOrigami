import os
import numpy as np
from oxDNA_analysis_tools.UTILS.RyeReader import describe, get_confs, inbox

def parse_dna_origami_data(filepath, topology_filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"file {filepath} does not exist.")
    if not os.path.exists(topology_filepath):
        raise FileNotFoundError(f"file {topology_filepath} does not exist.")

    #Parse the trajectory file
    top_info, traj_info = describe(topology_filepath, filepath)
    n_confs = traj_info.nconfs
    start_conf = 0
    confs = get_confs(top_info, traj_info, start_conf, n_confs)
    confs = [inbox(conf, center=True) for conf in confs]

    pos = [conf.positions for conf in confs]
    a1 = [conf.a1s for conf in confs]
    a3 = [conf.a3s for conf in confs]
    n_particles = pos[0].shape[0]
    trajectory_data = np.array([pos, a1, a3]).reshape(n_confs,n_particles, -1)
    
    # Parse the topology file
    with open(topology_filepath, 'r') as f:
        topology_lines = f.readlines()

    # Extract number of nucleotides and strands from the first line
    first_line = topology_lines[0].strip()
    num_nucleotides, num_strands = map(int, first_line.split())

    topology_data = []
    base_encoding = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
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
    
    # Combine the data
    combined_data = np.concatenate((trajectory_data, np.array(topology_data)), axis=2)

    return np.array(combined_data)

# Example usage
# combined_data, num_nucleotides, num_strands = parse_dna_origami_data('path/to/trajectory.dat', 'path/to/topology.dat')