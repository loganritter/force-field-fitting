import numpy as np
import os

def create_co2_molecule(center, orientation):
    bond_length = 1.155028  # CO bond length in Angstroms
    orientation = orientation / np.linalg.norm(orientation)
    
    c_pos = np.array(center)
    o1_pos = c_pos + orientation * bond_length
    o2_pos = c_pos - orientation * bond_length
    
    return np.vstack([c_pos, o1_pos, o2_pos])

def write_xyz(filename, molecules, comment=""):
    with open(filename, 'w') as f:
        f.write(f"{len(molecules) * 3}\n")
        f.write(f"{comment}\n")
        for mol in molecules:
            f.write(f"C {mol[0,0]:.6f} {mol[0,1]:.6f} {mol[0,2]:.6f}\n")
            f.write(f"O {mol[1,0]:.6f} {mol[1,1]:.6f} {mol[1,2]:.6f}\n")
            f.write(f"O {mol[2,0]:.6f} {mol[2,1]:.6f} {mol[2,2]:.6f}\n")

def generate_configurations(distances):
    configurations = {
        "parallel": lambda d: (
            create_co2_molecule([0, 0, 0], [0, 0, 1]),
            create_co2_molecule([d, 0, 0], [0, 0, 1])
        ),
        "t_shaped": lambda d: (
            create_co2_molecule([0, 0, 0], [0, 0, 1]),
            create_co2_molecule([d, 0, 0], [0, 1, 0])
        ),
        "crossed": lambda d: (
            create_co2_molecule([0, 0, 0], [0, 0, 1]),
            create_co2_molecule([d/np.sqrt(2), d/np.sqrt(2), 0], [1, 0, 0])
        ),
        "slipped_parallel": lambda d: (
            create_co2_molecule([0, 0, 0], [0, 0, 1]),
            create_co2_molecule([d, 0, d/2], [0, 0, 1])
        ),
        "linear": lambda d: (
            create_co2_molecule([0, 0, 0], [0, 0, 1]),
            create_co2_molecule([0, 0, d*1.75], [0, 0, 1])
        )
    }

    base_dir = "co2_configurations"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    for config_name, config_func in configurations.items():
        config_dir = os.path.join(base_dir, config_name)
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)

        all_configs_of_type = []

        for i, distance in enumerate(distances, 1):
            mol1, mol2 = config_func(distance)
            all_configs_of_type.append((mol1, mol2))

            numbered_dir = os.path.join(config_dir, str(i))
            if not os.path.exists(numbered_dir):
                os.makedirs(numbered_dir)

            write_xyz(
                os.path.join(numbered_dir, "input.xyz"),
                [mol1, mol2],
                f"CO2-CO2 {config_name} configuration at {distance:.2f} Angstroms"
            )

        with open(os.path.join(config_dir, "out.xyz"), 'w') as f:
            for i, (mol1, mol2) in enumerate(all_configs_of_type):
                f.write("6\n")
                f.write(f"CO2-CO2 {config_name} configuration at {distances[i]:.2f} Angstroms\n")
                f.write(f"C {mol1[0,0]:.6f} {mol1[0,1]:.6f} {mol1[0,2]:.6f}\n")
                f.write(f"O {mol1[1,0]:.6f} {mol1[1,1]:.6f} {mol1[1,2]:.6f}\n")
                f.write(f"O {mol1[2,0]:.6f} {mol1[2,1]:.6f} {mol1[2,2]:.6f}\n")
                f.write(f"C {mol2[0,0]:.6f} {mol2[0,1]:.6f} {mol2[0,2]:.6f}\n")
                f.write(f"O {mol2[1,0]:.6f} {mol2[1,1]:.6f} {mol2[1,2]:.6f}\n")
                f.write(f"O {mol2[2,0]:.6f} {mol2[2,1]:.6f} {mol2[2,2]:.6f}\n")

distances = np.arange(2.5, 6.1, 0.1)
generate_configurations(distances)
