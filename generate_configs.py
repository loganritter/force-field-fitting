import numpy as np
import os
from scipy.spatial.transform import Rotation as R

class PHASTFitConfigurations:
    """
    Creates input files for MOFs that generate configurations of gases
    near metal centers or randomly in pores. Supports fragments or periodic systems.
    
    Used in conjunction with FFit.
    """
    def __init__(self, input_file, charges_file, gas_list, target_metal_index,
                 start_distance, step, num_steps, axis='y', force_field_type='H2CNO'):
        """
        Initialize the processor with configuration parameters.
        
        Args:
            input_file (str): Path to XYZ coordinate file
            charges_file (str): Path to atomic charges file
            gas_list (list): List of gases to process
            target_metal_index (int): Index of the target metal atom
            start_distance (float): Initial distance for gas placement
            step (float): Distance increment between configurations
            num_steps (int): Number of configurations to generate
            axis (str): Axis for gas placement (1D, 2D, or 3D movement)
            force_field_type (str): Type of force field ('H2CNO' or 'AA')
        """
        self.input_file = input_file
        self.charges_file = charges_file
        self.gas_list = gas_list
        self.target_metal_index = target_metal_index
        self.start_distance = start_distance
        self.step = step
        self.num_steps = num_steps
        self.force_field_type = force_field_type
        
        # UFF radii
        self.vdw_radii = {
            'H': 2.571133701, 'He': 2.104302772, 'Li': 2.183592758, 'Be': 2.445516981, 'B': 3.637539466, 'C': 3.430850964, 'N': 3.260689308, 'O': 3.118145513, 'F': 2.996983288, 'Ne': 2.889184543, 
            'Na': 2.657550876, 'Mg': 2.691405028, 'Al': 4.008153333, 'Si': 3.826409994, 'P': 3.694556984, 'S': 3.594776328, 'Cl': 3.51637724, 'Ar': 3.445996242, 'K': 3.396105914, 'Ca': 3.028164743, 
            'Sc': 2.935511276, 'Ti': 2.82860343, 'V': 2.80098557, 'Cr': 2.693186825, 'Mn': 2.637951104, 'Fe': 2.594297067, 'Co': 2.558661118, 'Ni': 2.524806967, 'Cu': 3.11369102, 'Zn': 2.461553158, 
            'Ga': 3.904809082, 'Ge': 3.813046514, 'As': 3.768501578, 'Se': 3.74622911, 'Br': 3.73197473, 'Kr': 3.689211592, 'Rb': 3.665157326, 'Sr': 3.243762233, 'Y': 2.980056212, 'Zr': 2.783167595, 
            'Nb': 2.819694443, 'Mo': 2.719022888, 'Tc': 2.670914357, 'Ru': 2.639732902, 'Rh': 2.609442345, 'Pd': 2.582715384, 'Ag': 2.804549165, 'Cd': 2.537279549, 'In': 3.976080979, 'Sn': 3.91282717, 
            'Sb': 3.937772334, 'Te': 3.98231727, 'I': 4.009044232, 'Xe': 3.923517955, 'Cs': 4.02418951, 'Ba': 3.298997953, 'La': 3.137745285, 'Ce': 3.168035842, 'Pr': 3.212580778, 'Nd': 3.184962917, 
            'Pm': 3.160017753, 'Sm': 3.135963488, 'Eu': 3.111909222, 'Gd': 3.000546883, 'Tb': 3.074491476, 'Dy': 3.054000806, 'Ho': 3.03707373, 'Er': 3.021037553, 'Tm': 3.005892275, 'Yb': 2.988965199, 
            'Lu': 3.242871334, 'Hf': 2.798312874, 'Ta': 2.824148937, 'W': 2.734168166, 'Re': 2.631714813, 'Os': 2.779604001, 'Ir': 2.53015236, 'Pt': 2.45353507, 'Au': 2.933729479, 'Hg': 2.409881033, 
            'Tl': 3.872736728, 'Pb': 3.828191792, 'Bi': 3.893227398, 'Po': 4.195242064, 'At': 4.231768911, 'Rn': 4.245132392, 'Fr': 4.365403719, 'Ra': 3.275834587, 'Ac': 3.098545742, 'Th': 3.025492047, 
            'Pa': 3.050437211, 'U': 3.024601148, 'Np': 3.050437211, 'Pu': 3.050437211, 'Am': 3.012128566, 'Cm': 2.963129137, 'Bk': 2.97471082, 'Cf': 2.951547453, 'Es': 2.939074871, 'Fm': 2.927493188, 
            'Md': 2.916802403, 'No': 2.893639037, 'Lr': 2.882948252
        }
        
        self.axis_map = {
            'x': [0], 'y': [1], 'z': [2],
            'xy': [0, 1], 'xz': [0, 2], 'yz': [1, 2],
            'xyz': [0, 1, 2]
        }
        if axis not in self.axis_map:
            raise ValueError(f"Invalid axis: {axis}. Choose from 'x', 'y', 'z', 'xy', 'xz', 'yz', or 'xyz'.")
        self.axes = self.axis_map[axis]

    def read_xyz(self):
        """
        Read atomic coordinates and cell information from XYZ file.
        
        Returns:
            tuple: (atoms, cell_info, cell_dims) where atoms is list of (atom, position) tuples,
                  cell_info is string, and cell_dims is list of cell dimensions or None
        """
        with open(self.input_file, 'r') as file:
            lines = file.readlines()
        num_atoms = int(lines[0].strip())
        cell_info = lines[1].strip() if len(lines) > 1 else ""
        atoms = []
        for line in lines[2:2+num_atoms]:
            parts = line.split()
            atom = parts[0]
            position = np.array(list(map(float, parts[1:])))
            atoms.append((atom, position))
        
        cell_dims = None
        if cell_info:
            try:
                cell_dims = list(map(float, cell_info.split()))
                if len(cell_dims) != 6:
                    cell_dims = None
            except ValueError:
                cell_dims = None
        return atoms, cell_info, cell_dims

    def read_charges(self):
        """
        Read atomic charges from file.
        
        Returns:
            list: List of atomic charges as floats
        """
        with open(self.charges_file, 'r') as file:
            return [float(line.strip()) for line in file]

    @staticmethod
    def apply_pbc(position, box_dims):
        """
        Apply periodic boundary conditions to atomic positions.
        
        Args:
            position (np.ndarray): Atomic position
            box_dims (list): Box dimensions
            
        Returns:
            np.ndarray: Position after PBC application
        """
        if box_dims is not None:
            for i in range(3):
                position[i] %= box_dims[i]
        return position

    def transform_fragment(self, atoms, cell_dims=None):
        """
        Transform molecular fragment, optionally with PBC.
        
        Args:
            atoms (list): List of (atom, position) tuples
            cell_dims (list): Optional cell dimensions for PBC
            
        Returns:
            list: Transformed atomic positions
        """
        target_position = atoms[self.target_metal_index][1]
        if cell_dims:
            box_center = np.array(cell_dims[:3]) / 2
            translation_vector = np.zeros(3)
            for ax in self.axes:
                translation_vector[ax] = target_position[ax] - box_center[ax]
            return [(atom, self.apply_pbc(position - translation_vector, cell_dims[:3]))
                    for atom, position in atoms]
        else:
            translation_vector = np.zeros(3)
            for ax in self.axes:
                translation_vector[ax] = target_position[ax]
            return [(atom, position - translation_vector) for atom, position in atoms]

    def add_noble_gas(self, atoms, target_position, noble_gas, cell_dims):
        """
        Generate configurations with noble gas at different distances.
        
        Args:
            atoms (list): List of (atom, position) tuples
            target_position (np.ndarray): Position of target metal atom
            noble_gas (str): Noble gas element symbol
            cell_dims (list): Optional cell dimensions for PBC
            
        Returns:
            list: List of configurations with noble gas at different distances
        """
        configurations = []
        for i in range(self.num_steps):
            distance = self.start_distance + i * self.step
            position = target_position.copy()
            for ax in self.axes:
                position[ax] -= distance  # Move along all specified axes
            if cell_dims is not None:
                position = self.apply_pbc(position, cell_dims[:3])
            configurations.append(atoms + [(noble_gas, position)])
        return configurations
        
    def add_dimer(self, atoms, target_position, cell_dims, dimer_type="H2", num_rotations=10):
        """
        Generate configurations with a dimer gas (H2, N2, etc.) at different distances, with random rotations.
        
        Args:
            atoms (list): List of (atom, position) tuples.
            target_position (np.ndarray): Position of the target metal atom.
            cell_dims (list): Optional cell dimensions for PBC.
            dimer_type (str): Type of dimer gas ("H2" or "N2").
            num_rotations (int): Number of random rotations per step.
        
        Returns:
            list: List of configurations with the dimer gas at different distances and rotations.
        """
        if dimer_type == "H2":
            dimer_atoms = ["H2DA", "H2H", "H2H"]
            dimer_positions = [
                (0.000, 0.000, 0.000),  # H2DA center of mass
                (0.371, 0.000, 0.000),  # H2H position 1
                (-0.371, 0.000, 0.000)  # H2H position 2
            ]
        elif dimer_type == "N2":
            dimer_atoms = ["N2DA", "N2N", "N2N"]
            dimer_positions = [
                (0.000, 0.000, 0.000),  # N2DA center of mass
                (0.5507, 0.000, 0.000),  # N2N position 1
                (-0.5507, 0.000, 0.000)  # N2N position 2
            ]
        else:
            raise ValueError(f"Unsupported dimer type: {dimer_type}")

        dimer_positions = [np.array(pos) for pos in dimer_positions]

        configurations = []

        # Generate configurations at each distance step
        for i in range(self.num_steps):
            distance = self.start_distance + i * self.step
            dimer_center_position = target_position.copy()
            dimer_center_position[self.axes] -= distance  # Move dimer center along the specified axis

            # Apply PBC to dimer center if applicable
            if cell_dims is not None:
                dimer_center_position = self.apply_pbc(dimer_center_position, cell_dims[:3])

            # Create the base configuration for the dimer at each distance
            base_dimer_positions = [
                (dimer_atoms[j], dimer_center_position + dimer_positions[j])
                for j in range(len(dimer_atoms))
            ]

            # Apply random rotations to the dimer
            for _ in range(num_rotations):
                rotated_dimer_positions = self.random_rotate_molecule(base_dimer_positions)
                configurations.append(atoms + rotated_dimer_positions)

        return configurations
        
    def random_rotate_molecule(self, dimer):
        """
        Apply a random rotation to a dimer molecule around its center of mass.
        
        Args:
            dimer (list): List of (atom, position) tuples for dimers
        
        Returns:
            list: List of rotated (atom, position) tuples for dimers
        """
        # Extract center of mass and shift positions to it
        com = dimer[0][1]
        rel_positions = [(atom, pos - com) for atom, pos in dimer]
        
        # Generate a random rotation and apply it
        quaternion = R.random().as_quat()
        rotation = R.from_quat(quaternion)
        
        rotated_positions = [(atom, rotation.apply(pos) + com) for atom, pos in rel_positions]
        return rotated_positions
        
    def get_pore_bounds(self, atoms, cell_dims):
        """
        Calculate the boundaries of the pore space.
        
        Args:
            atoms (list): List of (atom, position) tuples
            cell_dims (list): Cell dimensions for PBC
            
        Returns:
            tuple: (min_bounds, max_bounds) for the pore space
        """
        if cell_dims is not None:
            min_bounds = np.zeros(3)
            max_bounds = np.array(cell_dims[:3])
        else:
            positions = np.array([pos for _, pos in atoms])
            min_bounds = np.min(positions, axis=0)
            max_bounds = np.max(positions, axis=0)
        return min_bounds, max_bounds
        
    def check_vdw_distance(self, position, atoms, min_factor=0.75, max_factor=1.5):
        """
        Check if a position satisfies van der Waals distance criteria.
        
        Args:
            position (np.ndarray): Position to check
            atoms (list): List of (atom, position) tuples
            min_factor (float): Minimum factor of sum of vdW radii
            max_factor (float): Maximum factor of sum of vdW radii
            
        Returns:
            bool: True if position satisfies vdW criteria
        """
        for atom_name, atom_pos in atoms:
            distance = np.linalg.norm(position - atom_pos)
            vdw_sum = self.vdw_radii.get(atom_name, 3.0)  # Default to 3.0 if unknown
            min_dist = vdw_sum * min_factor
            max_dist = vdw_sum * max_factor
            if distance < min_dist:
                return False
        return True
        
    def generate_random_position(self, atoms, cell_dims):
        """
        Generate a random position within the pore that satisfies vdW criteria.
        
        Args:
            atoms (list): List of (atom, position) tuples
            cell_dims (list): Cell dimensions for PBC
            
        Returns:
            np.ndarray: Valid random position
        """
        min_bounds, max_bounds = self.get_pore_bounds(atoms, cell_dims)
        max_attempts = 1000
        
        for _ in range(max_attempts):
            position = np.random.uniform(min_bounds, max_bounds)
            if self.check_vdw_distance(position, atoms):
                return position
                
        raise RuntimeError("Could not find valid position after maximum attempts")

    def add_noble_gas_aa(self, atoms, noble_gas, cell_dims):
        """
        Generate configurations with noble gas at random positions satisfying vdW criteria.
        
        Args:
            atoms (list): List of (atom, position) tuples
            noble_gas (str): Noble gas element symbol
            cell_dims (list): Optional cell dimensions for PBC
            
        Returns:
            list: List of configurations with noble gas at different positions
        """
        configurations = []
        for _ in range(self.num_steps):
            position = self.generate_random_position(atoms, cell_dims)
            if cell_dims is not None:
                position = self.apply_pbc(position, cell_dims[:3])
            configurations.append(atoms + [(noble_gas, position)])
        return configurations

    def add_dimer_aa(self, atoms, cell_dims, dimer_type="H2"):
        """
        Generate configurations with a dimer gas at random positions with random orientations.
        
        Args:
            atoms (list): List of (atom, position) tuples
            cell_dims (list): Optional cell dimensions for PBC
            dimer_type (str): Type of dimer gas ("H2" or "N2")
            
        Returns:
            list: List of configurations
        """
        if dimer_type == "H2":
            dimer_atoms = ["H2DA", "H2H", "H2H"]
            bond_length = 0.742     # H2 bond length
        elif dimer_type == "N2":
            dimer_atoms = ["N2DA", "N2N", "N2N"]
            bond_length = 1.1014    # N2 bond length
        else:
            raise ValueError(f"Unsupported dimer type: {dimer_type}")

        configurations = []
        for _ in range(self.num_steps):
            center_position = self.generate_random_position(atoms, cell_dims)
            
            # Base dimer positions
            base_positions = [
                (dimer_atoms[0], center_position),
                (dimer_atoms[1], center_position + np.array([bond_length/2, 0, 0])),
                (dimer_atoms[2], center_position + np.array([-bond_length/2, 0, 0]))
            ]
            
            # Generate 5 random orientations for each position
            for _ in range(5):
                rotated_positions = self.random_rotate_molecule(base_positions)
                if cell_dims is not None:
                    rotated_positions = [(atom, self.apply_pbc(pos, cell_dims[:3])) 
                                       for atom, pos in rotated_positions]
                configurations.append(atoms + rotated_positions)
        
        return configurations
        
    @staticmethod
    def write_files(file, atoms, comment, cell_info=None, charges=None, omit_cm=False):
        """
        Write molecular configurations.
        
        Args:
            file: File object to write to
            atoms (list): List of (atom, position) tuples
            comment (str): Comment line for XYZ file
            cell_info (str): Optional cell information
            charges (list): Optional atomic charges
            omit_cm (bool): Omits writing H2DA/N2DA in input.xyz
        """
        gas_charges = {
            "H2DA": -0.846166, "H2H": 0.423083,
            "N2DA": 0.94194, "N2N": -0.47103,
            "He": 0.0, "Ne": 0.0, "Ar": 0.0, "Kr": 0.0, "Xe": 0.0, "Rn": 0.0
        }
        
        atom_count = len(atoms) - (2 if omit_cm else 0)
        file.write(f"{atom_count}\n")
        file.write(f"{comment} {cell_info}\n" if cell_info else f"{comment}\n")
        
        charge_index = 0
        
        for atom, position in atoms:
            if omit_cm and atom in ["H2DA", "N2DA"]:
                continue
            atom_name = atom
            if omit_cm and atom in ["H2H", "N2N"]:
                atom_name = "H" if atom == "H2H" else "N"
            position_str = ' '.join(f"{coord:.5f}" for coord in position)
            
            if charges:
                if atom in gas_charges:
                    charge = gas_charges[atom]
                else:
                    charge = charges[charge_index]
                    charge_index += 1
                line = f"{atom_name} {position_str} {charge:.5f}"
            else:
                line = f"{atom_name} {position_str}"

            file.write(line + "\n")

    def process(self):
        """
        Main processing method to generate all configurations and write output files.
        """
        atoms, cell_info, cell_dims = self.read_xyz()
        charges = self.read_charges()
        
        atoms = self.transform_fragment(atoms, cell_dims)
        target_position = atoms[self.target_metal_index][1]

        for gas in self.gas_list:
            if self.force_field_type == 'H2CNO':
                if gas in ['H2', 'N2']:
                    configurations = self.add_dimer(atoms, target_position, cell_dims, dimer_type=gas, num_rotations=10)
                else:
                    configurations = self.add_noble_gas(atoms, target_position, gas, cell_dims)
            else:  # AA force field type
                if gas in ['H2', 'N2']:
                    configurations = self.add_dimer_aa(atoms, cell_dims, dimer_type=gas)
                else:
                    configurations = self.add_noble_gas_aa(atoms, gas, cell_dims)
            
            # Create folder structure and write files
            os.makedirs(gas, exist_ok=True)
            
            # Write configs.fit
            with open(os.path.join(gas, "configs.fit"), 'w') as configs_file:
                for config in configurations:
                    self.write_files(configs_file, config, "XXX", cell_info, charges)
            
            # Write out.xyz
            with open(os.path.join(gas, "out.xyz"), 'w') as out_file:
                for config in configurations:
                    self.write_files(out_file, config, cell_info)
            
            # Write individual configuration files in subfolders
            for idx, config in enumerate(configurations, start=1):
                subfolder = os.path.join(gas, str(idx))
                os.makedirs(subfolder, exist_ok=True)
                with open(os.path.join(subfolder, "input.xyz"), 'w') as input_file:
                    self.write_files(input_file, config, cell_info, omit_cm=True)

        print(f"Configuration generation complete using the {self.force_field_type} force field type.")

if __name__ == "__main__":
    processor = PHASTFitConfigurations(
        input_file="Cu-MOF-74.xyz",
        charges_file="Cu74_charges.txt",
        gas_list=["Ne", "Ar", "Kr", "Xe", "H2", "N2"],
        target_metal_index=78,
        start_distance=3.0,
        step=0.1,
        num_steps=501,
        axis='x',
        force_field_type='AA'  # Can be 'H2CNO' or 'AA'
    )
    processor.process()
