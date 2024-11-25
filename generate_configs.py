import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from math import pi, cos, sin, sqrt

class PBC:
    def __init__(self, a, b, c, alpha, beta, gamma):
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        basis00 = a
        basis01 = 0.0
        basis02 = 0.0
        basis10 = b * cos(pi / 180.0 * gamma)
        basis11 = b * sin(pi / 180.0 * gamma)
        basis12 = 0.0
        basis20 = c * cos(pi / 180.0 * beta)
        basis21 = ((b * c * cos(pi / 180.0 * alpha)) - (basis10 * basis20)) / basis11
        basis22 = sqrt(c * c - basis20 * basis20 - basis21 * basis21)

        self.basis_matrix = np.array([
            [basis00, basis01, basis02],
            [basis10, basis11, basis12],
            [basis20, basis21, basis22],
        ])

        self.volume = basis00 * (basis11 * basis22 - basis12 * basis21)
        self.volume += basis01 * (basis12 * basis20 - basis10 * basis22)
        self.volume += basis02 * (basis10 * basis21 - basis11 * basis20)

        self.inverse_volume = 1.0 / self.volume

        reciprocal_basis00 = self.inverse_volume * (basis11 * basis22 - basis12 * basis21)
        reciprocal_basis01 = self.inverse_volume * (basis02 * basis21 - basis01 * basis22)
        reciprocal_basis02 = self.inverse_volume * (basis01 * basis12 - basis02 * basis11)
        reciprocal_basis10 = self.inverse_volume * (basis12 * basis20 - basis10 * basis22)
        reciprocal_basis11 = self.inverse_volume * (basis00 * basis22 - basis02 * basis20)
        reciprocal_basis12 = self.inverse_volume * (basis02 * basis10 - basis00 * basis12)
        reciprocal_basis20 = self.inverse_volume * (basis10 * basis21 - basis11 * basis20)
        reciprocal_basis21 = self.inverse_volume * (basis01 * basis20 - basis00 * basis21)
        reciprocal_basis22 = self.inverse_volume * (basis00 * basis11 - basis01 * basis10)

        self.reciprocal_basis_matrix = np.array([
            [reciprocal_basis00, reciprocal_basis01, reciprocal_basis02],
            [reciprocal_basis10, reciprocal_basis11, reciprocal_basis12],
            [reciprocal_basis20, reciprocal_basis21, reciprocal_basis22],
        ])

    def wrap(self, dx):
        img = np.matmul(dx, self.reciprocal_basis_matrix)
        img = np.round(img)
        di = np.matmul(img, self.basis_matrix)
        dx_return = dx - di
        return dx_return

class PHASTFitConfigurations:
    def __init__(self, input_file, charges_file, gas_list, target_metal_index,
                 start_distance, step, num_steps, axis='y', force_field_type='H2CNO'):

        self.input_file = input_file
        self.charges_file = charges_file
        self.gas_list = gas_list
        self.target_metal_index = target_metal_index
        self.start_distance = start_distance
        self.step = step
        self.num_steps = num_steps
        self.force_field_type = force_field_type
        self.pbc = None
        
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
        
        if force_field_type not in ['H2CNO', 'AA']:
            raise ValueError("force_field_type must be either 'H2CNO' or 'AA'")
        
        self.axis_map = {
            'x': [0], 'y': [1], 'z': [2],
            'xy': [0, 1], 'xz': [0, 2], 'yz': [1, 2],
            'xyz': [0, 1, 2]
        }
        if axis not in self.axis_map:
            raise ValueError(f"Invalid axis: {axis}. Choose from 'x', 'y', 'z', 'xy', 'xz', 'yz', or 'xyz'.")
        self.axes = self.axis_map[axis]

    def read_xyz(self):
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
        
        if cell_info:
            try:
                cell_params = list(map(float, cell_info.split()))
                if len(cell_params) == 6:
                    a, b, c = cell_params[:3]
                    alpha, beta, gamma = cell_params[3:]
                    self.pbc = PBC(a, b, c, alpha, beta, gamma)
                    return atoms, cell_info, self.pbc.basis_matrix
            except ValueError:
                pass
        return atoms, cell_info, None

    def read_charges(self):
        with open(self.charges_file, 'r') as file:
            return [float(line.strip()) for line in file]

    def apply_pbc(self, position, cell_matrix):
        if cell_matrix is not None and self.pbc is not None:
            return self.pbc.wrap(position)
        return position

    def transform_fragment(self, atoms, cell_matrix=None):
        target_position = atoms[self.target_metal_index][1]
        if cell_matrix is not None:
            box_center = np.array([0.5, 0.5, 0.5])
            box_center_cart = np.dot(box_center, cell_matrix)
            
            translation_vector = np.zeros(3)
            for ax in self.axes:
                translation_vector[ax] = target_position[ax] - box_center_cart[ax]
            
            return [(atom, self.apply_pbc(position - translation_vector, cell_matrix))
                    for atom, position in atoms]
        else:
            translation_vector = np.zeros(3)
            for ax in self.axes:
                translation_vector[ax] = target_position[ax]
            return [(atom, position - translation_vector) for atom, position in atoms]
            
    def get_pore_bounds(self, atoms, cell_matrix):
        if cell_matrix is not None and self.pbc is not None:
            min_bounds = np.zeros(3)
            max_bounds = np.ones(3)
            return (
                np.dot(min_bounds, self.pbc.basis_matrix),
                np.dot(max_bounds, self.pbc.basis_matrix)
            )
        else:
            positions = np.array([pos for _, pos in atoms])
            return np.min(positions, axis=0), np.max(positions, axis=0)

    def add_noble_gas(self, atoms, target_position, noble_gas, cell_matrix):
        configurations = []
        
        if self.force_field_type == 'H2CNO':
            for i in range(self.num_steps):
                distance = self.start_distance + i * self.step
                position = target_position.copy()
                for ax in self.axes:
                    position[ax] -= distance
                if cell_matrix is not None:
                    position = self.apply_pbc(position, cell_matrix)
                configurations.append(atoms + [(noble_gas, position)])
        else:
            for _ in range(self.num_steps):
                position = self.generate_random_position(atoms, cell_matrix)
                if cell_matrix is not None:
                    position = self.apply_pbc(position, cell_matrix)
                configurations.append(atoms + [(noble_gas, position)])
        
        return configurations
        
    def add_dimer(self, atoms, target_position, cell_matrix, dimer_type="H2", num_rotations=10):
        if dimer_type == "H2":
            dimer_atoms = ["H2DA", "H2H", "H2H"]
            bond_length = 0.742
            base_positions = [
                (0.000, 0.000, 0.000),
                (0.371, 0.000, 0.000),
                (-0.371, 0.000, 0.000)
            ]
        elif dimer_type == "N2":
            dimer_atoms = ["N2DA", "N2N", "N2N"]
            bond_length = 1.1014
            base_positions = [
                (0.000, 0.000, 0.000),
                (0.5507, 0.000, 0.000),
                (-0.5507, 0.000, 0.000)
            ]
        else:
            raise ValueError(f"Unsupported dimer type: {dimer_type}")

        base_positions = [np.array(pos) for pos in base_positions]
        configurations = []
        
        if self.force_field_type == 'H2CNO':
            for i in range(self.num_steps):
                distance = self.start_distance + i * self.step
                dimer_center = target_position.copy()
                for ax in self.axes:
                    dimer_center[ax] -= distance

                if cell_matrix is not None:
                    dimer_center = self.apply_pbc(dimer_center, cell_matrix)

                base_dimer = [
                    (dimer_atoms[j], dimer_center + base_positions[j])
                    for j in range(len(dimer_atoms))
                ]

                for _ in range(num_rotations):
                    rotated_dimer = self.random_rotate_molecule(base_dimer)
                    if cell_matrix is not None:
                        rotated_dimer = [(atom, self.apply_pbc(pos, cell_matrix)) 
                                       for atom, pos in rotated_dimer]
                    configurations.append(atoms + rotated_dimer)
        
        else: 
            rotations_per_position = 5
            
            for _ in range(self.num_steps):
                center_position = self.generate_random_position(atoms, cell_matrix)
                
                base_dimer = [
                    (dimer_atoms[0], center_position),
                    (dimer_atoms[1], center_position + np.array([bond_length/2, 0, 0])),
                    (dimer_atoms[2], center_position + np.array([-bond_length/2, 0, 0]))
                ]
                
                for _ in range(rotations_per_position):
                    rotated_dimer = self.random_rotate_molecule(base_dimer)
                    if cell_matrix is not None:
                        rotated_dimer = [(atom, self.apply_pbc(pos, cell_matrix)) 
                                       for atom, pos in rotated_dimer]
                    configurations.append(atoms + rotated_dimer)
        
        return configurations
        
    def generate_random_position(self, atoms, cell_matrix):
        if cell_matrix is not None and self.pbc is not None:
            max_attempts = 1000
            for _ in range(max_attempts):
                frac_pos = np.random.uniform(0, 1, 3)
                cart_pos = np.dot(frac_pos, self.pbc.basis_matrix)
                if self.check_vdw_distance(cart_pos, atoms):
                    return cart_pos
        else:
            min_bounds, max_bounds = self.get_pore_bounds(atoms, None)
            max_attempts = 1000
            for _ in range(max_attempts):
                position = np.random.uniform(min_bounds, max_bounds)
                if self.check_vdw_distance(position, atoms):
                    return position

        raise RuntimeError("Could not find valid position after maximum attempts")
        
    def random_rotate_molecule(self, dimer):
        com = dimer[0][1]
        rel_positions = [(atom, pos - com) for atom, pos in dimer]
        
        quaternion = R.random().as_quat()
        rotation = R.from_quat(quaternion)
        
        rotated_positions = [(atom, rotation.apply(pos) + com) for atom, pos in rel_positions]
        return rotated_positions
        
    def check_vdw_distance(self, position, atoms, min_factor=0.75, max_factor=1.5):
        for atom_name, atom_pos in atoms:
            distance = np.linalg.norm(position - atom_pos)
            vdw_sum = self.vdw_radii.get(atom_name, 3.0)  # Default to 3.0 if unknown
            min_dist = vdw_sum * min_factor
            max_dist = vdw_sum * max_factor
            if distance < min_dist:
                return False
        return True
        
    @staticmethod
    def write_files(file, atoms, comment, cell_info=None, charges=None, omit_cm=False):
        gas_charges = {
            "H2DA": -0.846166, "H2H": 0.423083,
            "N2DA": 0.94194, "N2N": -0.47103,
            "He": 0.0, "Ne": 0.0, "Ar": 0.0, "Kr": 0.0, "Xe": 0.0, "Rn": 0.0
        }
        
        atoms_to_write = [atom for atom in atoms if not (omit_cm and atom[0] in ["H2DA", "N2DA"])]
        atom_count = len(atoms_to_write)
        
        file.write(f"{atom_count}\n")
        file.write(f"{comment} {cell_info}\n" if cell_info else f"{comment}\n")
        
        charge_index = 0
        original_atom_count = sum(1 for atom in atoms if atom[0] not in ["H2DA", "N2DA", "H2H", "N2N", "He", "Ne", "Ar", "Kr", "Xe", "Rn"])
        
        for idx, (atom, position) in enumerate(atoms_to_write, 1):
            atom_name = atom
            if omit_cm and atom in ["H2H", "N2N"]:
                atom_name = "H" if atom == "H2H" else "N"
            
            position_str = ' '.join(f"{coord:.5f}" for coord in position)
            
            type_column = 1 if idx <= original_atom_count else 2
            
            if charges:
                if atom in gas_charges:
                    charge = gas_charges[atom]
                else:
                    charge = charges[charge_index]
                    charge_index += 1
                line = f"{atom_name} {type_column} {position_str} {charge:.5f}"
            else:
                line = f"{atom_name} {type_column} {position_str}"

            file.write(line + "\n")

    def process(self):
        atoms, cell_info, cell_matrix = self.read_xyz()
        charges = self.read_charges()
        
        if cell_matrix is not None:
            atoms = [(atom, self.apply_pbc(position, cell_matrix)) for atom, position in atoms]
        
        atoms = self.transform_fragment(atoms, cell_matrix)
        target_position = atoms[self.target_metal_index][1]

        for gas in self.gas_list:
            if gas in ['H2', 'N2']:
                configurations = self.add_dimer(atoms, target_position, cell_matrix, dimer_type=gas)
            else:
                configurations = self.add_noble_gas(atoms, target_position, gas, cell_matrix)
            
            os.makedirs(gas, exist_ok=True)
            
            with open(os.path.join(gas, "configs.fit"), 'w') as configs_file:
                for config in configurations:
                    if cell_matrix is not None:
                        config = [(atom, self.apply_pbc(pos, cell_matrix)) for atom, pos in config]
                    self.write_files(configs_file, config, "XXX", cell_info, charges)
            
            with open(os.path.join(gas, "out.xyz"), 'w') as out_file:
                for config in configurations:
                    if cell_matrix is not None:
                        config = [(atom, self.apply_pbc(pos, cell_matrix)) for atom, pos in config]
                    self.write_files(out_file, config, cell_info)
            
            for idx, config in enumerate(configurations, start=1):
                subfolder = os.path.join(gas, str(idx))
                os.makedirs(subfolder, exist_ok=True)
                with open(os.path.join(subfolder, "input.xyz"), 'w') as input_file:
                    if cell_matrix is not None:
                        config = [(atom, self.apply_pbc(pos, cell_matrix)) for atom, pos in config]
                    self.write_files(input_file, config, cell_info, omit_cm=True)

        print(f"Configuration generation complete using the {self.force_field_type} force field type.")

if __name__ == "__main__":
    processor = PHASTFitConfigurations(
        input_file="T3T-opt.xyz",
        charges_file="charges.txt",
        gas_list=["Ne", "Ar", "Kr", "Xe", "H2", "N2"],
        target_metal_index=72,
        start_distance=3.0,
        step=0.1,
        num_steps=51,
        axis='x',
        force_field_type='H2CNO'  # Can be 'H2CNO' or 'AA'
    )
    processor.process()
