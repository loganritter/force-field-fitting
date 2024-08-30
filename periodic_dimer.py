#!/usr/bin/python3
import os
from numpy import *
import copy

class Atom:
    def __init__(self,x,y,z,name,element):
        self.name = name
        self.element = element
        self.x = array([x,y,z])
        self.vdw = self.get_uff_radii()

    def get_uff_radii(self):
        if (self.element=="H"):
            return 2.886/1.12246204831
        elif (self.element=="C"):
            return 3.851/1.12246204831
        elif (self.element=="N"):
            return 3.660/1.12246204831
        elif (self.element=="O"):
            return 3.500/1.12246204831
        elif (self.element=="He"):
            return 2.362/1.12246204831
        elif (self.element=="Ne"):
            return 3.243/1.12246204831
        elif (self.element=="Ar"):
            return 3.868/1.12246204831
        elif (self.element=="Kr"):
            return 4.141/1.12246204831
        elif (self.element=="Xe"):
            return 4.404/1.12246204831
        elif (self.element=="Cu"):
            return 3.495/1.12246204831
        elif (self.element=="Cr"):
            return 3.023/1.12246204831
        elif (self.element=="Ni"):
            return 2.834/1.12246204831
        elif (self.element=="Si"):
            return 4.295/1.12246204831
        elif (self.element=="F"):
            return 3.364/1.12246204831
        elif (self.element=="Te"):
            return 4.470/1.12246204831
        elif (self.element=="DA"):
            return 0.0
        raise ValueError('Unknown element in find_uff_radii {}'.format(self.element))

    def find_mass(self):
        if (self.element=="H"):
            return 1.00794
        if (self.element=="C"):
            return 12.0107
        if (self.element=="O"):
            return 15.9994
        if (self.element=="N"):
            return 14.0067
        if (self.element=="S"):
            return 32.065
        raise ValueError('Unknown element in find_mass {}'.format(self.element))

class Molecule:
    def __init__(self,name,charge,mult):
        self.name = name
        self.charge = charge
        self.mult = mult
        self.atoms = []

    def __len__(self):
        return len(self.atoms)

    def len(self):
        return len(self.atoms)

    def __getitem__(self, key):
        return self.atoms[key]

    def append(self, thing):
        self.atoms.append(thing)

    def remove(self, thing):
        self.atoms.remove(thing)

class Quaternion:
    def normalize(self):
        magnitude = sqrt(dot(self.x,self.x))
        self.x = self.x/magnitude

    def get_conjugate(self):
        result = Quaternion(-self.x[0],-self.x[1],-self.x[2],self.x[3])
        return result

    def axis_angle(self,x,y,z,degrees):
        angle = degrees/57.2957795
        magnitude = sqrt(dot(array([x,y,z]),array([x,y,z])))
        self.x[0] = x*sin(angle/2.0)/magnitude
        self.x[1] = y*sin(angle/2.0)/magnitude
        self.x[2] = z*sin(angle/2.0)/magnitude
        self.x[3] = cos(angle/2.0)

    def random_rotation(self):
        self.x[0] = random.random()*2.0-1.0
        sum = self.x[0]*self.x[0]
        self.x[1] = sqrt(1.0-sum)*(random.random()*2.0-1.0)
        sum += self.x[1]*self.x[1]
        self.x[2] = sqrt(1.0-sum)*(random.random()*2.0-1.0)
        sum += self.x[2]*self.x[2]
        self.x[3] = sqrt(1.0-sum)*(-1.0 if random.random()>0.5 else 1.0)

    def __init__(self,x,y,z,w):
        self.x = array([x,y,z,w])

    def __mul__(self, other):
        x = self.x[3]*other.x[0] + other.x[3]*self.x[0] + self.x[1]*other.x[2] - self.x[2]*other.x[1]
        y = self.x[3]*other.x[1] + other.x[3]*self.x[1] + self.x[2]*other.x[0] - self.x[0]*other.x[2]
        z = self.x[3]*other.x[2] + other.x[3]*self.x[2] + self.x[0]*other.x[1] - self.x[1]*other.x[0]
        w = self.x[3]*other.x[3] - self.x[0]*other.x[0] - self.x[1]*other.x[1] - self.x[2]*other.x[2]
        result = Quaternion(x,y,z,w)
        return result


class PBC:
    def __init__(self,a,b,c,alpha,beta,gamma):
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        b00 = a
        b01 = b * cos(pi / 180.0 * gamma)
        b02 = c * cos(pi / 180.0 * beta)

        b10 = 0
        b11 = b * sin(pi / 180.0 * gamma)
        b12 = ((b * c * cos(pi / 180.0 * alpha)) - (b01 * b02)) / b11

        b20 = 0
        b21 = 0
        b22 = sqrt(c * c - b02 * b02 - b12 * b12)

        self.basis00 = b00
        self.basis01 = b10
        self.basis02 = b20

        self.basis10 = b01
        self.basis11 = b11
        self.basis12 = b21

        self.basis20 = b02
        self.basis21 = b12
        self.basis22 = b22

        self.volume = self.basis00 * (self.basis11 * self.basis22 - self.basis12 * self.basis21)
        self.volume += self.basis01 * (self.basis12 * self.basis20 - self.basis10 * self.basis22)
        self.volume += self.basis02 * (self.basis10 * self.basis21 - self.basis11 * self.basis20)

        self.inverse_volume = 1.0 / self.volume

        self.reciprocal_basis00 = self.inverse_volume * (self.basis11 * self.basis22 - self.basis12 * self.basis21)
        self.reciprocal_basis01 = self.inverse_volume * (self.basis02 * self.basis21 - self.basis01 * self.basis22)
        self.reciprocal_basis02 = self.inverse_volume * (self.basis01 * self.basis12 - self.basis02 * self.basis11)

        self.reciprocal_basis10 = self.inverse_volume * (self.basis12 * self.basis20 - self.basis10 * self.basis22)
        self.reciprocal_basis11 = self.inverse_volume * (self.basis00 * self.basis22 - self.basis02 * self.basis20)
        self.reciprocal_basis12 = self.inverse_volume * (self.basis02 * self.basis10 - self.basis00 * self.basis12)

        self.reciprocal_basis20 = self.inverse_volume * (self.basis10 * self.basis21 - self.basis11 * self.basis20)
        self.reciprocal_basis21 = self.inverse_volume * (self.basis01 * self.basis20 - self.basis00 * self.basis21)
        self.reciprocal_basis22 = self.inverse_volume * (self.basis00 * self.basis11 - self.basis01 * self.basis10)

    def min_image(self,dx,dy,dz):

        img0 = 0.
        img1 = 0.
        img2 = 0.

        img0 += self.reciprocal_basis00 * dx
        img0 += self.reciprocal_basis10 * dy
        img0 += self.reciprocal_basis20 * dz
        img0 = round(img0)

        img1 += self.reciprocal_basis01 * dx
        img1 += self.reciprocal_basis11 * dy
        img1 += self.reciprocal_basis21 * dz
        img1 = round(img1)

        img2 += self.reciprocal_basis02 * dx
        img2 += self.reciprocal_basis12 * dy
        img2 += self.reciprocal_basis22 * dz
        img2 = round(img2)

        di0 = 0.
        di1 = 0.
        di2 = 0.

        di0 += self.basis00 * img0
        di0 += self.basis10 * img1
        di0 += self.basis20 * img2

        di1 += self.basis01 * img0
        di1 += self.basis11 * img1
        di1 += self.basis21 * img2

        di2 += self.basis02 * img0
        di2 += self.basis12 * img1
        di2 += self.basis22 * img2

        dx = dx - di0
        dy = dy - di1
        dz = dz - di2

        r = sqrt(dx*dx+dy*dy+dz*dz)
        return r

    def min_image_vector(self,dx,dy,dz):

        img0 = 0.
        img1 = 0.
        img2 = 0.

        img0 += self.reciprocal_basis00 * dx
        img0 += self.reciprocal_basis10 * dy
        img0 += self.reciprocal_basis20 * dz
        img0 = round(img0)

        img1 += self.reciprocal_basis01 * dx
        img1 += self.reciprocal_basis11 * dy
        img1 += self.reciprocal_basis21 * dz
        img1 = round(img1)

        img2 += self.reciprocal_basis02 * dx
        img2 += self.reciprocal_basis12 * dy
        img2 += self.reciprocal_basis22 * dz
        img2 = round(img2)

        di0 = 0.
        di1 = 0.
        di2 = 0.

        di0 += self.basis00 * img0
        di0 += self.basis10 * img1
        di0 += self.basis20 * img2

        di1 += self.basis01 * img0
        di1 += self.basis11 * img1
        di1 += self.basis21 * img2

        di2 += self.basis02 * img0
        di2 += self.basis12 * img1
        di2 += self.basis22 * img2

        dx = dx - di0
        dy = dy - di1
        dz = dz - di2
        
        result = array([dx,dy,dz])
        return result

def rotate_3vector(x,q):
    vec = Quaternion(x[0],x[1],x[2],0.)

    result = vec * q.get_conjugate()
    result = q * result
    return result

def write_xyz_file(giant_array_of_frames):
    file = open("out.xyz","w")
    for frame in giant_array_of_frames:
        file.write(str(len(frame[0])+len(frame[1]))+"\n\n")
        for mol in frame:
            for atom in mol:
                file.write(atom.name+" "+str(atom.x[0])+" "+str(atom.x[1])+" "+str(atom.x[2])+"\n")
    file.close()

def write_cp2k_files(giant_array_of_frames,pbc):
    i = 1

    array_copy = copy.deepcopy(giant_array_of_frames)

    for frame in array_copy:
        for mol in frame:
            for atom in mol:
                if (atom.name == "DA"):
                    mol.remove(atom)

    for frame in array_copy:
        filename = "cp2k/"+str(i)+"/"
        if not os.path.exists(filename):
            os.makedirs(filename)
        f = open(filename+"cp2k.inp","w")
        f.write("&GLOBAL\n  PROJECT BSSE\n  RUN_TYPE BSSE\n  PRINT_LEVEL LOW\n&END GLOBAL\n\n&FORCE_EVAL\n  METHOD QS\n  &BSSE\n")
        f.write("    &FRAGMENT\n      LIST 1..{}\n    &END FRAGMENT\n".format(frame[0].len()))
        f.write("    &FRAGMENT\n      LIST {}..{}\n    &END FRAGMENT\n".format(frame[0].len()+1,frame[0].len()+frame[1].len()))
        f.write("    &CONFIGURATION\n      GLB_CONF 1 0\n      SUB_CONF 1 0\n      CHARGE {}\n      MULTIPLICITY {}\n    &END\n".format(frame[0].charge,frame[0].mult))
        f.write("    &CONFIGURATION\n      GLB_CONF 0 1\n      SUB_CONF 0 1\n      CHARGE {}\n      MULTIPLICITY {}\n    &END\n".format(frame[1].charge,frame[1].mult))
        f.write("    &CONFIGURATION\n      GLB_CONF 1 1\n      SUB_CONF 1 0\n      CHARGE {}\n      MULTIPLICITY {}\n    &END\n".format(frame[0].charge,frame[0].mult))
        f.write("    &CONFIGURATION\n      GLB_CONF 1 1\n      SUB_CONF 0 1\n      CHARGE {}\n      MULTIPLICITY {}\n    &END\n".format(frame[1].charge,frame[1].mult))
        f.write("    &CONFIGURATION\n      GLB_CONF 1 1\n      SUB_CONF 1 1\n      CHARGE {}\n      MULTIPLICITY {}\n    &END\n".format(frame[0].charge+frame[1].charge,frame[0].mult+frame[1].mult-1))
        f.write("  &END BSSE\n  &DFT\n    BASIS_SET_FILE_NAME BASIS_MOLOPT_UZH\n    POTENTIAL_FILE_NAME POTENTIAL_UZH\n    &QS\n      METHOD GPW\n    &END QS\n    &SCF\n      MAX_SCF 400\n      EPS_SCF 1.0E-06\n      &OT\n        MINIMIZER CG\n        LINESEARCH 2PNT\n        PRECONDITIONER FULL_ALL\n      &END OT\n      &OUTER_SCF\n        MAX_SCF 100\n        EPS_SCF 1.0E-06\n      &END OUTER_SCF\n    &END SCF\n    &XC\n      &XC_FUNCTIONAL\n        &PBE\n          PARAMETRIZATION ORIG\n        &END PBE\n      &END XC_FUNCTIONAL\n      &XC_GRID\n        XC_SMOOTH_RHO NN50\n        XC_DERIV NN50_SMOOTH\n      &END XC_GRID\n      &VDW_POTENTIAL\n        POTENTIAL_TYPE NON_LOCAL\n        &NON_LOCAL\n          TYPE RVV10\n          KERNEL_FILE_NAME rVV10_kernel_table.dat\n          CUTOFF 300\n          PARAMETERS 6.3 0.0093\n        &END NON_LOCAL\n      &END VDW_POTENTIAL\n    &END XC\n    &MGRID\n      NGRIDS 5\n      CUTOFF 1250\n      REL_CUTOFF 60\n    &END MGRID\n")
#        f.write("    UKS T\n")
        f.write("  &END DFT\n  &SUBSYS\n    &CELL\n")
        f.write("      ABC {} {} {}\n".format(pbc.a,pbc.b,pbc.c))
        f.write("      ALPHA_BETA_GAMMA {} {} {}\n".format(pbc.alpha,pbc.beta,pbc.gamma))
        f.write("    &END\n    &TOPOLOGY\n      COORD_FILE_NAME ./input.xyz\n      COORD_FILE_FORMAT XYZ\n    &END\n")

        unique_atoms = []

        for mol in frame:
            for atom in mol:
                if (atom.element not in unique_atoms):
                    unique_atoms.append(atom.element)
        
        for atom in unique_atoms:
            if (atom == "H"):
                basis = "TZVP-MOLOPT-PBE-GTH-q1"
                potential = "GTH-PBE-q1"
            elif (atom == "Al"):
                basis = "TZVP-MOLOPT-PBE-GTH-q3"
                potential = "GTH-PBE-q3"
            elif (atom == "C" or atom == "Sn" or atom == "Pb" or atom == "Si"):
                basis = "TZVP-MOLOPT-PBE-GTH-q4"
                potential = "GTH-PBE-q4"
            elif (atom == "N" or atom == "P"):
                basis = "TZVP-MOLOPT-PBE-GTH-q5"
                potential = "GTH-PBE-q5"
            elif (atom == "O" or atom == "S" or atom == "Se" or atom == "Te"):
                basis = "TZVP-MOLOPT-PBE-GTH-q6"
                potential = "GTH-PBE0-q6"
            elif (atom == "F" or atom == "Cl" or atom == "Br" or atom == "I"):
                basis = "TZVP-MOLOPT-PBE-GTH-q7"
                potential = "GTH-PBE0-q7"
            elif (atom == "He" or atom == "Ne" or atom == "Ar" or atom == "Kr" or atom == "Xe"):
                basis = "TZVP-MOLOPT-PBE-GTH-q8"
                potential = "GTH-PBE-q8"
            elif (atom == "Mg" or atom == "Ca" ):
                basis = "TZVP-MOLOPT-PBE-GTH-q10"
                potential = "GTH-PBE-q10"
            elif (atom == "Cu"):
                basis = "TZVP-MOLOPT-PBE-GTH-q11"
                potential = "GTH-PBE-q11"
            elif (atom == "Zn" or atom == "Zr" or atom == "Cd"):
                basis = "TZVP-MOLOPT-PBE-GTH-q12"
                potential = "GTH-PBE-q12"
            elif (atom == "Cr"):
                basis = "TZVP-MOLOPT-PBE-GTH-q14"
                potential = "GTH-PBE-q14"
            elif (atom == "Mn"):
                basis = "TZVP-MOLOPT-PBE-GTH-q15"
                potential = "GTH-PBE-q15"
            elif (atom == "Fe"):
                basis = "TZVP-MOLOPT-PBE-GTH-q16"
                potential = "GTH-PBE-q6"
            elif (atom == "Co"):
                basis = "TZVP-MOLOPT-PBE-GTH-q17"
                potential = "GTH-PBE-q17"
            elif (atom == "Ni"):
                basis = "TZVP-MOLOPT-PBE-GTH-q18"
                potential = "GTH-PBE-q18"
            else:
                basis = "FIX"
                aux = "FIX"
                potential = "FIX"

            f.write("    &KIND {}\n      BASIS_SET {}\n      POTENTIAL {}\n    &END KIND\n".format(atom,basis,potential))
            f.write("    &KIND {}_ghost\n      BASIS_SET {}\n      GHOST\n    &END KIND\n".format(atom,basis))

        f.write("  &END SUBSYS\n&END FORCE_EVAL\n\n")
        f.close()

        f = open(filename+"input.xyz","w")
        f.write(str(len(frame[0])+len(frame[1]))+"\n\n")
        for mol in frame:
            for atom in mol:
                f.write(atom.element+" "+str(atom.x[0])+" "+str(atom.x[1])+" "+str(atom.x[2])+"\n")
        f.close()

        i += 1

def write_mpmc_thing(giant_array_of_frames,pbc):
    file = open("configs.fit","w")

    i = 0
    for frame in giant_array_of_frames:
        file.write("Configuration "+str(i)+"\nXXX periodic {} {} {} {} {} {}\n".format(pbc.a,pbc.b,pbc.c,pbc.alpha,pbc.beta,pbc.gamma))
        mol_counter = 0
        n = 0
        for mol in frame:
            mol_counter += 1
            for atom in mol:
                file.write(atom.name+" "+str(mol_counter)+" "+str(atom.x[0])+" "+str(atom.x[1])+" "+str(atom.x[2])+" "+str(charges[n])+"\n")
                n += 1
        i += 1
    file.close()

def write_output_files(giant_array_of_frames,pbc):
    write_xyz_file(giant_array_of_frames)
    write_mpmc_thing(giant_array_of_frames,pbc)
    write_cp2k_files(giant_array_of_frames,pbc)

def generate_random_configs(molecule1,molecule2,num_of_configs_per_r,r_start,steps,r_step):
    r = r_start
    i = 0
    giant_array_of_frames = []

    for j in range(steps):
        for k in range(num_of_configs_per_r):
            i += 1
            q1 = Quaternion(0.0,0.0,0.0,1.0)
            q1.random_rotation()
            q2 = Quaternion(0.0,0.0,0.0,1.0)
            q2.random_rotation()
            mol1copy = copy.deepcopy(molecule1)
            mol2copy = copy.deepcopy(molecule2)
            for atom in mol1copy:
                rotated = rotate_3vector(atom.x,q1)
                atom.x = rotated.x[:3]
                atom.x[0] += r/2.0
            for atom in mol2copy:
                rotated = rotate_3vector(atom.x,q2)
                atom.x = rotated.x[:3]
                atom.x[0] -= r/2.0
            frame = [mol1copy,mol2copy]
            giant_array_of_frames.append(frame)
        r = r + r_step
    #write_output_files(giant_array_of_frames)
    return giant_array_of_frames

def generate_periodic_configs(n,mol1,mol2,pbc):
    giant_array_of_frames = []
    for i in range(n):
        overlapping = True
        while(overlapping):
            x = pbc.a*1000.*(random.random()-0.5)
            y = pbc.b*1000.*(random.random()-0.5)
            z = pbc.c*1000.*(random.random()-0.5)
            vec = pbc.min_image_vector(x,y,z)
            q = Quaternion(0.0,0.0,0.0,1.0)
            q.random_rotation()
            mol1copy = copy.deepcopy(mol1)
            mol2copy = copy.deepcopy(mol2)
            for atom in mol1copy:
                rotated = rotate_3vector(atom.x,q)
                atom.x = rotated.x[:3]
                atom.x[0] += vec[0]
                atom.x[1] += vec[1]
                atom.x[2] += vec[2]
            overlapping = False
            for atom1 in mol1copy:
                for atom2 in mol2:
                    r = pbc.min_image(atom1.x[0]-atom2.x[0],atom1.x[1]-atom2.x[1],atom1.x[2]-atom2.x[2])
                    if (r<0.8*0.5*(atom1.vdw+atom2.vdw)):
                        overlapping = True
            if (not overlapping):
                frame = [mol2copy,mol1copy]
                giant_array_of_frames.append(frame)
    return giant_array_of_frames

def wrap_mof(mol,pbc):
    for atom in mol:
        vec = pbc.min_image_vector(atom.x[0],atom.x[1],atom.x[2])
        atom.x[0] = vec[0]
        atom.x[1] = vec[1]
        atom.x[2] = vec[2]

ar = Molecule("Ar",0,1)
ar.append(Atom(0.,0.,0.,"Ar","Ar"))

ne = Molecule("Ne",0,1)
ne.append(Atom(0.,0.,0.,"Ne","Ne"))

kr = Molecule("Kr",0,1)
kr.append(Atom(0.,0.,0.,"Kr","Kr"))

h2 = Molecule("H2",0,1)
h2.append(Atom(0.,0.,0.,"DA","DA"))
h2.append(Atom(0.371,0.,0.,"H2","H"))
h2.append(Atom(-0.371,0.,0.,"H2","H"))

co2 = Molecule("CO2",0,1)
co2.append(Atom(0.,0.,0.,"CO2","C"))
co2.append(Atom(1.1625,0.,0.,"O2C","O"))
co2.append(Atom(-1.1625,0.,0.,"O2C","O"))

n2 = Molecule("N2",0,1)
n2.append(Atom(0.,0.,0.,"DA","DA"))
n2.append(Atom(0.550700,0.,0.,"N2","N"))
n2.append(Atom(-0.550700,0.,0.,"N2","N"))

def load_mol_from_xyz(filename):
    mol = Molecule(filename,0,1)
    if (mol.name == "SIFSIX-3-Cu.xyz"):
        mol.mult = 28
    elif (mol.name == "Ni-MOF-74.xyz"):
        mol.mult = 145
    elif (mol.name == "CROFOUR-3-Ni.xyz"):
        mol.mult = 37
    elif (mol.name == "Trip3Tez.xyz"):
        mol.mult = 1
    elif (mol.name == "cu-atc.xyz"):
        mol.mult = 5
    elif (mol.name == "Cu74_hopt.xyz"):
        mol.mult = 73
    f = open(filename,"r")
    f.readline()
    f.readline()
    line = f.readline()
    while(line):
        tokens = line.split()
        mol.append(Atom(float(tokens[1]),float(tokens[2]),float(tokens[3]),tokens[0],tokens[0]))
        line = f.readline()
    return mol

files = ["Cu74_hopt.xyz"]
charges = loadtxt("charges.txt", dtype='float')

root = os.getcwd()
for filename in files:
    mof = load_mol_from_xyz(filename)
    pbc = None
    if (filename == "CC3.xyz"):
        pbc = PBC(24.805,24.805,24.805,90.,90.,90.)
    elif (filename == "SIFSIX-3-Cu.xyz"):
        pbc = PBC(20.7558,20.7558,23.7183,90.,90.,90.)
    elif (filename == "Ni-MOF-74.xyz"):
        pbc = PBC(25.7856,25.7856,27.0804,90.,90.,120.)
    elif (filename == "CROFOUR-3-Ni.xyz"):
        pbc = PBC(20.5310,20.5310,34.8160,90.,90.,120.)
    elif (filename == "1x1x3.xyz"):
        pbc = PBC(21.423575,21.423596,22.713522,90.000002,89.999998,120.048457)
    elif (filename == "cu-atc.xyz"):
        pbc = PBC(8.457215,8.457348,14.485151,89.999504,89.999794,90.000972)
    elif (filename == "Cu74_hopt.xyz"):
        pbc = PBC(25.9972,25.9972,25.0348,90.0000,90.0000,120.0000)
    wrap_mof(mof,pbc)
    folder = filename.split(".")[0]+"/"
    try:
        os.stat(folder)
    except:
        os.mkdir(folder)
    os.chdir(folder)
    for sorbate in [ne,ar,kr]:
        folder_sorbate = sorbate.name
        cwd = os.getcwd()
        try:
            os.stat(folder_sorbate)
        except:
            os.mkdir(folder_sorbate)
        os.chdir(folder_sorbate)
        giant_array_of_frames = generate_periodic_configs(1000,sorbate,mof,pbc)
        write_output_files(giant_array_of_frames,pbc)
        os.chdir(cwd)
    os.chdir(root)
