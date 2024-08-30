import os
from numpy import *
import copy

class Atom:
    def __init__(self,x,y,z,name,element):
        self.name = name
        self.element = element
        self.x = array([x,y,z])

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


def write_cp2k_files(giant_array_of_frames):
    i = 1
    for frame in giant_array_of_frames:
        filename = "cp2k/"+str(i)+"/"
        if not os.path.exists(filename):
            os.makedirs(filename)

        f = open(filename+"input.xyz","w")
        f.write(str(len(frame[0])+len(frame[1]))+"\n\n")
        for mol in frame:
            for atom in mol:
                f.write(atom.element+" "+str(atom.x[0])+" "+str(atom.x[1])+" "+str(atom.x[2])+"\n")
        f.close()

        i += 1

def write_mpmc_thing(giant_array_of_frames):
    file = open("configs.fit","w")
    i = 0
    for frame in giant_array_of_frames:
        file.write(str(len(frame[0])+len(frame[1]))+"\nXXX periodic 21.423575 21.423596 7.571174 90.000002 89.999998 120.048457\n")
        mol_counter = 0
        n = 0
        for mol in frame:
            mol_counter += 1
            for atom in mol:
                file.write(atom.name+" "+str(mol_counter)+" "+str(atom.x[0])+" "+str(atom.x[1])+" "+str(atom.x[2])+" "+str(charges[n])+"\n")
                n += 1
        i += 1
    file.close()

def write_output_files(giant_array_of_frames):
    write_xyz_file(giant_array_of_frames)
    write_mpmc_thing(giant_array_of_frames)
    write_cp2k_files(giant_array_of_frames)

def generate_random_configs_dont_move_first(molecule1,molecule2,num_of_configs_per_r,r_start,steps,r_step):
    r = r_start
    i = 0
    giant_array_of_frames = []

    for j in range(steps):
        for k in range(num_of_configs_per_r):
            i += 1
            q2 = Quaternion(0.0,0.0,0.0,1.0)
            q2.random_rotation()
            mol1copy = copy.deepcopy(molecule1)
            mol2copy = copy.deepcopy(molecule2)
            for atom in mol1copy:
                atom.x[1] += r/2.0
            for atom in mol2copy:
                rotated = rotate_3vector(atom.x,q2)
                atom.x = rotated.x[:3]
                atom.x[1] -= r/2.0
            frame = [mol1copy,mol2copy]
            giant_array_of_frames.append(frame)
        r = r + r_step
    #write_output_files(giant_array_of_frames)
    return giant_array_of_frames

def read_in_file(molecule1,molecule2,filename):
    file = open(filename,"r")
    lines = file.readlines()
    lines_per_frame = len(molecule1)+len(molecule2)+2
    num_of_frames = len(lines)/lines_per_frame
    giant_array_of_frames = []
    for i in range(num_of_frames):
        mol1copy = copy.deepcopy(molecule1)
        mol2copy = copy.deepcopy(molecule2)
        for j in range(len(mol1copy)):
            line = lines[i*lines_per_frame+2+j]
            line = line.split()
            mol1copy[j].x[0] = line[1]
            mol1copy[j].x[1] = line[2]
            mol1copy[j].x[2] = line[3]
        for j in range(len(mol2copy)):
            line = lines[i*lines_per_frame+2+j+len(mol1copy)]
            line = line.split()
            mol2copy[j].x[0] = line[1]
            mol2copy[j].x[1] = line[2]
            mol2copy[j].x[2] = line[3]
        frame = [mol1copy,mol2copy]
        giant_array_of_frames.append(frame)
    return giant_array_of_frames

def find_mass(element):
    if (element=="H"):
        return 1.00794
    if (element=="C"):
        return 12.0107
    if (element=="O"):
        return 15.9994
    if (element=="N"):
        return 14.0067
    if (element=="S"):
        return 32.065
    if (element=="Te"):
        return 127.6
    raise ValueError('Unknown element in find_mass')
    return

def move_to_com(molecule):
    com = [0.0,0.0,0.0]
    total_mass = 0
    for atom in molecule:
        com[0] += atom.x[0]*find_mass(atom.element)
        com[1] += atom.x[1]*find_mass(atom.element)
        com[2] += atom.x[2]*find_mass(atom.element)
        total_mass += find_mass(atom.element)
    com[0] /= total_mass
    com[1] /= total_mass
    com[2] /= total_mass
    for atom in molecule:
        atom.x[0] -= com[0]
        atom.x[1] -= com[1]
        atom.x[2] -= com[2]


def point_molecule_down_xaxis(molecule, atom_pointer):

    atom_copy = copy.deepcopy(atom_pointer)

    xangle = arctan2(atom_copy.x[1],atom_copy.x[0])
    qx = Quaternion(0.0,0.0,0.0,1.0)
    qx.axis_angle(0,0,1,-xangle*57.2957795)
    
    rotated = rotate_3vector(atom_copy.x,qx)
    atom_copy.x = rotated.x[:3]

    yangle = arctan2(atom_copy.x[2],atom_copy.x[0])
    qy = Quaternion(0.0,0.0,0.0,1.0)
    qy.axis_angle(0,1,0,yangle*57.2957795)
    
    for atom in molecule:
        rotated = rotate_3vector(atom.x,qx)
        atom.x = rotated.x[:3]
        rotated = rotate_3vector(atom.x,qy)
        atom.x = rotated.x[:3]

# Guest molecule to insert
He = []
He.append(Atom(0.,0.,0.,"He","He"))

Ne = []
Ne.append(Atom(0.,0.,0.,"Ne","Ne"))

Ar = []
Ar.append(Atom(0.,0.,0.,"Ar","Ar"))

Kr = []
Kr.append(Atom(0.,0.,0.,"Kr","Kr"))

Xe = []
Xe.append(Atom(0.,0.,0.,"Xe","Xe"))

H2 = []
H2.append(Atom(0.,0.,0.,"H2DA","DA"))
H2.append(Atom(0.371,0.,0.,"H2","H"))
H2.append(Atom(-0.371,0.,0.,"H2","H"))

N2 = []
N2.append(Atom(0.,0.,0.,"N2DA","DA"))
N2.append(Atom(0.5507,0.,0.,"N2","N"))
N2.append(Atom(-0.5507,0.,0.,"N2","N"))

# Insert structure below with the following format:
#cd.append(Atom(	-0.0109882282,16.18205299,4.935645022	, "C", "C" ))  

cd = []

move_to_com(cd)

point_molecule_down_xaxis(cd, cd[144])

copy_atom = copy.deepcopy(cd[144])

for atom in cd:
    atom.x = atom.x - copy_atom.x

# Read in charges from separate file
charges = loadtxt("charges.txt", dtype='float')
cwd = os.getcwd()

folder = "ne"
try:
    os.stat(folder)
except:
    os.mkdir(folder)
os.chdir(folder)
giant_array_of_frames = generate_random_configs_dont_move_first(cd,Ne,1,3.0,51,0.1)
write_output_files(giant_array_of_frames)
os.chdir(cwd)

folder = "ar"
try:
    os.stat(folder)
except:
    os.mkdir(folder)
os.chdir(folder)
giant_array_of_frames = generate_random_configs_dont_move_first(cd,Ar,1,3.0,21,0.1)
write_output_files(giant_array_of_frames)
os.chdir(cwd)

folder = "kr"
try:
    os.stat(folder)
except:
    os.mkdir(folder)
os.chdir(folder)
giant_array_of_frames = generate_random_configs_dont_move_first(cd,Kr,1,3.0,21,0.1)
write_output_files(giant_array_of_frames)
os.chdir(cwd)

folder = "xe"
try:
    os.stat(folder)
except:
    os.mkdir(folder)
os.chdir(folder)
giant_array_of_frames = generate_random_configs_dont_move_first(cd,Xe,1,3.0,21,0.1)
write_output_files(giant_array_of_frames)
os.chdir(cwd)

folder = "h2"
try:
    os.stat(folder)
except:
    os.mkdir(folder)
os.chdir(folder)
giant_array_of_frames = generate_random_configs_dont_move_first(cd,H2,10,3.0,21,0.1)
write_output_files(giant_array_of_frames)
os.chdir(cwd)

folder = "n2"
try:
    os.stat(folder)
except:
    os.mkdir(folder)
os.chdir(folder)
giant_array_of_frames = generate_random_configs_dont_move_first(cd,N2,10,3.0,21,0.1)
write_output_files(giant_array_of_frames)
os.chdir(cwd)