import subprocess
import numpy as np
import random
import math
from scipy.spatial.transform import Rotation as R

filename = 'data.net2422R0'  ## datafile name?
subprocess.run(f'cp /ocean/projects/che150015p/xwei/PNN_heat/set_ref/R0/data.net2422R0_r2 ./{filename}', shell=True, capture_output=True, text=True)

#with open(filename, 'r') as f:
#    lines = f.readlines()
#lines[9] =    '29 dihedral types\n'
#lines[11] = '-500 500 xlo xhi\n'
#lines[12] = '-500 500 ylo yhi\n'
#lines[13] = '-500 500 zlo zhi\n'
#lines[17] = '1 12.011\n'
#lines[18] = '2 12.011\n'
#lines[19] = '3 1.008\n'
#lines[21] = '5 1.008\n'
#lines[22] = '6 12.011\n'

#with open(filename, 'w') as f:
#    f.writelines(lines)

import matplotlib.pyplot as plt

# Your latest points
points = [
    (-0.19967117090029568,-0.13840381379172118,1),
    (-0.061651409667908816,0.046249308084957134,1),
    (0.08430987727110051,0.19372830702602856,1),
    (0.7472579197940861,-0.36771145282667866,0),
    (0.7386685283033411,0.3338090505537078,0),
    (-0.5638946154521689, -0.4374498012543361, 0),
]

def write_in1(write1):
    config_text_part1 = f"""
#set thermostats on AuNPs
variable name index {infile_name}
log log.${{name}}
    variable Tset equal  {T}
    variable rand1 equal {rand1}
    variable T1 equal    {T1}
    variable T2 equal    {T2}
    variable T3 equal    {T3}
    variable T4 equal    {T4}
    variable T5 equal    {T5}
    variable T6 equal    {T6}
    variable T7 equal    {T7}
    variable T8 equal    {T8}
    variable T9 equal    {T9}
    variable T10 equal    {T10}
variable vstep equal 0.25
variable Pdamp  equal  ${{vstep}}*1000

variable Tdamp1  equal  ${{vstep}}*10000
variable Tdamp2  equal  ${{vstep}}*10000
variable Tdamp3  equal  ${{vstep}}*10000
variable Tdamp4  equal  ${{vstep}}*10000
variable Tdamp5  equal  ${{vstep}}*10000
variable Tdamp6  equal  ${{vstep}}*10000
variable Tdamp7  equal  ${{vstep}}*10000
variable Tdamp8  equal  ${{vstep}}*10000
variable Tdamp9  equal  ${{vstep}}*10000
variable Tdamp10  equal  ${{vstep}}*10000

# ----------------- Init Section -----------------

    units real
    atom_style full
    bond_style hybrid harmonic
    angle_style hybrid harmonic
    dihedral_style hybrid opls
    pair_style hybrid lj/cut/coul/cut 10.0 10.0 lj/cut 10.0 morse 8.0 
    pair_modify mix arithmetic
    special_bonds lj/coul 0.0 0.0 0.5
    #kspace_style pppm 0.0001

# ----------------- Atom Definition Section -----------------

read_data ../{filename}
#read_restart res.
# ----------------- Settings Section -----------------

    pair_coeff 1 1 lj/cut/coul/cut 0.066 3.5
    pair_coeff 2 2 lj/cut/coul/cut 0.066 3.5
    pair_coeff 3 3 lj/cut/coul/cut 0.03 2.5
    pair_coeff 4 4 lj/cut/coul/cut 0.25 3.55
    pair_coeff 5 5 lj/cut/coul/cut 0.0 0.0
    pair_coeff 6 6 lj/cut/coul/cut 0.066 3.5
    pair_coeff 7 7 lj/cut 5.29 2.951  #Au-other LJ 0.039 2.935  #Au-Au morse 10.954 1.583 3.024 8 
    pair_coeff 4 7 morse 8.763 1.47 2.65 8    #Au-S
pair_coeff  1   7   lj/cut 0.050734604 3.2175  
pair_coeff  2   7   lj/cut 0.050734604 3.2175
pair_coeff  3   7   lj/cut 0.034205263 2.7175
pair_coeff  5   7   lj/cut 0   1.4675
pair_coeff  6   7   lj/cut 0.050734604 3.2175


    bond_coeff 1 harmonic 268.0 1.529 #c2-c1
    bond_coeff 2 harmonic 268.0 1.529 #c2-c2
    bond_coeff 3 harmonic 222.0 1.81
    bond_coeff 4 harmonic 340.0 1.09
    bond_coeff 5 harmonic 340.0 1.09

    angle_coeff 1 harmonic 58.35 112.7 #c2-c2-c1
    angle_coeff 2 harmonic 58.35 112.7 #c2-c2-c2
    angle_coeff 3 harmonic 50.0 108.6
    angle_coeff 4 harmonic 33.0 107.8
    angle_coeff 5 harmonic 33.0 107.8
    angle_coeff 6 harmonic 35.0 109.5
    angle_coeff 7 harmonic 37.5 110.7
    angle_coeff 8 harmonic 37.5 110.7
    angle_coeff 9 harmonic 37.5 110.7

    dihedral_coeff 1 opls 1.3 -0.05 0.2 0.0 #c2-c2-c1-c2
    dihedral_coeff 2 opls 1.3 -0.05 0.2 0.0 #c2-c2-c2-c2
    dihedral_coeff 3 opls 1.262 -0.198 0.465 0.0
    dihedral_coeff 4 opls 0.0 0.0 0.3 0.0
    dihedral_coeff 5 opls 0.0 0.0 0.3 0.0
    dihedral_coeff 6 opls 0.0 0.0 0.3 0.0
    dihedral_coeff 7 opls 0.0 0.0 0.452 0.0
    dihedral_coeff 8 opls 0.0 0.0 0.3 0.0
    dihedral_coeff 9 opls 0.0 0.0 0.3 0.0

    dihedral_coeff 10 opls {k1} -0.05 {k1_2} 0.0 #c2-c2-c2-c2
    dihedral_coeff 11 opls {k2} -0.05 {k2_2} 0.0 #c2-c2-c2-c2
    dihedral_coeff 12 opls {k3} -0.05 {k3_2} 0.0 #c2-c2-c2-c2
    dihedral_coeff 13 opls {k4} -0.05 {k4_2} 0.0 #c2-c2-c2-c2
    dihedral_coeff 14 opls {k5} -0.05 {k5_2} 0.0 #c2-c2-c2-c2
    dihedral_coeff 15 opls {k6} -0.05 {k6_2} 0.0 #c2-c2-c2-c2

    dihedral_coeff 16 opls {k7} -0.05 {k7_2} 0.0 #c2-c2-c2-c2
    dihedral_coeff 17 opls {k8} -0.05 {k8_2} 0.0 #c2-c2-c2-c2
    dihedral_coeff 18 opls {k9} -0.05 {k9_2} 0.0 #c2-c2-c2-c2
    dihedral_coeff 19 opls {k10} -0.05 {k10_2} 0.0 #c2-c2-c2-c2
    dihedral_coeff 20 opls {k11} -0.05 {k11_2} 0.0 #c2-c2-c2-c2
    dihedral_coeff 21 opls {k12} -0.05 {k12_2} 0.0 #c2-c2-c2-c2

    dihedral_coeff 22 opls {k13} -0.05 {k13_2} 0.0 #c2-c2-c2-c2
    dihedral_coeff 23 opls {k14} -0.05 {k14_2} 0.0 #c2-c2-c2-c2
    dihedral_coeff 24 opls {k15} -0.05 {k15_2} 0.0 #c2-c2-c2-c2
    dihedral_coeff 25 opls {k16} -0.05 {k16_2} 0.0 #c2-c2-c2-c2
    
    dihedral_coeff 26 opls {k17} -0.05 {k17_2} 0.0 #c2-c2-c2-c2
    dihedral_coeff 27 opls {k18} -0.05 {k18_2} 0.0 #c2-c2-c2-c2
    dihedral_coeff 28 opls {k19} -0.05 {k19_2} 0.0 #c2-c2-c2-c2
    dihedral_coeff 29 opls {k20} -0.05 {k20_2} 0.0 #c2-c2-c2-c2
"""
# Write the entire configuration to the main config file
    with open(infile, 'w') as file:
        file.write(config_text_part1)

def write_in2(write2):
    config_text_part2 = f"""
# ----------------- Run Section -----------------

restart 100000 res.1 res.2 

#write_data data.

timestep ${{vstep}}  
###
group p1 molecule 1
group p2 molecule 3
group p3 molecule 29
group p4 molecule 31

group p5 molecule 30
group p6 molecule 32
group p7 molecule 15
group p8 molecule 17
#
group p9 molecule 4
group p10 molecule 5

group p11 molecule 2
group p12 molecule 7

group p13 molecule 19
group p14 molecule 18

group p15 molecule 21
group p16 molecule 16
#
group p17 molecule 8
group p18 molecule 20

group p19 molecule 6
group p20 molecule 22
###
group NP1 molecule 9
group NP2 molecule 23

group NP3 molecule 10
group NP4 molecule 11
group NP5 molecule 24
group NP6 molecule 25

group NP7 molecule 12
group NP8 molecule 26

group NP9 molecule 14
group NP10 molecule 28
###
group polys type 1 2 3 4 5 6 
#group NPs type 7 


set group p1 dihedral 10
set group p2 dihedral 11
set group p3 dihedral 12
set group p4 dihedral 13
set group p5 dihedral 14
set group p6 dihedral 15
set group p7 dihedral 16
set group p8 dihedral 17
set group p9 dihedral 18
set group p10 dihedral 19
set group p11 dihedral 20
set group p12 dihedral 21
set group p13 dihedral 22
set group p14 dihedral 23
set group p15 dihedral 24
set group p16 dihedral 25
set group p17 dihedral 26
set group p18 dihedral 27
set group p19 dihedral 28
set group p20 dihedral 29

#dump 1 p1 xyz 100000 ${{name}}_p1.xyz
#dump_modify 1 element C C H S H C Au
#dump 2 p2 xyz 100000 ${{name}}_p2.xyz
#dump_modify 2 element C C H S H C Au
#dump 3 p3 xyz 100000 ${{name}}_p3.xyz
#dump_modify 3 element C C H S H C Au
#dump 4 p4 xyz 100000 ${{name}}_p4.xyz
#dump_modify 4 element C C H S H C Au
#dump 5 p5 xyz 100000 ${{name}}_p5.xyz
#dump_modify 5 element C C H S H C Au
#dump 6 p6 xyz 100000 ${{name}}_p6.xyz
#dump_modify 6 element C C H S H C Au
#dump 7 p7 xyz 100000 ${{name}}_p7.xyz
#dump_modify 7 element C C H S H C Au
#dump 8 p8 xyz 100000 ${{name}}_p8.xyz
#dump_modify 8 element C C H S H C Au
#dump 9 p9 xyz 100000 ${{name}}_p9.xyz
#dump_modify 9 element C C H S H C Au
#dump 10 p10 xyz 100000 ${{name}}_p10.xyz
#dump_modify 10 element C C H S H C Au
#dump 11 p11 xyz 100000 ${{name}}_p11.xyz
#dump_modify 11 element C C H S H C Au
#dump 12 p12 xyz 100000 ${{name}}_p12.xyz
#dump_modify 12 element C C H S H C Au
#dump 13 p13 xyz 100000 ${{name}}_p13.xyz
#dump_modify 13 element C C H S H C Au
#dump 14 p14 xyz 100000 ${{name}}_p14.xyz
#dump_modify 14 element C C H S H C Au
#dump 15 p15 xyz 100000 ${{name}}_p15.xyz
#dump_modify 15 element C C H S H C Au
#dump 16 p16 xyz 100000 ${{name}}_p16.xyz
#dump_modify 16 element C C H S H C Au
#dump 17 p17 xyz 100000 ${{name}}_p17.xyz
#dump_modify 17 element C C H S H C Au
#dump 18 p18 xyz 100000 ${{name}}_p18.xyz
#dump_modify 18 element C C H S H C Au
#dump 19 p19 xyz 100000 ${{name}}_p19.xyz
#dump_modify 19 element C C H S H C Au
#dump 20 p20 xyz 100000 ${{name}}_p20.xyz
#dump_modify 20 element C C H S H C Au

#dump 21 NPs xyz 100000 ${{name}}_NPs.xyz
#dump_modify 21 element C C H S H C Au 
dump 22 all xyz 1000000 ${{name}}_polys.xyz
dump_modify 22 element C C H S H C Au 

neighbor           2.5 bin
neigh_modify every 1 delay 0 check yes 
thermo 1000

fix NVE all nve

compute ke0 all ke/atom
variable temp0 atom c_ke0/0.00297881
compute ctemp all reduce ave v_temp0
variable temp0c equal c_ctemp

compute temp1 NP1 temp
variable vtemp1 equal c_temp1
compute temp2 NP2 temp
variable vtemp2 equal c_temp2

compute temp3 NP3 temp
variable vtemp3 equal c_temp3
compute temp4 NP4 temp
variable vtemp4 equal c_temp4
compute temp5 NP5 temp
variable vtemp5 equal c_temp5
compute temp6 NP6 temp
variable vtemp6 equal c_temp6

compute temp7 NP7 temp
variable vtemp7 equal c_temp7
compute temp8 NP8 temp
variable vtemp8 equal c_temp8

compute temp9 NP9 temp
variable vtemp9 equal c_temp9
compute temp10 NP10 temp
variable vtemp10 equal c_temp10

fix fix1 NP1 langevin ${{T1}} ${{T1}} ${{Tdamp1}} ${{rand1}}+1 tally yes
fix fix2 NP2 langevin ${{T2}} ${{T2}} ${{Tdamp2}} ${{rand1}}+2 tally yes
fix fix3 NP3 langevin ${{T3}} ${{T3}} ${{Tdamp3}} ${{rand1}}+3 tally yes
fix fix4 NP4 langevin ${{T4}} ${{T4}} ${{Tdamp4}} ${{rand1}}+4 tally yes
fix fix5 NP5 langevin ${{T5}} ${{T5}} ${{Tdamp5}} ${{rand1}}+5 tally yes
fix fix6 NP6 langevin ${{T6}} ${{T6}} ${{Tdamp6}} ${{rand1}}+6 tally yes
fix fix7 NP7 langevin ${{T7}} ${{T7}} ${{Tdamp7}} ${{rand1}}+7 tally yes
fix fix8 NP8 langevin ${{T8}} ${{T8}} ${{Tdamp8}} ${{rand1}}+8 tally yes
fix fix9 NP9 langevin ${{T9}} ${{T9}} ${{Tdamp9}} ${{rand1}}+9 tally yes
fix fix10 NP10 langevin ${{T10}} ${{T10}} ${{Tdamp10}} ${{rand1}}+10 tally yes

fix ftempout all ave/time 1 10000 10000 v_temp0c f_fix1 f_fix2 f_fix3 f_fix4 f_fix5 f_fix6 f_fix7 f_fix8 f_fix9 f_fix10 file ${{name}}_flux.txt mode scalar

log ${{name}}_heat.log
thermo 10000
thermo_style custom step temp epair pe etotal
thermo_modify flush yes
run 12000000
unfix NVE

write_data data.${{name}}
"""
# Write the entire configuration to the main config file
    with open(infile, 'a') as file:
        file.write(config_text_part2)
        
######################################################################################
######################################################################################
import itertools
import random
import math
import time
import csv
import os

######################### set x1 x2 using 6 random (3in+3out) data ###########
data1 = 1
x1 = points[data1-1][0]*25
x2 = points[data1-1][1]*25

random.seed(42)   # set the seed (any integer)

# Create separate generator for loop randomness
random_gen = random.Random()
random_gen.seed(int(time.time() * 1000000) % (2**32))

# Total number of combinations (20 choose 10)
W_choices = math.comb(20, 10)
print(f"Total number of W choices: {W_choices}")

W_all_combos = list(itertools.combinations(range(1, 21), 10))
W_sampled_combos = random.sample(W_all_combos, 100)

T_total = math.comb(8, 4)
print(f"Total number of T choices: {T_total}")

T_all_combos = list(itertools.combinations(range(1, 9), 4))
T_sampled_combos = random.sample(T_all_combos, 8)

# Build all 800 pairings
pairings = []
for i, T_combo in enumerate(T_sampled_combos, 1):
    for j, W_combo in enumerate(W_sampled_combos, 1):
        pairings.append((f"T{i}", T_combo, f"W{j}", W_combo))

# Randomly select 400 from 800 pairings
sampled_pairings = random.sample(pairings, 400)

print(f"\nProcessing {len(sampled_pairings)} pairings:")
print("=" * 80)

# Example points data (replace with your actual points)
#points = [(1, 2), (3, 4), (5, 6)]
# Prepare data for CSV
csv_data = []

# Process each pairing
for pairing_num, (t_label, t_combo, w_label, w_combo) in enumerate(sampled_pairings, 1):
    # Initialize delT variables (delT1-delT8)
    delT = [0.0] * 8  # indices 0-7 correspond to delT1-delT8
    
    # Set selected delT indices to 5K bias
    for t_index in t_combo:
        delT[t_index - 1] = round(5 * random.uniform(-1, 1), 2)
    
    # Initialize W variables (W1-W20)
    W = [0.0] * 20  # indices 0-19 correspond to W1-W20
    
    # Set selected W indices to 9.0
    for w_index in w_combo:
        W[w_index - 1] = round(9 * random.uniform(0, 1), 3)  # convert 1-based to 0-based indexing
    
    # Follow-up calculations for this pairing
    #data1 = 1
    #x1 = points[data1-1][0]*25
    #x2 = points[data1-1][1]*25
    
    # Calculate T1-T10
    rand1 = random_gen.randint(1, 9999999)
    T  = 300
    T1 = round(x1+300,2)
    T2 = round(x2+300,2)
    T3 = delT[0]+250  # delT1
    T4 = delT[1]+250  # delT2
    T5 = delT[2]+250  # delT3
    T6 = delT[3]+250  # delT4
    T7 = delT[4]+200  # delT5
    T8 = delT[5]+200  # delT6
    T9 = delT[6]+150  # delT7
    T10= delT[7]+150  # delT8
    
    # Create k1 through k20
    k = [round(W[i],3) for i in range(20)]
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20 = k
    
    # Create k12, k22, k32, ... k202 (the second set)
    k2_array = [round(k[i]/5,3) for i in range(20)]
    k1_2, k2_2, k3_2, k4_2, k5_2, k6_2, k7_2, k8_2, k9_2, k10_2, k11_2, k12_2, k13_2, k14_2, k15_2, k16_2, k17_2, k18_2, k19_2, k20_2 = k2_array
    
    # Create row for CSV
    row = [pairing_num, rand1, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10,
           k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20]
    
    csv_data.append(row)

    # Add any other calculations here using T1-T10, k1-k20, k12-k202
    # For example:
    # result = your_function(T1, T2, T3, ..., k1, k2, k3, ..., k12, k22, k32, ...)
    # Create directory for this pairing
    dir_name = f'set{pairing_num}'
    os.makedirs(dir_name, exist_ok=True)  # exist_ok=True prevents error if directory already exists
    # Now you can write files inside the directory
    infile = os.path.join(dir_name, f'in.net2422R0_set{pairing_num}')
    infile_name = f'net2422R0_set{pairing_num}'
    #infile = f'net2422R0_set{pairing_num}'
    write_in1(infile)
    write_in2(infile)

    # Copy submit.sh to the folder
    copy_cmd = f'cp /ocean/projects/che150015p/xwei/PNN_heat/set_ref/R0/submitB.sh {dir_name}/submitB.sh'
    result = subprocess.run(copy_cmd, shell=True, capture_output=True, text=True)
    #if result.returncode != 0:
    #    print(f"Error copying to {dir_name}: {result.stderr}")
    #    continue

    sed_cmd = f"sed -i '3s|.*|#SBATCH -J {dir_name}|' {dir_name}/submitB.sh"
    result = subprocess.run(sed_cmd, shell=True, capture_output=True, text=True)
    sed_cmd = f"sed -i '4s|.*|#SBATCH --output=\"PNN_PEdihedral_R0{dir_name}.%j.%N.out\"|' {dir_name}/submitB.sh"
    result = subprocess.run(sed_cmd, shell=True, capture_output=True, text=True)
    sed_cmd = f"sed -i '9s|.*|#SBATCH --ntasks-per-node=4|' {dir_name}/submitB.sh"
    result = subprocess.run(sed_cmd, shell=True, capture_output=True, text=True)
    # Edit line 14 in submitB.sh
    sed_cmd = f"sed -i '14s|.*|mpirun -n $SLURM_NTASKS lmp -in in.net2422R0_{dir_name}|' {dir_name}/submitB.sh"
    result = subprocess.run(sed_cmd, shell=True, capture_output=True, text=True)
    #if result.returncode != 0:
    #    print(f"Error editing submitB.sh in {dir_name}: {result.stderr}")
    #    continue

    # Submit the job using sbatch
    submit_cmd = f"sbatch --chdir={dir_name} submitB.sh"
    result = subprocess.run(submit_cmd, shell=True, capture_output=True, text=True)

    # Display results for first few pairings
    if pairing_num <= 5:
        print(f"\nPairing {pairing_num}: {t_label} {t_combo} --- {w_label} {w_combo}")
        print(f"T1={T1}, T2={T2}, T3={T3}, T4={T4}, T5={T5}")
        delT_selected = [f"delT{i+1}={delT[i]}" for i in range(8) if delT[i] != 0.0]
        print(f"Selected delT: {', '.join(delT_selected)}")
    elif pairing_num == 6:
        print(f"\n... Processing remaining {len(sampled_pairings)-5} pairings ...")

# Write to CSV file
csv_filename = 'pairing_results.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write header
    header = ['pairing_num', 'rand1', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10',
              'k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'k9', 'k10', 
              'k11', 'k12', 'k13', 'k14', 'k15', 'k16', 'k17', 'k18', 'k19', 'k20']
    writer.writerow(header)
    
    # Write all data
    writer.writerows(csv_data)
