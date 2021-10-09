#!/usr/bin/env python3
import os
import mdtraj as md
import numpy as np
import glob
import pickle
import pathlib
import re
import shutil
import time
import heapq
import argparse
import MDAnalysis as mda


"""
This file can make rid dir for given molecular.
Please modify 'pdbname'.
Last update date: 2021/2/24
Author: Dongdong Wang, Yanze Wang.
"""
num_sol = None
box_size = []
num_Na, num_Cl = None, None

def replace(file_name, pattern, subst):
    """
    Replace string in file_name. from pattern to subst. pattern is written by Regular Expression.
    """
    file_handel = open(file_name, 'r')
    file_string = file_handel.read()
    file_handel.close()
    file_string = (re.sub(pattern, subst, file_string))
    file_handel = open(file_name, 'w')
    file_handel.write(file_string)
    file_handel.close()


def get_all_dihedral_index(file_path):
    u = mda.Universe(file_path)
    all_res_list = []
    for seg in u.segments:
        chain_res_list = seg.residues.resindices
        if len(chain_res_list) <= 2:
            continue
        else:
            all_res_list += chain_res_list[1:-1].tolist()
    print("The dihedral angle indexes selected are:", all_res_list)
    return all_res_list


def change_his(pdbname):
    """
    This function can change all the HIS residues to HSD residues in pdbfile(pdbname). 
    it's used to meet the need of force field file. Some pdbs have different H+ sites on HIS residues, varying from HSD to HSP.
    """
    with open(pdbname, 'r') as pdb:
        ret = pdb.read()
    ret = ret.replace('HIS', 'HSD')
    with open(pdbname, 'w') as pdb:
        pdb.write(ret)


def run_md(pdbname, loop=0):
    """
    Let molecule in pdb files go into the equilibrium state through em, nvt and npt simulations. The boxes information, solvent, ions are added too.
    All initial structures and walkers have the exact same solvent number, ion number, ion type and box size. Concentration of saline is set as 0.15M.
    For this purpose, we record the information of the first structure as the tamplate.
    """
    global num_sol, box_size, num_Na, num_Cl
    initial_file = pdbname
    
    # if not os.path.exists("topol.top"):
    print("topol.top not found, generate one.")
    # os.system('echo -e "1\n1\n" | gmx pdb2gmx -f %s.pdb -o processed.gro -ignh -heavyh > grompp.log 2>&1' % pdbname)
    os.system('echo -e "1\n1\n" | gmx pdb2gmx -f %s.pdb -o processed.gro -ignh > grompp.log 2>&1' % pdbname)
    initial_file = "processed.gro"
    
    if loop == 0:
        print('gmx editconf -f {} -o newbox.gro -d 0.9 -c -bt triclinic'.format(initial_file))
        os.system(
            'gmx editconf -f {} -o newbox.gro -d 0.9 -c -bt triclinic'.format(initial_file))
        print('gmx solvate -cp newbox.gro -cs spc216.gro -o solv.gro -p topol.top > sol.log 2>&1')
        os.system(
            'gmx solvate -cp newbox.gro -cs spc216.gro -o solv.gro -p topol.top > sol.log 2>&1')
        with open('solv.gro', 'r') as sol_gro:
            for line in sol_gro.readlines():
                info = line.split()
                # print(info)
                if len(info) == 3:
                    if all([all([j.isdigit() for j in i.split('.')]) for i in info]):
                        box_size = [float(k)+0.10000 for k in info]

        with open('topol.top', 'r') as top:
            for line in top.readlines():
                line_sp = line.split()
                if line_sp == []:
                    continue
                if line.split()[0] == 'SOL' and line_sp[1].isdigit():
                    num_sol = line_sp[1]
        print('Max number of solvents is:', num_sol)
        os.system(
            'gmx grompp -f ions.mdp -c solv.gro -p topol.top -o ions.tpr -maxwarn 2 > grompp_ion.log 2>&1')
        os.system(
            'echo -e "13\n" | gmx genion -s ions.tpr -o solv_ions.gro -p topol.top -pname NA -nname CL -neutral -conc 0.15')
        with open('topol.top', 'r') as top:
            for line in top.readlines():
                line_sp = line.split()
                if line_sp == []:
                    continue
                if line.split()[0] == 'NA':
                    num_Na = line_sp[1]
                if line.split()[0] == 'CL':
                    num_Cl = line_sp[1]
        with open('../box_information.txt', 'w') as box_info:
            box_info.write('num_sol={}\nbox_size={},{},{}\nnum_Na={}\nnum_Cl={}'.format(
                num_sol, box_size[0], box_size[1], box_size[2], num_Na, num_Cl))
    else:
        print('gmx editconf -f {} -o newbox.gro -box {} {} {} -c -bt triclinic'.format(
            initial_file, box_size[0], box_size[1], box_size[2]))
        os.system('gmx editconf -f {} -o newbox.gro -box {} {} {} -c -bt triclinic'.format(
            initial_file, box_size[0], box_size[1], box_size[2]))
        print('gmx solvate -cp newbox.gro -cs spc216.gro -o solv.gro -p topol.top > sol.log 2>&1')
        os.system(
            'gmx solvate -cp newbox.gro -cs spc216.gro -maxsol {} -o solv.gro -p topol.top > sol.log 2>&1'.format(int(num_sol)))

        with open('topol.top', 'r') as top:
            for line in top.readlines():
                line_sp = line.split()
                if line_sp == []:
                    continue
                if line.split()[0] == 'SOL' and line_sp[1].isdigit():
                    print('Max number of solvents is:', line_sp[1])

        os.system(
            'gmx grompp -f ions.mdp -c solv.gro -p topol.top -o ions.tpr -maxwarn 2 > grompp_ion.log 2>&1')
        os.system('echo -e "13\n" | gmx genion -s ions.tpr -o solv_ions.gro -p topol.top -pname NA -nname CL -neutral -np {} -nn {}'.format(num_Na, num_Cl))

    os.system('gmx grompp -f minim.mdp -c solv_ions.gro -p topol.top -o em.tpr -maxwarn 1 > grompp_em.log 2>&1')
    # os.system('gmx mdrun -deffnm em -v -nt 4')
    os.system('gmx mdrun -deffnm em -v -ntmpi 1 -nt 4')
    os.system('gmx grompp -f nvt.mdp -c em.gro -p topol.top -o nvt.tpr -r em.gro -maxwarn 1 > grompp_nvt.log 2>&1')
    command = 'gmx mdrun -deffnm nvt -ntmpi 1 -v -nt 4'
    os.system(command)
    os.system('gmx grompp -f npt.mdp -c nvt.gro -t nvt.cpt -p topol.top -o npt.tpr -r nvt.gro -maxwarn 1 > grompp_npt.log 2>&1')
    command = 'gmx mdrun -deffnm npt -ntmpi 1 -v -nt 4'
    os.system(command)
    # os.system('gmx mdrun -deffnm npt -v -nt 4')
    os.system('cp topol.top topol.top.bak')


def mk_posre(dirname, dih_list, bottom_width=0.4):
    # 1~119  122~228
    # print(list_biased_ang)
    os.system('cp %s/source/jsons/phipsi_selected.json ./' % dirname)
    replace('phipsi_selected.json', '.*selected_index.*',
            '    "selected_index":  %s,' % dih_list)
    structure = 'nvt.gro'
    #   kappa=0.025      #kcal/mol/A2   *4.184*100
    # kappa=15             #kj/mol/nm2
    t_ref = md.load(structure, top=structure)
    topology = t_ref.topology
    ca_atoms = topology.select('name CA')+1
    wf = open('posre.itp.templ', 'w')
    wf.write('[ position_restraints ]\n;  i funct       g         r(nm)       k\n')
    for i in range(len(ca_atoms)):
        wf.write('%d    2        1          %f       TEMP\n' %
                 (ca_atoms[i], bottom_width))
    wf.close()


def mk_rid(dirname, pdbname):
    mol_dir = os.path.join(dirname, 'source/mol/', pdbname)
    print('mol_dir', mol_dir)
    print('pdbname', pdbname)
    print('dirname', dirname)
    pathlib.Path(mol_dir).mkdir(parents=True, exist_ok=True)
    case_path_list = [x for x in glob.glob("./*") if os.path.isdir(x)]
    assert len(case_path_list) > 0
    os.system('cp %s/topol.top %s' % (case_path_list[0], mol_dir))
    os.system('cp %s/*.itp %s' % (case_path_list[0], mol_dir))
    
    for i in range(len(case_path_list)):
        os.system('cp %s/npt.gro %s/conf00%d.gro' % (case_path_list[i], mol_dir, i))
    if len(case_path_list) < 8:
        for j in range(len(case_path_list), 8):
            os.system('cp %s/npt.gro %s/conf00%d.gro' % (case_path_list[0], mol_dir, j))
    os.system('cp %s/npt.gro %s/conf.gro' % (case_path_list[0], mol_dir))
    os.system('cp %s/posre.itp.templ %s/posre.itp' % (case_path_list[0], mol_dir))
    os.system('cp %s/source/mol/*.mdp %s' % (dirname, mol_dir))
    os.chdir('%s/source/' % dirname)#ooi
    os.system('python gen.py rid ./jsons/default_gen.json %s/%s/%s/phipsi_selected.json ./mol/%s/ -o %s/%s.run06' %
              (dirname, pdbname, os.path.basename(case_path_list[0]), pdbname, dirname, pdbname))
    all_itp = glob.glob(os.path.join(dirname, "source", "mol", pdbname, "*.itp"))
    for itp in all_itp:
        shutil.copy(itp, "{}/{}.run06/template/mol".format(dirname, pdbname))
    os.chdir('%s/%s' % (dirname, pdbname))


def mk_score(where_rw_dir, where_evo_path, target):
    '''
    generate rwplus dir in *.run dir. 3 files (calRWplus, rw.dat, scb,dat) should be in where_rw_dir.
    Args:
            where_sco_dir: containing rwplus files.
            target: name of protein.
    '''
    score_dir = './{}.run06/score'.format(target)  # where they will be copied to.
    if os.path.exists(score_dir):
        shutil.rmtree(score_dir)
    os.mkdir(score_dir)
    os.system('cp -r {}/calRWplus {}'.format(where_rw_dir, score_dir))
    os.system('cp -r {}/rw.dat {}'.format(where_rw_dir, score_dir))
    os.system('cp -r {}/scb.dat {}'.format(where_rw_dir, score_dir))
    os.system('cp -r {}/TMscore {}'.format(os.path.join(where_rw_dir, '..'), score_dir))
    return


def main(target_name, file_path, dih_list, bottom_width):
    pdb_path_list = glob.glob(os.path.join(file_path,"*.pdb"))
    pp = target_name.strip()
    pathlib.Path(pp).mkdir(parents=True, exist_ok=True)
    os.chdir(pp)  # at R0949/
    for num, rr in enumerate(pdb_path_list):
        pdb_name = os.path.basename(rr).split(".pdb")[0]
        if os.path.exists(pdb_name):
            shutil.rmtree(pdb_name)
        pathlib.Path(pdb_name).mkdir(parents=True, exist_ok=True)
        os.chdir(pdb_name)
        os.system('cp %s ./' % (rr))
        os.system('cp %s/topol.top ./' % (file_path))
        os.system('cp %s/*.itp ./' % (file_path))
        os.system('cp %s/mdp/* ./' % dirname)
        os.system('cp -r %s/charmm36-mar2019.ff ./' % dirname)
        change_his('./%s.pdb' % pdb_name)
        run_md(pdb_name, loop=num)        
        replace('topol.top', '.*charmm36-mar2019.ff',
                '#include "{}/charmm36-mar2019.ff'.format(dirname))
        
        mk_posre(dirname, dih_list, bottom_width=bottom_width)
        os.chdir('..')
    
    mk_rid(dirname, target_name)
    os.chdir('..')
    mk_score(where_rw_dir='/home/dongdong/wyz/rwplus/RWplus', where_evo_path="/home/dongdong/wyz/EvoEF2-master/EvoEF2", target=pp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make Refinement Directory')
    parser.add_argument('TASK', type=str, help="the task name")
    # parser.add_argument('--mol', type=str)
    parser.add_argument('--mol', type=str)
    parser.add_argument('--all-dihedral', action="store_true")
    parser.add_argument('-c', '--dihedral-index', nargs='+', type=int, default=None, help="the indexes of selected dihedral angles.")
    parser.add_argument('-d', '--bottom-width', type=float, default=0.4, help="the width of the bottom of the flat bottom harmonic potential(nm)")
    parser.add_argument('-r', '--config', type=str, default=None, help='config file')
    args = parser.parse_args()
    dirname = os.getcwd()
    if args.all_dihedral:
        _pdb_list = glob.glob(os.path.join(args.mol, "*.pdb"))
        if len(_pdb_list) == 0:
            raise RuntimeError("No pdb exists within {}.".format(args.mol))
        else:
            _pdb = _pdb_list[0]
        dih_list = get_all_dihedral_index(_pdb)
    else:
        if args.dihedral_index is None:
            raise RuntimeError("Please set dihedral angle indexes for CVs.")
        dih_list = args.dihedral_index


    main(target_name = args.TASK, file_path=args.mol, dih_list=dih_list, bottom_width=args.bottom_width)
    # python mk_rid_refinement.py --mol /home/dongdong/wyz/refinement3_2chain/PDB/test_2chain


