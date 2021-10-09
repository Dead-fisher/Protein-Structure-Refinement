import os
import numpy as np
import glob
import argparse
import random
import MDAnalysis as mda
from MDAnalysis.analysis import rms


ll = '/home/dongdong/wyz/analysis/score_function_comparison/average/R0968s1/R0968s1_gnn_gdtha_comparison.npy'

def his2hsd(file_in, file_out):
    with open(file_in, 'r') as clu:
        ret = clu.read()
    ret = ret.replace('HIS', 'HSD')
    with open(file_out, 'w') as clu:
        clu.write(ret)

def mkfname(finput, suffix='faspr'):
    if '.pdb' in finput:
        name = finput.split('.pdb')[0]
        return name + '_' + suffix + '.pdb'
    elif '.gro' in finput:
        name = finput.split('.gro')[0]
        return name + '_' + suffix + '.gro'


def restraint_em(finput, foutput='em.pdb', force_const=100, restraint_group='C-alpha'):
    if restraint_group == 'C-alpha':
        res_idx = 3
    elif restraint_group == 'backbone':
        res_idx = 4
    elif restraint_group == 'protein':
        res_idx = 2
    os.system('echo -e "1\n1\n" | gmx pdb2gmx -f {} > opt.log'.format(finput))
    os.remove('posre.itp')
    os.system('gmx editconf -f conf.gro -o newbox.gro -d 0.9 -c -bt triclinic >> opt.log')
    if not os.path.exists('conf_init.gro'):
        os.system("cp newbox.gro conf_init.gro")
    os.system('echo "3\nq\n" | gmx make_ndx -f newbox.gro -o posre.ndx >> opt.log')
    os.system('echo "3\n1\n" | gmx trjconv -f newbox.gro -s conf_init.gro -o newbox_fit.gro -fit rot+trans >> opt.log')
    assert os.path.exists('posre.ndx')
    assert os.path.exists('newbox_fit.gro')
    os.system('echo "{}\n" | gmx genrestr -f newbox_fit.gro -n posre.ndx -o posre.itp -fc {} {} {} >> opt.log'.format(res_idx, force_const, force_const, force_const))
    
    assert os.path.exists('minim.mdp')
    os.system('gmx grompp -f minim.mdp -c newbox_fit.gro -r conf_init.gro -p topol.top -o em.tpr -maxwarn 1 >> opt.log')
    os.system('gmx mdrun -deffnm em -v -ntmpi 1 -nt 4')
    os.system('echo -e "1\n" | gmx trjconv -s em.tpr -f em.gro -o {} -pbc mol -ur compact >> opt.log'.format(foutput))


def faspr(finput, foutput):
    faspr_path = '/home/dongdong/wyz/FASPR/FASPR'
    faspr_cmd = "{} -i {} -o {}".format(faspr_path, finput, foutput)
    os.system(faspr_cmd)


def restraint_equ(finput, ref_file, foutput=None, ndx_file=None, top_file='topol.top', force_const=1000):
    # if '.gro' in finput:
    os.system('echo "3\n1\n" | gmx trjconv -f {} -s {} -o {} -fit rot+trans'.format(finput, ref_file, mkfname(finput, suffix='fit')))
    os.system('gmx solvate -cp {} -cs spc216.gro -o {} -p {}'.format(mkfname(finput, suffix='fit'), mkfname(finput, suffix='sol'), top_file))
    assert os.path.exists('ions.mdp')
    os.system('gmx grompp -f ions.mdp -c {} -p {} -o ions.tpr -maxwarn 2'.format(mkfname(finput, suffix='sol'), top_file))
    os.system('echo -e "13\n" | gmx genion -s ions.tpr -o {} -p {} -pname NA -nname CL -neutral -conc 0.15'.format(mkfname(finput, suffix='ion'), top_file))
    assert os.path.exists(mkfname(finput, suffix='ion'))
    if ndx_file is None:
        os.system('echo "3\nq\n" | gmx make_ndx -f {} -o posre.ndx'.format(mkfname(finput, suffix='ion')))
        ndx_file = 'posre.ndx'
    
    assert os.path.exists(ndx_file)
    
    os.system('echo "3\n" | gmx genrestr -f {} -n {} -o posre.itp -fc {} {} {}'.format(mkfname(finput, suffix='ion'), ndx_file, force_const, force_const, force_const))
    posre_itp = 'posre.itp'
    assert os.path.exists('minim.mdp')
    os.system('gmx grompp -f minim.mdp -c {} -p {} -r {} -o em.tpr -maxwarn 2'.format(mkfname(finput, suffix='ion'), top_file, ref_file))
    os.system('gmx mdrun -deffnm em -v -ntmpi 1 -nt 4')
    os.system('gmx grompp -f nvt.mdp -c em.gro -p topol.top -o nvt.tpr -r {} -maxwarn 2'.format(ref_file))
    os.system('gmx mdrun -deffnm nvt -ntmpi 1 -v -nt 4')
    os.system('gmx grompp -f npt.mdp -c nvt.gro -p topol.top -o npt.tpr -r {} -maxwarn 2'.format(ref_file))
    os.system('gmx mdrun -deffnm npt -ntmpi 1 -v -nt 4')
    return

def restraint_sample(finput='npt.gro', ref_file='conf_init.gro', sample_title='1', top_file='topol.top', temp=300, clean=True, force_const=1000):
    with open('restraint_replica.mdp', 'r') as mdp:
        ret = mdp.read()
    ret = ret.replace('300', str(temp))
    with open('restraint{}.mdp'.format(sample_title), 'w') as mdp2:
        mdp2.write(ret)
    os.system('echo "3\n" | gmx genrestr -f {} -n posre.ndx -o posre.itp -fc {} {} {}'.format(finput, force_const, force_const, force_const))
    os.system('gmx grompp -f restraint{}.mdp -c {} -p topol.top -o sample{}.tpr -r {} -maxwarn 2'.format(sample_title, finput, sample_title, ref_file))
    os.system('gmx mdrun -deffnm sample{} -ntmpi 1 -v -nt 4'.format(sample_title))
    if clean:
        os.remove('restraint{}.mdp'.format(sample_title))

def post_sampling(sample_title='1', data_pool='opt_data'):
    if not os.path.exists(data_pool):
        os.mkdir(data_pool)
    os.system('cp sample{}.xtc ./{}/'.format(sample_title, data_pool))
    os.system('cp sample{}.tpr ./{}/'.format(sample_title, data_pool))
    os.chdir(data_pool)
    os.system('echo -e "1\n" | gmx trjconv -s sample{}.tpr -f sample{}.xtc -o sample{}_nopbc.xtc -pbc mol -ur compact'.format(sample_title, sample_title, sample_title))
    os.system('echo -e "1\n" | gmx trjconv -sep -f sample{}.xtc -o conf_{}_.pdb -s sample{}.tpr'.format(sample_title, sample_title, sample_title))
    os.chdir('../')

def select(data_pool):
    file_list = glob.glob('./{}/conf*.pdb'.format(data_pool))
    score_list = []
    rmsd_list = []
    init_tpo = mda.Universe(args.fin)
    init_bb = init_tpo.select_atoms('name CA').positions
    for ff in file_list:
        score_list.append(float(evoef2(ff)))
        tar_tpo = mda.Universe(ff)
        _rmsd = rms.rmsd(tar_tpo.select_atoms('name CA').positions,  # coordinates to align
                    init_bb,  # reference coordinates
                    center=True,  # subtract the center of geometry
                    superposition=True)  # superimpose coordinates
        rmsd_list.append(_rmsd)
    
    np.save(str(args.fin)+'score.npy', np.array(score_list))
    np.save(str(args.fin)+'rmsd.npy', np.array(rmsd_list))


def evoef2(fname, title_init='test'):
    title = '{}_{}'.format(title_init, random.random())
    evo_path = '/home/dongdong/wyz/EvoEF2-master/EvoEF2'
    evo_cmd = '{} --command=ComputeStability --pdb={} > {}.log'.format(evo_path, fname, title)
    os.system(evo_cmd)
    with open("{}.log".format(title), 'r') as evo:
        for line in evo.readlines():
            if 'Total                 =' in line:
                score = line.split()[-1]
                break
    os.remove('{}.log'.format(title))
    return str(score)


def clean_mid_file():
    file_list = glob.glob('*.ndx') + glob.glob('*.top') + glob.glob('*.itp')
    for ff in file_list:
        os.remove(ff)


def clean_file():
    file_list = glob.glob('#*#') + glob.glob('*.ndx') + glob.glob('*.top') + glob.glob('*.itp') + glob.glob('*.gro')
    file_list += glob.glob('*.tpr')
    file_list += glob.glob('*.log')
    file_list += glob.glob('*.edr')
    file_list += glob.glob('*.trr')
    file_list += glob.glob('*.cpt')
    file_list += glob.glob('*.xtc')
    for ff in file_list:
        os.remove(ff)


def clean_mid_structure():
    if os.path.exists('em.pdb'):
        os.remove('em.pdb')
    for i in range(2):
        if os.path.exists('em{}.pdb'.format(i+1)):
            os.remove('em{}.pdb'.format(i+1))
            os.remove('em{}_faspr.pdb'.format(i+1))
    if os.path.exists(mkfname(args.fin)):
        os.remove(mkfname(args.fin))
    if os.path.exists('mdout.mdp'):
        os.remove('mdout.mdp')


def gdtha(fname, target, title_init='test'):
    title = '{}_{}'.format(title_init, random.random())
    tm_path = '/home/dongdong/wyz/rwplus/TMscore'
    native_path = '/home/dongdong/wyz/Native'
    tm_cmd = '{} {} {}/{}.pdb > {}.log'.format(tm_path, fname, native_path, target, title)
    os.system(tm_cmd)
    with open("{}.log".format(title), 'r') as tm:
        score = tm.readlines()[19].split()[1]
    os.remove('{}.log'.format(title))
    return str(score)


def main1():
    os.chdir(base_path)
    faspr(fin, mkfname(fin))
    restraint_em(mkfname(fin), foutput='em1.pdb', force_const=100, restraint_group='protein')
    # , restraint_group='backbone'
    clean_mid_file()
    faspr('em1.pdb', mkfname('em1.pdb'))
    restraint_em(mkfname('em1.pdb'), foutput='em2.pdb', force_const=200, restraint_group='backbone')
    clean_mid_file()
    faspr('em2.pdb', mkfname('em2.pdb'))
    restraint_em(mkfname('em2.pdb'), foutput='em3.pdb', force_const=500)
    clean_mid_file()
    faspr('em3.pdb', mkfname('em3.pdb'))
    restraint_em(mkfname('em3.pdb'), foutput=fout, force_const=1000)
    clean_mid_file()
    clean_file()
    clean_mid_structure()
    if not is_cur_path:
        os.remove(fin)
    os.chdir(cur_dir)
    return


def main_debug():
    os.chdir(base_path)
    if not os.path.exists('em1.pdb'):
        faspr(fin, mkfname(fin))
        restraint_em(mkfname(fin), foutput='em1.pdb', force_const=100, restraint_group='backbone')
        clean_mid_file()
    if not os.path.exists('em2.pdb'):
        faspr('em1.pdb', mkfname('em1.pdb'))
        restraint_em(mkfname('em1.pdb'), foutput='em2.pdb', force_const=200, restraint_group='C-alpha')
        clean_mid_file()
    if not os.path.exists('em3.gro'):
        faspr('em2.pdb', mkfname('em2.pdb'))
        restraint_em(mkfname('em2.pdb'), foutput='em3.gro', force_const=500)
    if not os.path.exists('npt.gro'):
        restraint_equ('em3.gro', 'conf_init.gro', ndx_file='posre.ndx', top_file='topol.top', force_const=1000)
    temp_list = [300, 150, 100, 20]
    data_pool = 'opt_data'
    for ii in range(len(temp_list)):
        if not os.path.exists('sample{}.xtc'.format(str(ii))):
            restraint_sample(finput='npt.gro', ref_file='conf_init.gro', sample_title=str(ii), top_file='topol.top', temp=temp_list[ii])
            post_sampling(sample_title=str(ii), data_pool=data_pool)
    
    select(data_pool)
    os.chdir(cur_dir)
    return


def test():
    restraint_md('pre_md.gro', 'conf_init.gro', ndx_file='posre.ndx', posre_itp='posre.itp', top_file='topol.top', force_const=1000)
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize the input structrue locally.')
    # parser.add_argument('--help', '-h', dest='fin', type=str)
    parser.add_argument('--inputfile', '-i', dest='fin', type=str, help='input pdb file (path)')
    parser.add_argument('--outputfile', '-o', dest='fout', type=str, default=None, help='output file name')
    parser.add_argument('--clean', '-c', dest='clean', type=bool, default=False, help='clean files. default True')
    parser.add_argument('--min', '-m', dest='m', type=bool, default=True, help='minimization step default True')
    args = parser.parse_args()

    cur_dir = os.getcwd()
    base_path = os.path.abspath(os.path.dirname(__file__))
    input_dir = os.path.abspath(os.path.dirname(args.fin))

    if args.m:
        if input_dir == base_path:
            fin = args.fin
            is_cur_path = True
        else:
            os.system("cp {} {}".format(args.fin, base_path))
            is_cur_path = False
            fin = args.fin.split('/')[-1]
            
        if args.fout is None:
            fout = mkfname(fin, suffix='opt')
            fout = os.path.join(cur_dir, fout)
        else:
            if args.fout.split('.')[-1] == 'pdb':
                fout = os.path.abspath(args.fout)
            elif '/' in args.fout:
                fout = os.path.abspath(os.path.join(args.fout, mkfname(fin, suffix='opt')))

        main_debug()
        if args.clean:
            clean_file()

        # target = "R0974s1"
        # print('initial structure gdtha:', gdtha('clusters_opt.pdb', target))
        # # print('final structure gdtha:', gdtha(fout, target))
        # print('final structure EvoEF2', evoef2('clusters_opt.pdb'))
    
    
