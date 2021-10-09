import os
import glob
import numpy as np

from dpdispatcher.lazy_local_context import LazyLocalContext
from dpdispatcher.submission import Submission, Job, Task, Resources
from dpdispatcher.slurm import Slurm

score_resources = Resources(number_node=1, cpu_per_node=2, gpu_per_node=0, queue_name="GPU_2080Ti", group_size=1, if_cuda_multi_devices=False) 


def hsd2his(input_file):
    with open(input_file, 'r') as pdb:
        ret = pdb.read()
    ret = ret.replace('HSD', 'HIS')
    with open(input_file, 'w') as pdb:
        pdb.write(ret)
    return


def his2hsd(input_file):
    with open(input_file, 'r') as pdb:
        ret = pdb.read()
    ret = ret.replace('HIS', 'HSD')
    with open(input_file, 'w') as pdb:
        pdb.write(ret)
    return


def gnn_refine(input_name, out_name=None, output_path='.'):
    where_is_GNN = '{}/GNNRefine/GNNRefine.py'.format(base_path)
    os.system('/home/dongdong/anaconda3/envs/GNNRefine/bin/python {} {} {}/'.format(where_is_GNN, input_name, output_path))
    if out_name is None:
        return
    else:
        os.system('mv {}/{}.refined.pdb {}/{}'.format(output_path, input_name, output_path, out_name))
        return


def check_file(file_name):
    if os.path.exists(file_name):
        pass
    else:
        raise FileNotFoundError('{} File not found'.format(file_name))


def get_num_files(idx_iter):
    os.system('ls | grep conf_iter{:06d} > filename.txt'.format(idx_iter))
    num_list = []
    with open('./filename.txt', 'r') as files:
        for ff in files.readlines():
            if len(ff.split()) == 0:
                continue
            if ff.split('.')[-2] == 'pdb':
                continue
            try:
                num_list.append(int(ff.split('.')[-2].split('_')[-1]))
            except ValueError:
                print(ff.split('.')[-2].split('_')[-1])
                print('Encouter unknown file name. Ignore it.')
                continue
    return max(num_list) + 1


def cal_score(num_walker, idx_iter, num_of_pdb, task_work_path='./', local_path='./'):
    where_is_evo = '/home/dongdong/wyz/EvoEF2-master/EvoEF2'
    iter_name = "iter.{:06d}".format(idx_iter)
    if os.path.exists("{}/score.done".format(iter_name)):
        return
    lazy_local_context = LazyLocalContext(local_root=local_path)
    slurm = Slurm(context=lazy_local_context)
    rw_task = []
    # RWplus
    for idx_walker in range(num_walker):
        if os.path.exists("{}/rw_{:06d}_{:03d}.done".format(iter_name, idx_iter, idx_walker)):
            continue
        score_command = ''
        if os.path.exists("rw_{:03d}.txt".format(idx_walker)):
            os.remove("rw_{:03d}.txt".format(idx_walker))
        for ii in range(num_of_pdb):
            score_command += 'echo "conf_iter{:06d}_{:03d}_{:d}.pdb" >> rw_{:03d}.txt\n'.format(idx_iter, idx_walker, ii, idx_walker)
            score_command += './calRWplus ./data/conf_iter{:06d}_{:03d}_{:d}.pdb >> rw_{:03d}.txt\n'.format(idx_iter, idx_walker, ii, idx_walker)
        rw_task.append(Task(command=score_command, task_work_path=task_work_path,outlog='rwplus.out', errlog='rwplus.err'))
    
    # EvoEF2
    for idx_walker in range(num_walker):
        if os.path.exists("{}/evo_{:06d}_{:03d}.done".format(iter_name, idx_iter, idx_walker)):
            continue
        score_command = ''
        if os.path.exists("evo_{:03d}.txt".format(idx_walker)):
            os.remove("evo_{:03d}.txt".format(idx_walker))
        for ii in range(num_of_pdb):
            score_command += 'echo "conf_iter{:06d}_{:03d}_{:d}.pdb" >> evo_{:03d}.txt\n'.format(idx_iter, idx_walker, ii, idx_walker)
            # '{} --command=ComputeStability --pdb={} > {}'.format(self.evo, ff, self.evo_log)
            score_command += '{} --command=ComputeStability --pdb=./data/conf_iter{:06d}_{:03d}_{:d}.pdb > evo_{:03d}.log\n'.format(where_is_evo, idx_iter, idx_walker, ii, idx_walker)
            score_command += "tail -3 evo_{:03d}.log | head -1 >> evo_{:03d}.txt\n".format(idx_walker, idx_walker)
        rw_task.append(Task(command=score_command, task_work_path=task_work_path,outlog='evo.out', errlog='evo.err'))
    
    if len(rw_task) == 0:
        return

    rw_submission = Submission(work_base='./', resources=score_resources, machine=slurm, task_list=rw_task)
    rw_submission.run_submission()
    
    for idx_walker in range(num_walker):
        assert os.path.exists('rw_{:03d}.txt'.format(idx_walker)), "no rw_{:03d}.txt file found".format(idx_walker)
        assert os.path.exists('evo_{:03d}.txt'.format(idx_walker)), "no evo_{:03d}.txt file found".format(idx_walker)
        os.system('cat rw_{:03d}.txt >> rw_all.txt'.format(idx_walker))
        os.system('rm rw_{:03d}.txt'.format(idx_walker))
        os.system("touch {}/rw_{:06d}_{:03d}.done".format(iter_name, idx_iter, idx_walker))
        os.system("rm rwplus.err")
        os.system("rm rwplus.out")

        os.system('cat evo_{:03d}.txt >> evo_all.txt'.format(idx_walker))
        os.system('rm evo_{:03d}.txt'.format(idx_walker))
        os.system('rm evo_{:03d}.log'.format(idx_walker))
        os.system("rm evo.err")
        os.system("rm evo.out")
        os.system("touch {}/evo_{:06d}_{:03d}.done".format(iter_name, idx_iter, idx_walker))
    os.system("touch {}/score.done".format(iter_name))
    print('Calculation done')
    return



def load_rw(rw_file='rw_all.txt'):
    all_rwplus = {}
    with open('./{}'.format(rw_file), 'r') as rw:
        for line in rw.readlines():
            info = line.split()
            if len(info) == 0:
                continue
            if len(info) == 1:
                name = info[0]
            else:
                all_rwplus[name] = float(info[3])
    return sorted(all_rwplus.items(), key=lambda item:item[1])


def load_evo(evo_file='evo_all.txt'):
    all_evo = {}
    with open('./{}'.format(evo_file), 'r') as evo:
        for line in evo.readlines():
            info = line.split()
            if len(info) == 0:
                continue
            if len(info) == 1:
                name = info[0]
            else:
                all_evo[name] = float(info[-1].strip())
    return sorted(all_evo.items(), key=lambda item:item[1])


def load_rmsd(rmsd_file='rmsd_all.txt'):
    all_rmsd = {}
    with open('./{}'.format(rmsd_file), 'r') as rmsd:
        for line in rmsd.readlines():
            info = line.split()
            if len(info) == 0:
                continue
            if 'conf' in line:
                name = info[0]
            else:
                all_rmsd[name] = float(info[-1].strip())
    return all_rmsd



def read_config(dirname, target):
    box_info = {}
    with open('{}/box_information.txt'.format(os.path.abspath(os.path.join(dirname, '../%s'%target))), 'r') as _box_info:
        for i in _box_info.readlines():
            info = i.rstrip("\n").split('=')
            box_info[info[0]] = info[1]
    return box_info['box_size'].split(','), box_info['num_sol'], box_info['num_Na'], box_info['num_Cl']


def clean_dpdispatcher(clean_path):
    all_task = glob.glob(clean_path + "/*_job_id")
    all_task += glob.glob(clean_path + "/*_finished")
    all_task += glob.glob(clean_path + "/*.sub")
    all_task += glob.glob(clean_path + "/*.json")
    all_task += glob.glob(clean_path + "/*.sub.o*")
    all_task += glob.glob(clean_path + "/slurm*out")
    for ff in all_task:
        os.remove(ff)

def cal_rmsd(idx_iter, num_walker=8):
    if os.path.exists('./iter.{:06d}/rmsd.done'.format(idx_iter)):
        return
    cwd = os.getcwd()
    os.chdir("data")
    for idx_walker in range(num_walker):
        xtc_name = 'md_nopbc_iter{:06d}_{:03d}.xtc'.format(idx_iter, idx_walker)
        os.system('echo "3\n3\n" | gmx rms -f {} -s topol.tpr -fit rot+trans -o rmsd_{:06d}_{:03d}.xvg'.format(xtc_name, idx_iter, idx_walker))
    os.chdir(cwd)
    all_rms = open("rmsd_all.txt", 'a')
    for idx_walker in range(num_walker):
        frame = 0
        with open('./data/rmsd_{:06d}_{:03d}.xvg'.format(idx_iter, idx_walker), 'r') as xvg:
            for line in xvg.readlines():
                if ("#" in line) or ("@" in line) or (len(line) == 0):
                    continue
                elif len(line.split()) == 2:
                    info = line.split()
                    all_rms.write('conf_iter{:06d}_{:03d}_{:d}.pdb\n'.format(idx_iter, idx_walker, frame))
                    all_rms.write('{}\n'.format(info[-1]))
                    frame += 1
    os.system("touch ./iter.{:06d}/rmsd.done".format(idx_iter))
    pass


def score_process(idx_iter, num_walker):
    '''
    score Rwplus for .xtc file. deperiodic -> segement -> cluster -> pdb2gmx.
    Args:
            idx_iter: which iteration.
            num_walker: number of walkers.
    '''
    lowest_rate_list = [0.03, 0.1]
    lowest_num_list = [int(ii * 8000 * (idx_iter + 1)) for ii in lowest_rate_list]
    print('score processing...')
    dirname = os.getcwd()     # *.run/
    base_dir = os.path.dirname(dirname)
    target = dirname.split('/')[-1].split('.')[0]  # R0949
    run_dir = dirname + '/iter.{:06d}/00.enhcMD/{:03d}/'  # where .xtc file is.
    source_dir = os.path.abspath(os.path.join(dirname, '../{}/{}'.format(target, target)))+'/'  # where the source/ dir is.
    xtc = run_dir + 'traj_comp.xtc'
    trjname = 'traj_comp'
    outtrj = 'md_nopbc'
    iter_dir = dirname + "/score/" + "iter.{:06d}".format(idx_iter)
    data_pool = dirname + "/score/data"
    tm_path = os.path.join(dirname, "./score/TMscore")
    init_conf = base_dir + "/source/mol/{}/conf.gro".format(target)

    if not os.path.exists(iter_dir):
        os.mkdir(iter_dir)  # make iteration dir. we are at .run/ now
    if not os.path.exists(data_pool):
        os.mkdir(data_pool)
    
    # copy .ff file to gmx dir.
    ff_name = 'charmm36-mar2019.ff'
    if not os.path.exists('./score/iter.{:06d}/{}'.format(idx_iter, ff_name)):
        os.system('cp -r ../{} ./score/iter.{:06d}/'.format(ff_name, idx_iter))
    
    os.chdir('./score/data')  # change path to .run/rwplus/data
    for idx_walker in range(num_walker):  # for all walker converting pdbs.
        print('Process for walker {}...'.format(idx_walker))
        xtc_path = xtc.format(idx_iter, idx_walker)
        os.system('cp {} ./{}_iter{:06d}_{:03d}.xtc'.format(xtc_path, trjname, idx_iter, idx_walker))
        if not os.path.exists('./topol.tpr'):
            topol = run_dir + 'topol.tpr'
            topol_path = topol.format(idx_iter, idx_walker)
            os.system('cp {} ./topol.tpr'.format(topol_path))
        print('Segmenting...')
        if not os.path.exists("seg_iter{:06d}_{:03d}.done".format(idx_iter, idx_walker)):
            os.system('echo -e "1\n" | gmx trjconv -s topol.tpr -f {}_iter{:06d}_{:03d}.xtc -o {}_iter{:06d}_{:03d}.xtc -pbc mol -ur compact'.format(trjname, idx_iter, idx_walker, outtrj, idx_iter, idx_walker))
            os.system('echo -e "1\n" | gmx trjconv -sep -f {}_iter{:06d}_{:03d}.xtc -o conf_iter{:06d}_{:03d}_.pdb'.format(outtrj, idx_iter, idx_walker, idx_iter, idx_walker))
            os.system("touch seg_iter{:06d}_{:03d}.done".format(idx_iter, idx_walker))
    os.system("rm -rf ./#*#")
        
    # num_to_score = get_num_files(idx_iter)
    num_to_score = 1000
    
    os.chdir('..')  # at .run/score/
    score_file = 'score.rwplus'
    if (not os.path.exists('./calRWplus')):
        raise FileNotFoundError('calRWplus or EvoEF2 not found')
        # check if scores file exists. if true, clean the content in it.
    
    cal_score(num_walker=num_walker, idx_iter=idx_iter, num_of_pdb=num_to_score)
    cal_rmsd(idx_iter, num_walker=num_walker)
    clean_dpdispatcher("./")

    rmsd_list = load_rmsd()
    ranked_rw = load_rw()[:max(lowest_num_list)+1]
    ranked_evo = load_evo()[:max(lowest_num_list)+1]
    
    box_size, num_sol, num_Na, num_Cl = read_config(dirname, target)

    os.chdir(iter_dir)
    for jj, sel_num in enumerate(lowest_num_list):
        if not os.path.exists("./selected_{}_{}".format(jj, sel_num)):
            os.mkdir("./selected_{}_{}".format(jj, sel_num))
        os.chdir("selected_" + str(jj) + "_" + str(sel_num))
        evo_sel = [tar[0] for tar in ranked_rw[:sel_num]]
        rw_sel = [tar[0] for tar in ranked_evo[:sel_num]]
        intersection = [ff for ff in evo_sel if ff in rw_sel]
        intersection.sort()
        file_num = len(intersection)
        rmsd_list_inter = np.array([float(rmsd_list[tar]) for tar in intersection ])
        ave = np.mean(rmsd_list_inter)
        scale = 1.15
        benchmark = ave * scale
        print("set RMSD benchmark as", benchmark)
        sub_rmsd_list = np.array([_rm for _rm in rmsd_list_inter if _rm < benchmark])
        select_ratio = len(sub_rmsd_list) / file_num
        new_intersection = []
        if select_ratio < 0.85:
            top_rmsd = sorted(rmsd_list_inter)
            percent = 0.85
            file_num = int(percent * file_num)
            for _rm in top_rmsd[:file_num]:
                idx = np.argwhere(rmsd_list_inter == _rm).flatten()[0]
                new_intersection.append(intersection[idx])
        else:
            for _rm in sub_rmsd_list:
                idx = np.argwhere(rmsd_list_inter == _rm).flatten()[0]
                new_intersection.append(intersection[idx])
        if len(new_intersection) == 0:
            raise RuntimeError("something wrong")
        file_num = len(new_intersection)
        if not os.path.exists('all_{}.pdb'.format(file_num)):
            if new_intersection is None:
                raise RuntimeError
            if file_num <= 1000:
                files = ' '.join([os.path.join(data_pool, new_intersection[ii]) for ii in range(file_num)])
                os.system("cat {} > all_{}.pdb".format(files, file_num))
            else:
                with open('all_{}.pdb'.format(file_num), 'w') as allpdb:
                    for fname in [os.path.join(data_pool, new_intersection[ii]) for ii in range(file_num)]:
                        print(fname + "\r")
                        ff = open(fname, 'r')
                        allpdb.write(ff.read())
                        allpdb.write('\n')
                        ff.close()
            assert os.path.exists("all_{}.pdb".format(file_num)), "all_{}.pdb not found".format(file_num)
        if not os.path.exists('/topol.tpr'):
            os.system("cp {}/topol.tpr ./topol.tpr".format(data_pool))
        if not os.path.exists('clusters.pdb'):
            os.system('echo -e "3\\n1\\n" | gmx cluster -f all_{}.pdb -s topol.tpr -cutoff 20 -method gromos -av'.format(file_num))
            assert os.path.exists("clusters.pdb"), "clusters.gro not found"
        if not os.path.exists('charmm36-mar2019.ff'):
            os.system('cp -r ../charmm36-mar2019.ff ./')
            assert os.path.exists("charmm36-mar2019.ff"), "proper force field not found"
        if not os.path.exists('ions.mdp'):
            os.system('cp {}/ions.mdp ./ions.mdp'.format(base_dir+"/mdp"))
            assert os.path.exists("ions.mdp"), "ions.mdp not found"
        print('Average done.')
        
        if not os.path.exists("conf_init.gro"):
            os.system('echo -e "1\n1\n" | gmx pdb2gmx -f clusters.pdb -o processed_{}.gro -ignh -heavyh > grompp_{}.log 2>&1'.format(sel_num, sel_num))
            os.system('gmx editconf -f processed_{}.gro -o newbox_{}.gro -box {} {} {} -c -bt triclinic'.format(sel_num, sel_num, box_size[0], box_size[1], box_size[2]))
            os.system('gmx solvate -cp newbox_{}.gro -cs spc216.gro -maxsol {} -o solv_{}.gro -p topol.top > grompp_sol_{}.log 2>&1'.format(sel_num, num_sol, sel_num, sel_num))
            os.system('gmx grompp -f ions.mdp -c solv_{}.gro -p topol.top -o ions_{}.tpr -maxwarn 2 > grompp_ion_{}.log 2>&1'.format(sel_num, sel_num, sel_num))
            os.system('echo -e "13\n" | gmx genion -s ions_{}.tpr -o solv_ions_{}.gro -p topol.top -pname NA -nname CL -neutral -np {} -nn {} > grompp_genion_{}.log 2>&1'.format(sel_num, sel_num, num_Na, num_Cl, sel_num))
            # align
            os.system('echo "3\n0\n" | gmx trjconv -f solv_ions_{}.gro -s {} -o conf_init.gro -fit rot+trans > fit.log'.format(sel_num, init_conf))
        os.system("rm -rf ./#*#")
        os.chdir(iter_dir)
    os.chdir(dirname)
    pass




if __name__ == '__main__':
    score_process(0, 8)