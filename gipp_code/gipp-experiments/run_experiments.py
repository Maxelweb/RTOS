import argparse
import shutil
import os
from datetime import datetime
from toolbox.io import write_std_header, write_data, write_runtime, write_configuration
import exp.ecrts20 as ecrts20

RESULTS_DATA_WORKING_DIR = "./data_working/"
RESULTS_DATA_COMPLETE_DIR = "./data_complete/"
RESULTS_DATA_IMPOSSIBLE_DIR = "./data_impossible/"

# ecrts20 --> sched-experiments/exp/ecrts20.py

def run(
    num_cpus,   
    load,
    n,
    n_r,
    n_ls,
    n_ls_r,
    num_res_nls,
    num_res_ls,
    acc_max_nls,
    acc_max_ls,
    group_size_nls,
    group_type_nls,
    asym):

    conf_gen = ecrts20.generate_csl_configs(
        _cpus=num_cpus,
        _n=n,
        _n_ratio=n_r,
        _ls=n_ls,
        _ls_ratio=n_ls_r,
        _load=load,
        _acc_max_nls=acc_max_nls,
        _acc_max_ls=acc_max_ls,
        _g_size_nls=group_size_nls,
        _g_type_nls=group_type_nls,
        _tp=[0.5],
        _num_res_nls=num_res_nls,
        _num_res_ls=num_res_ls,
        _asymmetric=asym,
        _samples=100,
    )

    # manual run
    # conf_gen = ecrts20.generate_csl_configs(
    #     _cpus=[4],
    #     _n=[12],
    #     _n_ratio=False,
    #     _ls=[0],
    #     _ls_ratio=False,
    #     _load=[0.4],
    #     _acc_max_nls=[3],
    #     _acc_max_ls=[2],
    #     _g_size_nls=[1],
    #     _g_type_nls=[0],
    #     _tp=[0.5],
    #     _num_res_nls=[12],
    #     _num_res_ls=[3],
    #     _asymmetric=[False],
    #     _samples=20
    # )

    for (_, conf) in conf_gen:
        run_experiment(conf)

def run_experiment(conf):
    
    print conf.output_file

    # Controlli di sanità dei file (se presenti o meno)

    if os.path.exists(RESULTS_DATA_COMPLETE_DIR + conf.output_file):
        print "configuration already complete. skipping."
        return

    if os.path.exists(RESULTS_DATA_IMPOSSIBLE_DIR + conf.output_file):
        print "configuration already deemed impossible. skipping."
        return

    experiment_driver = ecrts20.run_csl_config
    conf.output = open(RESULTS_DATA_WORKING_DIR + conf.output_file, "w+")
    write_std_header(conf.output)

    start  = datetime.now()
    result = experiment_driver(conf)
    end = datetime.now()
    print conf.output_file
    print "total time: ", (end - start)

    write_configuration(conf.output, conf)
    conf.output.close()

    if result:
        shutil.move(
            RESULTS_DATA_WORKING_DIR + conf.output_file,
            RESULTS_DATA_COMPLETE_DIR + conf.output_file
        )
    else:
        shutil.move(
            RESULTS_DATA_WORKING_DIR + conf.output_file,
            RESULTS_DATA_IMPOSSIBLE_DIR + conf.output_file
        )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GIPP experiments.')
    parser.add_argument(
                    '--num_cpus', 
                    metavar='m', 
                    type=int,
                    default=[4, 8, 16],
                    nargs='+',
                    help='a list of numbers of cpus to be used')
    parser.add_argument(
                    '--load', 
                    metavar='lo', 
                    type=float,
                    default=[0.4, 0.6],
                    nargs='+',
                    help='a list of load values to be used')
    parser.add_argument(
                    '-n',
                    metavar='n',
                    type=float,
                    default=[2.0, 3.0],
                    nargs="+",
                    help=""
    )
    parser.add_argument(
                    '--n_r', 
                    metavar='n_r', 
                    type=bool, 
                    const=True,
                    default=False,
                    nargs="?",
                    help='denotes if n is a ratio to the number of cpus')
    parser.add_argument(
                    '--n_ls', 
                    metavar='ls', 
                    type=float,
                    default = [0.0, 0.5, 1.0],
                    nargs='+',
                    help='')
    parser.add_argument(
                    '--n_ls_r', 
                    metavar='n_ls_r', 
                    type=bool,
                    const=True,
                    default=False,
                    nargs="?",
                    help='')
    parser.add_argument(
                    '--num_res_nls', 
                    metavar='num_res_nls', 
                    type=int,
                    default=[3],
                    nargs='+',
                    help='number of resources nls tasks access')
    parser.add_argument(
                    '--num_res_ls', 
                    metavar='num_res_ls', 
                    type=int, 
                    nargs='+',
                    help='number of resources ls tasks access')
    parser.add_argument(
                    '--acc_max_nls', 
                    metavar='acc_max_nls', 
                    type=int, 
                    nargs='+',
                    help='number of accesses a nls task makes')
    parser.add_argument(
                    '--acc_max_ls', 
                    metavar='acc_max_ls', 
                    type=int, 
                    nargs='+',
                    help='number of accesses a ls task makes')
    parser.add_argument(
                    '--group_size_nls', 
                    metavar='g_size_nls', 
                    type=int,
                    nargs='+',
                    help='group size for nls tasks')
    parser.add_argument(
                    '--group_type_nls', 
                    metavar='g_type_nls', 
                    type=int, 
                    nargs='+',
                    help='group type (0 for wide, 1 for deep) for nls tasks')
    parser.add_argument(
                    '--asym', 
                    metavar='asym', 
                    type=int, 
                    nargs='+',
                    help='asymmetric resource accesses patterns (0 for false, one for true) for nls tasks')

    args = parser.parse_args()
    run(
        args.num_cpus,
        args.load,
        args.n,
        args.n_r,
        args.n_ls,
        args.n_ls_r,
        args.num_res_nls,
        args.num_res_ls,
        args.acc_max_nls,
        args.acc_max_ls,
        args.group_size_nls,
        args.group_type_nls,
        args.asym)
