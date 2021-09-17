#!/bin/bash

# RTOS: Special call similar to generate C-OMLP similar to paper exrts13b.pdf

declare -a load=(0.5)
declare -a num_nls_ratio=(2.5)
declare -a num_ls_ratio=(0.75)
declare -a num_res_nls=(10)
declare -a num_res_ls=(3)
declare -a acc_max_nls=(3)
declare -a acc_max_ls=(2)
declare -a group_size_nls=(1 2 3 4)
declare -a group_type_nls=(0 1)
declare -a asymm=(0)

rm ./experiment_calls.sh

for l in "${load[@]}"
do

    for n in "${num_nls_ratio[@]}"
    do

        for n_ls in "${num_ls_ratio[@]}"
        do

            for q in "${num_res_nls[@]}"
            do

                for qn in "${num_res_ls[@]}"
                do

                    for acc_nls in "${acc_max_nls[@]}"
                    do

                        for acc_ls in "${acc_max_ls[@]}"
                        do

                            for gs in "${group_size_nls[@]}"
                            do

                                for gt in "${group_type_nls[@]}"
                                do

                                    for as in "${asymm[@]}"
                                    do

                                    echo "python2 ./run_experiments.py --num_cpus 8 --load $l -n $n --n_r --n_ls $n_ls --n_ls_r --num_res_nls $q --num_res_ls $qn --acc_max_nls $acc_nls --acc_max_ls $acc_ls --group_size_nls $gs --group_type_nls $gt --asym $as &" >> ./experiment_calls.sh

                                    done

                                done

                            done

                        done

                    done

                done

            done

        done

    done

done