#!/bin/bash
set -e

. job_pool.sh

function myecho {
    echo -e "[`date`] ${1}" >> ${LOG}
    echo -e "\033[1;35m[`date`] \033[0;35m${1}\033[0m"
}

function wait_enter {
    while [ ! -d ${1} ]
    do
        sleep 2
    done
    cd ${1}
}

function to_parallel {
    cd ${2}
    # it
    myecho "  {${1}} Running homology for all labels of train-${1}.txt"
    tmp_dir="temp_${1}"
    padded=`printf "%05d" $(( ${1} / PERIOD ))`
    diag_name="${padded}-diag.png"
    mkdir -p ${tmp_dir}
    cd ${tmp_dir}

    cp ../encoded/train-${1}.txt full_encoded.txt
    for j in ${LABELS}
    do
        myecho "  {${1}} Label ${j}:"

        label_dir="../diagrams/label${j}"
        mkdir -p ${label_dir}
        if [ -f "${label_dir}/${diag_name}" ] ; then
            continue
        fi

        if [ ! -f "${label_dir}/pairs-${1}.txt" ] ; then
            myecho "    {${1}} Preparing data"
            python ${ROOT}/phom/generate_mnist_input.py "full_encoded.txt" ${j} "data.txt" >> ${LOG}

            myecho "    {${1}} Running ${DIM}D homology analysis"
            ${ROOT}/phom/homology_${DIM}d data.txt --percentage 0.9 >> ${LOG}
            cp pairs.txt ${label_dir}/pairs-${1}.txt
        else
            cp ${label_dir}/pairs-${1}.txt pairs.txt
        fi

        myecho "    {${1}} Generating diagram"
        python ${ROOT}/phom/plot_alpha2.py pairs.txt --lim ${DIAG_LIM} --nonums --label "iteration: ${1}" >> ${LOG}

        cp plot_pairs.txt.png "${label_dir}/${diag_name}" >> ${LOG}
    done

    cd ..
    rm -rf ${tmp_dir}

    true
}

if [ -z ${2} ] ; then
    echo "Parameters: name_of_the_run number_of_iterations"
    echo "Usage example: run_tests results 10"
    exit 0
fi

ROOT=${PWD}

MAIN_DIR=${1}
RUNS_NUM=${2}
DIAG_LIM=${3}

cd ${MAIN_DIR}

LOG="${PWD}/log.txt"

DIM=`jq -r .dim params.json`
PERIOD=`jq -r .encoding_period params.json`
LABELS="0 1 2 3 4 5 6 7 8 9" # all

if [ "`jq .filter_labels params.json`" != "null" ] ; then
    LABELS="`jq .filter_labels params.json | jq 'map(tostring)' | jq -r 'join(" ")'`"
fi
if [ "`jq .iter_labels params.json`" != "null" ] ; then
    LABELS="`jq .iter_labels params.json | jq 'map(tostring)' | jq -r 'join(" ")'`"
fi
myecho "Labels: ${LABELS}"

# Initialize thread pool

job_pool_init 4 1

for (( i = 1 ; i <= ${RUNS_NUM} ; i++ ))
do
    myecho "${i}'th iteration"

    run_folder="run${i}"
    wait_enter ${run_folder}


    it=0
    while true ; do
        waiting=true
        while [ ! -f encoded/train-$((it+PERIOD)).txt ] && [ ! -f train_encoded.txt ] ; do
            if ${waiting} ; then
                waiting=false
                myecho "  Waiting for train-${it}.txt ..."
            fi
            sleep 2
        done
        if [ ! -f encoded/train-${it}.txt ] ; then
            myecho "  Exiting"
            break
        fi

        job_pool_run to_parallel ${it} "${PWD}"

        it=$((it+PERIOD))
    done

    cd ..

done

job_pool_shutdown


cd ..


myecho "Done!"
