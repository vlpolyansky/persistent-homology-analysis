#!/bin/bash
set -e
set -o pipefail

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

ROOT=${PWD}
export PYTHONPATH=${ROOT}

MAIN_DIR=${1}
RUNS_NUM=${2}
PERCENTAGE=${3}


cd ${MAIN_DIR}

LOG="${PWD}/log.txt"

DIM=`jq -r .dim params.json`
LABELS="0 1 2 3 4 5 6 7 8 9" # all

if [ "`jq .filter_labels params.json`" != "null" ] ; then
    LABELS="`jq .filter_labels params.json | jq 'map(tostring)' | jq -r 'join(" ")'`"
fi
if [ "`jq .hom_labels params.json`" != "null" ] ; then
    LABELS="`jq .hom_labels params.json | jq 'map(tostring)' | jq -r 'join(" ")'`"
fi
myecho "Labels: ${LABELS}"


for (( i = 1 ; i <= ${RUNS_NUM} ; i++ ))
do
    myecho "${i}'th iteration"

    run_folder="run${i}"
    while [ ! -d "run$((i))" ]
    do
        sleep 2
    done
    wait_enter ${run_folder}
    while [ ! -f "train_encoded.txt" ]
    do
        sleep 2
    done

    for j in ${LABELS}
    do
        myecho "  Label ${j}:"

        label_dir="label${j}train"
        mkdir -p ${label_dir}
        cd ${label_dir}

        if [ -f "plot_pairs.txt.png" ] ; then
            cd ..
            continue
        fi

        if [ ! -f "data.txt" ] ; then
            myecho "    Preparing data"
            python ${ROOT}/phom/generate_mnist_input.py "../train_encoded.txt" ${j} "data.txt" |& tee -a ${LOG}
        fi

        if [ ! -f "pairs.txt" ] ; then
            myecho "    Running ${DIM}D homology analysis"
            ${ROOT}/phom/homology_${DIM}d data.txt --percentage ${PERCENTAGE} |& tee -a ${LOG}
        fi

        myecho "    Generating diagram"
        python ${ROOT}/phom/plot_alpha2.py pairs.txt --nocut --label "Run: ${i} | Label: ${j}" |& tee -a ${LOG}

        cp plot_pairs.txt.png "../../_pics/${run_folder}${label_dir}.png" |& tee -a ${LOG}

         cd ..
    done

    cd ..
done