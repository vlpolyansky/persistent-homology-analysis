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
    # it
    myecho "  {${1}} Calculating distance matrices for all runs and labels of train-${1}.txt"
    tmp_dir="temp_${1}"
    padded=`printf "%05d" $(( ${1} / PERIOD ))`
    matrix_name="${padded}-matrix.txt"
    mkdir -p ${tmp_dir}
    cd ${tmp_dir}

    for j in ${LABELS}
    do
        matrix_dir="../label${j}"
        mkdir -p ${matrix_dir}
        if [ -f "${matrix_dir}/${matrix_name}" ] ; then
            continue
        fi

        myecho "  {${1}} Label ${j}:"

        rm -f files.txt
        touch files.txt
        for (( i = 1 ; i <= ${RUNS_NUM} ; i++ ))
        do
            echo "../../run${i}/diagrams/label${j}/pairs-${1}.txt" >> files.txt
        done

        ${ROOT}/phom/dist_matrix files.txt --min_value 0.0001 --no_zero_dim >> ${LOG}
        mv homology_distances.txt "${matrix_dir}/${matrix_name}"
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

mkdir -p _distances
cd _distances

# Initialize thread pool

job_pool_init 4 1


it=0
while true ; do
    if [ ! -f ../run1/encoded/train-${it}.txt ] ; then
        myecho "  Exiting"
        break
    fi

    job_pool_run to_parallel ${it}

    it=$((it+PERIOD))
done

job_pool_shutdown

cd ${ROOT}

myecho "Done!"
