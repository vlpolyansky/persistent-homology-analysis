#!/bin/bash
set -e

function myecho {
    echo -e "[`date`] ${1}" >> ${LOG}
    echo -e "\033[1;35m[`date`] \033[0;35m${1}\033[0m"
}

if [ -z ${2} ] ; then
    echo "Parameters: name_of_the_run run_num label alpha"
    exit 0
fi

ROOT=${PWD}
export PYTHONPATH=${ROOT}

MAIN_DIR=${1}
RUN_ID=${2}
label_folder=${3}
ALPHA=${4}


cd ${MAIN_DIR}

LOG="${PWD}/log.txt"

DIM=`jq -r .dim params.json`


run_folder="run${RUN_ID}"
cd ${run_folder}

dens_folder="dens_${ALPHA}_${label_folder}"
mkdir -p ${dens_folder}
cd ${dens_folder}

cp ../${label_folder}/data.txt ./data.txt

if [ ! -f pairs.txt ] ; then
    myecho "Starting ${DIM}d-filtering with alpha ${ALPHA}"
    ${ROOT}/phom/density_filtration_${DIM}d data.txt --alpha ${ALPHA} &>> ${LOG}
fi

myecho "Generating diagram"
python ${ROOT}/phom/plot_alpha2.py pairs.txt --nocut --label "Run: ${RUN_ID} | ${label_folder} | ALPHA: ${ALPHA}" &>> ${LOG}



myecho "Done!"
