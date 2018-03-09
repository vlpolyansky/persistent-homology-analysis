#!/bin/bash
set -e
set -o pipefail

function myecho {
    echo -e "[`date`] ${1}" |& tee -a ${LOG}
    echo -e "\033[1;35m[`date`] \033[0;35m${1}\033[0m"
}

if [ -z ${3} ] ; then
    myecho "Parameters: params_file name_of_the_run number_of_iterations"
    myecho "Usage example: run_tests params_fc.json results 10"
    exit 0
fi

ROOT=${PWD}
export PYTHONPATH=${ROOT}

PARAMS_FILE=${1}
MAIN_DIR=${2}
RUNS_NUM=${3}

mkdir -p ${MAIN_DIR}
cd ${MAIN_DIR}

LOG="${PWD}/log.txt"

mkdir -p _pics
mkdir -p _pairs_files_lists

if [ ! -f ./params.json ]; then
    cp ${ROOT}/${PARAMS_FILE} ./params.json
fi

ALGO=`jq -r .algo params.json`
DIM=`jq -r .dim params.json`
LABELS="0 1 2 3 4 5 6 7 8 9" # all

if [ "`jq .filter_labels params.json`" != "null" ] ; then
    LABELS="`jq .filter_labels params.json | jq 'map(tostring)' | jq -r 'join(" ")'`"
fi
myecho "Labels: ${LABELS}"

if [ ! -d _data ] ; then
    myecho "Preparing data"
    python ${ROOT}/main.py prepare params.json |& tee -a ${LOG}
fi

for (( i = 1 ; i <= ${RUNS_NUM} ; i++ ))
do
    myecho "${i}'th iteration"

    run_folder="run${i}"
    mkdir -p ${run_folder}
    cd ${run_folder}

    if [ -f ../train_encoded.txt ]; then
        myecho "  Using existing encoding"
        cp ../train_encoded.txt ./
        cp ../test_encoded.txt ./
    elif [ ! -f ./train_encoded.txt ]; then
        case "${ALGO}" in
            "FCAutoencoder" | "ConvAutoencoder" | "FCClassifier" | "MixedClassifier" | "MixedAutoencoder" )
                myecho "  Training/encoding ${ALGO}"
                python ${ROOT}/main.py train_encode ../params.json |& tee -a ${LOG}

                myecho "  Running final encoder"
                python ${ROOT}/main.py encode ../params.json |& tee -a ${LOG}
                ;;
            "PCA" )
                myecho "  Running PCA"
                python ${ROOT}/main.py pca ../params.json |& tee -a ${LOG}
                ;;
            "Identity" )
                myecho "  Keeping input as is"
                python ${ROOT}/main.py identity ../params.json |& tee -a ${LOG}
                ;;
        esac
        if [ "`jq .postprocess ../params.json`" != "null" ] ; then
            myecho "  Postprocessing"
            python ${ROOT}/main.py process ../params.json |& tee -a ${LOG}
        fi
    else
        myecho "  Found existing encoding"
    fi

    cd ..
done

cd ..

myecho "Done!"
