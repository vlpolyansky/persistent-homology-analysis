#!/bin/bash
set -e

function myecho {
    echo -e "[`date`] ${1}" >> ${LOG}
    echo -e "\033[1;35m[`date`] \033[0;35m${1}\033[0m"
}

if [ -z ${3} ] ; then
    myecho "Parameters: params_file name_of_the_run number_of_iterations"
    myecho "Usage example: run_tests params_fc.json results 10"
    exit 0
fi

PARAMS_FILE=${1}
MAIN_DIR=${2}
RUNS_NUM=${3}

mkdir -p ${MAIN_DIR}
cd ${MAIN_DIR}

LOG="${PWD}/log.txt"

mkdir -p _pics
mkdir -p _pairs_files_lists
#if [ -f ./params.json ]; then
#    if ! cmp --silent ./params.json ../params.json; then
#        echo "Different params.json files"
#        exit 1
#    fi
#else
#    cp ../params.json ./
#fi
if [ ! -f ./params.json ]; then
    cp ../${PARAMS_FILE} ./params.json
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
    python ../main.py prepare params.json >> ${LOG}
fi

for (( i = 1 ; i <= ${RUNS_NUM} ; i++ ))
do
    myecho "${i}'th iteration"

#    if [ -d "run${i}" ]; then
#        echo "  Directory already exists, skipping..."
#        continue
#    fi

    run_folder="run${i}"
    mkdir -p ${run_folder}
    cd ${run_folder}

    if [ -f ../train_encoded.txt ]; then
        myecho "  Using existing encoding"
        cp ../train_encoded.txt ./
        cp ../test_encoded.txt ./
    elif [ ! -f ./train_encoded.txt ]; then
        case "${ALGO}" in
            "FCAutoencoder" | "ConvAutoencoder" | "FCClassifier" )
                myecho "  Training ${ALGO}"
                python ../../main.py train ../params.json >> ${LOG}

                myecho "  Running encoder"
                python ../../main.py encode ../params.json >> ${LOG}
                ;;
            "PCA" )
                myecho "  Running PCA"
                python ../../main.py pca ../params.json >> ${LOG}
                ;;
        esac
    else
        myecho "  Found existing encoding"
    fi

    myecho "  Running homology for all labels"
    for j in ${LABELS}
    do
        for k in "train" "test"
        do
            myecho "  Label ${j} (${k}):"

            label_folder="label${j}${k}"
            mkdir -p ${label_folder}
            cd ${label_folder}

            if [ ! -f data.txt ]; then
                myecho "    Preparing ${k} data"
                python ../../../phom/generate_mnist_input.py "../${k}_encoded.txt" ${j} "data.txt" >> ${LOG}
            else
                myecho "    ${k} data found"
            fi

            if [ ! -f pairs.txt ]; then
                myecho "    Running ${DIM}D homology analysis"
                ../../../phom/homology_${DIM}d "data.txt" >> ${LOG}
            else
                myecho "    Existing persistence pairs found"
            fi

            if [ ! -f  plot_pairs.txt.png ]; then
                myecho "    Generating diagram"
                python ../../../phom/plot_alpha2.py "pairs.txt" >> ${LOG}

                cp plot_pairs.txt.png ../../_pics/run${i}label${j}${k}.png >> ${LOG}
            else
                myecho "    Persistence diagram found"
            fi

            # Save reference to label (todo: replace this govnokod)
            touch "../../_pairs_files_lists/pairs_${label_folder}.txt"
            if grep -Fxq "../../${run_folder}/${label_folder}/pairs.txt" "../../_pairs_files_lists/pairs_${label_folder}.txt"
            then
                myecho "    Pairs file was already logged"
            else
                myecho "    Logging pairs file"
                echo "../../${run_folder}/${label_folder}/pairs.txt" >> "../../_pairs_files_lists/pairs_${label_folder}.txt"
            fi

            cd ..
        done
    done

    cd ..
done


myecho "Running embedding for all homologies"
mkdir -p _embed_2d
cd _embed_2d

for j in ${LABELS}
do
    for k in "train" "test"
    do
        label_folder="label${j}${k}"
        myecho "  ${label_folder}"

        mkdir ${label_folder}
        cd ${label_folder}

        ../../../phom/dist_matrix "../../_pairs_files_lists/pairs_${label_folder}.txt" --min_value 0.001 >> ${LOG}
        python ../../../phom/embed_homologies.py >> ${LOG}

        cd ..
    done
done

cd ..

cd ..

myecho "Done!"
