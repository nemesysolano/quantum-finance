#!/usr/bin/env bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )"
PYTHON_ENV="${SCRIPT_DIR}/pydevs"
FOURASTRO_MODULE_PATH="${SCRIPT_DIR}/qf"
MODELS_FOLDER="${SCRIPT_DIR}/models"
TEST_RESULTS_FOLDER="${SCRIPT_DIR}/test-results"

if [ ! -d "${PYTHON_ENV}" ]; then
    mkdir -p "${PYTHON_ENV}"
    python3 -m venv "${PYTHON_ENV}"    
fi

if [ ! -d "${MODELS_FOLDER}" ]; then
    mkdir -p "${MODELS_FOLDER}"
fi


if [ ! -d "${TEST_RESULTS_FOLDER}" ]; then
    mkdir -p "${TEST_RESULTS_FOLDER}"
fi

source "${PYTHON_ENV}/bin/activate"
pip show pandas
STATUS=$?

if [ ${STATUS} -ne 0 ]
then
    pip install -r ${SCRIPT_DIR}/packages.txt
fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${PYTHON_ENV}/lib/python3.9/site-packages/tensorflow-plugins"
export PYTHONPATH="${PYTHONPATH}:${FOURASTRO_MODULE_PATH}"

