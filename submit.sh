#!/bin/bash
#SBATCH -p highmem
source /data/soconnell/phd/test_scripts/set_up_env.sh tf_latest
file=$1
epochs=$2
afn=$3
name=$4
run() {
python3.7 autoencoder_train.py -f $file -sc -ep $epochs -afn $afn -name $name -n 50
}
run 2>&1

