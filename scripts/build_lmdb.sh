MODEL=$1
DATASET=$2
CONFIG_NAME=$3

for SPLIT in test train; do
  RAW=dataset/${DATASET}_raw/${SPLIT}
  LMDB=dataset/${DATASET}/${SPLIT}
  echo ${LMDB}
  python scripts/build_lmdb.py --config configs/projects/${MODEL}/${DATASET}/${CONFIG_NAME} --data_root ${RAW} --output_root ${LMDB} --overwrite
done