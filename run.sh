GPUS=$1
NB_COMMA=`echo ${GPUS} | tr -cd , | wc -c`
NB_GPUS=$((${NB_COMMA} + 1))
PORT=$((9000 + $RANDOM % 1000))

shift 1
echo "Launching exp on $GPUS... PORT $PORT"
MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=${GPUS} torchrun --master_port ${PORT} --nproc_per_node=${NB_GPUS} main.py $@ >error.txt