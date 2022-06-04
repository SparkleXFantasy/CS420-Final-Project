CL_DATE=`date '+%y-%m-%d_%H-%M'`

#CL_MODEL="densenet161"
#CL_MODEL="resnet101"
CL_MODEL="resnet50"
#CL_MODEL="sketchanet"

CL_CKPT_PREFIX="quickdraw_${CL_MODEL}"

CL_DATASET="quickdraw"
CL_DATASET_ROOT="../../dataset"
CL_LOG_DIR="logs"

CL_RUNNAME="${CL_DATE}-${CL_DATASET}-r2cnn-${CL_MODEL}"
mkdir "${CL_LOG_DIR}/${CL_RUNNAME}"

export PATH="$/usr/local/cuda-11.3/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH"

nvcc -V

CUDA_VISIBLE_DEVICES=0 python3 quickdraw_r2cnn_train.py \
    --ckpt_prefix "${CL_LOG_DIR}/${CL_CKPT_PREFIX}" \
    --dataset_fn ${CL_DATASET} \
    --dataset_root "${CL_DATASET_ROOT}" \
    --intensity_channels 8 \
    --log_dir "${CL_LOG_DIR}/${CL_RUNNAME}" \
    --model_fn ${CL_MODEL} \
    --num_epochs 1 \
2>&1 | tee -a "${CL_LOG_DIR}/${CL_RUNNAME}/train.log"

#nvidia-docker run --rm \
#    --network=host \
#    --shm-size 8G \
#    -v /:/host \
#    -v /tmp/torch_extensions:/tmp/torch_extensions \
#    -v /tmp/torch_models:/root/.torch \
#    -w "/host$PWD" \
#    -e PYTHONUNBUFFERED=x \
#    -e CUDA_CACHE_PATH=/host/tmp/cuda-cache \
#    craigleili/sketch-r2cnn:latest \


