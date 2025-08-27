
# Check if GPU ids are passed as an argument; if not, ask for it.
if [ -z "$1" ]; then
    read -p "Enter GPU IDs (comma-separated for multiple GPUs): " gpu_ids
else
    gpu_ids=$1
fi

CKPT_TYPE=plum-13b_kld_0.5_dice_2
DATASET_DIR=/shared/nas/data/m1/jk100/code/OpenAttrLibrary/LISA/dataset
MODEL_CKPT_PATH=/shared/nas2/jk100/partonomy_private/src/models/PLUM/runs/${CKPT_TYPE}/merged_model
VISION_MODEL_PATH=/shared/nas/data/m1/jk100/code/OpenAttrLibrary/LISA/weights/sam_vit_h_4b8939.pth

deepspeed --include localhost:${gpu_ids} --master_port=24999 plum_train_ds.py \
	  --version=$MODEL_CKPT_PATH \
	  --dataset_dir=$DATASET_DIR \
	  --vision_pretrained=$VISION_MODEL_PATH \
	  --val_dataset="ReasonSeg|val" \
	  --exp_name=$CKPT_TYPE \
	  --eval_only

# --val_dataset="ReasonSeg|val" \
# --version="xinlai/LISA-13B-llama2-v1" \