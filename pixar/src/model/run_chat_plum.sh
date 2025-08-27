# Check if GPU ids are passed as an argument; if not, ask for it.
if [ -z "$1" ]; then
    read -p "Enter GPU IDs (comma-separated for multiple GPUs): " gpu_ids
else
    gpu_ids=$1
fi

#MODEL_CKPT_PATH=/shared/nas2/jk100/partonomy_private/src/models/PLUM/runs/plum-13b_kld_0.1_focal_tversky_8_v1_0shot/merged_model
MODEL_CKPT_PATH=/shared/nas2/jk100/partonomy_private/src/models/PLUM/runs/plum-13b_kld_0.1_focal_tversky_8_v1_partonomy_ft/merged_model_joint

CUDA_VISIBLE_DEVICES=${gpu_ids} python chat.py \
	  --version=$MODEL_CKPT_PATH \
	  --precision='bf16' \
      --use_bidir_bio \
      --use_feedback_loop \
      --prompt_user_input

# --load_in_8bit \
# --version="xinlai/LISA-13B-llama2-v1" \

# input text: What parts does the commercial airplane have in common with a military fighter jet?
# input text: What parts do you see in this airplane?
# input image: /shared/nas/data/m1/jk100/code/OpenAttrLibrary/LISA/dataset/vlpart/pascal_part/VOCdevkit/VOC2010/JPEGImages/2007_000033.jpg

# input text: What is the part of the plane that allows the plane to fly?

# input image: '/shared/nas/data/m1/jk100/code/OpenAttrLibrary/LISA/dataset/partonomy/images/boats--banana boat/0d4dc0bcc13e52bb8a73d396b9415a045e14f3f69885f801d7ef1613c8a3e021.jpg'
