# Check if GPU ids are passed as an argument; if not, ask for it.
if [ -z "$1" ]; then
    read -p "Enter GPU IDs (comma-separated for multiple GPUs): " gpu_ids
else
    gpu_ids=$1
fi

# export PYTHONPATH="/home/jeonghkim/projects/pixar:$PYTHONPATH"
export PYTHONPATH="/shared/nas2/jk100/PixaR/:$PYTHONPATH"

# CKPT_PARENT_PATH=/home/jeonghkim/projects/pixar/pixar/src/model/runs
CKPT_PARENT_PATH=/shared/nas2/jk100/partonomy_private/src/models/PLUM/runs
# MODEL_CKPT_PATH=$CKPT_PARENT_PATH/pixar-13b_kld_0.1_focal_tversky_8_pixar_ft/merged_model
MODEL_CKPT_PATH=$CKPT_PARENT_PATH/plum-13b_kld_0.1_focal_tversky_8_v1_0shot_w_reasonseg/merged_model_9551

CUDA_VISIBLE_DEVICES=${gpu_ids} python chat_pixar_search.py \
	  --version=$MODEL_CKPT_PATH \
	  --precision='bf16' \
      --use_bidir_bio \
      --use_feedback_loop \
      --conv_type pixar_0shot_v1


# input text: What parts does the commercial airplane have in common with a military fighter jet?
# input text: What parts do you see in this airplane?
# input image: /shared/nas/data/m1/jk100/code/OpenAttrLibrary/LISA/dataset/vlpart/pascal_part/VOCdevkit/VOC2010/JPEGImages/2007_000033.jpg

# input text: What is the part of the plane that allows the plane to fly?
# input image: '/shared/nas/data/m1/jk100/code/OpenAttrLibrary/LISA/dataset/partonomy/images/boats--banana boat/0d4dc0bcc13e52bb8a73d396b9415a045e14f3f69885f801d7ef1613c8a3e021.jpg'

# input image: /home/jeonghkim/projects/pixar/data/infoseek/images/oven_01963180.JPEG
# input image: /shared/nas/data/m1/jk100/code/OpenAttrLibrary/LISA/dataset/pixar/infoseek/images/oven_01963180.JPEG
# input text: Which place is this animal endemic to?

# input image: /shared/nas/data/m1/jk100/code/OpenAttrLibrary/LISA/dataset/pixar/infoseek/images/oven_01857671.JPEG
# input text: What is the conservation status of this animal?

# input image: /shared/nas/data/m1/jk100/code/OpenAttrLibrary/LISA/dataset/pixar/infoseek/images/oven_02959375.JPEG
# input text: Who is the manufacturer of this vehicle?