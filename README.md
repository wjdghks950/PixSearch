# PixSearch

PixSearch is a segmentation-enabled multimodal model that can retrieve evidence
using text, a whole image, or a predicted image region while generating an answer.
The implementation retains the historical `pixar` directory and `PIXAR` class
names for checkpoint compatibility.

This repository contains the preparation and modeling code. Datasets, pretrained
backbones, generated annotations, FAISS indexes, and trained PixSearch weights are
not included. Train and export your own checkpoint before running model evaluation.

## Workflow and layout

| Location | Purpose |
| --- | --- |
| `src/build_multimodal_index.py` | Encode Wikipedia text/images and build FAISS indexes |
| `src/local_search_api.py` | Serve those indexes through an HTTP search API |
| `src/vqa_dataset_construct.py` | Generate answers, search decisions, and region supervision |
| `src/calculate_dataset_stats.py` | Classify annotated questions before retrieval interleaving |
| `src/construct_gt_interleaved.py` | Produce the ground-truth-enriched training variant |
| `pixar/src/model/pixar_train_ds.py` | Stage 1 segmentation training and Stage 2 retrieval fine-tuning |
| `pixar/src/model/validate_vqa.py` | VQA evaluation with or without search |
| `pixar/src/model/validate_seg.py` | Segmentation evaluation |
| `scripts/model.sh` | Portable launcher for training, export, evaluation, and chat |

Run the examples from the repository root. Use absolute paths for data and
checkpoints: the model launcher changes into `pixar/src/model` to support the
legacy relative resources. Core `run_*pixar*.sh` wrappers now accept explicit Python CLI arguments; old
positional recipes are no longer supported. Other legacy baseline scripts are
retained for reference. Use the commands below for a new setup.

## Installation

Use Linux, Python 3.10, and NVIDIA CUDA GPUs. Training a 13B model requires
substantial GPU memory; start with batch size 1 and increase gradient accumulation
or the number of GPUs. Full Wikipedia indexing also requires substantial host RAM
and disk space. No minimum-memory claim is made for an untested hardware setup.

Use **two environments**: the legacy LLaVA model and the newer DINOv3 retrieval
encoder require different Transformers versions. The model calls the retrieval
server over HTTP, so they do not need to share an environment.

```bash
git clone https://github.com/wjdghks950/PixSearch.git
cd PixSearch
export PIXSEARCH_ROOT="$PWD"
export DATA_DIR="$PIXSEARCH_ROOT/data"
export WEIGHTS_DIR="$PIXSEARCH_ROOT/weights"
mkdir -p "$DATA_DIR" "$WEIGHTS_DIR"

# Model training / inference (LLaVA/Llama-2 path)
conda create -n pixsearch-model python=3.10 -y
conda activate pixsearch-model
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements/model-lock.txt

# Preparation / indexing / search server
conda create -n pixsearch-data python=3.10 -y
conda activate pixsearch-data
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements/data-lock.txt
```

The `*-lock.txt` files record the resolved dependency versions; the shorter
`model.txt` and `data.txt` files describe direct dependencies. Both sets were
installed in isolated CPU environments and their main entry points passed `--help`
checks. This does not validate CUDA kernels, GroundingDINO compilation, or full training.

These are pip requirements, not a `conda create --file` export. Choose a driver
compatible with the indicated CUDA wheels. GroundingDINO additionally needs a
working compiler/CUDA toolkit to build its extension. Install its upstream code
and SAM in the **data** environment for region annotation:

```bash
mkdir -p external
git clone https://github.com/IDEA-Research/GroundingDINO.git external/GroundingDINO
python -m pip install --no-build-isolation -e external/GroundingDINO
python -m pip install 'git+https://github.com/facebookresearch/segment-anything.git'
```

Follow the upstream [GroundingDINO installation and checkpoint instructions](https://github.com/IDEA-Research/GroundingDINO#install)
if CUDA extension compilation fails. DINOv3 may require accepting its Hugging Face
access terms and logging in with `huggingface-cli login`. The optional CRAG-MM
retriever has its own [installation and index instructions](https://github.com/facebookresearch/CRAG-MM/blob/main/docs/search_api.md);
install `cragmm-search-pipeline` only when using `--retriever cragmm` or
`--search_api cragmm`, and check its dependency constraints in that environment.
The Llama-3.2 and other baseline scripts are experimental and are not covered by
the legacy model dependency set above.

## Download datasets and weights

Download only the datasets needed for your experiment. Keep official train and
validation/test splits separate. The source links below contain the downloads,
licenses, and any access requirements.

| Resource | Download instructions | Expected local location / use |
| --- | --- | --- |
| Wiki6M | [InfoSeek release](https://github.com/open-vision-language/infoseek) | `data/infoseek/Wiki6M_ver_1_0.jsonl`; retrieval corpus |
| InfoSeek | [Official annotations](https://github.com/open-vision-language/infoseek) | `data/pixar/infoseek/infoseek_train.jsonl`, `infoseek_val.jsonl` |
| OVEN and shared InfoSeek images | [OVEN release and image download guide](https://github.com/open-vision-language/oven) | `data/pixar/oven/oven_entity_train.jsonl`, `oven_entity_val.jsonl`, `oven_query_val.jsonl`; extracted images under `data/infoseek/images/` |
| A-OKVQA | [HuggingFaceM4/A-OKVQA](https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA) | Builder downloads the selected split and caches images |
| OK-VQA | [HuggingFaceM4/OK-VQA](https://huggingface.co/datasets/HuggingFaceM4/OK-VQA) | Builder downloads the selected split and caches images |
| TextVQA | [Official release](https://textvqa.org/dataset/) / [HF mirror](https://huggingface.co/datasets/lmms-lab/textvqa) | Builder uses HF; evaluation requires `data/pixar/textvqa/TextVQA_0.5.1_val.json` and `data/textvqa/train_images/` |
| WebQA (optional) | [Official release](https://github.com/WebQnA/WebQA) / [HF mirror](https://huggingface.co/datasets/TreezzZ/WebQA) | Provide image TSV, line index, and image cache through the builder's `--webqa_*` arguments |
| CRAG-MM (optional) | [Official benchmark](https://github.com/facebookresearch/CRAG-MM) | HF loader downloads `crag-mm-2025/crag-mm-single-turn-public`; model evaluation uses validation revision `v0.1.2` |
| Stage 1 segmentation + instruction data | [LISA data preparation](https://github.com/JIA-Lab-research/LISA#training) | ADE20K, COCO-Stuff, PACO-LVIS, Pascal-Part, Mapillary, RefCOCO family, ReasonSeg, and LLaVA instructions under `data/` |
| SAM ViT-H | [SAM checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints) | `weights/sam_vit_h_4b8939.pth` |
| GroundingDINO Swin-T | [GroundingDINO checkpoints](https://github.com/IDEA-Research/GroundingDINO#install) | `weights/groundingdino_swint_ogc.pth`; use config from `external/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py` |
| LLaVA base | [13B backbone](https://huggingface.co/liuhaotian/llava-llama-2-13b-chat-lightning-preview) | Downloaded by Transformers, or pass a local directory via `--version` |

For Stage 1, follow the upstream archive extraction instructions and preserve this
layout (the loaders do not download these training archives):

```text
data/
  ade20k/{images,annotations}/{training,validation}/
  coco/{train2017,val2017}/
  cocostuff/train2017/
  vlpart/paco/annotations/paco_lvis_v1_train.json
  vlpart/pascal_part/train.json
  mapillary/
  refer_seg/
  reason_seg/ReasonSeg/{train,val}/
  llava_dataset/llava_v1_5_mix665k.json
  gqa/images/
  ocr_vqa/images/
  textvqa/train_images/
  vg/VG_100K/
  vg/VG_100K_2/
  infoseek/images/
  pixar/{oven,infoseek,aokvqa,okvqa,textvqa,webqa}/
```

LLaVA mixed-instruction records contain image paths relative to `data/`; extract
all their image sources, not just COCO. The default Stage 1 mixture uses all five
semantic datasets listed above. To train a smaller subset, explicitly change
`--sem_seg_data`, `--refer_seg_data`, `--dataset`, and `--sample_rates` together.

## 1. Build and serve the retrieval index

Activate `pixsearch-data`. The Wiki6M JSONL must contain `wikipedia_title`,
`wikipedia_content`, and `wikipedia_image_url` (with optional `wikipedia_summary`).
Build **both** encoders into the same collection. Torch encoders run on CUDA;
FAISS indexes remain in CPU memory so `faiss-cpu` is sufficient.

```bash
bash src/scripts/run_mm_index_construct.sh \
  --input_jsonl "$DATA_DIR/infoseek/Wiki6M_ver_1_0.jsonl" \
  --output_dir "$DATA_DIR/wiki6m_index" \
  --model_str sentence-transformers/all-mpnet-base-v2 --batch_size 32

bash src/scripts/run_mm_index_construct.sh \
  --input_jsonl "$DATA_DIR/infoseek/Wiki6M_ver_1_0.jsonl" \
  --output_dir "$DATA_DIR/wiki6m_index" \
  --model_str facebook/dinov3-vitb16-pretrain-lvd1689m \
  --cache_wiki_images --batch_size 32

bash src/scripts/run_local_searchapi.sh \
  --base-dir "$DATA_DIR" --collection wiki6m_index --device cuda:0 \
  --host 127.0.0.1 --port 7000
```

Each encoder directory contains `wiki6m_faiss.index` and `wiki6m_meta.arrow`, plus
per-rank checkpoint files. Image indexing downloads the Wikipedia image URLs;
missing or failed images are skipped. Use `--limit_total 100` and a **different
output collection** for a smoke run. Set `NPROC=2` (or more) to distribute encoding;
use `--resume` only with the same corpus, encoder, chunk settings, and world size.
Do not mix a smoke collection with a full run. Keep server and evaluation on GPUs
with enough free memory; `CUDA_VISIBLE_DEVICES` controls each process separately.

```bash
curl http://127.0.0.1:7000/health
```

The HTTP API accepts image file paths, so the server must be able to read the
same paths as the client. For separate machines, mount the images at identical
paths or adapt the API before use.

## 2. Construct PixSearch training data

Synthetic annotation and question classification use OpenAI API calls. Supply
`OPENAI_API_KEY` through your environment; select a model your account can access
using `--llm_model` / `--model`. Start with `--limit_num 10` to inspect a small run.
Annotation also runs GroundingDINO + SAM to produce region masks.

The pipeline has three ordered stages; this avoids requiring classification
statistics before annotation files exist. This example uses OVEN train:

```bash
# In pixsearch-data, with OPENAI_API_KEY already set:
bash src/scripts/run_vqa_dataset_construct.sh \
  --stage annotate --datasets oven --split train \
  --output_dir "$DATA_DIR" --dataset_name pixar --version_num 1 \
  --sam_checkpoint "$WEIGHTS_DIR/sam_vit_h_4b8939.pth" \
  --dino_config "$PIXSEARCH_ROOT/external/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
  --dino_checkpoint "$WEIGHTS_DIR/groundingdino_swint_ogc.pth" \
  --llm_model gpt-4.1-mini --limit_num 10

python src/calculate_dataset_stats.py \
  --data_root "$DATA_DIR/pixar" --out_json "$DATA_DIR/pixar/question_types.json" \
  --fig_dir "$PIXSEARCH_ROOT/results/question_types" --version_num 1 --model gpt-4.1-mini

# Requires the local index server from step 1.
bash src/scripts/run_vqa_dataset_construct.sh \
  --stage retrieve --datasets oven --output_dir "$DATA_DIR" \
  --dataset_name pixar --version_num 1 --retriever local \
  --search_api_url http://127.0.0.1:7000 --search_api_collection wiki6m_index \
  --use_llm_for_span_extraction --topk 5

python src/construct_gt_interleaved.py \
  --in "$DATA_DIR/pixar/oven/oven_retrieve_k_5_interleaved_v2.jsonl" \
  --out "$DATA_DIR/pixar/oven/oven_retrieve_k_5_gt_interleaved_v2.jsonl" \
  --k 3 --seed 42 --force_style mixed --model gpt-4.1-mini
```

The outputs, in order, are `*_search_annotated_v1.jsonl`,
`question_types.json`, `*_search_interleaved_v1.jsonl`,
`*_retrieve_k_5_interleaved_v2.jsonl`, and
`*_retrieve_k_5_gt_interleaved_v2.jsonl`. The retrieval stage historically increments
the version by one; this is why annotation v1 becomes training v2.
`construct_gt_interleaved.py --k` is a helpfulness-score bound, not retrieval top-k.
Inspect `_regen_meta.error` in its output before training; failed rewrites preserve
the original row.

For a full run, increase/remove `--limit_num` (nonpositive means unlimited) and set
`--infoseek_limit_num` as needed (default 20,000 for local OVEN/InfoSeek input).
Pass multiple names to `--datasets`, for example
`oven infoseek aokvqa okvqa textvqa`, and run the final GT-enrichment command for each.
Only include generated datasets in the training `--pixar_data` mixture. Existing
annotations resume by source row index; keep the source split/order unchanged.
After adding annotations, rerun classification and rebuild downstream interleaving
files if their inputs changed. Do not combine train and evaluation data in one
output directory. CRAG-MM defaults to validation and must not silently become
training data for a CRAG-MM validation experiment.

The training loader currently expects exactly
`data/pixar/NAME/NAME_retrieve_k_5_gt_interleaved_v2.jsonl`.
Rows include a cached `image_path`, question, answer with `<search>` and
`<information>` spans, and `search_calls` containing region masks and span offsets.
Keep the generated image cache alongside the JSONL. When original cached paths are unavailable, training supports the legacy
TextVQA image layout `data/textvqa/images/`; link it to `train_images/` if needed.

## 3. Train PixSearch

Activate `pixsearch-model`. Stage 1 learns segmentation and instruction following;
Stage 2 adds retrieval supervision. The following are starting recipes, not a
claim of reproducing an unpublished checkpoint or benchmark score.

```bash
export BASE_MODEL=liuhaotian/llava-llama-2-13b-chat-lightning-preview

GPUS=0,1 bash scripts/model.sh train \
  --version "$BASE_MODEL" --dataset_dir "$DATA_DIR" \
  --vision_pretrained "$WEIGHTS_DIR/sam_vit_h_4b8939.pth" \
  --dataset 'sem_seg||refer_seg||vqa||reason_seg' --sample_rates '9,9,6,1' \
  --batch_size 1 --grad_accumulation_steps 10 --precision bf16 \
  --use_bidir_bio --use_feedback_loop --train_mask_prompt_encoder \
  --dice_type focal_tversky --dice_loss_weight 8 --bce_loss_weight 2 \
  --seg_cls_loss_weight 2 --kld_loss_weight 0.1 \
  --model_max_length 1024 --epochs 20 --exp_name pixsearch-stage1 \
  --log_dir "$PIXSEARCH_ROOT/runs" --auto_resume

# Set to the actual Stage 1 DeepSpeed checkpoint DIRECTORY printed by training.
export STAGE1_CKPT="$PIXSEARCH_ROOT/runs/REPLACE_WITH_STAGE1_CHECKPOINT_DIRECTORY"
GPUS=0,1 bash scripts/model.sh train \
  --version "$BASE_MODEL" --dataset_dir "$DATA_DIR" \
  --vision_pretrained "$WEIGHTS_DIR/sam_vit_h_4b8939.pth" \
  --zero_shot_ckpt_path "$STAGE1_CKPT" \
  --dataset 'sem_seg||refer_seg||vqa||reason_seg||pixar' --sample_rates '3,3,5,1,9' \
  --pixar_data 'oven' --random_seed 42 \
  --batch_size 1 --grad_accumulation_steps 10 --precision bf16 \
  --use_bidir_bio --use_feedback_loop --train_mask_prompt_encoder \
  --dice_type focal_tversky --dice_loss_weight 8 --bce_loss_weight 2 \
  --seg_cls_loss_weight 2 --kld_loss_weight 0.1 \
  --info_mask_type pad --lr 3e-4 --model_max_length 512 --epochs 20 \
  --exp_name pixsearch-stage2 --log_dir "$PIXSEARCH_ROOT/runs" --auto_resume
```

`--pixar_data 'oven'` matches the example preparation above. For the broader mixture,
generate all five datasets, then use `'oven||infoseek||aokvqa||okvqa||textvqa'`.
`--no_eval` disables epoch validation; otherwise prepare ADE20K validation.
Add `--wandb` only if you want Weights & Biases logging. `GPUS` selects physical
DeepSpeed GPU IDs, and `MASTER_PORT` overrides the launcher's default 29500.

## 4. Export a checkpoint

DeepSpeed training checkpoints are not directly loadable through `--version`.
Consolidate the desired Stage 2 checkpoint into a state dict and merge LoRA:

```bash
export TRAIN_CKPT="$PIXSEARCH_ROOT/runs/REPLACE_WITH_STAGE2_CHECKPOINT_DIRECTORY"
python "$TRAIN_CKPT/zero_to_fp32.py" "$TRAIN_CKPT" "$WEIGHTS_DIR/pixsearch_fp32.bin"

bash scripts/model.sh merge \
  --version "$BASE_MODEL" --weight "$WEIGHTS_DIR/pixsearch_fp32.bin" \
  --save_path "$WEIGHTS_DIR/pixsearch-merged" \
  --vision_pretrained "$WEIGHTS_DIR/sam_vit_h_4b8939.pth" \
  --use_bidir_bio --use_feedback_loop --train_mask_prompt_encoder \
  --bidir_dim_feedforward 2048 --seg_cls_loss_weight 2 --model_max_length 512
```

Use the same backbone, LoRA rank/targets, and architecture flags as training.
Review missing/unexpected-key diagnostics during export. Keep the exported
configuration, tokenizer, and weight files together.

## 5. Evaluate and interact

Evaluation consumes official held-out data, not the GT-enriched training JSONL.
Prepare the validation annotations/images from the download table. GQA additionally
expects `data/gqa/questions/val_balanced_questions.json` and `data/gqa/images/`;
POPE expects its three `coco_pope_*.json` files under `data/pope/coco/` and COCO val2014 images under `data/pope/coco/val2014/`.
See [GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html) and
[POPE](https://github.com/RUCAIBox/POPE) for these optional benchmarks.

```bash
export MODEL_DIR="$WEIGHTS_DIR/pixsearch-merged"

# Answer without retrieval (one GPU).
CUDA_VISIBLE_DEVICES=0 bash scripts/model.sh eval \
  --version "$MODEL_DIR" --dataset_dir "$DATA_DIR" \
  --val_dataset 'oven|val' --oven_mode query --no_search \
  --predictions_path "$PIXSEARCH_ROOT/results/oven/no-search" --model_max_length 512

# Answer with region/image/text retrieval; keep the data-environment server running.
CUDA_VISIBLE_DEVICES=0 bash scripts/model.sh eval \
  --version "$MODEL_DIR" --dataset_dir "$DATA_DIR" \
  --val_dataset 'infoseek|val' --search_api local \
  --search_base_url http://127.0.0.1:7000 --search_collection wiki6m_index \
  --search_k 5 --retrieve_type title \
  --predictions_path "$PIXSEARCH_ROOT/results/infoseek/search" --model_max_length 512

# Interactive retrieval-augmented answers.
CUDA_VISIBLE_DEVICES=0 bash scripts/model.sh chat \
  --version "$MODEL_DIR" --search_collection wiki6m_index --prompt_user_input

# Segmentation evaluation on ADE20K validation.
CUDA_VISIBLE_DEVICES=0 bash scripts/model.sh seg \
  --version "$MODEL_DIR" --dataset_dir "$DATA_DIR" --val_dataset 'ade20k|validation'
```

VQA dataset selectors include `textvqa|val`, `gqa|val`, `pope|val`,
`infoseek|val`, `oven|val`, and `cragmm|validation`.
Use `--oven_mode entity` for OVEN's entity split. `--limit_batches 10` limits a smoke
evaluation. Predictions are JSONL files under `--predictions_path`; keep separate
output directories for different checkpoints, splits, retrieval settings, and
scoring methods. `--use_llm_as_judge --llm_model ...` enables API-based judging;
`--format_with_llm` also makes API calls. Record these options when reporting scores.
For CRAG-MM search, install the optional pipeline first and use `--search_api cragmm`.

## Verification and troubleshooting

```bash
python -m unittest discover -s tests -v
bash scripts/model.sh help
```

The lightweight tests cover preparation stage ordering, dataset selection, unlimited
annotation, resume after skipped rows, and safe chunk-window bounds without GPUs
or API calls. Full training, dataset generation with paid APIs, and benchmark runs
are separate validation steps and are not exercised by these tests.

- Missing `question_types.json`: run annotation, then classification, then retrieval.
- Missing training JSONL: check the required `k_5_gt_interleaved_v2` filename and dataset list.
- Missing image: check the extraction layout and preserve the generated image cache.
- Missing index: build both encoders and use the same collection in server and client.
- CUDA out of memory: reduce batch size / sequence length or assign search to another GPU.
- Legacy model import errors: activate `pixsearch-model`; do not upgrade its Transformers
  dependency to the DINOv3 environment version.

Model code builds on LLaVA, SAM, LISA, and PLUM/PARTONOMY. Preserve upstream notices
and comply with the licenses of the source datasets, models, and dependencies.
