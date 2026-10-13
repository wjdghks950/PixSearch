import glob
import os
import random
import copy

import cv2
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide

from .conversation import get_default_conv_template
from .data_processing import get_mask_from_json
from .reason_seg_dataset import ReasonSegDataset
from .refer import REFER
from .refer_seg_dataset import ReferSegDataset
from .sem_seg_dataset import SemSegDataset
from .pixar_dataset import PixarDataset
from .explanatory_seg_datasets_adapter import ExplanatorySegDatasetsAdapter
from .utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX,
                    INFORMATION_START_TOKEN, INFORMATION_END_TOKEN,
                    SHORT_QUESTION_LIST, PLUM_ANSWER_LIST,
                    _info_char_spans, _search_char_spans, clamp_conversation_tensors_to_maxlen)
from .vqa_dataset import VQADataset


def collate_fn(
    batch,
    tokenizer=None,
    conv_type="llava_v1",
    use_mm_start_end=True,
    local_rank=-1,
    model_max_length=1024,
    info_mask_type="pad"  # or "attn"
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0

    conv_dataset_names = []
    conv_search_calls_norm = []

    per_token_label_dict_list = []
    search_calls_norm_list = []
    inferences = []
    dataset_names = []
    for (
        image_path,
        images,
        images_clip,
        conversations,
        masks,
        label,
        resize,
        questions,
        sampled_classes,
        per_token_label_dict,  # stores the concept label to mask (e.g., [{'airplane engine': mask, 'airplane wing': mask}])
        search_calls_norm,
        inference,
        dataset_name,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations) # NOTE: this will create sum(number of conversations over batches) - need 'offset_list' to keep track of the start index of each batch
        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)  # e.g., len(conversations[0]) == 4, len(conversations[1]) == 2, offset_list = [0, 4, 6]
        per_token_label_dict_list.append(per_token_label_dict)  # len(per_token_label_dict_list) == bsz
        search_calls_norm_list.append(search_calls_norm)
        inferences.append(inference)
        dataset_names.append(dataset_name)

        n_convs_this_sample = len(conversations)
        conv_dataset_names.extend([dataset_name] * n_convs_this_sample)
        conv_search_calls_norm.extend([search_calls_norm] * n_convs_this_sample)

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    # input_ids.shape = [num_conversations, max_len]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()
    qtype_labels = torch.full_like(input_ids, fill_value=IGNORE_INDEX)

    info_masks_list = []  # one [seq_len] mask per conversation

    if conv_type in ["llava_v1", "pixar_0shot_v1"]:
        sep = conv.sep + conv.roles[1] + ": "
    else:
        sep = "[/INST] "

    if (dataset_names[0] in [
        "textvqa",
        "gqa",
        "pope",
        "cragmm",
        "infoseek",
        "oven",
        "hotpotqa",
        "nq",
        "triviaqa",
        "musique",
        "popqa"
        ] and inferences[0] == True):  # handle VQA validation case
        targets = sampled_classes_list
    else:
        for conv_idx, (conversation, target) in enumerate(zip(conversation_list, targets)):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())
            info_mask_conv = torch.ones_like(target, dtype=torch.long)

            rounds = conversation.split(conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                assert len(parts) == 2, f"parts: {parts} | rounds: {rounds}"
                parts[0] += sep

                if DEFAULT_IMAGE_TOKEN in conversation:
                    round_len = len(tokenizer_image_token(rou, tokenizer))
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX  # ignore the user/inst segment

                is_pixar_conv = (conv_dataset_names[conv_idx] == "pixar")
                if is_pixar_conv:  # NOTE: also test whether such token-masking indeed helps model stability
                    assistant_text = parts[1]
                    enc = tokenizer(assistant_text, add_special_tokens=False, return_offsets_mapping=True)
                    asst_ids = enc.input_ids
                    offsets = enc.offset_mapping

                    # find char spans of <information> payloads
                    info_tok_include = True  # set to True to prevent the model from generating <information> tokens
                    info_spans = _info_char_spans(assistant_text, info_tok_include=info_tok_include) # <information> ... </information>

                    if len(info_spans) > 0 and len(asst_ids) > 0:
                        keep_mask = np.ones(len(asst_ids), dtype=bool)
                        for (s_char, e_char) in info_spans:
                            for tok_idx, (s_tok, e_tok) in enumerate(offsets):
                                if (s_tok < e_char) and (e_tok > s_char):
                                    tok_slice = assistant_text[s_tok:e_tok]
                                    if not info_tok_include and ("<" in tok_slice or ">" in tok_slice):
                                        continue  # not masking <information> tokens - NOTE: testing
                                    keep_mask[tok_idx] = False

                        asst_start = cur_len + instruction_len
                        max_take = min(len(keep_mask), target.size(0) - asst_start)
                        if max_take > 0:
                            ignore_positions = np.where(~keep_mask[:max_take])[0].tolist()
                            for pos in ignore_positions:
                                target[asst_start + pos] = IGNORE_INDEX
                                info_mask_conv[asst_start + pos] = 0

                        # <search> token qtype label assignment
                        search_tok_spans = _search_char_spans(assistant_text)
                        if len(search_calls_norm_list) == len(conversation_list) and len(search_tok_spans) > 0:
                            search_call = search_calls_norm_list[conv_idx]
                            if len(search_call) == 0:
                                continue
                            for sc_idx, sc in enumerate(search_call):
                                if sc["qtype"] == "REGION":
                                    sc_lbl = 2
                                elif sc["qtype"] == "IMAGE":
                                    sc_lbl = 1
                                else:  # sc["qtype"] == "TEXT"
                                    sc_lbl = 0
                                s_char, e_char = search_tok_spans[sc_idx]
                                if s_char > 0 and assistant_text[s_char - 1].isspace():
                                    s_char -= 1  # take care of `_<` token to match `<search>`
                                for tok_idx, (s_tok, e_tok) in enumerate(offsets):
                                    if s_tok >= s_char and e_tok <= e_char:
                                        qtype_labels[conv_idx, asst_start + tok_idx] = sc_lbl
                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                try:
                    assert cur_len == total_len, f"cur_len: {cur_len}, total_len: {total_len}"
                except Exception as e:
                    print(f"conversation: {conversation}")
                    print(f"target: {target}")
                    print(f"input_ids: {input_ids}")
                    print(f"attention_masks: {attention_masks}")
                    print(f"per_token_labels: {per_token_labels}")
                    print(f"info_masks: {info_masks}")
                    breakpoint()

            info_masks_list.append(info_mask_conv)

    if inferences[0] == False:
        im_start_id = tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)
        has_image_token = (input_ids == im_start_id).any().item()
        reserve = 255 if has_image_token else 0
        truncate_len = model_max_length - reserve  # hack for adding 255 image patch tokens
        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    # construct the BIO label
    per_token_labels_list = []  # list of lists (one per conversation)
    mask_positions_in_input_ids = []  # list of starting token indices for each conversation

    # find the text span within the tokenized input texts that corresponds to the part_text span (this is for embedding sampling for SAM later on)
    BIO_NO_LBL = 0
    BIO_START_LBL = 1
    BIO_INTERM_LBL = 2
    for batch_idx in range(len(offset_list) - 1):
        start = offset_list[batch_idx]
        end = offset_list[batch_idx + 1]
        # NOTE: len(part_to_mask_list) is the number of conversations
        # each dict (e.g., part_to_mask_list[i]) contains part_text as key and the corresponding mask as value
        # part_to_mask_list[i] can have multiple parts (for 'explanatory_seg'), or single part (for 'reason_seg', 'sem_seg', 'refer_seg') per conversation
        part_to_mask_list = per_token_label_dict_list[batch_idx]  # e.g., [{'engine': mask, 'wing': mask}]

        for conv_idx in range(start, end):
            part_mask_idx = conv_idx - start
            part_to_mask = part_to_mask_list[part_mask_idx]

            conversation = conversation_list[conv_idx]
            per_token_labels = [BIO_NO_LBL] * input_ids.size(1)
            mask_positions = set()

            if len(part_to_mask_list) != len(conversation_list[start:end]):
                print(f">> Warning: len(part_to_mask_list) != len(conversation_list[start:end])")
                print(f">> part_to_mask_list: {part_to_mask_list}")
                print(f">> conversation_list[start:end]: {conversation_list[start:end]}")
                breakpoint()

            ASSISTANT_START_TOK = 'ASSISTANT:'
            TOK_START_CHAR = '▁'
            if len(part_to_mask) != 0:
                conversation_input_ids = input_ids[conv_idx].clone()  # shape [seq_len]
                seq_len = conversation_input_ids.size(0)
                assistant_start_tok_ids = tokenizer(ASSISTANT_START_TOK)['input_ids'][1:]
                conversation_input_ids = torch.where(
                    (conversation_input_ids == IMAGE_TOKEN_INDEX) | (conversation_input_ids == IGNORE_INDEX),
                    torch.tensor(tokenizer.pad_token_id, device=conversation_input_ids.device),
                    conversation_input_ids,
                )
                input_ids_tokens = tokenizer.convert_ids_to_tokens(conversation_input_ids.tolist())
                # If part_text falls before "ASSISTANT:", then it's not a match
                for assistant_start_tok_idx in range(len(conversation_input_ids) - len(assistant_start_tok_ids) + 1):
                    sub_str_ids = conversation_input_ids[assistant_start_tok_idx : assistant_start_tok_idx + len(assistant_start_tok_ids)].tolist()
                    assistant_str_ids = assistant_start_tok_ids
                    if sub_str_ids == assistant_str_ids:
                        break

                for part_text in sorted(part_to_mask, key=len, reverse=True):
                    part_token_ids = tokenizer(part_text)['input_ids'][1:]  # skip the <s> token
                    part_token_ids_tokens = tokenizer.convert_ids_to_tokens(part_token_ids)
                    substr_found = False
                    for j in range(len(input_ids_tokens) - len(part_token_ids_tokens) + 1):
                        sub_str = input_ids_tokens[j : j + len(part_token_ids_tokens)]
                        part_tok = part_token_ids_tokens
                        if sub_str == part_tok:
                            if j <= assistant_start_tok_idx:  # do not include any match that falls before <image> token
                                continue
                            elif TOK_START_CHAR not in input_ids_tokens[j + len(part_token_ids_tokens)] and input_ids_tokens[j + len(part_token_ids_tokens)][0].isalnum():  # check if part_text is a whole word and not a subtoken
                                continue
                            elif per_token_labels[j] != BIO_NO_LBL:  # skip if the part_text is already matched to a previous span
                                continue
                            per_token_labels[j] = BIO_START_LBL
                            for k in range(1, len(part_tok)):
                                per_token_labels[j + k] = BIO_INTERM_LBL
                            mask_positions.add(j)
                            substr_found = True
                            break  # Break after finding the first valid match for this part_text

                    if not substr_found:
                        pass

            per_token_labels_list.append(torch.tensor(per_token_labels, dtype=torch.long))
            mask_positions_in_input_ids.append(sorted(list(mask_positions)))

    # just pad to the same dim as input_ids
    max_len = input_ids.size(1)
    padded_per_token_labels = []
    for labels in per_token_labels_list:
        if labels.size(0) < max_len:
            pad_size = max_len - labels.size(0)
            labels = torch.cat([labels, torch.zeros(pad_size, dtype=torch.long)], dim=0)
        else:
            labels = labels[:max_len]
        padded_per_token_labels.append(labels.unsqueeze(0))
    per_token_labels_tensor = torch.cat(padded_per_token_labels, dim=0)  # shape: [num_convs (across the batches), seq_len]

    padded_info_masks = []
    if len(info_masks_list) > 0:
        for mask in info_masks_list:
            if mask.size(0) < max_len:
                pad = torch.ones(max_len - mask.size(0), dtype=mask.dtype, device=mask.device)
                mask = torch.cat([mask, pad], dim=0)
            else:
                mask = mask[:max_len]
            padded_info_masks.append(mask.unsqueeze(0))
    else:
        padded_info_masks = [torch.zeros(max_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)] * input_ids.size(0)

    info_mask_tensor = torch.cat(padded_info_masks, dim=0)  # shape: [num_convs (across the batches), seq_len]

    # clamp any sequence > model_max_len
    input_ids, targets, attention_masks, per_token_labels_tensor, info_mask_tensor, mask_positions_in_input_ids = \
        clamp_conversation_tensors_to_maxlen(
            input_ids=input_ids,
            labels=targets if isinstance(targets, torch.Tensor) else None,
            attention_masks=attention_masks,
            per_token_labels=per_token_labels_tensor,
            info_mask_tensors=info_mask_tensor,
            mask_positions_in_input_ids=mask_positions_in_input_ids,
            max_len=model_max_length,
        )

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
        "per_token_labels": per_token_labels_tensor,  # BIO ids per token (same dim as input_ids)
        "info_masks": info_mask_tensor,  # mask for <information> </information> payload tokens
        "mask_positions_in_input_ids": mask_positions_in_input_ids,  # list of lists of starting token indices per sample
        "qtype_labels": qtype_labels,  # qtype labels per <search> token (same dim as input_ids)
    }


class HybridDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        dataset="sem_seg||refer_seg||vqa||reason_seg",
        sample_rate=[9, 5, 5, 1],
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        vqa_data="llava_v1_5_mix665k",
        reason_seg_data="ReasonSeg|train",
        pixar_data="oven||infoseek||aokvqa||okvqa||textvqa||webqa",
        explanatory=0.1,
    ):
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.datasets = dataset.split("||")

        assert len(sample_rate) == len(self.datasets)

        self.all_datasets = []
        for dataset in self.datasets:
            if dataset == "pixar":
                self.all_datasets.append(
                    PixarDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        precision,
                        image_size,
                        pixar_data,
                    )
                )
            elif dataset == "sem_seg":
                self.all_datasets.append(
                    SemSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        sem_seg_data,
                    )
                )
            elif dataset == "refer_seg":
                self.all_datasets.append(
                    ReferSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        refer_seg_data,
                    )
                )
            elif dataset == "vqa":
                self.all_datasets.append(
                    VQADataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        vqa_data,
                    )
                )
            elif dataset == "reason_seg":
                self.all_datasets.append(
                    ReasonSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        reason_seg_data,
                        explanatory,
                    )
                )

    def __len__(self):
        if len(self.all_datasets) == 1:
            return len(self.all_datasets[0])
        return self.samples_per_epoch

    def __getitem__(self, idx):
        if len(self.all_datasets) == 1:
            ind = 0
            data = self.all_datasets[0]
            local_idx = idx
        else:
            ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
            data = self.all_datasets[ind]
            local_idx = np.random.randint(len(data))

        inference = False
        return *data[local_idx], inference, self.datasets[ind]


class ValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset,
        image_size=1024,
    ):
        self.base_image_dir = base_image_dir
        splits = val_dataset.split("|")
        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = PLUM_ANSWER_LIST  # corresponds to ANSWER_LIST in LISA

        if len(splits) == 2:
            ds, split = splits
            if ds == "reason_seg":
                images = glob.glob(os.path.join(self.base_image_dir, ds, "ReasonSeg", split, "*.jpg"))
                self.images = images
                self.data_type = "reason_seg"
                # since reason_seg lacks proper output response in text, we load `reformulated_responses.json`
                self.reformulated_text_path = os.path.join(base_image_dir, ds, "ReasonSeg", split, "val_reformulated_responses.json")
                with open(self.reformulated_text_path, "r") as f:
                    self.reformulated_text_dict = json.load(f)

            elif ds == "ade20k":
                with open("utils/ade20k_classes.json", "r") as f:
                    ade20k_classes = json.load(f)
                ade20k_classes = np.array(ade20k_classes)
                image_ids = sorted(os.listdir(os.path.join(base_image_dir, "ade20k/images", split)))
                ade20k_images = [os.path.join(base_image_dir, "ade20k/images/validation", image_path) for image_path in image_ids if image_path.endswith(".jpg")]
                ade20k_labels = [x.replace(".jpg", ".png").replace("images", "annotations") for x in ade20k_images]
                print(f"Number of ade20k images: {len(ade20k_images)}")
                self.images = ade20k_images
                self.dataset_type = "ade20k"
                self.classes = ade20k_classes
                self.labels = ade20k_labels
                self.data_type = "sem_seg"

            elif ds == "cocostuff":
                cocostuff_classes = []
                with open("utils/cocostuff_classes.txt") as f:
                    for line in f.readlines()[1:]:
                        cocostuff_classes.append(line.strip().split(": ")[-1])
                cocostuff_classes = np.array(cocostuff_classes)
                cocostuff_labels = glob.glob(os.path.join(base_image_dir, "cocostuff", "val2017", "*.png"))
                cocostuff_images = [x.replace(".png", ".jpg").replace("cocostuff", "coco") for x in cocostuff_labels]
                print(f"Number of cocostuff images: {len(cocostuff_images)}")
                self.images = cocostuff_images
                self.dataset_type = "cocostuff"
                self.classes = cocostuff_classes
                self.labels = cocostuff_labels
                self.data_type = "sem_seg"
                self.cocostuff_class2index = {
                    c: i for i, c in enumerate(cocostuff_classes)
                }

            elif ds == "mapillary":
                mapillary_data_root = os.path.join(base_image_dir, "mapillary")
                with open(os.path.join(mapillary_data_root, "config_v2.0.json")) as f:
                    mapillary_classes = json.load(f)["labels"]
                mapillary_classes = [x["readable"].lower() for x in mapillary_classes]
                mapillary_classes = np.array(mapillary_classes)
                mapillary_labels = sorted(glob.glob(os.path.join(mapillary_data_root, "validation", "v2.0", "labels", "*.png")))
                mapillary_images = [x.replace(".png", ".jpg").replace("v2.0/labels", "images") for x in mapillary_labels]
                print(f"Number of mapillary images: {len(mapillary_images)}")
                self.images = mapillary_images
                self.dataset_type = "mapillary"
                self.classes = mapillary_classes
                self.labels = mapillary_labels
                self.data_type = "sem_seg"

            elif ds == "paco_lvis":
                raise NotImplementedError("paco_lvis is not implemented yet")

        elif len(splits) == 3:  # refer_seg
            ds, splitBy, split = splits
            DATA_DIR = os.path.join(base_image_dir, "refer_seg")
            refer_api = REFER(DATA_DIR, ds, splitBy)
            ref_ids_val = refer_api.getRefIds(split=split)
            images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
            refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        DATA_DIR, "images/saiapr_tc-12", item["file_name"]
                    )
                elif ds in ["refcoco", "refcoco+", "refcocog", "grefcoco"]:
                    item["file_name"] = os.path.join(
                        DATA_DIR,
                        "images/mscoco/images/train2014",
                        item["file_name"],
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_val
            self.reformulated_text_dict = None

            img2refs = {}
            for ref in refs_val:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_ds = refer_seg_ds
            self.data_type = "refer_seg"

        self.ds = ds
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)


    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        else:
            return len(self.images)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        if self.data_type == "refer_seg":
            refer_seg_ds = self.refer_seg_ds
            images = refer_seg_ds["images"]
            annotations = refer_seg_ds["annotations"]
            img2refs = refer_seg_ds["img2refs"]

            image_info = images[idx]
            image_path = image_info["file_name"]
            image_id = image_info["id"]

            refs = img2refs[image_id]
            if len(refs) == 0:
                raise ValueError("image {} has no refs".format(image_id))

            sents = []
            ann_ids = []
            for ref in refs:
                for sent in ref["sentences"]:
                    sents.append(sent["sent"].strip().lower())
                    ann_ids.append(ref["ann_id"])

            sampled_sents = sents
            sampled_ann_ids = ann_ids
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            is_sentence = False

        elif self.data_type == "sem_seg":
            image_path = self.images[idx]
            label_path = self.labels[idx]
            label = Image.open(label_path)
            label = np.array(label)
            if self.ds in ["ade20k", "mapillary"]:
                label[label == 0] = 255
                label -= 1
                label[label == 254] = 255
            elif self.ds == "cocostuff":
                for c, i in self.cocostuff_class2index.items():
                    if "-" in c:
                        label[label == i] = 255

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            image = self.transform.apply_image(image)  # preprocess image for sam
            resize = image.shape[:2]
            unique_label = np.unique(label).tolist()
            if 255 in unique_label:
                unique_label.remove(255)
            if len(unique_label) == 0:
                return self.__getitem__(0)

            sampled_classes = [self.classes[class_id] for class_id in unique_label]
            questions = []
            answers = []
            class_ids = []
            for sampled_cls in sampled_classes:
                text = sampled_cls
                assert len(text.split("||")) == 1
                question_template = random.choice(self.short_question_list)
                questions.append(question_template.format(class_name=text.lower()))
                answer_template = random.choice(self.answer_list)
                answers.append(answer_template.format(class_name=text.lower()))

                # For ["ade20k", "cocostuff", "mapillary"], we need the class_ids to be in the label mask
                class_id = self.classes.tolist().index(sampled_cls)
                class_ids.append(class_id)

            is_sentence = False
            sampled_sents = sampled_classes

        else:  # reason_seg
            image_path = self.images[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            json_path = image_path.replace(".jpg", ".json")
            json_id = os.path.basename(json_path).replace(".json", "")
            response_texts = self.reformulated_text_dict[json_id]
            mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
            sampled_sents = [sampled_sents[0]]


        ####
        # TODO: After implementing the self.data_type == "sem_seg", we need to revise the conversations logic
        ####

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        i = 0
        chosen_response_texts = []
        while i < len(sampled_sents):
            conv.messages = []
            text = sampled_sents[i].strip()
            print(">> (ValDataset) sampled_sents: ", text)
            if is_sentence:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n {} Please explain why.".format(text),
                )
                # conv.append_message(conv.roles[1], " ")
                chosen_response = random.choice(response_texts)
                conv.append_message(conv.roles[1], chosen_response)
                chosen_response_texts.append(chosen_response)
            else:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n What is {} in this image? Please highlight the region in the image.".format(
                        text
                    ),
                )
                if self.data_type == "reason_seg":
                    chosen_response = random.choice(response_texts)
                    conv.append_message(conv.roles[1], chosen_response)
                    chosen_response_texts.append(chosen_response)
                else:
                    conv.append_message(conv.roles[1], text)

            conversations.append(conv.get_prompt())
            i += 1

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        masks = []
        if self.data_type == "refer_seg":
            for i, ann_id in enumerate(sampled_ann_ids):
                ann = annotations[ann_id]
                if len(ann["segmentation"]) == 0 and sampled_sents[i] != "":
                    m = np.zeros((image_info["height"], image_info["width"], 1))
                else:
                    if type(ann["segmentation"][0]) == list:  # polygon
                        rle = mask.frPyObjects(
                            ann["segmentation"],
                            image_info["height"],
                            image_info["width"],
                        )
                    else:
                        rle = ann["segmentation"]
                        for i in range(len(rle)):
                            if not isinstance(rle[i]["counts"], bytes):
                                rle[i]["counts"] = rle[i]["counts"].encode()
                    m = mask.decode(rle)
                m = np.sum(
                    m, axis=2
                )  # sometimes there are multiple binary map (corresponding to multiple segs)
                m = m.astype(np.uint8)  # convert to np.uint8
                masks.append(m)

        elif self.data_type == "sem_seg":
            if self.ds in ["paco_lvis", "pascal_part"]:
                pass # TODO: Implement this after ade20k - need `sampled_anns` loaded for this
            else: # ade20k, cocostuff, mapillary
                label = torch.from_numpy(label).long()
                for class_id in class_ids:
                    masks.append(label == class_id)
        else:
            masks = [mask_json]

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        sampled_sents = chosen_response_texts if self.data_type == "reason_seg" else sampled_sents

        per_token_label_dict = [{k: v} for k, v in zip(sampled_sents, masks)]

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            labels,
            resize,
            None,
            sampled_sents,
            per_token_label_dict,
            None,
            inference,
            self.data_type
        )
