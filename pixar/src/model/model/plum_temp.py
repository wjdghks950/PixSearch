def model_forward(
    self,
    images: torch.FloatTensor,
    images_clip: torch.FloatTensor,
    input_ids: torch.LongTensor,
    labels: torch.LongTensor,
    attention_masks: torch.LongTensor,
    offset: torch.LongTensor,
    masks_list: List[torch.FloatTensor],  # List of ground truth segmentation masks per sample; each element: [num_pos, H, W]
    label_list: List[torch.Tensor],  # auxiliary info for postprocessing (e.g., original sizes)
    resize_list: List[tuple],  # list of target sizes for masks (one per sample, or per positive token)
    per_token_labels: torch.LongTensor,  # [B, seq_len] binary labels for segmentation-relevant tokens
    inference: bool = False,
    **kwargs,
):
    # Get image embeddings from the visual encoder (SAM’s image encoder)
    image_embeddings = self.get_visual_embs(images)  # shape: [B, C, H, W]
    batch_size = image_embeddings.shape[0]
    assert batch_size == len(offset) - 1

    if inference:
        # In inference mode, we use the parent's forward (which handles caching, etc.)
        # Note: In inference, we assume a single image_clip is provided.
        n_batch = 1
        length = input_ids.shape[0]
        assert images_clip.shape[0] == 1
        images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()
        
        output_hidden_states = []
        ce_losses = []
        for i in range(n_batch):
            start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
            outputs = super().forward(
                images=images_clip_extend[: end_i - start_i],
                attention_mask=attention_masks[start_i:end_i],
                input_ids=input_ids[start_i:end_i],
                labels=input_ids[start_i:end_i],
                output_hidden_states=True,
                output_per_instance_loss=False  # Adjust if needed for your multiple-choice case
            )
            output_hidden_states.append(outputs.hidden_states)
            ce_losses.append(outputs.loss)
            torch.cuda.empty_cache()

        # Concatenate collected hidden states
        output_hidden_states = [torch.cat(output_hidden_states, dim=0)]
        output = None  # In inference, segmentation branch might be handled separately.
        ce_loss = None  # Not used during inference

    else:
        # Training branch:
        images_clip_list = []
        for i in range(len(offset) - 1):
            start_i, end_i = offset[i], offset[i + 1]
            images_clip_i = (
                images_clip[i]
                .unsqueeze(0)
                .expand(end_i - start_i, -1, -1, -1)
                .contiguous()
            )
            images_clip_list.append(images_clip_i)
        images_clip = torch.cat(images_clip_list, dim=0)

        outputs = super().forward(
            images=images_clip,
            attention_mask=attention_masks,
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=True,
        )
        output_hidden_states = outputs.hidden_states  # List; we use the last layer
        logits = outputs.logits  # Language modeling logits
        ce_loss = outputs.loss  # LM loss (cross-entropy)
    
    # Use the final hidden states from the last transformer layer: shape [B, seq_len, hidden_size]
    final_hidden = output_hidden_states[-1]
    # Project the hidden states to the SAM query space.
    sam_proj_hidden_states = self.model.text_hidden_fcs[0](final_hidden)  # [B, seq_len, hidden_size]
    # Also obtain segmentation relevance scores.
    seg_scores = self.model.token_to_mask_fcs[0](final_hidden)  # [B, seq_len, 1]
    # Compute the token-wise classification loss (binary cross-entropy)
    seg_cls_loss = F.binary_cross_entropy(seg_scores.squeeze(-1), per_token_labels.float())

    # For tokens labeled as positive (segmentation-relevant), compute KL divergence loss and segmentation loss.
    positive_mask = (per_token_labels == 1)  # [B, seq_len] boolean

    total_kl_loss = 0.0
    total_seg_loss = 0.0
    pos_token_count = 0

    # Loop over each sample in the batch.
    for b in range(final_hidden.size(0)):
        pos_indices = positive_mask[b].nonzero(as_tuple=False).squeeze(-1)  # Indices of positive tokens for sample b
        if pos_indices.numel() == 0:
            continue

        # For each positive token, assume ground truth segmentation masks are stored in masks_list[b]
        # with the same order as the positive tokens.
        gt_masks = masks_list[b]  # shape: [num_pos, H_gt, W_gt]
        # Optionally, get the target resize info for this sample from resize_list[b] if resizing is needed.
        for idx, token_idx in enumerate(pos_indices):
            # Original hidden state (reference) and projected query for token token_idx in sample b.
            h_ref = final_hidden[b, token_idx, :]  # [hidden_size]
            q = sam_proj_hidden_states[b, token_idx, :]  # [hidden_size]
            kl_loss_token = kl_divergence_loss(h_ref, q, sigma=self.model.config.get("sigma", 1.0))
            total_kl_loss += kl_loss_token
            pos_token_count += 1

            # Predict segmentation mask using the SAM mask decoder.
            # Here we assume that the SAM decoder takes a query (shape [1, hidden_size])
            # and image features (for sample b: image_embeddings[b]: [C, H, W])
            pred_mask = self.model.visual_model.mask_decoder(q.unsqueeze(0), image_embeddings[b].unsqueeze(0))
            # Optionally, resize pred_mask to match the ground truth size if necessary.
            # For simplicity, we assume pred_mask and gt_masks[idx] are the same spatial size.
            seg_loss_token = dice_loss(pred_mask, gt_masks[idx].unsqueeze(0).float(), num_masks=1)
            total_seg_loss += seg_loss_token

    # Average the KL and segmentation losses over the number of positive tokens.
    if pos_token_count > 0:
        avg_kl_loss = total_kl_loss / pos_token_count
        avg_seg_loss = total_seg_loss / pos_token_count
    else:
        avg_kl_loss = torch.tensor(0.0, device=final_hidden.device)
        avg_seg_loss = torch.tensor(0.0, device=final_hidden.device)

    # Combine all loss components.
    if inference:
        # During inference, we only return generation outputs.
        final_loss = None
    else:
        final_loss = ce_loss \
            + self.seg_cls_loss_weight * seg_cls_loss \
            + self.kld_loss_weight * avg_kl_loss \
            + self.dice_loss_weight * avg_seg_loss

    # Prepare the output dictionary
    return {
        "loss": final_loss,
        "lm_loss": ce_loss,
        "seg_cls_loss": seg_cls_loss,
        "kl_loss": avg_kl_loss,
        "seg_loss": avg_seg_loss,
        "logits": outputs.logits if not inference else None,
        "hidden_states": output_hidden_states,
    }
