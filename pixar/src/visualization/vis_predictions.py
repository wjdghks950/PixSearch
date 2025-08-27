import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # For custom legend creation
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
import torch
from evaluation.prediction import RLEDict
import pycocotools.mask as mask_utils
from data.utils import open_image
from evaluation import Prediction
import orjson
import numpy as np

def visualize_predictions(predictions: list[dict], output_dir: str, shuffle: bool = True, limit_outputs: int = None):
    """
    Visualize predictions with predicted masks on top row and ground truth masks on bottom row.

    Args:
        predictions: List of dicts containing prediction data
        output_dir: Directory to save visualization images
    """
    os.makedirs(output_dir, exist_ok=True)

    if shuffle:
        rng = np.random.default_rng(42)
        indices = rng.permutation(len(predictions))
        predictions = [predictions[i] for i in indices]
    else:
        indices = np.arange(len(predictions))

    if limit_outputs is not None:
        predictions = predictions[:limit_outputs]

    total = len(predictions)
    for i, prediction in tqdm(zip(indices, predictions), desc="Creating visualizations", total=total):
        # Extract data from dict
        image_path = prediction['image_path']
        gt_masks = prediction['gt_masks']
        predicted_masks = prediction['predicted_masks']
        part_names = prediction['gt_parts']

        # Load the original image
        image = open_image(image_path)
        image_t = pil_to_tensor(image)

        # Create figure with 2 rows and len(all_parts) columns
        n_cols = len(gt_masks)
        fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols, 8))

        # Handle case where there's only one column
        if n_cols == 1:
            axes = axes.reshape(2, 1)

        # Create dictionaries for quick lookup
        gt_mask_dict = {mask['label']: mask for mask in gt_masks}
        pred_mask_dict = {mask['label']: mask for mask in predicted_masks}

        for col, part_name in enumerate(part_names):
            # Top row: Predicted masks
            ax_pred = axes[0, col]
            if part_name in pred_mask_dict:
                # Decode and display predicted mask
                pred_rle = pred_mask_dict[part_name]
                pred_mask_decoded = torch.from_numpy(mask_utils.decode(pred_rle)).bool()
                overlaid_pred = draw_segmentation_masks(image_t, pred_mask_decoded.unsqueeze(0),
                                                       alpha=0.7, colors=['aqua'])
                ax_pred.imshow(to_pil_image(overlaid_pred))
            else:
                # No predicted mask for this part, show original image
                ax_pred.imshow(image)
                ax_pred.text(0.5, 0.5, 'No Prediction', transform=ax_pred.transAxes,
                           ha='center', va='center', fontsize=12,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax_pred.set_title(f'Part: {part_name}', fontsize=10, fontweight='bold')
            ax_pred.axis('off')

            # Bottom row: Ground truth masks
            ax_gt = axes[1, col]
            if part_name in gt_mask_dict:
                # Decode and display ground truth mask
                gt_rle = gt_mask_dict[part_name]
                gt_mask_decoded = torch.from_numpy(mask_utils.decode(gt_rle)).bool()
                overlaid_gt = draw_segmentation_masks(image_t, gt_mask_decoded.unsqueeze(0),
                                                     alpha=0.7, colors=['aqua'])
                ax_gt.imshow(to_pil_image(overlaid_gt))
            else:
                # No ground truth mask for this part, show original image
                ax_gt.imshow(image)
                ax_gt.text(0.5, 0.5, 'No Ground Truth', transform=ax_gt.transAxes,
                          ha='center', va='center', fontsize=12,
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax_gt.axis('off')

        # Add overall title
        image_filename = os.path.basename(image_path)
        fig.suptitle(f'Prediction {i+1}: {image_filename}', fontsize=14, fontweight='bold')

        # Create legend
        # predicted_patch = mpatches.Patch(color='aqua', label='Predicted Mask')
        # ground_truth_patch = mpatches.Patch(color='aqua', label='Ground Truth Mask')
        # fig.legend(handles=[predicted_patch, ground_truth_patch],
        #           loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=12)

        # Adjust layout and save
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.1)

        output_path = os.path.join(output_dir, f'prediction_{i+1:04d}.jpg')
        fig.savefig(output_path, bbox_inches='tight', dpi=200)
        plt.close(fig)

if __name__ == '__main__':
    # Example usage with default paths - modify as needed
    predictions_path = '/shared/nas2/blume5/sp25/partonomy/results/lisa/2025_07_28-20_33_19/partimagenet/identification_with_label.json'
    vis_output_dir = '/shared/nas2/blume5/sp25/partonomy/results/vis_outputs_comparison'

    shuffle: bool = True
    limit_outputs: int = 100

    # Load predictions from JSON file
    with open(predictions_path, 'rb') as f:
        data = orjson.loads(f.read())

    predictions = data['predictions']

    # Create visualizations
    visualize_predictions(predictions, vis_output_dir, shuffle, limit_outputs)
    print(f"Visualizations saved to: {vis_output_dir}")