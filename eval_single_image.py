import sys
# Make sure the parent directory is in the path to find custom modules
sys.path.append("..") 
import os
import cv2
import argparse
import numpy as np
import torch
import torch.nn as nn
import os.path as osp 
import torchvision.transforms as transforms
from PIL import Image
import tqdm
import re

# Assuming this class is defined in the specified file
from surgical_dinov2_query_guidance import SurgicalDINOCLassifier

# Define a color map for visualization (7 classes + background)
# Class IDs are 1-7, so index 0 is background.
COLOR_MAP = [
    [0, 0, 0],          # 0: Background (Black)
    [255, 0, 0],        # 1: Bipolar Forceps (Red)
    [0, 255, 0],        # 2: Prograsp Forceps (Green)
    [0, 0, 255],        # 3: Large Needle Driver (Blue)
    [255, 255, 0],      # 4: Vessel Sealer (Yellow)
    [0, 255, 255],      # 5: Grasping Retractor (Cyan)
    [255, 0, 255],      # 6: Monopolar Curved Scissors (Magenta)
    [128, 128,255],      # 7: Ultrasound Probe (Orange)
]



def create_endovis_masks_pred(binary_masks, H, W):
    """
    Given the dictionary containing all predicted binary masks, compute the final 
    prediction for the frame.
    
    Args:
        binary_masks (list): A list of dictionaries, each containing a mask, its predicted class, and score.
        H (int): Height of the final mask.
        W (int): Width of the final mask.
        
    Returns:
        np.ndarray: The final combined segmentation mask.
    """
    endovis_mask = np.zeros((H, W), dtype=np.uint8)

    # Sort masks by their quality/score in ascending order, so higher-score masks are drawn last
    binary_masks_list = sorted(binary_masks, key=lambda x: x["mask_quality"])

    for binary_mask in binary_masks_list:
        predicted_label = binary_mask["predicted_label"]
        mask = binary_mask["mask"]
        # 'Paint' the class label onto the final mask where the instance mask is 1
        endovis_mask[mask == 1] = predicted_label
        
    return endovis_mask

def main(args):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the same transformation used during validation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # DINOv2 models were trained with ImageNet normalization
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Initialize the model with the same architecture as during training
    print("======> Initializing Model")
    model = SurgicalDINOCLassifier(
        backbone_size=args.model_size,
        r=4,
        image_shape=(224, 224),
        decode_type='linear4',
        lora_layer=None,
        num_classes=7,
        use_avgpool=True,
        num_extra_queries=7,
    ).to(device)

    # Load the checkpoint
    print(f"======> Loading Checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle DataParallel wrapper if the model was saved with it
    state_dict = checkpoint['extractor_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        # Create a new state_dict without the 'module.' prefix
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    model.eval() # Set model to evaluation mode


    # Load the input image
    print(f"======> Loading Input Image from {args.image_path}")
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Error: Could not read image at {args.image_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W, _ = image.shape
    
    # Find all candidate masks
    mask_files = [f for f in os.listdir(args.masks_dir) if f.endswith('.png')]
    if not mask_files:
        print(f"Error: No candidate masks (.png files) found in {args.masks_dir}")
        return
    print(f"Found {len(mask_files)} candidate masks.")

    predictions = []

    # Process each mask
    print("======> Processing Candidate Masks")
    with torch.no_grad():
        for mask_file in tqdm.tqdm(mask_files):
            mask_path = os.path.join(args.masks_dir, mask_file)
            
            # Load the original full-size mask
            full_size_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if full_size_mask is None:
                continue
            
            # Binarize the mask to be safe
            _, full_size_mask_bin = cv2.threshold(full_size_mask, 127, 255, cv2.THRESH_BINARY)
            full_size_mask_bin = (full_size_mask_bin / 255).astype(np.uint8)

            # Find bounding box of the mask to crop the image
            coords = cv2.findNonZero(full_size_mask_bin)
            if coords is None:
                continue # Skip empty masks
            x, y, w, h = cv2.boundingRect(coords)

            # Crop the image using the bounding box
            cropped_image = image[y:y+h, x:x+w]
            
            #cv2.imwrite(f"cropped_{mask_file}",cropped_image)
            
            # Apply transformations to the cropped image
            pil_image = Image.fromarray(cropped_image)
            transformed_image = val_transform(pil_image).unsqueeze(0).to(device)

            # Get model prediction
            logits, _ = model(transformed_image)
            
            # Compute class probabilities and get the top prediction
            class_probs = torch.softmax(logits, dim=1)
            predicted_class_idx = torch.argmax(class_probs, dim=1).item()
            class_confidence = class_probs[0, predicted_class_idx].item()
            
            # The model predicts class indices 0-6. The dataset uses labels 1-7.
            predicted_label = predicted_class_idx + 1

            # Store the result
            predictions.append({
                "mask": full_size_mask_bin,
                "predicted_label": predicted_label,
                "mask_quality": class_confidence,
            })

    # Combine masks into a single prediction map
    print("======> Assembling Final Segmentation Map")
    final_mask = create_endovis_masks_pred(predictions, H, W)
    
    # Save the final grayscale mask
    output_path_gray = os.path.join(args.output_dir, "prediction_mask_gray.png")
    cv2.imwrite(output_path_gray, final_mask)
    print(f"Saved grayscale prediction mask to: {output_path_gray}")

    # Save a colorized version for easy visualization
    color_mask = np.zeros((H, W, 3), dtype=np.uint8)
    for class_id in range(1, 8):
        color_mask[final_mask == class_id] = COLOR_MAP[class_id]
        
    output_path_color = os.path.join(args.output_dir, "prediction_mask_color.png")
    cv2.imwrite(output_path_color,color_mask)
    print(f"Saved colorized prediction mask to: {output_path_color}")
    
    print("======> Inference Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run single-image inference for Surgical DINOv2 Classifier.")
    parser.add_argument('--image_path', type=str,  default="./data/endovis_2018/val/images/seq2/00078.png",help='Path to the input image file.')
    parser.add_argument('--checkpoint', type=str,  default="./work_dirs/dinov2_LQ_LORA_ev18_default_full_adamw_Base/checkpoint_epoch_19.pth", 
    help='Path to the model checkpoint file (.pth).')
    parser.add_argument('--masks_dir', type=str,  default="./segms_example_image/", help='Path to the directory containing candidate binary masks.')
    parser.add_argument('--output_dir', type=str, default="./visualization", help='Directory to save the output masks.')
    parser.add_argument('--model_size', type=str, default="base", choices=["base", "large", "giant"], help='Size of the DINOv2 backbone.')
    
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
