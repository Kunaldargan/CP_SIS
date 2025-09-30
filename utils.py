import numpy as np 
import cv2 
import torch 
import os 
import os.path as osp 
import re 
import tqdm
import pdb


def compute_sample_weights_multilabel(dataset, apply_gamma=False, gamma=0.9):

    """
    Given a PyTorch Dataset whose __getitem__ returns (x, y, z) with
    y a tensor of shape (n_classes,), compute a per-sample weight
    that up-weights under-represented classes.
    """
    
    num_samples = len(dataset)
    num_classes = 7  # Known from dataset structure
    
    # Initialize label matrix
    labels = np.zeros(num_samples, dtype=np.float32)

    for i in tqdm.tqdm(range(num_samples), desc="Extracting labels"):
        _, cls_id,_ = dataset[i]  # label should be torch.Tensor of shape (n_classes,)
        labels[i] = cls_id  # ensure 1D

    labels = labels.astype(np.int64)

    # 2) Count the total occurrences of each class
    # count how many samples of each class 1 through 7
    class_counts = np.bincount(labels.astype(int), minlength=num_classes)

    if apply_gamma:
        class_counts = class_counts**gamma  ## (Train counts ^ 0.9)
        
    # 3) Inverse frequency weighting
    class_weights = 1.0 / (class_counts + 1e-6)     # avoid div0
    

    # 4) Weight each sample by the sum of its class-weights
    sample_weights = class_weights[labels]
    
    print(set( sample_weights.tolist() ) )

    return sample_weights
    
    #all_labels.dot(class_weights)  # shape: (n_samples,)

#################### VERIFY #################################################################33
def compute_class_weights(dataset):
    """
    Computes a weight for each sample based on class frequencies.
    For each class, weight = 1 / (frequency of the class).
    For a sample, the weight is the weighted average of its active classes.
    """
    num_samples = len(dataset)
    num_classes = 7  # Known from dataset structure

    # Initialize label matrix
    labels = np.zeros((num_samples, num_classes), dtype=np.float32)

    # dataset.labels shape: (num_samples, num_classes)
    #labels = dataset.labels  # NumPy array
    
    # Collect labels from dataset
    for i in tqdm.tqdm(range(num_samples), desc="Extracting labels"):
        _, cls_id, _ = dataset[i]
        labels[i] = cls_id
        
        
    # Compute class frequencies (number of positive occurrences per class)
    class_counts = labels.sum(axis=0)  # shape: (num_classes,)
    
    # Avoid division by zero: if a class is not present, set count to 1
    class_counts[class_counts == 0] = 1
    
    # Inverse frequency: classes with fewer occurrences get higher weights
    class_weights = 1.0 / class_counts  # shape: (num_classes,)
    
    
    return class_weights



def compute_sample_weights(dataset):
    """
    Computes a weight for each sample based on class frequencies.
    For each class, weight = 1 / (frequency of the class).
    For a sample, the weight is the weighted average of its active classes.
    """
    num_samples = len(dataset)
    num_classes = 7  # Known from dataset structure

    # Initialize label matrix
    labels = np.zeros((num_samples, num_classes), dtype=np.float32)

    # dataset.labels shape: (num_samples, num_classes)
    #labels = dataset.labels  # NumPy array
    
    # Collect labels from dataset
    for i in tqdm.tqdm(range(num_samples), desc="Extracting labels"):
        _, cls_id, _ = dataset[i]
        labels[i] = cls_id
        
        
    # Compute class frequencies (number of positive occurrences per class)
    class_counts = labels.sum(axis=0)  # shape: (num_classes,)
    
    # Avoid division by zero: if a class is not present, set count to 1
    class_counts[class_counts == 0] = 1
    
    # Inverse frequency: classes with fewer occurrences get higher weights
    class_weights = 1.0 / class_counts  # shape: (num_classes,)
    
    sample_weights = []
    for sample in labels:
        # If no class is present, assign a default weight (e.g., 0.0)
        if sample.sum() == 0:
            weight = 0.0
        else:
            # Compute a weighted average for active classes
            weight = (sample * class_weights).sum() / sample.sum()
        sample_weights.append(weight)
    
    return sample_weights

def compute_pos_weights(dataset):
    num_classes = 7  # As defined in your __getitem__ logic
    pos_counts = np.zeros(num_classes)
    total_samples = len(dataset)

    for i in tqdm.tqdm(range(total_samples), desc="Computing pos_weights"):
        _, cls_id = dataset[i]  # cls_id is a binary array of shape (7,)
        pos_counts += cls_id  # count presence of each class

    neg_counts = total_samples - pos_counts
    pos_weights = neg_counts / (pos_counts + 1e-6)  # avoid division by zero

    return torch.tensor(pos_weights, dtype=torch.float32)

########################################################################################################
def save_endovis_masks(endovis_masks, output_folder="output"):
    """
    Saves the endovis masks in the specified output folder.

    Args:
        endovis_masks (dict): A dictionary containing the endovis masks.
        output_folder (str): The path to the output folder.
    """

    for mask_name, mask_array in endovis_masks.items():
        
        seq_name, frame_name = mask_name.split('/')
        frame_name_base = frame_name.split('.')[0]
        frame_name = frame_name.split('.')[0] + ".png" # ensures .png extension

        seq_output_folder = os.path.join(output_folder, seq_name)
        os.makedirs(seq_output_folder, exist_ok=True)

        output_path = os.path.join(seq_output_folder, frame_name_base+"_binary.png")
        cv2.imwrite(output_path, mask_array*255)
        
        output_path = os.path.join(seq_output_folder, frame_name)
        cv2.imwrite(output_path, mask_array*32)
        
         # Save each individual class mask
        for class_id in range(1, 8):  # Classes 1 to 7
            class_mask = (mask_array == class_id).astype(np.uint8) * 255
            class_mask_path = os.path.join(seq_output_folder, f"{frame_name_base}_class_{class_id}.png")
            cv2.imwrite(class_mask_path, class_mask)



def save_binary_masks(binary_masks, output_folder="binary_masks_output"):
    """
    Saves individual binary masks from the nested dictionary structure.

    Args:
        binary_masks (dict): Dictionary with structure binary_masks[seq][frame] = list of mask dicts.
        output_folder (str): Folder where the masks will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)

    for seq_name, frames in binary_masks.items():
        seq_folder = os.path.join(output_folder, seq_name)
        os.makedirs(seq_folder, exist_ok=True)

        for frame_name, masks_info in frames.items():
            for mask_info in masks_info:
            
                if "mask_label" in mask_info.keys():
            	    class_id = mask_info["mask_label"]
            	    score = mask_info["mask_label_score"]
                else:
                    class_id = int(re.search(r"class(\d+)",  mask_info["mask_name"]).group(1)) 
                    score = mask_info["mask_quality"]
               
                mask = mask_info["mask"].numpy().astype("uint8") * 255
                
                if score:
                    output_filename = f"{frame_name}_class_{class_id}_conf_{score}.png";
                else:
                    output_filename = f"{frame_name}_class_{class_id}.png"
                
                output_path = os.path.join(seq_folder, output_filename)

                cv2.imwrite(output_path, mask)

def create_binary_masks(binary_masks, preds, preds_quality, mask_names, thr):

    """Gather the predicted binary masks of different frames and classes into a dictionary, mask quality is also recorded

    Returns:
        dict: a dictionary containing all predicted binary masks organised based on sequence, frame, and mask name
    """
    preds = preds.cpu()
    preds_quality = preds_quality.cpu()
    
    pred_masks = (preds > thr).int()
    

    for pred_mask, mask_name, pred_quality in zip(pred_masks, mask_names, preds_quality):        
      
        seq_name = mask_name.split("/")[0]
        frame_name = osp.basename(mask_name).split("_")[0]
        
        if seq_name not in binary_masks.keys():
            binary_masks[seq_name] = dict()
        
        if frame_name not in binary_masks[seq_name].keys():
            binary_masks[seq_name][frame_name] = list()
            
        binary_masks[seq_name][frame_name].append({
            "mask_name": mask_name,
            "mask": pred_mask,
            "mask_quality": pred_quality.item()
        })
        
    return binary_masks

"""
binary_masks = {
    "seq_name": {
        "frame_name": [
            {
                "mask_name": str,
                "mask": torch.Tensor,
                "mask_quality": float
            },
            ...
        ]
    },
    ...
}

"""
    
def create_endovis_masks(binary_masks, H, W):
    """given the dictionary containing all predicted binary masks, compute final prediction of each frame and organise the prediction masks into a dictionary
       H - height of image 
       W - width of image
    
    Returns: a dictionary containing one prediction mask for each frame with the frame name as key and its predicted mask as value; 
             For each frame, the binary masks of different classes are conbined into a single prediction mask;
             The prediction mask for each frame is a 1024 x 1280 map with each value representing the class id for the pixel;
             
    """
    
    endovis_masks = dict()
    
    for seq in binary_masks.keys():
        
        for frame in binary_masks[seq].keys():
            
            endovis_mask = np.zeros((H, W))
    
            binary_masks_list = binary_masks[seq][frame]

            binary_masks_list = sorted(binary_masks_list, key=lambda x: x["mask_quality"])
           
            for binary_mask in binary_masks_list:
                mask_name  = binary_mask["mask_name"]
                predicted_label = int(re.search(r"class(\d+)", mask_name).group(1))
                mask = binary_mask["mask"].numpy()
                endovis_mask[mask==1] = predicted_label

            endovis_mask = endovis_mask.astype(int)

            endovis_masks[f"{seq}/{frame}.png"] = endovis_mask
    
    return endovis_masks


def eval_endovis(endovis_masks, gt_endovis_masks):
    """Given the predicted masks and groundtruth annotations, predict the challenge IoU, IoU, mean class IoU, and the IoU for each class
        
      ** The evaluation code is taken from the official evaluation code of paper: ISINet: An Instance-Based Approach for Surgical Instrument Segmentation
      ** at https://github.com/BCV-Uniandes/ISINet
      
    Args:
        endovis_masks (dict): the dictionary containing the predicted mask for each frame 
        gt_endovis_masks (dict): the dictionary containing the groundtruth mask for each frame 

    Returns:
        dict: a dictionary containing the evaluation results for different metrics 
    """

    endovis_results = dict()
    num_classes = 7
    
    all_im_iou_acc = []
    all_im_iou_acc_challenge = []
    cum_I, cum_U = 0, 0
    class_ious = {c: [] for c in range(1, num_classes+1)}
    class_ious_fn = {}
    for fn, pred in endovis_masks.items():
        class_ious_fn[fn] = {}
        for c in range(1, num_classes+1):
            class_ious_fn[fn][c]=0
    
    for file_name, prediction in endovis_masks.items():
       
        full_mask = gt_endovis_masks[file_name]
        
        im_iou = []
        im_iou_challenge = []
        target = full_mask.numpy()
        gt_classes = np.unique(target)
        gt_classes.sort()
        gt_classes = gt_classes[gt_classes > 0] 
        if np.sum(prediction) == 0:
            if target.sum() > 0: 
                all_im_iou_acc.append(0)
                all_im_iou_acc_challenge.append(0)
                for class_id in gt_classes:
                    class_ious[class_id].append(0)
                    class_ious_fn[file_name][class_id]=0
            continue

        gt_classes = torch.unique(full_mask)
        # loop through all classes from 1 to num_classes 
        for class_id in range(1, num_classes + 1): 

            current_pred = (prediction == class_id).astype(np.float64)
            current_target = (full_mask.numpy() == class_id).astype(np.float64)

            if current_pred.astype(np.float64).sum() != 0 or current_target.astype(np.float64).sum() != 0:
                i, u = compute_mask_IU_endovis(current_pred, current_target)     
                im_iou.append(i/u)
                cum_I += i
                cum_U += u
                class_ious[class_id].append(i/u)
                class_ious_fn[file_name][class_id]=i/u
                if class_id in gt_classes:
                    im_iou_challenge.append(i/u)
        
        ciou = [class_ious_fn[file_name][c] for c in class_ious_fn[file_name].keys()]
        #print(ciou)
        #print(f"{file_name} | {np.mean(ciou):.3f} | {class_ious_fn[file_name]}")
        if len(im_iou) > 0:
            all_im_iou_acc.append(np.mean(im_iou))

        if len(im_iou_challenge) > 0:
            all_im_iou_acc_challenge.append(np.mean(im_iou_challenge))

    # calculate final metrics
    final_im_iou = cum_I / (cum_U + 1e-15)
    mean_im_iou = np.mean(all_im_iou_acc)
    mean_im_iou_challenge = np.mean(all_im_iou_acc_challenge)

    final_class_im_iou = torch.zeros(9)
    cIoU_per_class = []
    for c in range(1, num_classes + 1):
        final_class_im_iou[c-1] = torch.tensor(class_ious[c]).float().mean()
        cIoU_per_class.append(round((final_class_im_iou[c-1]*100).item(), 3))
        
    mean_class_iou = torch.tensor([torch.tensor(values).float().mean() for c, values in class_ious.items() if len(values) > 0]).mean().item()
    
    endovis_results["challengIoU"] = round(mean_im_iou_challenge*100,3)
    endovis_results["IoU"] = round(mean_im_iou*100,3)
    endovis_results["mcIoU"] = round(mean_class_iou*100,3)
    endovis_results["mIoU"] = round(final_im_iou*100,3)
    
    endovis_results["cIoU_per_class"] = cIoU_per_class
    
    return endovis_results


def compute_mask_IU_endovis(masks, target):
    """compute iou used for evaluation
    """
    assert target.shape[-2:] == masks.shape[-2:]
    temp = masks * target
    intersection = temp.sum()
    union = ((masks + target) - temp).sum()
    return intersection, union


def read_gt_endovis_masks_b(data_root_dir = "../data/endovis_2018",
                          mode = "val", 
                          fold = None):
    
    """Read the annotation masks into a dictionary to be used as ground truth in evaluation.

    Returns:
        dict: mask names as key and annotation masks as value 
    """
    gt_endovis_masks = dict()
    
    if "2018" in data_root_dir:
        gt_endovis_masks_path = osp.join(data_root_dir, mode, "annotations_b")
        for seq in os.listdir(gt_endovis_masks_path):
            for mask_name in os.listdir(osp.join(gt_endovis_masks_path, seq)):
                full_mask_name = f"{seq}/{mask_name}"
                mask = torch.from_numpy(cv2.imread(osp.join(gt_endovis_masks_path, full_mask_name),cv2.IMREAD_GRAYSCALE)/32)
                gt_endovis_masks[full_mask_name] = mask
                
    elif "2017" in data_root_dir:
        if fold == "all":
            seqs = [1,2,3,4,5,6,7,8]
            
        elif fold in [0,1,2,3]:
            fold_seq = {0: [1, 3],
                        1: [2, 5],
                        2: [4, 8],
                        3: [6, 7]}
            
            seqs = fold_seq[fold]
        
        gt_endovis_masks_path = osp.join(data_root_dir, "0", "annotations")
        
        for seq in seqs:
            for mask_name in os.listdir(osp.join(gt_endovis_masks_path, f"seq{seq}")):
                full_mask_name = f"seq{seq}/{mask_name}"
                mask = torch.from_numpy(cv2.imread(osp.join(gt_endovis_masks_path, full_mask_name),cv2.IMREAD_GRAYSCALE))
                gt_endovis_masks[full_mask_name] = mask
            
            
    return gt_endovis_masks
    
def read_gt_endovis_masks(data_root_dir = "../data/endovis_2018",
                          mode = "val", 
                          fold = None):
    
    """Read the annotation masks into a dictionary to be used as ground truth in evaluation.

    Returns:
        dict: mask names as key and annotation masks as value 
    """
    gt_endovis_masks = dict()
    
    if "2018" in data_root_dir:
        gt_endovis_masks_path = osp.join(data_root_dir, mode, "annotations")
        for seq in os.listdir(gt_endovis_masks_path):
            for mask_name in os.listdir(osp.join(gt_endovis_masks_path, seq)):
                full_mask_name = f"{seq}/{mask_name}"
                mask = torch.from_numpy(cv2.imread(osp.join(gt_endovis_masks_path, full_mask_name),cv2.IMREAD_GRAYSCALE)/32)
                gt_endovis_masks[full_mask_name] = mask
                
    elif "2017" in data_root_dir:
        if fold == "all":
            seqs = [1,2,3,4,5,6,7,8]
            
        elif fold in [0,1,2,3]:
            fold_seq = {0: [1, 3],
                        1: [2, 5],
                        2: [4, 8],
                        3: [6, 7]}
            
            seqs = fold_seq[fold]
        
        gt_endovis_masks_path = osp.join(data_root_dir, "0", "annotations")
        
        for seq in seqs:
            for mask_name in os.listdir(osp.join(gt_endovis_masks_path, f"seq{seq}")):
                full_mask_name = f"seq{seq}/{mask_name}"
                mask = torch.from_numpy(cv2.imread(osp.join(gt_endovis_masks_path, full_mask_name),cv2.IMREAD_GRAYSCALE))
                gt_endovis_masks[full_mask_name] = mask
            
            
    return gt_endovis_masks


def print_log(str_to_print, log_file):
    """Print a string and meanwhile write it to a log file
    """
    print(str_to_print)
    with open(log_file, "a") as file:
        file.write(str_to_print+"\n")
