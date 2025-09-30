import sys
sys.path.append("..")
import os
import cv2
import re
import json
import os.path as osp 
import random 
import argparse
import numpy as np 
import torch 
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset_dinov2 import Endovis18Dataset_COCO_IMG_MASK, Endovis18Dataset_COCO, Endovis18Dataset_COCO_FULL, Endovis18Dataset_CLS, Endovis17Dataset, annToMask, nms_pytorch_mask_slow

from pytorch_metric_learning import losses
from surgical_dinov2_query_guidance import SurgicalDINOCLassifier
import tqdm

from sklearn.metrics import accuracy_score, classification_report

import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from utils import create_endovis_masks, eval_endovis, read_gt_endovis_masks, read_gt_endovis_masks_b, save_endovis_masks

from collections import defaultdict

def print_trainable_by_module(model):
    module_params = defaultdict(int)
    for name, param in model.named_parameters():
        if param.requires_grad:
            key = name.split('.')[0]
            module_params[key] += param.numel()
    
    total = 0
    for k, v in module_params.items():
        print(f"{k:30s} : {v:,}")
        total += v
    print(f"\nTotal trainable parameters: {total:,}")
       
def count_trainable_params(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")
    return total
    
    
def compute_sample_weights_multilabel(dataset, num_classes=7):
    num_samples = len(dataset)

    labels = dataset.get_labels()

    labels = np.array(labels, dtype=np.int64)

    class_counts = np.bincount(labels, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[labels]

    print("Unique sample weights:", set(sample_weights.tolist()))
    return sample_weights
    
def print_log(str_to_print, log_file):
    """Print a string and meanwhile write it to a log file
    """
    print(str_to_print)
    with open(log_file, "a") as file:
        file.write(str_to_print+"\n")
        
print("======> Process Arguments")
parser = argparse.ArgumentParser()
parser.add_argument('--dataset'    , type=str, default="endovis_2018", choices=["endovis_2018", "endovis_2017"], help='specify dataset')
parser.add_argument('--fold'       , type=int, default=0, choices=[0,1,2,3], help='specify fold number for endovis_2017 dataset')
parser.add_argument('--checkpoint' , action="store_true",   help='load best checkpoint to continue training')

args = parser.parse_args()

print("======> Set Parameters for Training" )
dataset_name = args.dataset
fold = args.fold
thr = 0
seed = 666  
data_root_dir = f"../data/{dataset_name}"
batch_size = 8 #64 #256 #160 #64 #512 #8

print("batch_size:", batch_size)
vit_mode = "h"

# set seed for reproducibility 
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
512
segm_masks={}
def create_binary_masks_pred(predictions, thr=0):

    """Gather the predicted binary masks of different frames and classes into a dictionary, mask quality is also recorded

    Returns:
        dict: a dictionary containing all predicted binary masks organised based on sequence, frame, and mask name
    """  
    
    binary_masks = {}
    ## accumulate predictions framewise and classwise
    for pred in tqdm.tqdm(predictions) :
        
        key   = pred["image_name"]
        seq_name, frame_name = key.split("/")
        label = pred["category_id"]
        segm  = pred["segmentation"] / 255
        segm  = segm.astype(int)
        ##print("segm.shape[1]",segm.shape, np.min(segm), np.max(segm))

        #print("segm.shape[2]",segm.shape, np.min(segm), np.max(segm))
        score = pred["score"]
        
        if seq_name not in binary_masks.keys():
            binary_masks[seq_name] = dict()
        
        if frame_name not in binary_masks[seq_name].keys():
            binary_masks[seq_name][frame_name] = list()
        
        binary_masks[seq_name][frame_name].append({
            "mask_name": f"{frame_name}_class{label+1}.png",
            "mask": segm, ##annToMask(pred["segmentation"]),
            "mask_quality":score,
        })
        
    return binary_masks


def create_endovis_masks_pred(binary_masks, gt_file_names, H, W):
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
                mask = binary_mask["mask"]
                endovis_mask[mask==1] = predicted_label

            endovis_mask = endovis_mask.astype(int)
            
            key = f"{seq}/{frame}"
            if key in gt_file_names:
                endovis_masks[key] = endovis_mask
            
    return endovis_masks
    
###### Define data transformations

# Additional augmentations for cropped candidate image
train_transform = transforms.Compose([
		    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random crop and resize
		    transforms.RandomHorizontalFlip(p=0.5),  # Flip horizontally
		    transforms.RandomVerticalFlip(p=0.5),  # Flip horizontally
		    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color variations
		    transforms.RandomRotation(degrees=60),  # Rotate image
		    transforms.ToTensor(),
		    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
		])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize larger to maintain aspect ratio
    transforms.ToTensor(),
    ##transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


##sam_model.image_encoder.img_size: 1024
##transform = ResizeAndPad(1024)


ann_file = "./data/endovis_2018/instances_val_sub.json"

gt_anns = json.load(open(ann_file))
id_to_fname = {}
for img in gt_anns["images"]:
	id_to_fname[img["id"]] = img["file_name"]
	
print("======> Load Dataset-Specific Parameters" )
if "18" in dataset_name:
    num_tokens = 2
    val_dataset = Endovis18Dataset_COCO_IMG_MASK(data_root_dir = data_root_dir, 
                                   	mode="val",
                                   	bbox_only=True
                                   )   
    print("val_dataset length:", len(val_dataset))
                                   
    val_dataset_b = Endovis18Dataset_COCO_IMG_MASK(data_root_dir = data_root_dir, 
                                   	mode="val_b",
                                   	bbox_only=True
                                   )    
    print("val_dataset_b length:", len(val_dataset_b))


    val_dataset_pred = Endovis18Dataset_COCO_IMG_MASK(data_root_dir = data_root_dir, 
                                   	mode="pred",
                                   	bbox_only=True
                                   )    

    print("val_dataset_pred length:", len(val_dataset_pred))

                                   
    gt_endovis_masks   = read_gt_endovis_masks(data_root_dir = data_root_dir, mode = "val")
    gt_endovis_masks_b = read_gt_endovis_masks_b(data_root_dir = data_root_dir, mode = "val")
    
    num_epochs = 50 #35
    lr = 0.001
    save_dir = "./work_dirs/dinov2_LQ_LORA_ev18_default_full_adamw_Base/"
    #"./work_dirs/dinov2_LQ_LORA_Mask_Prompting_ev18_default_adamw_Large_All_Blocks_mask_query_guidance/"
    #"./work_dirs/dinov2_LQ_LORA_ev18_default_adamw_Giant_All_Blocks/"
    #"./work_dirs/dinov2_LQ_LORA_Mask_Prompting_ev18_default_adamw_Giant_All_Blocks_mask_query_guidance/"

val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True )
val_dataloader_b = DataLoader(val_dataset_b, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True )
val_dataloader_pred = DataLoader(val_dataset_pred, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


extractor = SurgicalDINOCLassifier(
             backbone_size="base",#"large", ##"giant", ##"base", ##base
             r=4, 
             image_shape=(224, 224), 
             decode_type='linear4', 
             lora_layer=None, 
             num_classes=7, 
             use_avgpool=True,
             num_extra_queries=7,
             ##lora_layer=[0,1,2,3,4,5],
             ).cuda() 


print(f"Total Trainable Params: {count_trainable_params(extractor)}")
print(f"Module wise Trainable Params: {print_trainable_by_module(extractor)}")
##exit()

print("backbone_size;giant")
print(extractor)      
extractor = nn.DataParallel( extractor )  

print("======> Define Optmiser and Loss")
cls_criterion = nn.CrossEntropyLoss().cuda()

optimiser = torch.optim.AdamW([
            {'params': extractor.parameters(), 'lr': 5e-5}
        ], lr = lr, weight_decay = 0.0001)

def compute_metrics_report(predictions, labels):
	report = classification_report(labels, predictions, output_dict=True, zero_division=0)
	return {"accuracy": accuracy_score(labels, predictions), **report}
        
print("======> Set Saving Directories and Logs")
os.makedirs(save_dir, exist_ok = True) 
log_file = osp.join(save_dir, "log.txt")
print_log(str(args), log_file)

print("log_file", log_file)

print("======> Start Training and Validation" )
save_interval = 1
start_epoch   = 0 
loss_history = {"class_loss":[] ,"metric_results": []}  # Store loss per epoch

if args.checkpoint:
    checkpoint_path  = log_file = osp.join(save_dir, 'checkpoint_epoch_19.pth') ##"checkpoint_epoch_336.pth") #

    if osp.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        
        print("checkpoint.keys()", checkpoint.keys())

        extractor.load_state_dict(checkpoint['extractor_state_dict'])
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        print_log(f"Loaded weights from epoch {start_epoch}", log_file)


epoch = start_epoch

# validation 
extractor.eval() ## test mode
##extractor.debug = True
all_preds  = []
all_labels = []
epoch_val_cls_loss = 0


with torch.no_grad():

	for cnd_imgs, cand_masks, cand_mask_1024, cls_ids, mask_names in tqdm.tqdm(val_dataloader): 
	    
	    cls_ids = cls_ids.cuda() #torch.full((sam_feats.shape[0],), -1).cuda() #dummy placeholder
	    cand_masks= cand_masks.cuda()

	    #candidates
	    inst_imgs = cnd_imgs.cuda()

	    
	    # Select non-zero img_crops
	    logits , features = extractor(inst_imgs, cand_masks)
							    
	    # Compute softmax over logits to get class probabilities
	    class_probs = torch.softmax(logits, dim=1)  # Shape: (batch_size, num_classes)

	    # Get confidence score of the predicted class
	    predicted_classes = torch.argmax(class_probs, dim=1)  # Shape: (batch_size,)
	    class_confidence = class_probs[torch.arange(class_probs.shape[0]), predicted_classes]  # Shape: (batch_size,) 
	    
	    cls_ids = cls_ids-1
	    cls_loss = cls_criterion(logits, cls_ids)
	    
	    epoch_val_cls_loss += cls_loss.item()
	    
	    all_preds.extend(predicted_classes.cpu().numpy())
	    all_labels.extend(cls_ids.cpu().numpy())
	    
	metric_results = compute_metrics_report(all_preds, all_labels)
	epoch_val_cls_loss = epoch_val_cls_loss/ len(val_dataloader)

	print_log(f"Epoch {epoch}: VAL Class Loss = {epoch_val_cls_loss:.4f}", log_file)
	print_log(f"\n Epoch {epoch}: VAL Classification report={metric_results}", log_file)

	   
	# validation 
	extractor.eval() ## test mode

	all_preds  = []
	all_labels = []
	epoch_val_cls_loss = 0


with torch.no_grad():

	for cnd_imgs, cand_masks, cand_mask_1024, cls_ids, mask_names in tqdm.tqdm(val_dataloader_b): 
	    
	    cls_ids   = cls_ids.cuda() #torch.full((sam_feats.shape[0],), -1).cuda() #dummy placeholder
	    cand_masks= cand_masks.cuda()
	    
	    #candidates
	    inst_imgs = cnd_imgs.cuda()
	    
	    # Select non-zero img_crops
	    logits , features = extractor(inst_imgs, cand_masks)
							    
	    # Compute softmax over logits to get class probabilities
	    class_probs = torch.softmax(logits, dim=1)  # Shape: (batch_size, num_classes)

	    # Get confidence score of the predicted class
	    predicted_classes = torch.argmax(class_probs, dim=1)  # Shape: (batch_size,)
	    class_confidence = class_probs[torch.arange(class_probs.shape[0]), predicted_classes]  # Shape: (batch_size,) 
	    
	    cls_ids = cls_ids-1
	    cls_loss = cls_criterion(logits, cls_ids)
	    
	    epoch_val_cls_loss += cls_loss.item()
	    
	    all_preds.extend(predicted_classes.cpu().numpy())
	    all_labels.extend(cls_ids.cpu().numpy())
	    
	metric_results = compute_metrics_report(all_preds, all_labels)
	epoch_val_cls_loss = epoch_val_cls_loss/ len(val_dataloader_b)

	print_log(f"Epoch {epoch}: VAL (B) Class Loss = {epoch_val_cls_loss:.4f}", log_file)
	print_log(f"\n Epoch {epoch}: VAL (B) Classification report={metric_results}", log_file)


##"""
# validation pred
preds_per_fname = {}
predictions     = []

with torch.no_grad():

	for cnd_imgs, cand_masks, cand_mask_1024, cls_ids , mask_names in tqdm.tqdm(val_dataloader_pred): ##)val_dataloader_b: 
	    
	    #candidates
	    inst_imgs = cnd_imgs.cuda()
	    cand_masks= cand_masks.cuda()
	    
	    #print(inst_imgs.shape)
	    
	    # Select non-zero img_crops
	    logits , features = extractor(inst_imgs, cand_masks)
							    
	    # Compute softmax over logits to get class probabilities
	    class_probs = torch.softmax(logits, dim=1)  # Shape: (batch_size, num_classes)

	    # Get confidence score of the predicted class
	    predicted_classes = torch.argmax(class_probs, dim=1)  # Shape: (batch_size,) ##cls_ids.cuda() - 1
	    class_confidence  = class_probs[torch.arange(class_probs.shape[0]), predicted_classes]  # Shape: (batch_size,) 

	    # Save images to directories
	    for i in range(len(cnd_imgs)):
                seq, fname = mask_names[i].split("/")[-2],mask_names[i].split("/")[-1] 
                fname = fname.split("_")[0]+".png"
            
                key = osp.join(seq, fname)
                if key not in preds_per_fname.keys():
                    preds_per_fname[key]  = []

                preds_per_fname[key].append([predicted_classes[i], class_confidence[i], cand_mask_1024[i].numpy(), mask_names[i] ] )


	for key in preds_per_fname.keys():
	    for val in preds_per_fname[key] :

	        pred_cls, pred_score, segm, mask_name = val
	        ##print(segm.squeeze().shape)
	        
	        result = {
			"image_name":key,
			"category_id":pred_cls.cpu().item(),
			"score":torch.max(pred_score).cpu().item(),
			"segmentation":segm.squeeze(),
	    		"mask_name": mask_name,
		}
	        predictions.append(result)

	binary_masks = create_binary_masks_pred(predictions)
#####################################################################################################

endovis_masks   = create_endovis_masks_pred(binary_masks, gt_endovis_masks.keys(), 1024, 1280)
endovis_results = eval_endovis(endovis_masks, gt_endovis_masks)

save_endovis_masks(endovis_masks)

print("endovis_results : ", endovis_results )

endovis_masks   = create_endovis_masks_pred(binary_masks, gt_endovis_masks_b.keys(), 1024, 1280)
endovis_results_b = eval_endovis(endovis_masks, gt_endovis_masks_b)

print("endovis_results (b)", endovis_results_b )

