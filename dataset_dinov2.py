import torch
from torch.utils.data import Dataset
import os 
import os.path as osp
import re 
import numpy as np 
import cv2 
import json
from PIL import Image

import random

from torch.utils.data._utils.collate import default_collate
from segment_anything.utils.transforms import ResizeLongestSide

import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F


from pycocotools import mask as maskUtils

def annToRLE(ann):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    h, w = 1024, 1280 #1080,1920 
    segm = ann
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann
    return rle

def annToMask(ann):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = annToRLE(ann)
    m = maskUtils.decode(rle)
    return m

def annToBbox(ann):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = annToRLE(ann)
    bbox = maskUtils.toBbox(rle)
    return bbox
    

def mask_iou(gt, dt):
    """
    Compute mask iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :return: iou (float)
    """
    intersection = ((gt * dt) > 0).sum()
    union = ((gt + dt) > 0).sum()
    iou = intersection / union
    return iou


def nms_pytorch_mask_slow(masks: np.array, 
                          scores: np.array, 
                          iou_threshold: float,
                          ) -> np.array:
    """Apply non-maximum suppression to avoid detecting too many
    overlapping masks for a given object.

    Args:
        masks: (tensor) The predicted masks, Shape: [num_masks, height, width].
        scores: (tensor) The confidence scores, Shape: [num_masks].
        iou_threshold: (float) The overlap threshold for suppressing unnecessary masks.
        min_score_threshold: (float) The minimum score required to keep a mask.

    Returns:
        A list of filtered masks after NMS.
    """

    if len(masks) == 0:
        return []
    
    # print(len(areas), areas)
    # sort the prediction boxes in bboxes
    # according to their confidence scores
    order = scores.argsort() #decreasing order

    # initialise an empty list for 
    # filtered prediction boxes
    keep = []

    # print("scores:", scores)
    # print("order:", order)
    # print("scorted scores:",scores[order])
    # print(len(masks))

    while len(order) > 0:
        # extract the index of the 
        # prediction with highest score
        # we call this prediction S
        idx = order[-1]

        # remove S from P
        order = order[:-1]

        # push S in filtered predictions list
        keep.append(idx) #P[idx],

        instance_1= masks[idx] #.cpu().numpy()

        # sanity check
        if len(order) == 0:
            break

        suppress=[]
        
        for pos in range(len(order)):
            j = order[pos]

            instance_2= masks[j] #.cpu().numpy()

            # plt.imshow(instance_1.cpu().numpy())
            # plt.show()
            # plt.title("inst{1}")
            
            # plt.imshow(instance_2.cpu().numpy())
            # plt.show()
            # plt.title("inst{2}")
            # Compute IoU (Intersection over Union)
            iou = mask_iou(instance_1, instance_2)

            #interclass nms removal
            if iou > iou_threshold:
                suppress.append(pos)

            # subset = intersection / instance_2.sum()
            
            # #subset removal
            # if subset > 0.8:
            #     suppress.append(pos)


        # Remove suppressed boxes from indices list
        order[suppress] = -1
        order = order[order != -1]              

    return np.stack(keep, axis=0)

def perturb_mask(mask):
	"""Apply random morphological operations to perturb the mask."""
	# Random choice of operation
	operation = random.choice(['erode', 'dilate', 'affine'])

	kernel_size = random.choice([3, 5])
	kernel = np.ones((kernel_size, kernel_size), np.uint8)

	if operation == 'erode':
		perturbed = cv2.erode(mask, kernel, iterations=1)
	elif operation == 'dilate':
		perturbed = cv2.dilate(mask, kernel, iterations=1)
	else:  # affine
		#cv2.imwrite("mask_temp.png", mask)
		rows, cols = mask.shape
		pts1 = np.float32([[0,0], [cols-1,0], [0,rows-1]])
		shift = lambda: random.randint(-10, 15)
		pts2 = np.float32([[shift(),shift()], [cols-1+shift(), shift()], [shift(), rows-1+shift()]])
		M = cv2.getAffineTransform(pts1, pts2)
		perturbed = cv2.warpAffine(mask, M, (cols, rows), borderValue=0)
		perturbed = cv2.dilate(perturbed, kernel, iterations=1)
		#cv2.imwrite("mask_temp_perturb.png", perturbed)

	return perturbed


    
def mask_to_bbox(mask):
	""" Convert a binary mask to a bounding box (x, y, w, h) """
	y_indices, x_indices = np.where(mask > 0)

	if len(y_indices) == 0 or len(x_indices) == 0:
		return None  # Return None if no object in mask

	x_min, x_max = np.min(x_indices), np.max(x_indices)
	y_min, y_max = np.min(y_indices), np.max(y_indices)

	return [x_min, y_min, x_max, y_max]


class ResizeAndPad:

	def __init__(self, target_size):
		self.target_size = target_size
		self.transform = ResizeLongestSide(target_size)
		self.to_tensor = transforms.ToTensor()

	def __call__(self, image, masks, bboxes):
		# Resize image and masks
		og_h, og_w, _ = image.shape
		image = self.transform.apply_image(image)
		masks = [torch.tensor(self.transform.apply_image(mask)) for mask in masks]
		image = self.to_tensor(image)
		#print("image.shape:", image.shape) ## [3,819,1024]

		# Pad image and masks to form a square
		_, h, w = image.shape
		max_dim = max(w, h)
		pad_w = (max_dim - w) // 2
		pad_h = (max_dim - h) // 2

		padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
		image = transforms.Pad(padding)(image)
		masks = [transforms.Pad(padding)(mask) for mask in masks]

		# Adjust bounding boxes
		bboxes = self.transform.apply_boxes(bboxes, (og_h, og_w))
		bboxes = [[bbox[0] + pad_w, bbox[1] + pad_h, bbox[2] + pad_w, bbox[3] + pad_h] for bbox in bboxes]

		return image, masks, bboxes

class Endovis18Dataset_Visual_Prompt(Dataset):
	
	def __init__(self, data_root_dir = "./data/endovis_2018",  
			mode = "train", 
			vit_mode = "h",
			version = 0,
			transform=None,
			sam_transform=None):

		""" Define the Endovis18 dataset

			Args:
				data_root_dir (str, optional): root dir containing all data for Endovis18. Defaults to "../data/endovis_2018".
				mode (str, optional): either in "train" or "val" mode. Defaults to "val".
				vit_mode (str, optional): "h", "l", "b" for huge, large, and base versions of SAM. Defaults to "h".
				version (int, optional): augmentation version to use. Defaults to 0.
				
			Desc: Creates a combined list of candidate instances [ Predicted Candidates and Ground Truth Candidates ]
		"""

		self.vit_mode = vit_mode
		self.mode = mode
		
		self.transform = sam_transform
		self.cnd_img_transform = transform

		
		
		# directory containing all binary annotations
		if self.mode == "train":
			self.mask_dir = osp.join(data_root_dir, mode, str(version), "annotations")
			self.img_dir  = osp.join(data_root_dir, mode, str(version), "images")
			self.cand_dir = osp.join(data_root_dir, mode, str(version), "instance_pred_candidates")
			self.binary_mask_dir = osp.join(data_root_dir, mode, str(version), "binary_annotations")
			cand_match_fn = open(osp.join(data_root_dir, mode, str(version), f"cand_match_{str(version)}.txt"))
			
			self.bbox_shift = 5
		
		else: ## val mode
			#mode = "val_surgical_sam"
			self.mask_dir = osp.join(data_root_dir, mode, "annotations")
			self.img_dir  = osp.join(data_root_dir, mode, "images")
			self.cand_dir = osp.join(data_root_dir, mode, "instance_pred_candidates") #"./data/endovis_2018/val_instances_results_maskdino_pred_candidates/"
			self.binary_mask_dir = osp.join(data_root_dir, mode, "binary_annotations")
			cand_match_fn = open(osp.join(data_root_dir, mode, f"cand_match.txt"))
			
			self.bbox_shift = 0
		
							
		self.cand_matches = {}
		for line in cand_match_fn.readlines():
			inst_file, cls_id = line.strip().split(",")
			self.cand_matches[inst_file] = int(cls_id)
			
		# put all binary masks into a list
		self.mask_list = []
		for seq in os.listdir(self.mask_dir):
			for file in os.listdir(osp.join(self.mask_dir, seq) ):
				self.mask_list.append(osp.join(seq, file))

		self.cands = {}

		for file in self.mask_list:
			seq, fn   = file.split("/")
			folder_fn = fn.split(".")[0]
			folder_fn = seq + "/" + folder_fn 

			if os.path.exists(os.path.join(self.cand_dir, folder_fn) ):
				if mode == "train":
					self.cands[file] = [osp.join(folder_fn, img) for img in os.listdir(osp.join(self.cand_dir, folder_fn))   ] 
					## Keep only forground masks
				else: ##val
					self.cands[file] = [osp.join(folder_fn, img) for img in os.listdir(osp.join(self.cand_dir, folder_fn))  ] ## keep all ## if self.cand_matches[osp.join(folder_fn, img)] > 0

			
		## setup
		self.mask_cands = []
		self.cand_prompt = 0
		self.cls_prompt  = 1
		
		for mask_name in self.mask_list:
			"""
			if mask_name in self.cands.keys():
				candidates = self.cands[mask_name]
				for cand in candidates:
					self.mask_cands.append( (self.cand_prompt, cand, mask_name) )
			"""
			
			#if self.mode == "train" :
			seq, mask_name_fn = mask_name.split("/")
			
			## class prompt mode
			for binary_mask_name in os.listdir(os.path.join(self.binary_mask_dir,seq) ):
				#print("binary_mask_name:", binary_mask_name)
				f_name, class_id = binary_mask_name.split("_")
				class_id = class_id.split("class")[0]

				if mask_name_fn.split(".")[0] == f_name:
					self.mask_cands.append( (self.cls_prompt,  seq+"/"+binary_mask_name, mask_name ) )		

		print("Number of mask candidates:", len(self.mask_cands))
		
	def __len__(self):
		return len(self.mask_cands)
		
	def perturb(self,bbox, H, W ):
		
		# Add random perturbation to bounding box coordinates
		min_x = max(0, bbox[0] - random.randint(0, self.bbox_shift))
		max_x = min(W, bbox[2] + random.randint(0, self.bbox_shift))
		min_y = max(0, bbox[1]  - random.randint(0, self.bbox_shift))
		max_y = min(H, bbox[3]  + random.randint(0, self.bbox_shift))
		
		bbox = [min_x, min_y, max_x, max_y]
		
		return bbox
	
		
	def process_candidate(self, mask, prompt_type, cand_name, mask_name):
		""" Process Candidates

			Args:
				mask: multi-class GT Mask containing 7 classes.
				prompt_type (int): self.cand_prompt = 0, self.cls_prompt  = 1
				cand_name (str): Candidate mask name
				mask_name (str): Annotations mask name
				
			Desc: Process candidates and return class_emb, candidate mask, class id of candidate, candidate bbox, class mask (1 channel binary)
		"""
		
		if self.mode == "val":
			"""
			# Load candidate mask (assuming it's a binary or RGB image)
			cnd_mask = cv2.imread(osp.join(self.cand_dir, cand_name), cv2.IMREAD_GRAYSCALE)
			class_id = self.cand_matches[cand_name]
			bbox = mask_to_bbox(cnd_mask)

			#class_id = -1 # placeholder
			class_mask = np.zeros(cnd_mask.shape)
			class_embedding = np.zeros(256)
			
			return class_embedding, cnd_mask, class_id, bbox, class_mask
			"""
			
			# get ground-truth mask
			mask_path = osp.join(self.binary_mask_dir, cand_name)
			cnd_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
			class_id = int(re.search(r"class(\d+)", cand_name).group(1))

			# get class embedding
			class_embedding_path = osp.join(self.mask_dir.replace("annotations", f"class_embeddings_{self.vit_mode}"), cand_name.replace("png","npy"))
			class_embedding = np.load(class_embedding_path)
			
			
	
		if prompt_type == self.cls_prompt:
			# get ground-truth mask
			mask_path = osp.join(self.binary_mask_dir, cand_name)
			cnd_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
			class_id = int(re.search(r"class(\d+)", cand_name).group(1))

			# get class embedding
			class_embedding_path = osp.join(self.mask_dir.replace("annotations", f"class_embeddings_{self.vit_mode}"), cand_name.replace("png","npy"))
			class_embedding = np.load(class_embedding_path)
		else:
			# Load candidate mask (assuming it's a binary or RGB image)
			cnd_mask = cv2.imread(osp.join(self.cand_dir, cand_name), cv2.IMREAD_GRAYSCALE)
			class_id = self.cand_matches[cand_name]


			# get class embedding
			class_embedding_path = osp.join(self.mask_dir.replace("annotations", f"class_embeddings_{self.vit_mode}"), mask_name.split(".")[0]+"_class"+str(class_id)+".npy")
			class_embedding = np.load(class_embedding_path)
			
		bbox = mask_to_bbox(cnd_mask)
		
		# Define image dimensions (H, W)
		H, W = mask.shape
		
		if self.bbox_shift:
			bbox = self.perturb(bbox, H, W )
			
		class_mask = (mask == class_id).astype(np.uint8)

		return class_embedding, cnd_mask, class_id, bbox, class_mask
	

	def __getitem__(self, index):

		prompt_type, cand_name, mask_name  = self.mask_cands[index]

		## Select image
		img_fn  = osp.join(self.img_dir , mask_name)
		mask_fn = osp.join(self.mask_dir, mask_name) 

		## get ground-truth mask and image
		img  = cv2.imread(img_fn , 1)
		mask = cv2.imread(mask_fn, cv2.IMREAD_GRAYSCALE) 

		mask = (mask/ 32).astype('uint8') ## normalize

		# get pre-computed sam feature 
		feat_dir = osp.join(self.mask_dir.replace("annotations", f"sam_features_{self.vit_mode}"), mask_name.split(".")[0] + ".npy")
		sam_feat = np.load(feat_dir)
		
		class_embedding, cnd_mask, class_id, bbox, class_mask = self.process_candidate(mask, prompt_type, cand_name, mask_name)
		
		# Apply bitwise OR to merge candidate mask with `img` and extract box
		cnd_img = cv2.bitwise_and(img,img,mask = cnd_mask) #img[bbox[1]:bbox[3], bbox[0]:bbox[2]] 

		if self.transform :  ## Should only happens with Candidate prompt in train set not otherwise
			## Transform requires PIL format
			im_pil  = Image.fromarray(cnd_img)
			cnd_img = self.cnd_img_transform(im_pil)
			#"""
			#if self.mode == "val": # debug
			
			if len(cand_name.split("/")) == 2: #seq7/00038_class1.png
				cand_name = cand_name.split("/")[0] + "/"+ cand_name.split("/")[1].split("_")[0] + "/" + cand_name.split("/")[1]
			
			os.makedirs("temp",exist_ok=True)
			os.makedirs("temp/"+cand_name.split("/")[0],exist_ok=True)
			os.makedirs("temp/"+cand_name.split("/")[0]+"/"+cand_name.split("/")[1],exist_ok=True)
			
			im_pil.save("temp/"+cand_name)
			#"""
			img, cnd_masks, bboxes = self.transform(img, [cnd_mask], np.array([bbox]) )

			cnd_mask   = cnd_masks[0]
			bbox       = bboxes[0]

		###    Sam feature, GT Mask name, Cls ID, Gt Mask for loss, Cls Embedding for loss, Cnd_Img for dinov2 if cand prompt, box prompt, mask prompt for Sam

		return sam_feat, mask_name, class_id, class_mask, class_embedding, cnd_img, np.array(bbox), cnd_mask


class Endovis18Dataset_Multilabel(Dataset):
    def __init__(self, data_root_dir = "../data/endovis_2018", 
                 mode = "val", 
                 transform=None,
                 version = 0,
                 save_transformed=False):
        
        """Define the Endovis18 dataset

        Args:
            data_root_dir (str, optional): root dir containing all data for Endovis18. Defaults to "../data/endovis_2018".
            mode (str, optional): either in "train" or "val" mode. Defaults to "val".
            version (int, optional): augmentation version to use. Defaults to 0.
        """
        
        self.transform = transform
        self.mode = mode

        # directory containing all binary annotations
        if mode == "train":
            self.mask_dir = osp.join(data_root_dir, mode, str(version), "annotations")
            self.img_dir  = osp.join(data_root_dir, mode, str(version), "images")
            self.cand_dir_binary = osp.join(data_root_dir, mode, str(version), "instance_pred_candidates_binary")


        else: ## val mode
            self.mask_dir = osp.join(data_root_dir, mode, "annotations")
            self.img_dir  = osp.join(data_root_dir, mode, "images")
            self.cand_dir_binary = osp.join(data_root_dir, mode, "instance_pred_candidates_binary")

        print(self.mask_dir)
        # put all binary masks into a list
        self.mask_list = []
        for subdir, _, files in os.walk(self.mask_dir):
            if len(files) == 0:
                continue 
            self.mask_list += [osp.join(osp.basename(subdir),i) for i in files]
        
        ##Debug
        self.save_transformed = save_transformed
        
        if self.save_transformed:
            self.output_dir="transformed_images"
            os.makedirs(self.output_dir, exist_ok=True)

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, index):
        mask_name = self.mask_list[index]
        
        # get ground-truth mask
        mask_path = osp.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)/32
        
        all_cls = (np.unique(mask)-1)[1:] ## from  0 to 6
        cls_id = np.isin(np.arange(7), all_cls).astype(np.float32)
        
        cand_mask_path = os.path.join(self.cand_dir_binary, mask_name)
        
        #if self.mode == "train":
        #    if os.path.exists(cand_mask_path) and random.random() > 0.2:
        #        #print(cand_mask_path)
        #        mask = cv2.imread(cand_mask_path, cv2.IMREAD_GRAYSCALE)
        #else:
        #    if os.path.exists(cand_mask_path):
        #        #print(cand_mask_path)
        #        mask = cv2.imread(cand_mask_path, cv2.IMREAD_GRAYSCALE)
       
        mask = (mask > 0).astype(np.uint8)
        
        # Load corresponding image
        img_path = osp.join(self.img_dir, mask_name)  # assumes same name
        img = cv2.imread(img_path, 1)
        
        #print(img_path, mask_path, cand_mask_path)

        # Apply mask: assume non-zero mask values indicate regions of interest
        masked_img = img #cv2.bitwise_and(img, img, mask=mask)
        
        # --- Apply transforms ---
        if self.transform is not None:         
            # Convert to PIL Image
            pil_img = Image.fromarray(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
            transformed_img = self.transform(pil_img)
            masked_img = transformed_img
            

            # Save transformed image if requested
            if self.save_transformed:
                transformed_np = transformed_img.permute(1, 2, 0).cpu().numpy()
                transformed_bgr = cv2.cvtColor(transformed_np, cv2.COLOR_RGB2BGR)
                
                seq, fn = mask_name.split("/")
                os.makedirs(self.output_dir + "/" + seq , exist_ok=True)
                output_filename = f"{os.path.splitext(mask_name)[0]}_transformed.png"
                output_path = osp.join(self.output_dir, output_filename)
                #print(output_path)
                cv2.imwrite(output_path, (transformed_bgr * 255).astype(np.uint8))

            

        return masked_img, cls_id


class Endovis18Dataset_COCO(Dataset):
	def __init__(self, data_root_dir = "../data/endovis_2018", 
				mode = "val", 
				vit_mode = "h",
				version = 0,
				transform=None,
				save_transform=False,
				ann_file = "", 
				result_file = "",
				perturb=False,
				test_mode=False,
				bbox_only=False,
				return_mask=False,
				det_threshold=0.01):

		"""Define the Endovis18 dataset

		Args:
		data_root_dir (str, optional): root dir containing all data for Endovis18. Defaults to "../data/endovis_2018".
		mode (str, optional): either in "train" or "val" mode. Defaults to "val".
		vit_mode (str, optional): "h", "l", "b" for huge, large, and base versions of SAM. Defaults to "h".
		version (int, optional): augmentation version to use. Defaults to 0.
		"""

		self.vit_mode = vit_mode
		self.mode = mode
		self.test = test_mode
		self.bbox_only =bbox_only

		# directory containing all binary annotations
		if mode == "train":
			self.mask_dir = osp.join(data_root_dir, mode, str(version), "binary_annotations")
			self.img_dir  = osp.join(data_root_dir, mode, str(version), "images")
		elif mode == "val":
			self.mask_dir = osp.join(data_root_dir, mode  , "binary_annotations_coco")
			self.img_dir  = osp.join(data_root_dir, mode  , "images")
		elif mode == "val_b":
			mode = "val"
			self.mask_dir = osp.join(data_root_dir, mode  , "binary_annotations_coco_b")
			self.img_dir  = osp.join(data_root_dir, mode  , "images")
		elif mode == "pred":
			self.mask_dir = osp.join(data_root_dir, mode, "binary_annotations")
			self.img_dir  = osp.join(data_root_dir, "val", "images")

		self.transform = transform
		self.save_transformed = save_transform
		self.perturb = perturb
		self.return_mask = return_mask;
		self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

		print(self.mask_dir)
		
		# put all binary masks into a list
		self.mask_list = []
		for subdir, _, files in os.walk(self.mask_dir):
			if len(files) == 0:
				continue 
			self.mask_list += [osp.join(osp.basename(subdir),i) for i in files]


		if mode == "pred":
			preds = json.load(open(result_file, 'r'))
			#preds = [p for p in preds if p["score"] >= 0.05]
			anns  = json.load(open(ann_file, 'r'))
			
			self.img_id_to_fname = {}
			self.fname_to_img_id = {}
			ann_per_img_id = {}

			for img in anns["images"]:
				img["file_name"] = img["file_name"].replace("frame","00")
				self.img_id_to_fname[img["id"]]= img["file_name"]
				self.fname_to_img_id[img["file_name"]]= img["id"]
			
			for ann in anns["annotations"]:
				if ann["image_id"] not in ann_per_img_id.keys():
					ann_per_img_id[ann["image_id"]]=[]
				
				ann_per_img_id[ann["image_id"]].append(ann)
			
			self.preds_per_filename = {}
			for idx, pred in enumerate(preds):        
				if pred["score"] > det_threshold:
					if pred["image_id"] not in self.preds_per_filename.keys():
						self.preds_per_filename[pred["image_id"]] = []

					
					bbox = pred["bbox"]
					segm = pred["segmentation"]
					pred_score= pred["score"]
					pred_label= pred["category_id"]
					bbox = [ bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3] ]
					bbox = [ int(b) for b in bbox ]
					
					best_iou = 0.0
					gt_label = None
					
					
					if pred["image_id"] in ann_per_img_id.keys():
						for gt in ann_per_img_id[pred["image_id"]]:
							iou = mask_iou(annToMask(segm), annToMask(gt["segmentation"]))
							if iou > best_iou:
								best_iou = iou
								gt_label = gt["category_id"]
								
						if gt_label:	
							self.preds_per_filename[pred["image_id"]].append((bbox, segm, pred_score, pred_label, idx, gt_label))   
						else:
							gt_label = -1
							self.preds_per_filename[pred["image_id"]].append((bbox, segm, pred_score, pred_label, idx, gt_label))   
						
			self.results = []
			for res_img_id in self.preds_per_filename.keys():
				for box, segm, pred_score, pred_label, res_idx, gt_label in self.preds_per_filename[res_img_id]:
					mask = annToMask(segm)
					if np.count_nonzero(mask) > 50: ## base threshold
						self.results.append((res_img_id, box, segm, pred_score, pred_label, res_idx, gt_label))


	def __len__(self):
		if self.mode == "train":
			return len(self.mask_list)
		elif self.mode == "val":
			return len(self.mask_list)
		elif self.mode == "val_b":
			return len(self.mask_list)
		elif self.mode == "pred":
			return len(self.results)

	# Return the label for one sample
	def get_label(self, idx: int) -> int:
		"""Return the int class‑id of sample `idx`."""
		mask_name = self.mask_list[idx]
		return int(re.search(r"class(\d+)", mask_name).group(1))

	# Return labels for samples 
	def get_labels(self) -> list:
		"""Return a list with one int class‑id per sample."""
		return [self.get_label(i) for i in range(len(self))]

	def __getitem__(self, index):

		if self.mode=="pred":

			res_img_id, bbox, segm, pred_score, pred_label, res_idx, gt_label = self.results[index]
			img_name  = self.img_id_to_fname[res_img_id]						
			img_path  = osp.join(self.img_dir, img_name)

			img  = cv2.imread( img_path, 1)  # (1024, 1280, 3) 
			
			mask = annToMask(segm)

			# get bbox
			bbox = mask_to_bbox(mask)
			
			if not self.bbox_only:
				cnd_img = cv2.bitwise_and(img,img,mask = mask)[bbox[1]:bbox[3], bbox[0]:bbox[2]]
			else:							
				cnd_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
				
			# --- Apply transforms ---
			if self.transform is not None:         
				# Convert to PIL Image
				pil_img = Image.fromarray(cv2.cvtColor(cnd_img, cv2.COLOR_BGR2RGB))
				cnd_img = self.transform(pil_img)
				
				if self.return_mask:
					mask = Image.fromarray(mask[bbox[1]:bbox[3], bbox[0]:bbox[2]])
					mask = self.transform(mask)
					cnd_img= self.normalize(cnd_img)

			if self.test:
				return cnd_img, img_path , res_idx, pred_score, pred_label, gt_label ##, bbox ##mask
			
			if not self.return_mask:
				return cnd_img, img_path , res_idx, gt_label
			else:
				return cnd_img, mask, img_path , res_idx, gt_label
	
	

		mask_name = self.mask_list[index]
		
		img_path = osp.join(self.img_dir, mask_name.split("_")[0]+".png")  # assumes same name
		img = cv2.imread(img_path, 1)
		
		# get class id from mask_name 
		cls_id = int(re.search(r"class(\d+)", mask_name).group(1))
		
		# get ground-truth mask
		mask_path = osp.join(self.mask_dir, mask_name)
		mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
		
		if self.perturb and random.random() > 0.5:
			mask = perturb_mask(mask)
		
		# get bbox
		bbox = mask_to_bbox(mask)
		
		if not self.bbox_only:
			cnd_img = cv2.bitwise_and(img,img,mask = mask)[bbox[1]:bbox[3], bbox[0]:bbox[2]]
		else:
			cnd_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
		
		# --- Apply transforms ---
		if self.transform is not None:         
			# Convert to PIL Image
			pil_img = Image.fromarray(cv2.cvtColor(cnd_img, cv2.COLOR_BGR2RGB))
			transformed_img = self.transform(pil_img)
			cnd_img = transformed_img
			
			if self.return_mask: 
				mask = Image.fromarray(mask[bbox[1]:bbox[3], bbox[0]:bbox[2]])
				mask = self.transform(mask)
				cnd_img= self.normalize(cnd_img)
			


			# Save transformed image if requested
			if self.save_transformed:
				transformed_np = transformed_img.permute(1, 2, 0).cpu().numpy()
				transformed_bgr = cv2.cvtColor(transformed_np, cv2.COLOR_RGB2BGR)

				seq, fn = mask_name.split("/")
				os.makedirs(self.output_dir + "/" + seq , exist_ok=True)
				output_filename = f"{os.path.splitext(mask_name)[0]}_transformed.png"
				output_path = osp.join(self.output_dir, output_filename)
				#print(output_path)
				cv2.imwrite(output_path, (transformed_bgr * 255).astype(np.uint8))

		
		if not self.return_mask:
			return cnd_img, cls_id, mask_name
		else:
			return cnd_img, mask, cls_id, mask_name
        

class Endovis18Dataset_COCO_FULL(Dataset):
	def __init__(self, data_root_dir = "../data/endovis_2018", 
				mode = "val", 
				vit_mode = "h",
				version = 0,
				transform=None,
				save_transform=False,
				ann_file = "", 
				result_file = "",
				perturb=False,
				test_mode=False,
				bbox_only=False,
				return_mask=False,
				det_threshold=0.01):

		"""Define the Endovis18 dataset

		Args:
		data_root_dir (str, optional): root dir containing all data for Endovis18. Defaults to "../data/endovis_2018".
		mode (str, optional): either in "train" or "val" mode. Defaults to "val".
		vit_mode (str, optional): "h", "l", "b" for huge, large, and base versions of SAM. Defaults to "h".
		version (int, optional): augmentation version to use. Defaults to 0.
		"""

		self.vit_mode = vit_mode
		self.mode = mode
		self.test = test_mode
		self.bbox_only =bbox_only
		self.return_mask=return_mask;

		# directory containing all binary annotations
		if mode == "train":
			self.mask_dir = osp.join(data_root_dir, mode) #str(version), "binary_annotations")
			self.img_dir  = osp.join(data_root_dir, mode) #str(version), "images")
		elif mode == "val":
			self.mask_dir = osp.join(data_root_dir, mode  , "binary_annotations_coco")
			self.img_dir  = osp.join(data_root_dir, mode  , "images")
		elif mode == "val_b":
			mode = "val"
			self.mask_dir = osp.join(data_root_dir, mode  , "binary_annotations_coco_b")
			self.img_dir  = osp.join(data_root_dir, mode  , "images")
		elif mode == "pred":
			self.mask_dir = osp.join(data_root_dir, mode, "binary_annotations")
			self.img_dir  = osp.join(data_root_dir, "val", "images")

		self.transform = transform
		self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
		self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
		self.save_transformed = save_transform
		self.perturb = perturb

		print(self.mask_dir)
		
		### If mode == "val" or "val_b"
		# put all binary masks into a list
		self.mask_list = []
		for subdir, _, files in os.walk(self.mask_dir):
			if len(files) == 0:
				continue 
			self.mask_list += [osp.join(osp.basename(subdir),i) for i in files]
		
				        
		if self.save_transformed:
			self.output_dir="transformed_images"
			os.makedirs(self.output_dir, exist_ok=True)
			
		if mode == "train":
			versions = os.listdir(osp.join(data_root_dir, mode))	
			# put all binary masks into a list
			self.mask_list = []
			for version in versions:
				mask_folder = os.path.join(self.mask_dir, str(version), "binary_annotations")
				for seq in os.listdir(mask_folder):
					self.mask_list+=[osp.join(str(version), "binary_annotations", seq, fn)  for fn in os.listdir(osp.join(mask_folder, seq))]
			##self.mask_list =self.mask_list[:10000]
			

	def __len__(self):
		if self.mode == "train":
			return len(self.mask_list)
		elif self.mode == "val":
			return len(self.mask_list)
		elif self.mode == "val_b":
			return len(self.mask_list)
		elif self.mode == "pred":
			return len(self.results)
			
	# Return the label for one sample
	def get_label(self, idx: int) -> int:
		"""Return the int class‑id of sample `idx`."""
		mask_name = self.mask_list[idx]
		return int(re.search(r"class(\d+)", mask_name).group(1))

	# Return labels for samples 
	def get_labels(self) -> list:
		"""Return a list with one int class‑id per sample."""
		return [self.get_label(i) for i in range(len(self))]
			

	def __getitem__(self, index):
		
		mask_name = self.mask_list[index]
		
		if self.mode == "train":
			img_path = osp.join(self.img_dir, mask_name.replace("binary_annotations", "images").split("_")[0]+".png")  # assumes same name
		else:
			img_path = osp.join(self.img_dir, mask_name.split("_")[0]+".png")  # assumes same name
		
		img = cv2.imread(img_path, 1)
		
		# get class id from mask_name 
		cls_id = int(re.search(r"class(\d+)", mask_name).group(1))
		
		# get ground-truth mask
		mask_path = osp.join(self.mask_dir, mask_name)
		mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
		
		if self.perturb and random.random() > 0.5:
			mask = perturb_mask(mask)
		
		# get bbox
		bbox = mask_to_bbox(mask)
		
		if not self.bbox_only:
		
			try:
			
				cnd_img = cv2.bitwise_and(img,img,mask = mask)[bbox[1]:bbox[3], bbox[0]:bbox[2]]
				
			except:
				print(mask_path)
				cnd_img = cv2.bitwise_and(img,img,mask = mask)
		else:
			try:
			# Define image dimensions (H, W)
			
				##print(mask_path)
				H, W = mask.shape
				cnd_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
			
			except:
				print(mask_path)
				cnd_img = cv2.bitwise_and(img,img,mask = mask)
		
		# --- Apply transforms ---
		if self.transform is not None:         
			# Convert to PIL Image
			pil_img = Image.fromarray(cv2.cvtColor(cnd_img, cv2.COLOR_BGR2RGB))
			transformed_img = self.transform(pil_img)
			cnd_img = transformed_img
			
			if self.return_mask:
				if bbox is not None:
					mask = Image.fromarray(mask[bbox[1]:bbox[3], bbox[0]:bbox[2]])
				else:
					mask = Image.fromarray(mask)
				mask = self.transform(mask)
				cnd_img=self.color_jitter(cnd_img)
				cnd_img= self.normalize(cnd_img)





			# Save transformed image if requested
			if self.save_transformed:
				transformed_np = transformed_img.permute(1, 2, 0).cpu().numpy()
				transformed_bgr = cv2.cvtColor(transformed_np, cv2.COLOR_RGB2BGR)
				
				transformed_np_mask = mask.permute(1, 2, 0).cpu().numpy()
				transformed_np_mask_bgr = cv2.cvtColor(transformed_np_mask, cv2.COLOR_RGB2BGR)
				
				seq = mask_name.split("/")[-2]
				os.makedirs(self.output_dir + "/" + seq , exist_ok=True)
				fn = os.path.splitext(mask_name)[0].replace("/","_")
				output_filename = f"{fn}_transformed.png"
				output_path = osp.join(self.output_dir, output_filename)
				#print(output_path)
				cv2.imwrite(output_path, (transformed_bgr * 255).astype(np.uint8))
				
				output_filename = f"{fn}_mask_transformed.png"
				output_path = osp.join(self.output_dir, output_filename)
				#print(output_path)
				cv2.imwrite(output_path, (transformed_np_mask_bgr * 255).astype(np.uint8))

		if not self.return_mask:
			return cnd_img, cls_id, mask_name
		else:
			return cnd_img, mask, cls_id, mask_name


class Endovis18Dataset_COCO_IMG_MASK(Dataset):
	def __init__(self, data_root_dir = "../data/endovis_2018", 
				mode = "val", 
				version = 0,
				transform=False,
				perturb=False,
				bbox_only=False):

		"""Define the Endovis18 dataset

		Args:
		data_root_dir (str, optional): root dir containing all data for Endovis18. Defaults to "../data/endovis_2018".
		mode (str, optional): either in "train" or "val" mode. Defaults to "val".
		version (int, optional): augmentation version to use. Defaults to 0.
		"""

		self.mode = mode
		self.transform = transform
		self.bbox_only=bbox_only;


		# directory containing all binary annotations
		if mode == "train":
			#self.mask_dir = osp.join(data_root_dir, mode, str(version), "binary_annotations")
			#self.img_dir  = osp.join(data_root_dir, mode, str(version), "images")
			self.mask_dir = osp.join(data_root_dir, mode) #str(version), "binary_annotations")
			self.img_dir  = osp.join(data_root_dir, mode) #str(version), "images")
		elif mode == "val":
			self.mask_dir = osp.join(data_root_dir, mode  , "binary_annotations_coco")
			self.img_dir  = osp.join(data_root_dir, mode  , "images")
		elif mode == "val_b":
			self.mask_dir = osp.join(data_root_dir, "val"  , "binary_annotations_coco_b")
			self.img_dir  = osp.join(data_root_dir, "val"  , "images")
		elif mode == "pred":
			self.mask_dir = "./segms/" ##osp.join(data_root_dir, mode, "binary_annotations")
			self.img_dir  = osp.join(data_root_dir, "val", "images")


		# ------------------------------------------------------------------ #
		# 2. Image‑level transforms (common to every sample)
		# ------------------------------------------------------------------ #
		self.shape_transforms = transforms.Compose([
			transforms.Resize((224, 224)),           # works on PIL images / tensors
			transforms.ToTensor(),                   # → [0,1] float tensor, C×H×W
		])
		
		self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
		
		self.perturb = perturb

		print(self.mask_dir)
		
		# put all binary masks into a list
		self.mask_list = []
		for subdir, _, files in os.walk(self.mask_dir):
			if len(files) == 0:
				continue 
			self.mask_list += [osp.join(osp.basename(subdir),i) for i in files]


		if mode == "train":
			versions = os.listdir(osp.join(data_root_dir, mode))	
			# put all binary masks into a list
			self.mask_list = []
			for version in versions:
				mask_folder = os.path.join(self.mask_dir, str(version), "binary_annotations")
				for seq in os.listdir(mask_folder):
					self.mask_list+=[osp.join(str(version), "binary_annotations", seq, fn)  for fn in os.listdir(osp.join(mask_folder, seq))]
			


	def __len__(self):
		return len(self.mask_list)

	# Return the label for one sample
	def get_label(self, idx: int) -> int:
		"""Return the int class‑id of sample `idx`."""
		mask_name = self.mask_list[idx]
		return int(re.search(r"class(\d+)", mask_name).group(1))

	# Return labels for samples 
	def get_labels(self) -> list:
		"""Return a list with one int class‑id per sample."""
		return [self.get_label(i) for i in range(len(self))]

	def __getitem__(self, index):

		mask_name = self.mask_list[index]
		
		if self.mode == "train":
			img_path = osp.join(self.img_dir, mask_name.replace("binary_annotations", "images").split("_")[0]+".png")  # assumes same name
		else:
			img_path = osp.join(self.img_dir, mask_name.split("_")[0]+".png")  # assumes same name
		
		img = cv2.imread(img_path, 1)
		
		if self.mode == "pred":
			cls_id = -1		
		else:
			# get class id from mask_name 
			cls_id = int(re.search(r"class(\d+)", mask_name).group(1))
		
		# get ground-truth mask
		mask_path = osp.join(self.mask_dir, mask_name)
		mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
		

		if self.perturb and random.random() > 0.5:
			mask = perturb_mask(mask)
			
		# get bbox
		bbox = mask_to_bbox(mask)
		
		if not self.bbox_only:
			cnd_img = cv2.bitwise_and(img,img,mask = mask)[bbox[1]:bbox[3], bbox[0]:bbox[2]]
		else:
			cnd_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
		
      
		# Convert to PIL Image
		pil_img = Image.fromarray(cv2.cvtColor(cnd_img, cv2.COLOR_BGR2RGB))
		transformed_img = self.shape_transforms(pil_img)
		cnd_img = transformed_img
		cnd_img= self.normalize(cnd_img)
		
		mask_1024 = mask
		mask    = Image.fromarray(mask[bbox[1]:bbox[3], bbox[0]:bbox[2]])
		mask = self.shape_transforms(mask)

		return cnd_img, mask, mask_1024, cls_id, mask_name



class Endovis18Dataset_CLS(Dataset):
	def __init__(self, data_root_dir = "../data/endovis_2018", 
				mode = "val", 
				vit_mode = "h",
				version = 0,
				transform=None,
				save_transform=False,
				ann_file = "", 
				result_file = "",
				perturb=False,
				test_mode=False,
				bbox_only=False):

		"""Define the Endovis18 dataset

		Args:
		data_root_dir (str, optional): root dir containing all data for Endovis18. Defaults to "../data/endovis_2018".
		mode (str, optional): either in "train" or "val" mode. Defaults to "val".
		vit_mode (str, optional): "h", "l", "b" for huge, large, and base versions of SAM. Defaults to "h".
		version (int, optional): augmentation version to use. Defaults to 0.
		"""

		self.vit_mode = vit_mode
		self.mode = mode
		self.test = test_mode
		self.bbox_only =bbox_only

		# directory containing all binary annotations
		if mode == "train":
			self.mask_dir = osp.join(data_root_dir, mode, str(version), "binary_annotations")
			self.img_dir  = osp.join(data_root_dir, mode, str(version), "images")
		elif mode == "val":
			self.mask_dir = osp.join(data_root_dir, mode, "binary_annotations")
			self.img_dir  = osp.join(data_root_dir, mode, "images")
		elif mode == "pred":
			self.mask_dir = osp.join(data_root_dir, mode, "binary_annotations")
			self.img_dir  = osp.join(data_root_dir, "val", "images")

		self.transform = transform
		self.save_transformed = save_transform
		self.perturb = perturb

		print(self.mask_dir)
		
		# put all binary masks into a list
		self.mask_list = []
		for subdir, _, files in os.walk(self.mask_dir):
			if len(files) == 0:
				continue 
			self.mask_list += [osp.join(osp.basename(subdir),i) for i in files]


		if mode == "pred":
			preds = json.load(open(result_file, 'r'))
			#preds = [p for p in preds if p["score"] >= 0.05]
			anns  = json.load(open(ann_file, 'r'))
			
			self.img_id_to_fname = {}
			self.fname_to_img_id = {}

			for img in anns["images"]:
				img["file_name"] = img["file_name"].replace("frame","00")
				self.img_id_to_fname[img["id"]]= img["file_name"]
				self.fname_to_img_id[img["file_name"]]= img["id"]
			
			self.preds_per_filename = {}
			for idx, pred in enumerate(preds):        
				if pred["score"] > 0.05:
					if pred["image_id"] not in self.preds_per_filename.keys():
						self.preds_per_filename[pred["image_id"]] = []

					bbox = pred["bbox"]
					segm = pred["segmentation"]
					pred_score= pred["score"]
					pred_label= pred["category_id"]
					bbox = [ bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3] ]
					bbox = [ int(b) for b in bbox ]
					self.preds_per_filename[pred["image_id"]].append((bbox, segm, pred_score, pred_label, idx))    

			self.results = []
			for res_img_id in self.preds_per_filename.keys():
				for box, segm, pred_score, pred_label, res_idx in self.preds_per_filename[res_img_id]:
					self.results.append((res_img_id, box, segm, pred_score, pred_label, res_idx))


	def __len__(self):
		if self.mode == "train":
			return len(self.mask_list)
		elif self.mode == "val":
			return len(self.mask_list)
		else: 
			return len(self.results)
			

	def __getitem__(self, index):

		if self.mode=="pred":

			res_img_id, bbox, segm, pred_score, pred_label, res_idx = self.results[index]
			img_name  = self.img_id_to_fname[res_img_id]						
			img_path  = osp.join(self.img_dir, img_name)

			img  = cv2.imread( img_path, 1)  # (1024, 1280, 3) 
			
			mask = annToMask(segm)

			# get bbox
			bbox = mask_to_bbox(mask)
			
			if not self.bbox_only:
				cnd_img = cv2.bitwise_and(img,img,mask = mask)[bbox[1]:bbox[3], bbox[0]:bbox[2]]
			else:							
				cnd_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
				
			# --- Apply transforms ---
			if self.transform is not None:         
				# Convert to PIL Image
				pil_img = Image.fromarray(cv2.cvtColor(cnd_img, cv2.COLOR_BGR2RGB))
				cnd_img = self.transform(pil_img)


			if self.test:
				return cnd_img, img_path , res_idx, pred_score, pred_label ##, bbox
			
			return cnd_img, img_path , res_idx
	

		mask_name = self.mask_list[index]
		
		img_path = osp.join(self.img_dir, mask_name.split("_")[0]+".png")  # assumes same name
		img = cv2.imread(img_path, 1)
		
		# get class id from mask_name 
		cls_id = int(re.search(r"class(\d+)", mask_name).group(1))
		
		# get ground-truth mask
		mask_path = osp.join(self.mask_dir, mask_name)
		mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
		
		if self.perturb and random.random() > 0.5:
			mask = perturb_mask(mask)
		
		# get bbox
		bbox = mask_to_bbox(mask)
		
		if not self.bbox_only:
			cnd_img = cv2.bitwise_and(img,img,mask = mask)[bbox[1]:bbox[3], bbox[0]:bbox[2]]
		else:
			cnd_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
		
		# --- Apply transforms ---
		if self.transform is not None:         
			# Convert to PIL Image
			pil_img = Image.fromarray(cv2.cvtColor(cnd_img, cv2.COLOR_BGR2RGB))
			transformed_img = self.transform(pil_img)
			cnd_img = transformed_img


			# Save transformed image if requested
			if self.save_transformed:
				transformed_np = transformed_img.permute(1, 2, 0).cpu().numpy()
				transformed_bgr = cv2.cvtColor(transformed_np, cv2.COLOR_RGB2BGR)

				seq, fn = mask_name.split("/")
				os.makedirs(self.output_dir + "/" + seq , exist_ok=True)
				output_filename = f"{os.path.splitext(mask_name)[0]}_transformed.png"
				output_path = osp.join(self.output_dir, output_filename)
				#print(output_path)
				cv2.imwrite(output_path, (transformed_bgr * 255).astype(np.uint8))

		return cnd_img, cls_id, mask_name

class Endovis18Dataset_CLS_FULL(Dataset):
	def __init__(self, data_root_dir = "../data/endovis_2018", 
				mode = "train", 
				vit_mode = "h",
				version = 0,
				transform=None,
				save_transform=False,
				ann_file = "", 
				result_file = "",
				perturb=False,
				bbox_only=False):

		"""Define the Endovis18 dataset

		Args:
		data_root_dir (str, optional): root dir containing all data for Endovis18. Defaults to "../data/endovis_2018".
		mode (str, optional): either in "train" or "val" mode. Defaults to "val".
		vit_mode (str, optional): "h", "l", "b" for huge, large, and base versions of SAM. Defaults to "h".
		version (int, optional): augmentation version to use. Defaults to 0.
		"""

		self.vit_mode = vit_mode
		self.mode = mode
		self.bbox_only=bbox_only
		self.bbox_shift=5
		
		versions = os.listdir(osp.join(data_root_dir, mode))
		
		# directory containing all binary annotations
		self.mask_dir = osp.join(data_root_dir, mode) #str(version), "binary_annotations")
		self.img_dir  = osp.join(data_root_dir, mode) #str(version), "images")
		
		self.transform = transform
		self.save_transformed = save_transform
		self.perturb = perturb

		print(self.mask_dir)
		
		# put all binary masks into a list
		self.mask_list = []
		for version in versions:
			mask_folder = os.path.join(self.mask_dir, str(version), "binary_annotations")
			for seq in os.listdir(mask_folder):
				self.mask_list+=[osp.join(str(version), "binary_annotations", seq, fn)  for fn in os.listdir(osp.join(mask_folder, seq))]
		
		
		random.shuffle(self.mask_list)
		
	
	def __len__(self):
		if self.mode == "train":
			return len(self.mask_list)
			
	def perturb_box(self,bbox, H, W ):
		
		# Add random perturbation to bounding box coordinates
		min_x = max(0, bbox[0] - random.randint(0, self.bbox_shift))
		max_x = min(W, bbox[2] + random.randint(0, self.bbox_shift))
		min_y = max(0, bbox[1]  - random.randint(0, self.bbox_shift))
		max_y = min(H, bbox[3]  + random.randint(0, self.bbox_shift))
		
		bbox = [min_x, min_y, max_x, max_y]
		
		return bbox
		
	def __getitem__(self, index):

		mask_name = self.mask_list[index]
		
		# get class id from mask_name 
		cls_id = int(re.search(r"class(\d+)", mask_name).group(1))
		
		# get ground-truth mask
		mask_path = osp.join(self.mask_dir, mask_name)
		mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
		
		img_path = osp.join(self.img_dir, mask_name.replace("binary_annotations", "images").split("_")[0]+".png")  # assumes same name
		img = cv2.imread(img_path, 1)

		
		if self.perturb and random.random() > 0.5:
			mask = perturb_mask(mask)
		
		# get bbox
		bbox = mask_to_bbox(mask)
		
		if not self.bbox_only:
		
			try:
			
				cnd_img = cv2.bitwise_and(img,img,mask = mask)[bbox[1]:bbox[3], bbox[0]:bbox[2]]
				
			except:
				print(mask_path)
				cnd_img = cv2.bitwise_and(img,img,mask = mask)
		else:
			try:
				# Define image dimensions (H, W)
				H, W = mask.shape
				
				bbox = self.perturb_box(bbox, H, W )
				cnd_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
			except:
				print(mask_path)
				cnd_img = cv2.bitwise_and(img,img,mask = mask)
				
			
		# --- Apply transforms ---
		if self.transform is not None:         
			# Convert to PIL Image
			pil_img = Image.fromarray(cv2.cvtColor(cnd_img, cv2.COLOR_BGR2RGB))
			transformed_img = self.transform(pil_img)
			cnd_img = transformed_img


			# Save transformed image if requested
			if self.save_transformed:
				transformed_np = transformed_img.permute(1, 2, 0).cpu().numpy()
				transformed_bgr = cv2.cvtColor(transformed_np, cv2.COLOR_RGB2BGR)

				seq, fn = mask_name.split("/")
				os.makedirs(self.output_dir + "/" + seq , exist_ok=True)
				output_filename = f"{os.path.splitext(mask_name)[0]}_transformed.png"
				output_path = osp.join(self.output_dir, output_filename)
				#print(output_path)
				cv2.imwrite(output_path, (transformed_bgr * 255).astype(np.uint8))

		return cnd_img, cls_id, mask_name


class ImageDataset(Dataset):
    def __init__(self, root_dir="./data/endovis_2018/train_instances_sampled", transform=None):
        """
        Args:
            root_dir (str): Path to the dataset (e.g., "Train/" or "Val/")
            transform (torchvision.transforms): Image transformations
        """
        self.root_dir  = root_dir
        self.transform = transform
        self.classes   = sorted(os.listdir(root_dir))  # Sort to maintain consistent mapping
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.samples = []

        # Load image file paths and labels
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for file in sorted(os.listdir(cls_dir)):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Supported formats
                    self.samples.append((os.path.join(cls_dir, file), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")  # Ensure 3-channel RGB
        
        if self.transform:
            image = self.transform(image)

        return image, label, img_path



class Endovis18Dataset(Dataset):
    def __init__(self, data_root_dir = "../data/endovis_2018", 
                 mode = "val", 
                 vit_mode = "h",
                 version = 0):
        
        """Define the Endovis18 dataset

        Args:
            data_root_dir (str, optional): root dir containing all data for Endovis18. Defaults to "../data/endovis_2018".
            mode (str, optional): either in "train" or "val" mode. Defaults to "val".
            vit_mode (str, optional): "h", "l", "b" for huge, large, and base versions of SAM. Defaults to "h".
            version (int, optional): augmentation version to use. Defaults to 0.
        """
        
        self.vit_mode = vit_mode
       
        # directory containing all binary annotations
        if mode == "train":
            self.mask_dir = osp.join(data_root_dir, mode, str(version), "binary_annotations")
        elif mode == "val":
            self.mask_dir = osp.join(data_root_dir, mode, "binary_annotations")
        else:
            self.mask_dir = osp.join(data_root_dir, mode, "binary_annotations")

        print(self.mask_dir)
        
        # put all binary masks into a list
        self.mask_list = []
        for subdir, _, files in os.walk(self.mask_dir):
            if len(files) == 0:
                continue 
            self.mask_list += [osp.join(osp.basename(subdir),i) for i in files]
                    

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, index):
        mask_name = self.mask_list[index]
        
        # get class id from mask_name 
        cls_id = int(re.search(r"class(\d+)", mask_name).group(1))
        
        # get pre-computed sam feature 
        feat_dir = osp.join(self.mask_dir.replace("binary_annotations", f"sam_features_{self.vit_mode}"), mask_name.split("_")[0] + ".npy")
        if not os.path.exists(feat_dir):
        	os.rename(feat_dir.split(".npy")[0]+"npy.npy", feat_dir)
        
        sam_feat = np.load(feat_dir)
        
        # get ground-truth mask
        mask_path = osp.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # get class embedding
        class_embedding_path = osp.join(self.mask_dir.replace("binary_annotations", f"class_embeddings_{self.vit_mode}"), mask_name.replace("png","npy"))
        class_embedding = np.load(class_embedding_path)
        

        return sam_feat, mask_name, cls_id, mask, class_embedding
 

class Endovis17Dataset(Dataset):
    def __init__(self, data_root_dir = "../data/endovis_2017", 
                 mode = "val",
                 fold = 0,  
                 vit_mode = "h",
                 version = 0):
                        
        self.vit_mode = vit_mode
        
        all_folds = list(range(1, 9))
        fold_seq = {0: [1, 3],
                    1: [2, 5],
                    2: [4, 8],
                    3: [6, 7]}
        
        if mode == "train":
            seqs = [x for x in all_folds if x not in fold_seq[fold]]     
        elif mode == "val":
            seqs = fold_seq[fold]

        self.mask_dir = osp.join(data_root_dir, str(version), "binary_annotations")
        
        self.mask_list = []
        for seq in seqs:
            seq_path = osp.join(self.mask_dir, f"seq{seq}")
            self.mask_list += [f"seq{seq}/{mask}" for mask in os.listdir(seq_path)]
            
    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, index):
        mask_name = self.mask_list[index]
        
        # get class id from mask_name 
        cls_id = int(re.search(r"class(\d+)", mask_name).group(1))
        
        # get pre-computed sam feature 
        feat_dir = osp.join(self.mask_dir.replace("binary_annotations", f"sam_features_{self.vit_mode}"), mask_name.split("_")[0] + ".npy")
        sam_feat = np.load(feat_dir)
        
        # get ground-truth mask
        mask_path = osp.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # get class embedding
        class_embedding_path = osp.join(self.mask_dir.replace("binary_annotations", f"class_embeddings_{self.vit_mode}"), mask_name.replace("png","npy"))
        class_embedding = np.load(class_embedding_path)
        
        return sam_feat, mask_name, cls_id, mask, class_embedding
        
        
class NpyDataset(Dataset):
    def __init__(self, data_root, mode='train', transform=None, bbox_shift=20):
    
        self.data_root = data_root
        self.base_path = osp.join(data_root,mode)

        self.gt_path_files = []
        if mode == "train":
            for version in range(41):
                binary_mask_dir = osp.join(self.base_path, str(version), "binary_annotations")
                for seq in os.listdir(binary_mask_dir):
                    self.gt_path_files +=[ join(str(version), "binary_annotations", seq, inst) for inst in os.listdir(join(binary_mask_dir,seq))]
        else:
            binary_mask_dir = osp.join(self.base_path, "binary_annotations")
            for seq in os.listdir(binary_mask_dir):
                self.gt_path_files +=[ join("binary_annotations", seq, inst) for inst in os.listdir(join(binary_mask_dir,seq))]

        self.transform = transform
        self.bbox_shift = bbox_shift
        print(f"number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        
        ## print("\n##########################################################")
        ## load npy image (1024, 1024, 3), [0,1]
        mask_name = self.gt_path_files[index]
        img_name  = mask_name.split("_class")[0]+".png"
        
        img_name  = img_name.replace("binary_annotations", "images")
        img_path  = ospjoin(self.base_path, img_name)
        img  = cv2.imread( img_path, 1)  # (1024, 1280, 3) 
        
        cls_id = mask_name.split("class")[-1].split(".")[0]
        
       	mask_name  = img_name.replace("images", "annotations")
        mask_path = osp.join(self.base_path, mask_name)
        gt = cv2.imread( mask_path, 0) # (1024, 1280, 1) ## grayscale multi class
        
        #print(img_path, mask_path)
        gt = (gt/ 32).astype('uint8') ## normalize to 0 to 7
        #print(cls_id, np.unique(gt))
        
        cls_id = int(cls_id)
        gt2D = (gt == cls_id).astype(np.uint8) ## binary mask
        
        #print(np.unique(gt2D))
        assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"
        
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bbox = [x_min, y_min, x_max, y_max]
        
        img_1024, cnd_masks, bboxes = self.transform(img, [gt2D], np.array([bbox]) )
        bbox = bboxes[0]
        gt2D = cnd_masks[0]

        
        return (
            img_1024,
            gt2D,
            np.array(bbox),
            img_name,
        )

    
        
class ResizeAndPadAugment:

    def __init__(self, target_size, augment=False):
        self.target_size = target_size
        self.augment = augment
        self.transform = ResizeLongestSide(target_size)
        self.to_tensor = transforms.ToTensor()

    def hflip_bboxes(self, bboxes, img_width):
        # Horizontally flip bounding boxes
        flipped_bboxes = []
        for bbox in bboxes:
            x1 = img_width - bbox[2]
            x2 = img_width - bbox[0]
            flipped_bboxes.append([x1, bbox[1], x2, bbox[3]])
        return flipped_bboxes
        
        
    def vflip_bboxes(self, bboxes, img_height):
        # Vertically flip bounding boxes
        flipped_bboxes = []
        for bbox in bboxes:
            y1 = img_height - bbox[3]
            y2 = img_height - bbox[1]
            flipped_bboxes.append([bbox[0], y1, bbox[2], y2])
        return flipped_bboxes
        
        
    def __call__(self, image, masks, bboxes):
        # Resize image and masks
        og_h, og_w, _ = image.shape
        image = self.transform.apply_image(image)
        masks = [torch.tensor(self.transform.apply_image(mask)) for mask in masks]
        image = self.to_tensor(image)
        #print("image.shape:", image.shape) ## [3,819,1024]

        # Pad image and masks to form a square
        _, h, w = image.shape
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2

        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        image = transforms.Pad(padding)(image)
        masks = [transforms.Pad(padding)(mask) for mask in masks]

        # Adjust bounding boxes
        bboxes = self.transform.apply_boxes(bboxes, (og_h, og_w))
        bboxes = [[bbox[0] + pad_w, bbox[1] + pad_h, bbox[2] + pad_w, bbox[3] + pad_h] for bbox in bboxes]
        
        # Apply augmentations if specified
        if self.augment:
            # Random Horizontal Flip
            if random.random() > 0.5:
                image = F.hflip(image)
                masks = [F.hflip(mask) for mask in masks]
                bboxes = self.hflip_bboxes(bboxes, image.shape[2])
                
            # Random Vertical Flip
            if random.random() > 0.5:
                image = F.vflip(image)
                masks = [F.vflip(mask) for mask in masks]
                bboxes = self.vflip_bboxes(bboxes, image.shape[1])

            color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            image = color_jitter(image)
            
            # Random additional brightness adjustment
            if random.random() > 0.5:
                brightness_factor = random.uniform(0.7, 1.3)
                image = F.adjust_brightness(image, brightness_factor)

            # Random Gaussian Noise
            if random.random() > 0.5:
                noise = torch.randn_like(image) * 0.05  # adjust noise level if needed
                image = torch.clamp(image + noise, 0., 1.)

            # Random Gaussian Blur
            if random.random() > 0.5:
                kernel_size = random.choice([3, 5])  # random 3x3 or 5x5 blur
                image = F.gaussian_blur(image, kernel_size=kernel_size)

        ##print(padding, (h, w), (og_h, og_w) ) ## (0, 102, 0, 103) (819, 1024) (1024, 1280)

        return image, masks, bboxes
