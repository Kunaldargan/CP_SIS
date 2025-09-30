import warnings
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


import argparse
from functools import partial
import json
import logging
import os
import sys
from typing import List, Optional

import numpy as np
from torch.nn.parallel import DistributedDataParallel

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import pdb

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

class DINOv2WithLearnableQueries(nn.Module):
    def __init__(self, base_model: nn.Module, num_extra_queries: int):
        super().__init__()
        self.model = base_model
        self.num_extra_queries = num_extra_queries
        self.q = nn.Parameter(torch.randn(num_extra_queries, base_model.embed_dim))
        nn.init.trunc_normal_(self.q, std=0.02)
        self.embed_dim = base_model.embed_dim
        D = self.embed_dim
        ##self.class_head = nn.Linear(self.model.backbone.embed_dim, num_extra_queries)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=base_model.embed_dim, num_heads=2, batch_first=True)
    
   
    def query_analysis(self, queries, block_stats, threshold=0.9, threshold_lower=0.1, save_dir="query_vis", label_batch=None):
    
        ## threshold (float): threshold to consider a query as activated
        os.makedirs(save_dir, exist_ok=True)
        with torch.no_grad():
            q_sigmoid = torch.sigmoid(queries)  # (B, Q, D)
            activated = (q_sigmoid > threshold).float()  # (B, Q, D)
            activated_lower = (q_sigmoid < threshold_lower).float()  # (B, Q, D)
            
            #activated_per_query = activated.mean(dim=(0, 2))  # (Q,) average over batch and D
            #fraction_activated  = activated_per_query.cpu()
	

            print("Sigmoid Q_ctx stats:")
            print("Mean:", q_sigmoid.mean().item())
            print("Max:", q_sigmoid.max().item())
            print("Min:", q_sigmoid.min().item())
            
            print("Mean per query:", q_sigmoid.mean(dim=2))
            print("Max per query:", q_sigmoid.max(dim=2))
            print("Min per query:", q_sigmoid.min(dim=2))
                 
            print("Num > 0.9:", activated.mean().item())
            print("Num < 0.1:", activated_lower.mean().item())
            
            print("###########################################")


            #
            #block_stat = {
	    #	'block': i,
	    #	'activated_per_query': activated_per_query.cpu(),  # shape (Q,)
	    #	'fraction_activated_per_query': fraction_activated
            #}
            #block_stats.append(block_stat)
            # Save Heatmap: mean query activation per query
            
            # Reduce over D to get activation per query (e.g., norm or mean)       
            activations = queries.norm(dim=-1).cpu().numpy()  # [B, Q]
            
            ## Per-Sample (row-wise) Normalization           
            # Normalize globally to [0, 1] for better contrast
            act_min, act_max = activations.min(), activations.max()
            if act_max - act_min < 1e-5:
                print("⚠️ Activations too flat for heatmap.")
            else:
                activations = (activations - act_min) / (act_max - act_min)
            plt.figure(figsize=(12, 6))
            sns.heatmap(activations, cmap="viridis", cbar=True)
            plt.title("Mean Activation per Query")
            plt.xlabel("Query Index")
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "query_activation_heatmap.png"))
            plt.close()

            # Save t-SNE plot
            B, Q, D = queries.shape
            q_flat = queries.reshape(-1, D).cpu().numpy()  # [B*Q, D]
            tsne = TSNE(n_components=2, random_state=42)
            q_tsne = tsne.fit_transform(q_flat)
            
            # Cluster
            n_clusters=7
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(q_flat)
            labels = kmeans.labels_
            
            plt.figure(figsize=(6, 6))
            scatter = plt.scatter(q_tsne[:, 0], q_tsne[:, 1],  c=labels, cmap='tab10', s=10, alpha=0.8)
            plt.title("t-SNE of Query Embeddings")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "query_tsne.png"))
            plt.close()

            print("Saved t-SNE and heatmap visualizations to:", save_dir)


        return block_stats

        
    def get_intermediate_layers(self,
        x: torch.Tensor,
        n: int,
        seg_mask=None,
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
        debug=False):
        
        debug=False
        ##pdb.set_trace();
        B = x.size(0)
                
        x = self.model.prepare_tokens_with_masks(x)
        L = x.shape[1]  # save the original token count before concat

        blocks_to_take = range(len(self.model.blocks) - n, len(self.model.blocks)) if isinstance(n, int) else n
        learnable_queries = self.q.unsqueeze(0).expand(B, -1, -1)  # (B, Q, D)

        outputs, qctx_outputs = [], []        
        
        if debug:
            block_stats=[]

        ########################################## CPA BLOCK #################################################               
        for i, blk in enumerate(self.model.blocks):
            
            #print(x.shape, i)
            x = blk(x)    
            
            #print(x.shape, i)    
            
            ##if i in blocks_to_take:                    
            # cross-attention between learnable queries
            q_ctx, _ = self.cross_attn(
            	query=learnable_queries, key=x, value=x, need_weights=False
            )                                         # (B, Q, D)
            
            if debug:
                self.query_analysis(q_ctx, block_stats)
                
            
            #########################################################################################################              

            if i in blocks_to_take:
                outputs.append(x)           # (B, L, D)
                qctx_outputs.append(q_ctx)  # (B, Q, D)   
                                    

        
        assert len(outputs) == len(blocks_to_take), f"only {len(outputs)} / {len(blocks_to_take)} blocks found"
        
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.model.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
            
        ##print("return_class_token:", return_class_token)
        if return_class_token:
            return list(zip(outputs, class_tokens, qctx_outputs))
        return list(zip(outputs, qctx_outputs))




class LinearClassifier(nn.Module):
    def __init__(self, out_dim, use_n_blocks=1, use_avgpool=False, num_classes=7, 
    		 num_queries=7,
                 pool_type="mean",
                 pool_queries=0):
        super().__init__()
        self.pool_type   = pool_type
        self.use_avgpool = use_avgpool
        self.use_n_blocks = use_n_blocks
        self.num_classes = num_classes

        if use_avgpool:
            self.avgpool = nn.AdaptiveAvgPool1d(1) # AvgPool over the sequence dimension
        self.fc = nn.Linear(out_dim, num_classes) # linear layer for classification
        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()
        
        #if pool_queries:
        #    # reduce Q×D → D_q
        #    self.query_proj = nn.Linear(num_queries, out_dim)
                  
    def forward(self, features):
        # features: list of tuples [(patch_tokens, class_token), ...] or a single tuple if n=1
        if self.use_n_blocks > 1: # linear4 case
            patch_tokens_list = [f[0] for f in features]        # extract patch tokens from each block
            patch_tokens = torch.cat(patch_tokens_list, dim=-1) # concatenate patch tokens along the embedding dimension
        else: # linear case
            patch_tokens = features[0][0] # take the first block and its patch_tokens

        ## print("features.shape", len(features), len(features[0]), features[0][0].shape, features[0][1].shape )
        ## features.shape 4 2 torch.Size([32, 256, 1536]) torch.Size([32, 1536])

        
        if self.use_avgpool:
            ## patch_tokens are [B, L, D] (Batch, Sequence Length, Dimension)
            ## print("patch_tokens.shape(1):", patch_tokens.shape)
            ## torch.Size([32, 256, 6144])
            
            patch_tokens = patch_tokens.transpose(1, 2) # [B, D, L] for AvgPool1d
            
            ## print("patch_tokens.shape(2):", patch_tokens.shape)
            ## patch_tokens.shape(2): torch.Size([32, 6144, 256])

            pooled_features = self.avgpool(patch_tokens).squeeze(-1) # [B, D, 1] -> [B, D]
        else:
            pooled_features = patch_tokens[:, 0] # Use the CLS token if no avgpool

        ## print("pooled_features.shape", pooled_features.shape)
        ## pooled_features.shape torch.Size([32, 6144])

        logits = self.fc(pooled_features) ##torch.softmax()
        return logits
        
    def pool_q(self, q_ctx):
        """
        q_ctx : (B, Q, D)
        → (B, Q*D)   (concatenate)  or  (B, D) (mean)
        """
        if self.pool_type == "concat":
            return q_ctx.flatten(1)
        else:  # "mean"
        
            return q_ctx.mean(dim=1)

    def pred_queries(self, features):
        """
        features = [(patch_tokens, q_ctx), ...]
        """
        patch_tokens, q_tokens = [], []
        
        ##print(len(features))
        ##print(len(features[0]))
        ##print(features)

        for patch,_, q_ctx in features:
            ##print(patch.shape, q_ctx.shape) ## torch.Size([16, 256, 1536]) torch.Size([16, 7, 1536])
            patch_tokens.append(patch)          # (B, L, D)
            q_tokens.append(self.pool_q(q_ctx)) # (B, D)  or (B, Q*D)

        # --- fuse across blocks (if linear4) ---------------------------
        patch_feat = torch.cat(patch_tokens, dim=-1) if self.use_n_blocks > 1 else patch_tokens[0]
        q_feat     = torch.cat(q_tokens,   dim=-1) if self.use_n_blocks > 1 else q_tokens[0]

        if self.use_avgpool:
            patch_feat = patch_feat.transpose(1, 2)      # (B, D, L)
            ##print(patch_feat.shape)
            patch_feat = self.avgpool(patch_feat).squeeze(-1)  # (B, D)

        # ---- simple late fusion: add or concat ------------------------
        ##print(patch_feat.shape) ## [B,D]
        ##print(self.query_proj(q_feat).shape) ## [B,D]
        ##exit()
        fused = patch_feat +  q_feat ##self.query_proj(q_feat)      # element-wise add
        ##print(fused.shape)
        logits = self.fc(fused)
        return logits


        
class _LoRA_qkv(nn.Module):
    """In Dinov2 it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim:] += new_v
        return qkv


class SurgicalDINOCLassifier(nn.Module):
    """Applies low-rank adaptation to Dinov2 model's image encoder.

    Args:
        backbone_size: the pretrained size of dinov2 model
        r: rank of LoRA
        image_shape: input image shape
        decode_type: the decode type of decode head, "linear" or ""

    """

    def __init__(self, backbone_size="base", 
                 r=4, 
                 image_shape=(224, 224), 
                 decode_type='linear', 
                 lora_layer=None, 
                 num_classes=7, 
                 use_avgpool=True,
                 num_extra_queries=7,
                 debug=False):
        
        super(SurgicalDINOCLassifier, self).__init__()
        self.debug = debug

        assert r > 0
        self.backbone_size = backbone_size
        self.backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }
        self.intermediate_layers = {
            "small": [2, 5, 8, 11],
            "base": [2, 5, 8, 11],
            "large": [4, 11, 17, 23],
            "giant": [9, 19, 29, 39],
        }
        self.embedding_dims = {
            "small": 384,
            "base" : 768,
            "large": 1024,
            "giant": 1536,
        }
        self.backbone_arch = self.backbone_archs[self.backbone_size]
        self.n = self.intermediate_layers[self.backbone_size] if decode_type == 'linear4' else 1 
        self.embedding_dim = self.embedding_dims[self.backbone_size]
        
        self.backbone_name = f"dinov2_{self.backbone_arch}"
        ##num_extra_queries = 7  # Queries for classes
        dinov2 = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=self.backbone_name)
            
        self.image_shape = image_shape
        self.decode_type = decode_type
        
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(
                range(len(dinov2.blocks)))  # Only apply lora to the image encoder by default
        
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []
        # freeze first
        for param in dinov2.parameters():
            param.requires_grad = False
        
        ##"""
        ## Initialize LORA Layers
        for t_layer_i, blk in enumerate(dinov2.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        self.reset_parameters()
        ##"""
        
        self.num_extra_queries = num_extra_queries
        if num_extra_queries:
            dinov2 = DINOv2WithLearnableQueries(dinov2, num_extra_queries=num_extra_queries )
        self.dinov2 = dinov2

        print("self.decode_type:",self.decode_type)
                    
        # Initialize the LinearClassifier decode head
        if self.decode_type == 'linear':
            n_blocks = 1
        elif self.decode_type == 'linear4':
            n_blocks = 4
        else:
            raise ValueError("Unsupported decode_type. Use 'linear' or 'linear4'.")
        
        # Compute output dimension based on number of blocks and whether avgpooling is used
        out_dim = self.embedding_dim * n_blocks #+ (self.embedding_dim if use_avgpool else 0)
        self.linear_head = LinearClassifier(out_dim=out_dim, use_n_blocks=n_blocks,
                                            use_avgpool=use_avgpool, num_classes=num_classes, pool_queries = num_extra_queries )


        
        # Set the classification loss function (cross-entropy)
        self.loss_fn = nn.CrossEntropyLoss()


    def forward(self, pixel_values, mask=None):
        """
        Forward pass for classification.
        
        Args:
            pixel_values: Input images.
            labels: Ground truth class labels as a LongTensor.
            
        Returns:
            A dictionary with 'logits' and 'loss' (if labels are provided).
        """
        ## Refer
        ##https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L298
        
        #print(pixel_values.shape) ## torch.Size([16, 3, 224, 224])  ## (B, 3, W, H)
        
        ## features.shape : len(tuple) : (1 (2))  ## patch_tokens : torch.Size([16, 256, 768]) , class_token : torch.Size([16, 768])
        ## Extract intermediate features: list of tuples [(patch_tokens, class_token), ...] 
        
        ##pdb.set_trace()
        if self.num_extra_queries:
            ##print("self.debug:",self.debug)
            features = self.dinov2.get_intermediate_layers(pixel_values, n=self.n,
                                                       reshape=False, return_class_token=True, norm=False, seg_mask=mask, debug=self.debug) ##reshape = True
                                                       
            ###print("here")
            # Obtain logits from the linear classifier head
            logits = self.linear_head.pred_queries(features)
            
        else:
            features = self.dinov2.get_intermediate_layers(pixel_values, n=self.n,
                                                       reshape=False, return_class_token=True, norm=False) ##reshape = True
                                                       
            # Obtain logits from the linear classifier head
            logits = self.linear_head(features)

        return logits, features
        
        
    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)
        
    def save_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        linear_head_tensors = {}

        # save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
        if isinstance(self.linear_head, torch.nn.DataParallel) or isinstance(self.linear_head, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.linear_head.module.state_dict()
        else:
            state_dict = self.linear_head.state_dict()
        for key, value in state_dict.items():
            linear_head_tensors[key] = value

        merged_dict = {**a_tensors, **b_tensors, **linear_head_tensors}
        torch.save(merged_dict, filename)

        print('saved lora parameters to %s.' % filename)

    def load_parameters(self, filename: str, device: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        state_dict = torch.load(filename, map_location=device)

        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = Parameter(saved_tensor)

        linear_head_dict = self.linear_head.state_dict()
        linear_head_keys = linear_head_dict.keys()

        # load linear head
        linear_head_keys = [k for k in linear_head_keys]
        linear_head_values = [state_dict[k] for k in linear_head_keys]
        linear_head_new_state_dict = {k: v for k, v in zip(linear_head_keys, linear_head_values)}
        linear_head_dict.update(linear_head_new_state_dict)

        self.linear_head.load_state_dict(linear_head_dict)

        print('loaded lora parameters from %s.' % filename)


