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


class DINOv2WithLearnableQueries(nn.Module):
    def __init__(self, base_model: nn.Module, num_extra_queries: int, visual_prompt_learning=False, mask_prompting=False, up_embed_dim=256):
        super().__init__()
        self.model = base_model
        self.num_extra_queries = num_extra_queries
        self.q = nn.Parameter(torch.randn(num_extra_queries, base_model.embed_dim))
        nn.init.trunc_normal_(self.q, std=0.02)
        
        ##self.class_head = nn.Linear(self.model.backbone.embed_dim, num_extra_queries)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=base_model.embed_dim, num_heads=2, batch_first=True
        )
        
        self.with_mask_attn=mask_prompting;
        D = base_model.embed_dim
        
        # Mask attention embedding dimension transform: 768 → 256 → 768
        if self.with_mask_attn:
            self.channel_up = nn.Conv2d(D, up_embed_dim, kernel_size=1)
            self.channel_down = nn.Conv2d(up_embed_dim, D, kernel_size=1)
            
            self.spatial_up = nn.Sequential(
		    nn.Conv2d(D, D, kernel_size=5, stride=4, padding=1),  # 16 → 64
		    nn.ReLU(),
		    nn.Conv2d(D, D, kernel_size=3, stride=2, padding=1),  # 64 → 128
		    nn.ReLU(),
		    nn.Upsample(size=(224, 224), mode="nearest")          # final 128 → 224
            )
            
            self.spatial_down = nn.Sequential(
                nn.Conv2d(D, D, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((16, 16))
            )
        
        D = base_model.embed_dim
        self.bottleneck = visual_prompt_learning

        # ----- 1×1-conv residual block (Conv-GELU-Conv) ------------------- #
        if self.bottleneck:
            self.fuse_conv = nn.Sequential(
		    nn.Conv1d(D, D, kernel_size=1),
		    nn.GELU(),
		    nn.Conv1d(D, D, kernel_size=1),
            )
    
    
    def mask_attn(self, x, seg_mask):
    
        cls, patches = (
                       x[:, :1],         # (B,1,D)
                        x[:, 1:]         # (B,L=256,D)
                    )
        B, L, D = patches.shape
        H = W = int(L ** 0.5)
                    
        ##pdb.set_trace()
        ##print("seg_mask.shape",seg_mask.shape )

        patches_2d = patches.permute(0, 2, 1).reshape(B, D, H, W)     # (B, 768, 16, 16)

        # Spatial upsample
        patches_upsampled = self.spatial_up(patches_2d)              # (B, 768, 224, 224)

        # Upsample: Embedding Dim to Projection Dim 
        patches_highdim = self.channel_up(patches_upsampled)  # (B, 256, 224, 224)

        # Apply feature masking
        mask_feats = patches_highdim * seg_mask  # (B, 256, 224, 224)

        # Channel downsample: Projection Dim → Embedding Dim
        mask_feats = self.channel_down(mask_feats)  # (B, 768, 224, 224)

        # Spatial downsample
        patches_masked = self.spatial_down(mask_feats).flatten(2).permute(0, 2, 1)   # (B, 768, 16, 16) --> # (B, 256, D)

        return torch.cat([cls, patches_masked], dim=1)
        
    def get_intermediate_layers(self,
        x: torch.Tensor,
        n: int,
        seg_mask=None,
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True):
        
        ##pdb.set_trace();
        B = x.size(0)
        
        gate = 0.3
        
        x = self.model.prepare_tokens_with_masks(x)

        blocks_to_take = range(len(self.model.blocks) - n, len(self.model.blocks)) if isinstance(n, int) else n
        learnable_queries = self.q.unsqueeze(0).expand(B, -1, -1)  # (B, Q, D)

        outputs, qctx_outputs = [], []

        for i, blk in enumerate(self.model.blocks):

            x = blk(x)
            
            ########################################## M A BLOCK #################################################
            if self.with_mask_attn and seg_mask is not None:
                m_x = self.mask_attn(x, seg_mask)
                x   = gate * m_x + (1 - gate) * x 
                                
            # cross-attention between learnable queries
            q_ctx, _ = self.cross_attn(
            	query=learnable_queries, key=x, value=x, need_weights=False
            )                                         # (B, Q, D)
                
                
            ##class_logits = self.class_head(q_ctx)
            if self.bottleneck:                              # Optional Resnet Connection
                x = x + self.residual_from_q(q_ctx, L=x.size(1))
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

    def residual_from_q(self, q_ctx: torch.Tensor, L: int) -> torch.Tensor:
        """
        q_ctx : (B, Q, D)  →  upsample to (B, L, D) with nearest-neighbor,
        pass through Conv-GELU-Conv, return in (B, L, D) layout.
        """
        add = q_ctx.permute(0, 2, 1)             # (B, D, Q)
        add = F.interpolate(add, size=L, mode="nearest")  # (B, D, L)
        add = self.fuse_conv(add)                # (B, D, L)
        return add.permute(0, 2, 1)              # (B, L, D)




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
                 num_extra_queries=0,
                 visual_prompt_learning=False,
                 mask_prompting=False):
        
        super(SurgicalDINOCLassifier, self).__init__()

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
            dinov2 = DINOv2WithLearnableQueries(dinov2, num_extra_queries=num_extra_queries, visual_prompt_learning=visual_prompt_learning, mask_prompting=mask_prompting )
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
            features = self.dinov2.get_intermediate_layers(pixel_values, n=self.n,
                                                       reshape=False, return_class_token=True, norm=False, seg_mask=mask) ##reshape = True
                                                       
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

