# CP-SIS: Learnable Class Prototypes for Efficient Surgical Instrument Segmentation

This project provides a framework for fine-tuning DINOv2 models for the task of surgical instrument classification. 
The implementation uses Low-Rank Adaptation (LoRA) for efficient training and a novel query-guided attention mechanism to improve classification performance.

## Key Features

  * **Powerful DINOv2 Backbone**: Easily configurable to use different DINOv2 model sizes (`base`, `large`, `giant`).
  * **Efficient Fine-Tuning**: Utilizes Low-Rank Adaptation (LoRA) to significantly reduce the number of trainable parameters, enabling faster training with lower memory requirements.
  * **Query-Guided Classification**: Implements learnable "class queries" that interact with image features via cross-attention.
  * **Dataset Support**: Includes complete data loaders and evaluation scripts for the Endovis 2018 and Endovis 2017 datasets.
  * **Comprehensive Workflow**: Provides scripts for training, evaluation, and visualizing predictions on a single image.

## Directory Structure

To use this repository, your project should be organized with the following directory structure. The training and evaluation scripts assume that the dataset is located in a parallel `data` directory.

```
.
├── data/
│   └── endovis_2018/
│       ├── train/
│       │   ├── 0/  (Augmentation version 0)
│       │   │   ├── images/
│       │   │   ├── annotations/
│       │   │   └── binary_annotations/
│       │   │  
│       │   └── ... (Other augmentation versions)
│       └── val/
│           ├── images/
│           ├── annotations/
│           ├── annotations_b/
│           ├── binary_annotations_coco/
│           ├── binary_annotations_coco_b/
│           └── instance_pred_candidates/
│
├── CP_SIS/ (Project Root)
│   ├── dataset_dinov2.py
│   ├── surgical_dinov2_query_guidance.py
│   ├── surgical_dinov2.py
│   ├── surgical_dinov2_all_blocks.py
│   ├── train_dinov2_default_full_adamw_mask_attn_gt_verify_base_mask_query.py
│   ├── train_dinov2_default_full_adamw_mask_attn_gt_verify_base_mask_query_inference.py
│   ├── eval_single_image.py
│   ├── eval.py
│   ├── utils.py
│   ├── work_dirs/            # For saving checkpoints and logs
│   ├── segms_example_image/  # Example candidate masks for single image inference
│   └── visualization/        # Output for single image inference
│
└── requirements.txt
```

## Setup and Installation

1.  **Clone the Repository**

    ```bash
    git clone <your-repository-url>
    cd CP_SIS
    ```

2.  **Create a Virtual Environment** (Recommended)

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**

    This project requires several libraries. You can install them using the provided `requirements.txt` file.

    ```bash
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt
    ```

    **requirements.txt:**

    ```
    numpy
    opencv-python
    Pillow
    scikit-learn
    pycocotools
    tqdm
    seaborn
    matplotlib
    ```


## Data Preparation

The provided `Dataset` classes are built for the **EndoVis 2018** and **EndoVis 2017** datasets.

Download and extract Experiments.zip file in the root directory: https://drive.google.com/file/d/1YLiOv6ZAwX_rA_BdVHOjoJqdKbSdQIbb/view?usp=sharing

1.  Download the dataset and organize it according to the Directory Structure shown above.
2.  The code expects pre-processed data, including:
      * Original images.
      * Annotation masks.
      * Binary masks for each instrument instance.
      * Candidate instance masks generated from a separate detection/segmentation model (e.g., Mask R-CNN, SAM).

## Usage


### Evaluation

To run inference and evaluation using a trained checkpoint:

1.  Make sure you have a trained checkpoint file in the `work_dirs/` directory.
2.  Run the inference script. This script will load the model, perform predictions on the validation set, evaluate the results against the ground truth, and save the final prediction masks in an `output/` directory.

<!-- end list -->

To run inference on the full validation dataset 
```bash
python eval.py --checkpoint 2>&1 | tee inference_log.txt
```

### Single Image Inference & Visualization

The `eval_single_image.py` script allows you to visualize the model's predictions on a single image, given a directory of candidate instance masks.

1.  Place your input image (e.g., `frame.png`).
2.  Place the corresponding binary candidate masks (as `.png` files) in a directory (e.g., `segms_example_image/`).
3.  Run the script:

<!-- end list -->

```bash
python eval_single_image.py \
    --image_path /path/to/your/frame.png \
    --masks_dir ./segms_example_image/ \
    --checkpoint ./work_dirs/dinov2_LQ_LORA_ev18_default_full_adamw_Base/checkpoint_epoch_19.pth \
    --output_dir ./visualization/ \
    --model_size base
```

This will produce two output files in the `visualization` directory:

  * `prediction_mask_gray.png`: A grayscale mask where pixel values correspond to the predicted class ID.
  * `prediction_mask_color.png`: A colorized mask for easy visualization.

-----


### Training

The main training script trains the model, runs validation after each epoch, and saves checkpoints.

To start training on the Endovis 2018 dataset:

```bash
python train_dinov2_default_full_adamw_mask_attn_gt_verify_base_mask_query.py --dataset endovis_2018
```

**Optional Arguments:**

  * `--checkpoint`: Use this flag to load the latest checkpoint from the `work_dirs` directory and resume training.
