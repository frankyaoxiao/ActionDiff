<h1 align='center'>ActionDiff: Diffusion-Based Action Recognition Generalizes to Untrained Domains</h1>
<div align='center'>
    Rogério Guimarães<sup>1*</sup> 
    Frank Xiao<sup>1*</sup> 
    Pietro Perona<sup>1</sup> 
    Markus Marks<sup>1</sup>
  </div>
  <div align='center'>
    <sup>1</sup>California Institute of Technology
  </div>
  <br>
  <div align="center">
    <a href="https://vision.caltech.edu/actiondiff/"><img src="https://img.shields.io/static/v1?label=Project&message=Page&color=purple"></a>  
    <a href="https://arxiv.org/pdf/2509.08908"><img src="https://img.shields.io/static/v1?label=Paper&message=ArXiv&color=red&logo=arxiv"></a>  
    <a href="https://github.com/frankyaoxiao/ActionDiff"><img src="https://img.shields.io/static/v1?label=Code&message=GitHub&color=blue&logo=github"></a>
  </div>

*Equal contribution


## Prerequisites
- Python 3.10.0
- CUDA-compatible GPU

## Installation

### Quick Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/rogerioagjr/extract_svd_feats
   cd extract_svd_feats
   ```

2. Run the automated setup script:
   ```bash
   chmod +x setup_sm90.sh
   ./setup_sm90.sh
   ```

### Manual Setup
1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up generative-models:
   ```bash
   cd generative-models
   pip install .
   pip install -r requirements/pt2.txt
   pip install 'numpy<2'
   ```

3. Download model checkpoints:
   ```bash
   mkdir -p checkpoints
   cd checkpoints
   wget https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/resolve/main/svd.safetensors
   wget https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt.safetensors
   cd ../..
   ```

## Data Preparation

Datasets should be placed under `generative-models/data`. Each dataset has its own annotation and data style, please refer to the corresponding dataset class for details.

## Usage

All experiments can be reproduced by running the corresponding bash script in the `generative-models/configs` directory.

## Results

The following table summarizes the key experimental results from our paper "Diffusion-Based Action Recognition Generalizes to Untrained Domains":

| Dataset | Train → Test | Model | Best mAP/Acc | Bash Command |
|---------|--------------|-------|--------------|---------------|
| Animal Kingdom | Full Dataset | ActionDiff | 80.79 mAP | `bash configs/transformergrid/ak.bash` |
| Animal Kingdom | Unseen Species | ActionDiff | 51.49 acc | `bash configs/transformergrid/akunseen.bash` |
| Charades-Ego | 1st → 1st | ActionDiff | 36.5 mAP | `bash configs/transformergrid/charades1-1.bash` |
| Charades-Ego | 3rd → 1st | ActionDiff | 30.2 mAP | `bash configs/transformergrid/charades3-1.bash` |
| UCF-HMDB | UCF → HMDB | ActionDiff | 77.6 acc | `bash configs/transformergrid/ucfhmdb.bash` |
| UCF-HMDB | HMDB → UCF | ActionDiff | 81.5 acc | `bash configs/transformergrid/hmdbucf.bash` |
| Animal Kingdom | Full Dataset | VideoMAEv2 | 63.90 mAP | `bash configs/videoMAE/animalkingdom.bash` |
| Animal Kingdom | Unseen Species | VideoMAEv2 | 37.25 acc | `bash configs/videoMAE/animalkingdom_unseen.bash` |
| Charades-Ego | 1st → 1st | VideoMAEv2 | 16.9 mAP | `bash configs/videoMAE/charades1-1.bash` |
| Charades-Ego | 3rd → 1st | VideoMAEv2 | 15.3 mAP | `bash configs/videoMAE/charades3-1.bash` |
| UCF-HMDB | UCF → HMDB | VideoMAEv2 | 23.8 acc | `bash configs/videoMAE/ucfhmdb.bash` |
| UCF-HMDB | HMDB → UCF | VideoMAEv2 | 26.7 acc | `bash configs/videoMAE/hmdbucf.bash` |
| Animal Kingdom | Full Dataset | V-JEPA | 78.64 mAP | `bash configs/vjepa/animalkingdom.bash` |
| Animal Kingdom | Unseen Species | V-JEPA | 51.40 acc | `bash configs/vjepa/animalkingdom_unseen.bash` |
| Charades-Ego | 1st → 1st | V-JEPA | 23.4 mAP | `bash configs/vjepa/charades1-1.bash` |
| Charades-Ego | 3rd → 1st | V-JEPA | 17.3 mAP | `bash configs/vjepa/charades3-1.bash` |
| UCF-HMDB | UCF → HMDB | V-JEPA | 51.75 acc | `bash configs/vjepa/ucfhmdb.bash` |
| UCF-HMDB | HMDB → UCF | V-JEPA | 60.58 acc | `bash configs/vjepa/hmdbucf.bash` |
| Animal Kingdom | Full Dataset | SDv2 | 78.66 mAP | `bash configs/sdv2/animalkingdom.bash` |
| Animal Kingdom | Unseen Species | SDv2 | 41.96 acc | `bash configs/sdv2/animalkingdom_unseen.bash` |
| Charades-Ego | 1st → 1st | SDv2 | 35.0 mAP | `bash configs/sdv2/charades1-1.bash` |
| Charades-Ego | 3rd → 1st | SDv2 | 29.4 mAP | `bash configs/sdv2/charades3-1.bash` |
| UCF-HMDB | UCF → HMDB | SDv2 | 76.5 acc | `bash configs/sdv2/ucfhmdb.bash` |
| UCF-HMDB | HMDB → UCF | SDv2 | 79.5 acc | `bash configs/sdv2/hmdbucf.bash` |


## Feature Extraction

To reproduce the results in our paper, you need to extract SVD features from the Stable Video Diffusion XT model. Each dataset has its own extraction subfolder containing scripts for different timesteps.

### Quick Start (t=20)

For the most commonly used timestep (t=20), run:

```bash
# UCF-101
bash scripts/extract_ucf101_svd_xt_grid_search/extract_ucf101_svd_xt_grid_search_7.sh

# HMDB-51  
bash scripts/extract_hmdb_svd_xt_grid_search/extract_hmdb_svd_xt_grid_search_7.sh

# Charades-Ego
bash scripts/extract_charades_ego_svd_xt_grid_search/extract_charades_ego_svd_xt_grid_search_7.sh

# Animal Kingdom
bash scripts/extract_animal_kingdom_svd_xt_grid_search/extract_animal_kingdom_svd_xt_grid_search_7.sh
```

### Complete Grid Search

To run all timesteps for each dataset, execute all scripts in the respective folders:

```bash
# UCF-101 (7 scripts: t=0,5,10,15,25,29,20)
bash scripts/extract_ucf101_svd_xt_grid_search/*.sh

# HMDB-51 (7 scripts: t=0,5,10,15,25,29,20)
bash scripts/extract_hmdb_svd_xt_grid_search/*.sh

# Charades-Ego (6 scripts: t=0,5,10,15,25,29)
bash scripts/extract_charades_ego_svd_xt_grid_search/*.sh

# Animal Kingdom (6 scripts: t=0,5,10,15,25,29)
bash scripts/extract_animal_kingdom_svd_xt_grid_search/*.sh
```

### Extraction Parameters

All scripts use:
- **Model**: Stable Video Diffusion XT (`svd_xt`)
- **Frames**: 25 frames per video
- **Feature stride**: 4




