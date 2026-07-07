# SKETCH: Semantic Key-Point Conditioning for Long-Horizon Vessel Trajectory Prediction

<div>
    <a href="https://arxiv.org/abs/2601.18537"><img src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg"></a>
    <a href="https://github.com/LinyongGAN/SKETCH"><img src="https://img.shields.io/badge/github-repo-blue?logo=github"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/LICENSE-MIT-red"></a>
    </div>
<br>

It is the official repo for the paper: _SKETCH: Semantic Key-Point Conditioning for Long-Horizon Vessel Trajectory Prediction_ accepted by **ICML 2026** in Seoul, South Korea. 

## 📖 Introduction
It is a project for predicting vessel trajectories. Accurate vessel trajectory prediction is a fundamental capability for maritime intelligence, supporting collision avoidance, port operation optimization, search and rescue, and fuel-efficient voyage planning. Our well-trained model can also serve as an off-the-shelf model for trajectory understanding and be applied to various downstream tasks. 

We recast long-horizon vessel trajectory prediction as a hierarchical forecasting problem that explicitly models navigational intent. Our contributions are summarized as follows:

1. We identify the lack of explicit global intent modeling as a fundamental limitation of existing long-horizon vessel trajectory prediction methods.
   
2. We introduce the Next Key Point (NKP) as a semantic intent variable and condition it to trajectory prediction, separating global navigational decisions from local motion dynamics within a hierarchical framework.
   
3. We propose an efficient training strategy for NKP-conditioned forecasting, allowing the model to generalize to open-set navigational targets rather than depending on a fixed closed set of ports.

4.  Experiments on large-scale AIS datasets show state-of-the-art performance, particularly in long-horizon prediction.

## 🚀 Quick Start
Please replace the code with the tag `[TODO]`. We have contained the public dataset and checkpoints in the repo, but the private dataset will not be revealed.

## 📦 Installation
Run the bash code below:
```bash
conda create -n SKETCH python=3.9
conda activate SKETCH
pip install -r requirement.txt
```

### Evaluation
|Instruction|function|
|--|--|
|`python evaluate_final_dataloader_public.py`|Evaluate on a public dataset|
|`python evaluate_final_dataloader.py`|evaluate on private (or other dataset)|
|`python eval_sft.py`|Evaluate the accuracy on stage II|
### Inference
|Instruction|function|
|--|--|
|`python inference_final_public.py`|Derive the visualization results for the public dataset. PNG and HTML are provided. You can specify the number of figures you want to sample|
|`python inference_final.py`|Derive the visualization results from the private (or other) dataset. |
### Training/Fine-tuning
|Instruction|function|
|--|--|
|`python train.py`|Train the stage I|
|`python sft.py`|SFT the stage II|
### Data Preprocessing
Running this section is not necessary. We provided sufficient files for evaluation and inference. 

In the 'data_preprocessing' folder, we provided two sample scripts for data preprocessing and database preparation. We cannot provide the Excel file with NKPs, but we can provide a sample file in the 'data' folder. Please collect it by yourself if needed. The evaluation datasets have been given. 

We also provided the code for database preparation for stage II. We have provided a sample database file for evaluation and inference. You can specify or enlarge it by yourself. 

For both tools, please replace the path in the code with the `[TODO]` tag to process the data. 

## 📊 Dataset Requirement
The training dataset should be a CSV file with columns: mmsi (to differentiate vessels only), lat, lon, sog, cog, next_lat, next_lon.

The evaluation/inference dataset should be a CSV file with columns: mmsi (to differentiate vessels only), lat, lon, sog, cog. 

## 💻 Implementation Details

```
SKETCH/
├── README.md                          # Project documentation
├── requirements.txt                   # Project dependencies
├── LICENSE                            # License file
├── train.py                           # Stage I pre-training script
├── sft.py                             # Stage II supervised fine-tuning script
├── eval_sft.py                        # Evaluate Stage II model accuracy
├── evaluate_final_dataloader.py       # Evaluate final model on private dataset
├── evaluate_final_dataloader_public.py # Evaluate final model on public dataset
├── inference_final.py                 # Visualization inference on private dataset
├── inference_final_public.py          # Visualization inference on public dataset
├── horizon_wise.py                    # Evaluation script for different prediction horizons
├── enrolled_trajectory.npy            # Enrolled trajectory data (for semantic key point retrieval)
├── models/                            # Model definitions directory
│   ├── model_minimind.py              # Base model architecture definition
│   ├── model_minimind_sft.py          # Stage II supervised fine-tuning model
│   └── model_minimind_final.py        # Final complete prediction model
├── utils/                             # Utility functions directory
│   ├── process.py                     # Data processing and data loaders
│   ├── dataloader_public.py           # Public dataset data loader
│   ├── earth_computation.py           # Earth science computation tools (coordinate conversion, distance calculation, etc.)
│   ├── metrics.py                     # Evaluation metric calculation (Frechet distance, curvature, etc.)
│   └── visualization.py               # Visualization tools (trajectory plotting)
├── data/                              # Data directory
│   ├── CapacityLargeModel_NKP_108.xlsx # Semantic key point data example
│   └── ne_10m_coastline/              # Coastline data (for visualization)
├── data_preprocessing/                # Data preprocessing directory
│   ├── data_processing.ipynb          # Data processing notebook
│   └── database_preperation.ipynb     # Database preparation notebook
├── data_1_13/                         # Public Dataset folder
├── demonstrations/                    # Demonstration/example directory
├── weights_pretrain/                  # Stage I pre-trained weights storage directory
└── weights_sft_new/                   # Stage II fine-tuned weights storage directory
```

## 🙏 Acknowledgement

We utilized the [MiniMind Model](https://github.com/jingyaogong/minimind) as the baseline. [ne_10m_coastline](https://github.com/nvkelso/natural-earth-vector/tree/master/10m_physical) was utilized to derive the distance from each coordinate to the nearest coastline. 

## 📚 Citations

```bibtex
@misc{gan2026sketchsemantickeypointconditioning,
      title={SKETCH: Semantic Key-Point Conditioning for Long-Horizon Vessel Trajectory Prediction}, 
      author={Linyong Gan and Zimo Li and Wenxin Xu and Xingjian Li and Jianhua Z. Huang and Enmei Tu and Shuhang Chen},
      year={2026},
      eprint={2601.18537},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2601.18537}, 
}
```
