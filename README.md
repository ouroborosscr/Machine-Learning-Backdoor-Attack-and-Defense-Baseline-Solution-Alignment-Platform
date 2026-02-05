# Machine-Learning-Backdoor-Attack-and-Defense-Baseline-Solution-Alignment-Platform

## 📖 Introduction

This project is a comprehensive benchmark and alignment platform designed to standardize the evaluation of Backdoor Attacks and Defenses in Deep Learning. It aims to bridge the gap between disparate experimental settings, enabling researchers to fairly compare methods and accelerate the reproduction of state-of-the-art (SOTA) results.

## 🎯 Motivation

In the process of conducting research and submitting papers, we identified several critical **pain points** regarding the reproduction and comparison of existing backdoor metrics:

**Framework Heterogeneity**: Existing methods often rely on different deep learning frameworks (e.g., PyTorch vs. TensorFlow), making direct integration and comparison difficult.

**Misalignment of Models & Datasets**: Reproducing results on "similar" but not identical dataset specifications significantly increases the workload for subsequent researchers and introduces confounding variables.

**Implementation Discrepancies**: Even when using the same model architecture, differences in internal node definitions or layer naming conventions cause significant difficulties in transferability and code reuse.

**Our Solution**: We provide a unified platform with aligned environments, standardized model definitions, and curated datasets to ensure fair and efficient benchmarking.

## 📊 Dataset Selection Strategy
Our choice of datasets and model structures is guided by three principles:

**Prevalence**: They are frequently used in top-tier academic papers.

**Diversity**: They vary in specifications (channel depth, resolution) to demonstrate the robustness of attacks/defenses across different scenarios.

**Applicability**: They reinforce the "real-world" relevance of the research, which is crucial for paper acceptance.

**Supported Datasets**
We have selected the following three datasets to cover a wide range of complexities:

| Dataset | Type | Dimensions | Classes | Purpose & Application |
| :--- | :--- | :--- | :--- | :--- |
| **MNIST** | Grayscale | `1 x 28 x 28` | 10 | **Rapid Validation.** Ideal for debugging and quick proof-of-concept experiments. |
| **GTSRB** | Color (RGB) | `3 x 32 x 32` | 43 | **Autonomous Driving.** German Traffic Sign Recognition Benchmark. Adds complexity with color and more classes. |
| **PUBFIG** | Color (RGB) | `3 x 224 x 224`* | 43 | **Face Recognition.** High-resolution validation adapted for large models (e.g., VGG16). |

* Note: The PUBFIG dataset in this framework is resized to 224x224 to fit standard VGG16 input requirements.

## 🚀 Getting Started
**Installation**
```
git clone https://github.com/ouroborosscr/Machine-Learning-Backdoor-Attack-and-Defense-Baseline-Solution-Alignment-Platform.git
```

## 🧩 Project Structure
```
.
├── Backdoor Attacks/   # Implementation of attack algorithms
│   └── BadNets
├── Backdoor Defenses/  # Implementation of defense algorithms
│   ├── SVD
│   ├── STRIP
│   └── NeuralCleanse
├── Demo Video/         # Demonstrate the usage of the WebUI through video
├── WebUI/              # Visualization Platform
└── README.md
```

## Demo Video


https://github.com/user-attachments/assets/2888e167-088f-442d-972d-d2c5e6e2bba1


https://github.com/user-attachments/assets/6ab99f96-a8d5-48b7-a71b-d1f12f92675e


https://github.com/user-attachments/assets/19ece644-c217-412c-99e8-80378a80abd6


https://github.com/user-attachments/assets/c24b11bd-5fe2-4e10-a9da-cbd14e86a0ce







## 📜 Citation
If you find this platform useful in your research, please consider citing our work:
```
@misc{sun2025isolatetriggerdetectingeliminating,
      title={Isolate Trigger: Detecting and Eliminating Adaptive Backdoor Attacks}, 
      author={Chengrui Sun and Hua Zhang and Haoran Gao and Shang Wang and Zian Tian and Jianjin Zhao and Qi Li and Hongliang Zhu and Zongliang Shen and Anmin Fu},
      year={2025},
      eprint={2508.04094},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2508.04094}, 
}
```
