

## LW-SRCNN: Lightweight Super-Resolution Convolutional Neural Network

LW-SRCNN (Lightweight Super-Resolution Convolutional Neural Network) is a deep learning model designed for efficient and high-quality single image super-resolution (SISR). This repository provides code, pre-trained models, and instructions for training and evaluating LW-SRCNN on standard image datasets.

---

## Overview

Single image super-resolution aims to reconstruct a high-resolution (HR) image from a low-resolution (LR) input. While deep learning models like SRCNN have achieved impressive results, their computational cost limits real-time and resource-constrained applications. LW-SRCNN addresses this by introducing a compact, hourglass-shaped CNN architecture that accelerates inference while maintaining or improving restoration quality compared to classical SRCNN models[1].

Key features:
- **End-to-end learning**: Direct mapping from LR to HR images without pre-interpolation.
- **Compact architecture**: Uses shrinking and expanding layers to reduce computational complexity.
- **Deconvolution upsampling**: Learns upsampling kernels, replacing fixed interpolation.
- **Real-time performance**: Achieves significant speedup, enabling real-time super-resolution on standard CPUs[1].

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

---

## Architecture

LW-SRCNN builds on the principles of FSRCNN, introducing several optimizations:
- **Feature Extraction**: Initial convolutional layer extracts features from the original LR image.
- **Shrinking Layer**: Reduces feature dimensionality for efficient mapping.
- **Mapping Layers**: Multiple 3x3 convolutional layers for non-linear mapping in a low-dimensional space.
- **Expanding Layer**: Restores feature dimensionality before reconstruction.
- **Deconvolution Layer**: Learns upsampling filters for HR image reconstruction[1].

**Activation:** Parametric ReLU (PReLU) is used to avoid dead features and improve model capacity.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/lwsrcnn.git
   cd lwsrcnn
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Download pre-trained weights from the releases section.

---

## Usage

### Super-Resolve an Image

```python
python main.py --input path/to/low_res.png --output path/to/high_res.png --model lwsrcnn
```

### Model Parameters

- `--scale`: Upscaling factor (e.g., 2, 3, 4)
- `--model`: Model type (default: lwsrcnn)
- `--weights`: Path to pre-trained weights

---

## Training

To train LW-SRCNN from scratch:

1. Prepare training data (paired LR-HR images).
2. Run the training script:
   ```python
   python train.py --dataset path/to/dataset --scale 3 --epochs 100
   ```

**Recommended datasets:** [Set5, Set14, BSD200][1]

---

## Evaluation

Evaluate the model on benchmark datasets:

```python
python evaluate.py --dataset path/to/testset --scale 3 --model lwsrcnn --weights path/to/weights.pth
```

Metrics reported: PSNR, SSIM

---

## Results

LW-SRCNN achieves a speedup of over 40x compared to SRCNN, with equal or superior PSNR on standard benchmarks. The model can run in real-time (24+ fps) on generic CPUs for common image sizes[1].

| Model    | PSNR (Set5, x3) | Time (s) | Parameters |
|----------|-----------------|----------|------------|
| SRCNN    | 32.83           | 1.32     | 8,032      |
| LW-SRCNN | 33.06           | 0.027    | 12,464     |

*See the paper and supplementary materials for detailed results and ablation studies.*

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

