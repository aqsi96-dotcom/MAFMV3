# MAFMv3

Implementation of **MAFMv3: An automated Multi-Scale Attention-Based Feature Fusion MobileNetv3 for spine lesion classification** (Image and Vision Computing, 2025).

## Architecture

- **MobileNetV3-Large** backbone
- **CBAM** (channel + spatial attention) on multi-stage features
- **ASPP** with dilations `{1, 6, 12, 18}` for multi-scale context
- **Three-stream fusion** into a joint feature map:
  - default `preprocess` mode: raw / normalized / histogram-equalized
  - optional `views` mode: axial / coronal / sagittal
- Conv head + fully connected classifier

## Setup

```bash
cd MAFMv3
pip install -r requirements.txt
```

## Data (VinDr-SpineXR style)

```
data/vindr_spinexr/train/normal/*.png
data/vindr_spinexr/train/abnormal/*.png
data/vindr_spinexr/val/normal/*.png
data/vindr_spinexr/val/abnormal/*.png
```

Public dataset: [VinDr-SpineXR](https://www.physionet.org/content/vindr-spinexr/1.0.0/)

Update `num_classes` / `class_names` in `configs/default.yaml` for multi-class labels.

## Train

```bash
# smoke test
python train.py --dummy-data --epochs 2 --batch-size 2

# full training (paper defaults: batch 32, Adam 1e-3, dropout 0.3, 100 epochs)
python train.py --config configs/default.yaml
```

## Inference

```bash
python predict.py --checkpoint checkpoints/mafmv3_best.pt --image path/to/xray.png
```

## Paper metrics (VinDr-SpineXR, with HE + augmentation)

| Metric    | Value   |
|-----------|---------|
| Accuracy  | 96.81%  |
| Precision | 98.38%  |
| Recall    | 97.95%  |
| F1-score  | 98.15%  |
| AUC       | 99.98%  |
| FNR       | 2.05%   |
