# ECE 285 Homework 01 — Variational Autoencoder for Chest X-ray Generation

A convolutional Variational Autoencoder (VAE) implemented in PyTorch and trained on the
[Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
dataset to generate 64×64 chest radiographs. Generated images are evaluated with the **Frechet
Inception Distance (FID)** and the **Inception Score (IS)** using the
[pytorch-ignite reference implementation](https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/).

## Contents

| File | Description |
|---|---|
| `hw01.py` | Complete self-contained script: training, generation, FID/IS evaluation, figure export |
| `ECE_285_HW01_Report.pdf` | Two-page report |
| `results/` | Metrics (`*.csv`, `metrics_summary.json`) and figures from the 50-epoch run |

## Requirements

Python 3.10+ with the following packages:

```
torch  torchvision  pytorch-ignite  numpy  pandas  matplotlib  pillow  scipy
```

`pytorch-ignite` provides the FID and Inception Score implementations, and `scipy` is required by
its FID metric for the matrix square root.

## Dataset

Download the dataset from Kaggle and place it next to `hw01.py`. Either layout works:

```
chest_xray/
    train/{NORMAL,PNEUMONIA}/*.jpeg
    test/{NORMAL,PNEUMONIA}/*.jpeg
```

or the nested form that the Kaggle archive extracts to:

```
chest_xray/chest_xray/
    train/{NORMAL,PNEUMONIA}/*.jpeg
    test/{NORMAL,PNEUMONIA}/*.jpeg
```

The loader locates the directory that directly contains `train/` and `test/`, so no path editing
is required. The 16-image official `val/` split is **ignored** — see *Method* below.

## Usage

```bash
python hw01.py
```

One command runs the whole pipeline: training → FID/IS evaluation → figures → export. On an Apple
Silicon GPU (MPS) the full 50 epochs take about 10 minutes.

Outputs:

```
results/
    figures/   loss_curves.png, fid_is_curve.png, generated_samples.png, reconstructions.png
    samples/   epoch_XXX.png  (64-sample grid every 5 epochs)
    training_history.csv, fid_is_history.csv, metrics_summary.json
checkpoints/
    best.pt    checkpoint with the lowest FID
    last.pt    final epoch
```

To change hyperparameters, edit the `Config` dataclass at the top of `hw01.py`. Every field is
serialised into each checkpoint, so a saved model can be rebuilt with `load_vae()` without
consulting the source.

## Method

**Model.** A symmetric convolutional VAE with 4 down/up-sampling stages (4×4 kernels, stride 2,
channel widths 32→64→128→256) and a 128-dimensional latent space, 2.96 M parameters in total. The
decoder ends in a sigmoid, matching the `[0, 1]` support of the data.

**Objective.** Negative ELBO per image in nats:

```
L = E_q[-log p(x|z)]  +  β · KL(q(z|x) || N(0, I))
```

with a factorised Bernoulli likelihood (BCE summed over pixels, divided by the batch size) and a
closed-form Gaussian KL. `β` is ramped linearly from 0 to 1 over the first 10 epochs, the standard
remedy for early posterior collapse.

**Data.** 5,863 AP/PA radiographs, officially split 5,216 / 16 / 624. Because the official
validation split has only 16 images, it is discarded and a **class-stratified 90/10 split** of the
official training set is used instead: 4,694 train / 522 validation images, preserving the 74.3 %
pneumonia prevalence. The 624-image test set is never optimised on; a fixed 512-image subset serves
as the *real* reference distribution for FID and IS.

No random horizontal flips are applied — chest radiographs are left/right asymmetric, so mirroring
would manufacture anatomically impossible images.

**Metrics.** FID uses the 2048-d global-pooling features of an ImageNet-pretrained InceptionV3, IS
uses its 1000-d softmax output. Grayscale inputs are replicated to three channels and resized to
299×299 with **PIL bilinear** interpolation rather than `F.interpolate`, because aliasing in the
latter is known to bias FID ([Parmar et al., 2022](https://arxiv.org/abs/2104.11222)).

## Results

Trained for 50 epochs on an Apple Silicon GPU (MPS).

| Metric | VAE (this work) | Real images (reference) |
|---|---|---|
| Inception Score ↑ | 1.618 | 2.289 |
| FID ↓ | 343.51 | 66.30 |

- Final objective **2388.75 nats/image** (0.583 nats/pixel), of which only **24.08 nats** (34.7 bits) are KL.
- Best FID **341.49** at epoch 45; reconstruction MSE **0.0034** on validation images.
- Note the FID floor: two *disjoint halves* of the real test set are already 66.30 apart, so the
  model's 343.51 should be read as 5.2× that floor rather than against zero.

Samples reproduce the global anatomy of a chest X-ray — rib cage, lung fields, mediastinal column,
diaphragm — but lack high-frequency detail. This is a property of the objective rather than of the
optimisation: under a Bernoulli likelihood the optimal prediction is the conditional mean of each
pixel given `z`, so averaging over residual uncertainty necessarily yields a smooth image. The
24-nat KL term (≈35 bits over 128 dimensions) is the other half of the explanation — the code
carries very little information about a 4,096-pixel image.

## Implementation notes

Three issues are easy to get wrong on this stack and are handled explicitly:

1. **Device split for the metrics.** `pytorch-ignite` accumulates FID/IS statistics in `float64`,
   which Apple MPS does not implement. The metrics therefore live on the CPU while the
   27 M-parameter InceptionV3 forward pass stays on the GPU; only the `(N, 2048)` feature matrix
   crosses devices. Ignite unconditionally calls `feature_extractor.to(metric_device)`, so a small
   `DeviceLockedExtractor` wrapper keeps the network where it was built.

2. **`InceptionScore` input.** It consumes only the generated images, while `FID.update` expects
   `(generated, real)` in that order.

3. **DataLoader workers.** The Inception preprocessing uses `Grayscale(num_output_channels=3)`
   rather than a `T.Lambda`, because lambdas are not picklable and break `num_workers > 0`.

## License

Coursework submission for ECE 285. Not intended for clinical use.
