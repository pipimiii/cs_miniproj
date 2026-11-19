# Self-Supervised Representation Learning on MNIST and CIFAR-10

Course project for Deep Learning on Computation (236781), Technion, winter semester 2024-2025.

The project compares three ways to train a convolutional image encoder in PyTorch. It then measures how useful the learned features are for classification.

## Methods

1. **Autoencoder pretraining.** A convolutional autoencoder learns to reconstruct each image with an L1 loss. A classifier is then trained on top of the frozen encoder.
2. **Supervised baseline.** The encoder and the classifier are trained together with cross entropy loss.
3. **Supervised contrastive learning (SupCon).** The encoder sees two augmented views of each image. It learns embeddings in which images of the same class are close together (temperature 0.07). A classifier is then trained on top of the frozen encoder.

The contrastive views use random resized crops, random affine transforms and Gaussian blur. The latent spaces are visualised with t-SNE.

## Results

Test accuracy:

| Method | MNIST | CIFAR-10 |
|---|---|---|
| Autoencoder, classifier on frozen encoder | 97.95% | 59.71% |
| Supervised baseline | 99.10% | 80.56% |
| Supervised contrastive, classifier on frozen encoder | **99.26%** | 80.27% |

Test reconstruction loss of the autoencoder: 0.0095 on MNIST and 0.0512 on CIFAR-10.

- On MNIST, supervised contrastive pretraining gives the best accuracy.
- On CIFAR-10, it matches the supervised baseline to within 0.3 percentage points.
- Autoencoder features keep pixel detail but carry less class information. They classify CIFAR-10 about 20 points worse.

## Repository layout

| Path | Content |
|---|---|
| `main.py` | Entry point: choose the dataset and the method |
| `models/` | Encoder, decoder, classifier and SupCon model definitions |
| `trainers/` | Training loops for the autoencoder, classifiers and SupCon loss |
| `scripts/` | MNIST experiment scripts |
| `notebooks/` | MNIST and CIFAR-10 runs with logs, plots and t-SNE figures |
| `utils.py` | t-SNE plotting helpers |

## How to run

```bash
conda env create -f environment.yml
conda activate aes
python main.py --mnist --method contrastive --data-path <path to data>
```

Use `--method autoencoder`, `supervised` or `contrastive`. The CIFAR-10 experiments are in `notebooks/`.
