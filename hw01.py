import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from ignite.metrics import FID, InceptionScore
from ignite.metrics.gan.utils import InceptionModel


# --------------------------------------------------------------------------- config
@dataclass
class Config:
    # data
    data_root: str = "chest_xray"
    image_size: int = 64
    batch_size: int = 64
    num_workers: int = 2
    val_fraction: float = 0.10
    # model
    in_channels: int = 1
    base_channels: int = 32
    latent_dim: int = 128
    # optimisation
    epochs: int = 50
    lr: float = 1e-3
    lr_min: float = 1e-5
    weight_decay: float = 0.0
    beta: float = 1.0            # final KL weight (beta-VAE)
    kl_warmup_epochs: int = 10   # linear 0 -> beta over this many epochs
    grad_clip: float = 5.0
    # evaluation
    eval_num_images: int = 512   # real reference images for FID / IS
    eval_batch_size: int = 32
    eval_every: int = 5          # epochs between FID/IS evaluations
    fid_num_features: int = 2048 # InceptionV3 global-pool feature size
    sample_every: int = 5        # epochs between saved sample grids
    # hardware
    device: str = "mps"          # "mps" | "cuda" | "cpu" | "auto"
    # bookkeeping
    seed: int = 42
    out_dir: str = "results"
    ckpt_dir: str = "checkpoints"


# --------------------------------------------------------------------------- setup
def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def has_mps() -> bool:
    return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()


def resolve_device(request: str) -> torch.device:
    """Map 'auto' | 'cuda' | 'mps' | 'cpu' to a device, falling back with a warning."""
    if request == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("mps") if has_mps() else torch.device("cpu")

    if request == "cuda" and not torch.cuda.is_available():
        print("warning: cuda unavailable, falling back to cpu")
        return torch.device("cpu")
    if request == "mps" and not has_mps():
        print("warning: mps unavailable, falling back to cpu")
        return torch.device("cpu")
    if request not in {"cuda", "mps", "cpu"}:
        raise ValueError(f"unknown device {request!r}")
    return torch.device(request)


def resolve_data_root(root: Path) -> Path:
    """Find the directory that directly contains train/ and test/."""
    for candidate in (root, root / "chest_xray"):
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate.resolve()
    for test_dir in sorted(root.rglob("test")):
        if (test_dir.parent / "train").is_dir():
            return test_dir.parent.resolve()
    raise FileNotFoundError(f"no train/ + test/ pair found under {root.resolve()}")


# --------------------------------------------------------------------------- data
def stratified_split(dataset: ImageFolder, val_fraction: float, seed: int):
    """Split by class so the PNEUMONIA:NORMAL ratio is preserved in both parts."""
    targets = np.asarray(dataset.targets)
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for cls in np.unique(targets):
        idx = np.where(targets == cls)[0]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_fraction))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return Subset(dataset, sorted(train_idx)), Subset(dataset, sorted(val_idx))


def build_loaders(cfg: Config, device: torch.device):
    data_root = resolve_data_root(Path(cfg.data_root))

    # Decoder outputs a Bernoulli mean, so keep pixels in [0, 1] (no +/-1 normalisation).
    train_transform = T.Compose(
        [
            T.Grayscale(num_output_channels=1),
            T.Resize((cfg.image_size, cfg.image_size), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.ToTensor(),
        ]
    )
    # InceptionV3 needs 3 channels at 299x299. Grayscale(num_output_channels=3) replicates the
    # channel and, unlike a T.Lambda, is picklable for DataLoader worker processes.
    inception_transform = T.Compose(
        [
            T.Grayscale(num_output_channels=3),
            T.Resize((299, 299), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.ToTensor(),
        ]
    )

    train_full = ImageFolder(data_root / "train", transform=train_transform)
    train_set, val_set = stratified_split(train_full, cfg.val_fraction, cfg.seed)

    # Real reference for FID/IS: never trained on, resized from the original JPEGs
    # rather than upsampled from the 64x64 crops.
    real_set = ImageFolder(data_root / "test", transform=inception_transform)
    real_set = Subset(real_set, list(range(min(cfg.eval_num_images, len(real_set)))))

    loader_kwargs = dict(
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
    )
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, drop_last=False, **loader_kwargs)
    real_loader = DataLoader(real_set, batch_size=cfg.eval_batch_size, shuffle=False, drop_last=True, **loader_kwargs)

    print(f"train {len(train_set)} | val {len(val_set)} | real reference {len(real_set)}")
    return train_loader, val_loader, real_loader, real_set


# --------------------------------------------------------------------------- model
class Encoder(nn.Module):
    """q_phi(z | x): convolutional stack producing (mu, logvar)."""

    def __init__(self, image_size: int, in_channels: int, base_channels: int, latent_dim: int):
        super().__init__()
        n_stages = int(math.log2(image_size)) - 2
        channels = [min(base_channels * 2**i, 512) for i in range(n_stages)]

        blocks, prev = [], in_channels
        for ch in channels:
            blocks += [
                nn.Conv2d(prev, ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            prev = ch
        self.blocks = nn.Sequential(*blocks)

        self.feat_size = image_size // (2**n_stages)
        flat_dim = channels[-1] * self.feat_size**2
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.blocks(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """p_theta(x | z): transposed-convolution stack ending in a sigmoid over [0, 1]."""

    def __init__(self, image_size: int, out_channels: int, base_channels: int, latent_dim: int):
        super().__init__()
        n_stages = int(math.log2(image_size)) - 2
        channels = [min(base_channels * 2**i, 512) for i in range(n_stages)]
        self.feat_size = image_size // (2**n_stages)
        self.init_channels = channels[-1]

        self.fc = nn.Linear(latent_dim, self.init_channels * self.feat_size**2)

        blocks, prev = [], self.init_channels
        for ch in reversed(channels[:-1]):
            blocks += [
                nn.ConvTranspose2d(prev, ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            ]
            prev = ch
        blocks.append(nn.ConvTranspose2d(prev, out_channels, kernel_size=4, stride=2, padding=1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, z: torch.Tensor):
        h = self.fc(z).view(-1, self.init_channels, self.feat_size, self.feat_size)
        return torch.sigmoid(self.blocks(h))


class VAE(nn.Module):
    def __init__(self, image_size: int = 64, in_channels: int = 1, base_channels: int = 32, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(image_size, in_channels, base_channels, latent_dim)
        self.decoder = Decoder(image_size, in_channels, base_channels, latent_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        return self.decoder(self.reparameterize(mu, logvar)), mu, logvar

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """Draw n images from the prior z ~ N(0, I)."""
        self.eval()
        return self.decoder(torch.randn(n, self.latent_dim, device=device))

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic reconstruction from the posterior mean."""
        self.eval()
        return self.decoder(self.encoder(x)[0])


def vae_loss(recon_x, x, mu, logvar, beta: float):
    """Negative ELBO per image in nats: BCE (summed over pixels) + beta * KL."""
    recon = F.binary_cross_entropy(recon_x, x, reduction="sum") / x.size(0)
    kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon + beta * kl, recon.detach(), kl.detach()


def beta_for_epoch(cfg: Config, epoch: int) -> float:
    """Linear KL warm-up, which avoids early posterior collapse."""
    if cfg.kl_warmup_epochs <= 0:
        return cfg.beta
    return cfg.beta * min(1.0, epoch / cfg.kl_warmup_epochs)


# --------------------------------------------------------------------------- metrics
class DeviceLockedExtractor(nn.Module):
    """Keeps InceptionV3 on its own device because ignite forces `.to(metric_device)`.

    ignite's _BaseInceptionMetric calls `feature_extractor.to(device)`, which would move the
    27M-parameter network to the metric device. FID/IS must accumulate in float64 (unsupported
    on MPS) on the CPU, but the 299x299 convolutions are much faster on the GPU. This wrapper
    ignores `.to()`, sends batches to the network's device, and returns float32 CPU features.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__()
        self.model = model
        self.extractor_device = torch.device(device)

    def to(self, *args, **kwargs):  # noqa: D102 - deliberately a no-op
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(x.to(self.extractor_device)).float().cpu()


def build_metrics(cfg: Config):
    """Create the FID/IS feature extractors. Returns (metric_device, fid_extractor, is_extractor)."""
    metric_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if torch.cuda.is_available():
        extractor_device = torch.device("cuda")
    elif has_mps():
        extractor_device = torch.device("mps")
    else:
        extractor_device = torch.device("cpu")

    feature_extractor = DeviceLockedExtractor(
        InceptionModel(return_features=True, device=extractor_device), extractor_device
    )
    # Fall back if any InceptionV3 kernel is missing on the accelerator.
    if extractor_device.type != "cpu":
        try:
            feature_extractor(torch.rand(1, 3, 299, 299))
        except Exception as exc:
            print(f"inception on {extractor_device.type} failed ({type(exc).__name__}), using cpu")
            extractor_device = torch.device("cpu")
            feature_extractor = DeviceLockedExtractor(
                InceptionModel(return_features=True, device=extractor_device), extractor_device
            )
    prob_extractor = DeviceLockedExtractor(
        InceptionModel(return_features=False, device=extractor_device), extractor_device
    )

    print(f"inception device {extractor_device} | metric statistics {metric_device}")
    return metric_device, feature_extractor, prob_extractor


def to_inception(batch: torch.Tensor) -> torch.Tensor:
    """[N, 1, H, W] in [0, 1] -> [N, 3, 299, 299] using PIL bilinear resampling.

    PIL is used instead of F.interpolate because aliasing in the latter biases FID
    (Parmar et al., 2022, as noted in the pytorch-ignite reference notebook).
    """
    out = []
    for img in batch.detach().float().cpu():
        pil_img = T.ToPILImage()(img).resize((299, 299), Image.BILINEAR)
        tensor = T.ToTensor()(pil_img)
        out.append(tensor.repeat(3, 1, 1) if tensor.shape[0] == 1 else tensor)
    return torch.stack(out)


@torch.no_grad()
def generate_inception_batch(model: VAE, n: int, device: torch.device) -> torch.Tensor:
    model.eval()
    return to_inception(model.decoder(torch.randn(n, model.latent_dim, device=device)))


def compute_fid_is(model, loader, cfg, metric_device, feature_extractor, prob_extractor, device):
    """Score `model` against the real reference loader. Returns (fid, is)."""
    fid = FID(num_features=cfg.fid_num_features, feature_extractor=feature_extractor, device=metric_device)
    is_score = InceptionScore(num_features=1000, feature_extractor=prob_extractor, device=metric_device)

    for batch in loader:
        real = batch[0]
        fake = generate_inception_batch(model, real.size(0), device)
        fid.update((fake, real))   # first argument: generated, second: real
        is_score.update(fake)      # IS only consumes generated images
    return float(fid.compute()), float(is_score.compute())


# --------------------------------------------------------------------------- train / eval
def save_grid(tensor: torch.Tensor, path: Path, nrow: int = 8, title: str | None = None, figsize=(6.4, 6.4)):
    """Save a batch [N, 1, H, W] as a grayscale grid."""
    grid = vutils.make_grid(tensor, nrow=nrow, padding=2, pad_value=1.0)
    # make_grid expands a single channel to three, so read the first channel directly.
    arr = grid[0].numpy() if grid.shape[0] == 1 else grid.permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train(cfg: Config, device: torch.device, loaders, metrics):
    train_loader, val_loader, real_loader, _ = loaders
    metric_device, feature_extractor, prob_extractor = metrics

    sample_dir = Path(cfg.out_dir) / "samples"
    ckpt_dir = Path(cfg.ckpt_dir)
    for d in (sample_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    model = VAE(cfg.image_size, cfg.in_channels, cfg.base_channels, cfg.latent_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"VAE parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min)

    def save_checkpoint(name: str, epoch: int, scores: dict):
        torch.save(
            {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "config": asdict(cfg), "metrics": scores},
            ckpt_dir / name,
        )

    history = {"epoch": [], "train_loss": [], "train_recon": [], "train_kl": [],
               "val_loss": [], "val_recon": [], "val_kl": [], "beta": [], "lr": []}
    iter_loss = []
    eval_epoch, eval_is, eval_fid = [], [], []
    best = {"fid": float("inf"), "epoch": None}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        beta = beta_for_epoch(cfg, epoch)
        sums = {"loss": 0.0, "recon": 0.0, "kl": 0.0}

        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            recon_x, mu, logvar = model(x)
            loss, recon, kl = vae_loss(recon_x, x, mu, logvar, beta)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            iter_loss.append(loss.item())
            for key, value in (("loss", loss), ("recon", recon), ("kl", kl)):
                sums[key] += value.item()

        scheduler.step()

        # validation ELBO
        model.eval()
        val_sums = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                recon_x, mu, logvar = model(x)
                loss, recon, kl = vae_loss(recon_x, x, mu, logvar, cfg.beta)
                for key, value in (("loss", loss), ("recon", recon), ("kl", kl)):
                    val_sums[key] += value.item()
        n_train, n_val = len(train_loader), len(val_loader)

        for key, target in (("loss", "train_loss"), ("recon", "train_recon"), ("kl", "train_kl")):
            history[target].append(sums[key] / n_train)
        for key, target in (("loss", "val_loss"), ("recon", "val_recon"), ("kl", "val_kl")):
            history[target].append(val_sums[key] / n_val)
        history["epoch"].append(epoch)
        history["beta"].append(beta)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(
            f"[{epoch:03d}/{cfg.epochs}] train {history['train_loss'][-1]:.2f} "
            f"(recon {history['train_recon'][-1]:.2f} + beta*KL {history['train_kl'][-1]:.2f}, beta={beta:.2f}) "
            f"| val {history['val_loss'][-1]:.2f} | lr {history['lr'][-1]:.2e}"
        )

        # FID / IS every eval_every epochs and on the last one
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            fid_value, is_value = compute_fid_is(
                model, real_loader, cfg, metric_device, feature_extractor, prob_extractor, device
            )
            eval_epoch.append(epoch)
            eval_is.append(is_value)
            eval_fid.append(fid_value)
            improved = fid_value < best["fid"]
            if improved:
                best.update(fid=fid_value, epoch=epoch)
                save_checkpoint("best.pt", epoch, {"is": is_value, "fid": fid_value})
            print(f"        IS {is_value:.3f} | FID {fid_value:.3f}" + ("  <- best" if improved else ""))

        save_checkpoint("last.pt", epoch, {"fid": best["fid"]})

        if epoch % cfg.sample_every == 0:
            samples = model.sample(64, device).cpu()
            save_grid(samples, sample_dir / f"epoch_{epoch:03d}.png")

    return model, history, iter_loss, (eval_epoch, eval_is, eval_fid), best, len(train_loader)


def load_vae(path: Path, device: torch.device) -> VAE:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    saved = Config(**ckpt["config"])
    model = VAE(saved.image_size, saved.in_channels, saved.base_channels, saved.latent_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# --------------------------------------------------------------------------- reporting
def plot_history(cfg: Config, history, iter_loss, eval_scores, iters_per_epoch: int):
    fig_dir = Path(cfg.out_dir) / "figures"

    def ema(values, weight=0.98):
        out = [values[0]]
        for v in values[1:]:
            out.append(out[-1] * weight + (1.0 - weight) * v)
        return out

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))

    ax = axes[0]
    iters = np.arange(1, len(iter_loss) + 1)
    ax.plot(iters, iter_loss, color="tab:blue", alpha=0.18, lw=0.8, label="train (per iteration)")
    ax.plot(iters, ema(iter_loss), color="tab:blue", lw=1.8, label="train (EMA 0.98)")
    ax.plot(np.arange(1, len(history["val_loss"]) + 1) * iters_per_epoch, history["val_loss"],
            color="tab:orange", lw=1.6, marker=".", ms=4, label="validation")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$-$ELBO per image (nats)")
    ax.set_title("Total loss")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(history["epoch"], history["train_recon"], "o-", ms=2.5, lw=1.3, color="tab:green", label="train recon")
    ax.plot(history["epoch"], history["val_recon"], "o--", ms=2.5, lw=1.3, color="tab:green", alpha=0.6, label="val recon")
    ax.plot(history["epoch"], history["train_kl"], "s-", ms=2.5, lw=1.3, color="tab:red", label="train KL")
    ax.plot(history["epoch"], history["val_kl"], "s--", ms=2.5, lw=1.3, color="tab:red", alpha=0.6, label="val KL")
    ax.set_xlabel("epoch")
    ax.set_ylabel("nats per image")
    ax.set_title(r"Reconstruction vs. KL term ($\beta$ warm-up)")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(fig_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    eval_epoch, eval_is, eval_fid = eval_scores
    fig, ax1 = plt.subplots(figsize=(6.4, 3.8))
    ax1.set_title("Inception Score and FID during training")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("Inception Score (higher is better)", color="tab:red")
    ax1.plot(eval_epoch, eval_is, "o-", color="tab:red", ms=4, lw=1.4)
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.grid(False)
    ax2.set_ylabel("FID (lower is better)", color="tab:blue")
    ax2.plot(eval_epoch, eval_fid, "s--", color="tab:blue", ms=4, lw=1.4)
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    fig.tight_layout()
    fig.savefig(fig_dir / "fid_is_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def reference_scores(cfg: Config, loaders, metrics):
    """IS of real images, and FID between two disjoint halves of the real set."""
    _, _, _, real_set = loaders
    metric_device, feature_extractor, prob_extractor = metrics

    is_real = InceptionScore(num_features=1000, feature_extractor=prob_extractor, device=metric_device)
    loader = DataLoader(real_set, batch_size=cfg.eval_batch_size, shuffle=False, drop_last=False)
    for batch in loader:
        is_real.update(batch[0])
    real_is = float(is_real.compute())

    # Equal-sized halves; drop_last keeps the two loaders in step for FID.update.
    half = (len(real_set) // 2 // cfg.eval_batch_size) * cfg.eval_batch_size
    indices = list(range(len(real_set)))
    loader_a = DataLoader(Subset(real_set, indices[:half]), batch_size=cfg.eval_batch_size, shuffle=False, drop_last=True)
    loader_b = DataLoader(Subset(real_set, indices[half:2 * half]), batch_size=cfg.eval_batch_size, shuffle=False, drop_last=True)

    fid_real = FID(num_features=cfg.fid_num_features, feature_extractor=feature_extractor, device=metric_device)
    for batch_a, batch_b in zip(loader_a, loader_b):
        fid_real.update((batch_a[0], batch_b[0]))
    return real_is, float(fid_real.compute())


# --------------------------------------------------------------------------- main
def main():
    cfg = Config()
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"torch {torch.__version__} | device {device}")

    out_dir = Path(cfg.out_dir)
    fig_dir = out_dir / "figures"
    for d in (out_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    loaders = build_loaders(cfg, device)
    metrics = build_metrics(cfg)

    model, history, iter_loss, eval_scores, best, iters_per_epoch = train(cfg, device, loaders, metrics)
    plot_history(cfg, history, iter_loss, eval_scores, iters_per_epoch)

    # ---- qualitative results from the best checkpoint
    ckpt = Path(cfg.ckpt_dir) / ("best.pt" if (Path(cfg.ckpt_dir) / "best.pt").exists() else "last.pt")
    best_model = load_vae(ckpt, device)
    print(f"loaded {ckpt}")

    samples = best_model.sample(64, device).cpu()
    save_grid(samples, fig_dir / "generated_samples.png", title="VAE samples (z ~ N(0, I))", figsize=(7, 7))

    imgs, _ = next(iter(loaders[1]))
    imgs = imgs[:8].to(device)
    recon = best_model.reconstruct(imgs).cpu()
    save_grid(torch.cat([imgs.cpu(), recon]), fig_dir / "reconstructions.png", nrow=8,
              title="top: real validation images | bottom: reconstructions", figsize=(11, 3))

    with torch.no_grad():
        mse = F.mse_loss(best_model.reconstruct(imgs), imgs).item()

    # ---- final metrics against the real reference set
    eval_fid, eval_is = compute_fid_is(
        best_model, loaders[2], cfg, metrics[0], metrics[1], metrics[2], device
    )
    real_is, real_fid = reference_scores(cfg, loaders, metrics)

    print(f"VAE (best epoch {best['epoch']}): IS {eval_is:.3f} | FID {eval_fid:.3f}")
    print(f"real images (reference)       : IS {real_is:.3f} | FID {real_fid:.3f}")
    print(f"reconstruction MSE (8 val)    : {mse:.5f}")

    # ---- export
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    eval_epoch, eval_is_hist, eval_fid_hist = eval_scores
    pd.DataFrame({"epoch": eval_epoch, "IS": eval_is_hist, "FID": eval_fid_hist}).to_csv(
        out_dir / "fid_is_history.csv", index=False
    )
    summary = {
        "config": asdict(cfg),
        "device": str(device),
        "best_epoch": best["epoch"],
        "best_fid": best["fid"],
        "final_is": eval_is,
        "final_fid": eval_fid,
        "reference_is_real": real_is,
        "reference_fid_real_vs_real": real_fid,
        "reconstruction_mse": mse,
    }
    with open(out_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(pd.DataFrame([
        {"model": "VAE (this work)", "IS (higher better)": round(eval_is, 3), "FID (lower better)": round(eval_fid, 3)},
        {"model": "real images (reference)", "IS (higher better)": round(real_is, 3), "FID (lower better)": round(real_fid, 3)},
    ]).to_string(index=False))


if __name__ == "__main__":
    main()
