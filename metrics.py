import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision.models import inception_v3, Inception_V3_Weights

# Metric tools authored by Cesar Cabrera


def _pick_device(device):
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_inception(device):
    model = inception_v3(
        weights=Inception_V3_Weights.IMAGENET1K_V1,
        aux_logits=False,
        transform_input=False,
    )
    model.fc = torch.nn.Identity()
    model.dropout = torch.nn.Identity()
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def _collect_features(loader, model, device, max_batches=None):
    features = []
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    for idx, batch in enumerate(loader):
        if isinstance(batch, dict):
            images = batch["pixel_values"]
        elif isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch

        images = images.to(device)
        images = (images + 1.0) * 0.5
        images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
        images = torch.clamp(images, 0, 1)
        images = (images - mean) / std

        outputs = model(images)
        features.append(outputs.cpu())

        if max_batches and (idx + 1) >= max_batches:
            break

    if not features:
        raise ValueError("Loader yielded no batches.")

    return torch.cat(features, dim=0)


def _covariance(feats):
    feats = feats.float()
    mu = feats.mean(dim=0, keepdim=True)
    centered = feats - mu
    cov = centered.t() @ centered / (feats.shape[0] - 1)
    return cov, mu.squeeze(0)


def _matrix_sqrt(mat):
    eigenvalues, eigenvectors = torch.linalg.eigh(mat)
    eigenvalues = torch.clamp(eigenvalues, min=0)
    sqrt_eig = torch.sqrt(eigenvalues)
    return eigenvectors @ torch.diag(sqrt_eig) @ eigenvectors.t()


def fid_from_loaders(real_loader, fake_loader, device=None, max_batches=None):
    """
    Computes Frechet Inception Distance between two loaders.
    Expects images normalized to [-1, 1] with shape [B, 3, H, W].
    """
    device = _pick_device(device)
    model = _build_inception(device)

    real_feats = _collect_features(real_loader, model, device, max_batches)
    fake_feats = _collect_features(fake_loader, model, device, max_batches)

    sigma_r, mu_r = _covariance(real_feats)
    sigma_f, mu_f = _covariance(fake_feats)

    eps = 1e-6
    eye = torch.eye(sigma_r.shape[0], device=sigma_r.device)
    sigma_r = sigma_r + eps * eye
    sigma_f = sigma_f + eps * eye

    diff = mu_r - mu_f
    covmean = _matrix_sqrt(sigma_r @ sigma_f)
    fid = diff.dot(diff) + torch.trace(sigma_r + sigma_f - 2 * covmean)

    return fid.item()


def collapse_report(fake_loader, device=None, max_batches=None):
    """
    Light heuristic for mode collapse using inception feature spread.
    Returns variance and mean pairwise distance.
    """
    device = _pick_device(device)
    model = _build_inception(device)

    feats = _collect_features(fake_loader, model, device, max_batches)
    cov, _ = _covariance(feats)
    var_score = torch.diag(cov).mean().item()

    pairwise = torch.pdist(feats, p=2) if feats.shape[0] > 1 else torch.tensor([0.0])
    mean_dist = pairwise.mean().item()

    flag = var_score < 0.05 or mean_dist < 15.0

    return {
        "feature_variance": var_score,
        "mean_pairwise_distance": mean_dist,
        "possible_collapse": flag,
    }


def compute_and_store_metrics(real_loader, fake_loader, save_path="outputs/logs/metrics.pt", device=None, max_batches=None):
    """
    Computes FID and collapse signal, then stores them for the dashboard.
    Saves a dict with keys: 'fid', 'collapse', 'collapse_detail'.
    """
    fid_val = fid_from_loaders(real_loader, fake_loader, device=device, max_batches=max_batches)
    collapse = collapse_report(fake_loader, device=device, max_batches=max_batches)

    metrics = {
        "fid": fid_val,
        "collapse": collapse["possible_collapse"],
        "collapse_detail": collapse,
    }

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(metrics, path)
    return metrics


class _NoiseImages(Dataset):
    """
    Simple synthetic set for quick checks. Keeps tests offline.
    """

    def __init__(self, count=64, seed=0, scale=1.0):
        gen = torch.Generator().manual_seed(seed)
        self.data = torch.randn(count, 3, 64, 64, generator=gen) * scale

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"pixel_values": self.data[idx]}


if __name__ == "__main__":
    print("Cesar Cabrera | Metrics smoke test (synthetic data).")

    real_loader = DataLoader(_NoiseImages(count=96, seed=2, scale=1.0), batch_size=16)
    fake_loader = DataLoader(_NoiseImages(count=96, seed=7, scale=1.2), batch_size=16)

    fid_score = fid_from_loaders(real_loader, fake_loader, device="cpu", max_batches=2)
    diversity = collapse_report(fake_loader, device="cpu", max_batches=2)

    print(f"FID on random tensors (should be high): {fid_score:.2f}")
    print(
        f"Diversity variance: {diversity['feature_variance']:.4f} | "
        f"mean L2: {diversity['mean_pairwise_distance']:.2f}"
    )
