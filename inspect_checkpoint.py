import torch
from pathlib import Path
import sys
import logging

logging.basicConfig(level=logging.INFO)

# try to import Generator from same places as your code
try:
    from train import Generator
except Exception:
    try:
        from models import Generator
    except Exception:
        try:
            from model import Generator
        except Exception as e:
            raise ImportError("Generator class not found. Ensure train.py/models.py/model.py exist and export Generator.") from e

def load_ckpt(path):
    ckpt = torch.load(path, map_location="cpu")
    return ckpt

def looks_like_state_dict(d):
    return isinstance(d, dict) and any(hasattr(v, "shape") for v in list(d.values())[:8])

def extract_state_dict(ckpt):
    # mirror your repo helper
    for key in ("generator_state_dict","model_state_dict","state_dict","netG","net","model"):
        if key in ckpt and isinstance(ckpt[key], dict):
            cand = ckpt[key]
            if any(k in cand for k in ("state_dict","model_state_dict","net","generator_state_dict")):
                return extract_state_dict(cand)
            return cand
    for v in ckpt.values() if isinstance(ckpt, dict) else []:
        if isinstance(v, dict) and looks_like_state_dict(v):
            return v
    if looks_like_state_dict(ckpt):
        return ckpt
    return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_checkpoint.py /path/to/checkpoint.pt [noise_dim]")
        sys.exit(1)
    ckpt_path = Path(sys.argv[1])
    nz = int(sys.argv[2]) if len(sys.argv) > 2 else 512

    print("Loading checkpoint:", ckpt_path)
    ckpt = load_ckpt(ckpt_path)
    state = extract_state_dict(ckpt) if isinstance(ckpt, dict) else ckpt
    if state is None:
        print("No state_dict found in checkpoint. Top keys:", list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt))
        sys.exit(2)

    print("Top-level state_dict keys:", len(state))
    sample_keys = list(state.keys())[:40]
    for k in sample_keys:
        print("  ", k, getattr(state[k], "shape", None))

    # instantiate model and compare
    G = Generator(nz=nz)
    model_state = G.state_dict()
    print("\nModel state keys:", len(model_state))
    sk = set(state.keys())
    mk = set(model_state.keys())
    common = sorted(list(sk & mk))
    print("Common keys:", len(common))
    if len(common) == 0:
        print("No common keys between checkpoint and model -> incompatible checkpoint/architecture.")
        sys.exit(3)

    # compare shapes and diffs for matched-shape keys
    matched = []
    diffs = []
    for k in common:
        s = state[k]
        m = model_state[k]
        if hasattr(s, "shape") and s.shape == m.shape:
            matched.append(k)
            diffs.append(float((m.cpu().float() - s.cpu().float()).norm().item()))
    print("Matched keys (name+shape):", len(matched))
    if diffs:
        import statistics
        print("Avg param-norm diff for matched keys:", statistics.mean(diffs), "median:", statistics.median(diffs))
        small = [k for k,d in zip(matched,diffs) if d < 1e-6][:10]
        if small:
            print("Keys with near-zero diff (first examples):", small)
    print("Example keys only in checkpoint (first 20):", list(sk - mk)[:20])
    print("Example keys only in model (first 20):", list(mk - sk)[:20])
    print("Done.")