import torch, os, sys
ckpt = "models/latest_checkpoint.pt"
if not os.path.exists(ckpt):
    ckpt = "models/checkpoint_epoch_1.pt"
    if not os.path.exists(ckpt):
        print("No checkpoint found in models/"); sys.exit(1)
d = torch.load(ckpt, map_location="cpu")
print("TOP TYPE:", type(d))
if isinstance(d, dict):
    print("Top keys:", list(d.keys()))
    sd = d.get("generator_state_dict", d.get("state_dict", d))
else:
    sd = d
if isinstance(sd, dict):
    print("Number of state keys:", len(sd))
    for k, v in list(sd.items())[:40]:
        try:
            print(k, getattr(v, "shape", None))
        except Exception:
            print(k, type(v))
else:
    print("State is not a dict; type:", type(sd))
print("INSPECTION DONE. Path:", ckpt)