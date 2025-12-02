import torch, os, sys
ckpt = "models/latest_checkpoint.pt"
if not os.path.exists(ckpt):
    ckpt = "models/checkpoint_epoch_1.pt"
    if not os.path.exists(ckpt):
<<<<<<< HEAD
        print("No checkpoint found"); sys.exit(1)
d = torch.load(ckpt, map_location="cpu")
print("TYPE:", type(d))
if isinstance(d, dict):
    print("Top keys:", list(d.keys())[:60])
    for candidate in ("generator_state_dict","model_state_dict","state_dict","model","netG"):
        if candidate in d:
            sd = d[candidate]
            print(f"\nFound candidate: {candidate} with {len(sd)} keys")
            for k,v in list(sd.items())[:40]:
                print(k, getattr(v,"shape", None))
            break
    else:
        # show some sample entries
        for k,v in list(d.items())[:40]:
            if isinstance(v, dict):
                print(f"\nNested dict at {k} with {len(v)} keys; sample keys:")
                print(list(v.keys())[:40])
                break
else:
    print("Checkpoint is not a dict.")
=======
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
>>>>>>> cesar/main
