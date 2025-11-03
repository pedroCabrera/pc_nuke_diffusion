import torch, os
filepath = "../nuke/pc_nuke_diffusion/models/ESRGAN/Public ESRGAN Models"
recursive = True
print(os.path.abspath(filepath))
if not recursive:
    for file in os.listdir(filepath):
        if file.endswith(".pth") or file.endswith(".pt"):
            print(f"Converting {file}...")
            checkpoint = torch.load(os.path.join(filepath, file), map_location="cpu")
            torch.save(checkpoint, os.path.join(filepath, "converted", file), _use_new_zipfile_serialization=True)
else:
    for root, dirs, files in os.walk(filepath):
        print(f"Entering directory: {root}")
        for file in files:
            if file.endswith(".pth") or file.endswith(".pt"):
                print(f"Converting {file}...")
                checkpoint = torch.load(os.path.join(root, file), map_location="cpu")
                relative_path = os.path.relpath(root, filepath)
                save_dir = os.path.join(filepath, "converted", relative_path)
                os.makedirs(save_dir, exist_ok=True)
                torch.save(checkpoint, os.path.join(save_dir, file), _use_new_zipfile_serialization=True)            