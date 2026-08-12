import os

def replace_in_file(path, old, new):
    with open(path, "r") as f:
        content = f.read()
    with open(path, "w") as f:
        f.write(content.replace(old, new))

files_to_patch = [
    "rl/policy_infer_torch.py",
    "agent/main.py",
    "scripts/build_submission.py"
]

for p in files_to_patch:
    replace_in_file(p, "torch.float16", "torch.float32")
    replace_in_file(p, "np.float16", "np.float32")
    replace_in_file(p, "FP16", "FP32")
    replace_in_file(p, "ptcg-torch-fp16-v1", "ptcg-torch-fp32-v1")
    replace_in_file(p, "bc_best_torch_fp16.pt", "bc_best_torch_fp32.pt")
    
# In policy_infer_torch.py, we also need to remove the hardcoded check for FP16
# Wait, replacing torch.float16 with torch.float32 automatically fixes the check!
# because the check was `tensor.dtype != torch.float16`, now it will be `tensor.dtype != torch.float32`
