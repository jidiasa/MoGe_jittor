from pathlib import Path
import torch
from huggingface_hub import hf_hub_download

# -----------------------------
# 1️⃣  选择模型仓库（下面是 MoGe-vitl14）
repo_id   = "Ruicheng/moge-vitl"
filename  = "model.pt"          # 仓库里就是这个文件名
# -----------------------------

# 2️⃣  下载（会自动缓存到 ~/.cache/huggingface）
ckpt_path = hf_hub_download(
    repo_id = repo_id,
    repo_type = "model",
    filename = filename,
    revision = None,        # 指定 tag / commit 时改这里
    resume_download = True, # 已下过会跳过
    local_dir = None,       # 自定义缓存目录可改
)

print(f"✅ checkpoint 已保存到 → {ckpt_path}")

# 3️⃣  仅 CPU 加载即可，read‐only
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

# 4️⃣  打印完整超参数字典
print("\n=== model_config 超参数 ===")
for k, v in ckpt["model_config"].items():
    print(f"{k:35} : {v}")

# 若还想看 state_dict 的键数量
print("\nstate_dict 参数总数:", len(ckpt["model"]))

# ------------------------------------------------------------------
# 附：如需把 config 保存成单独的 json / yaml 方便查阅 ↓
import json, yaml
json_path = Path(ckpt_path).with_suffix(".config.json")
yaml_path = Path(ckpt_path).with_suffix(".config.yaml")
json_path.write_text(json.dumps(ckpt["model_config"], indent=2))
yaml_path.write_text(yaml.dump(ckpt["model_config"]))
print(f"\n已导出 config 到\n  {json_path}\n  {yaml_path}")
