import argparse
import json
import os
import shutil
from safetensors.torch import load_file, save_file
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def _is_attn(key: str) -> bool:
    return any(s in key for s in ["q_proj", "k_proj", "v_proj", "o_proj"])

def _is_mlp(key: str) -> bool:
    return any(s in key for s in ["up_proj", "down_proj", "gate_proj", "gate_up_proj"])

def _scale_adapter(src_dir: str, dst_dir: str, alpha: float, alpha_attn: float | None, alpha_mlp: float | None) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    wa = load_file(os.path.join(src_dir, "adapter_model.safetensors"))
    wb = {}
    for k, v in wa.items():
        s = alpha
        if alpha_attn is not None and _is_attn(k):
            s = alpha_attn
        elif alpha_mlp is not None and _is_mlp(k):
            s = alpha_mlp
        wb[k] = v * s
    save_file(wb, os.path.join(dst_dir, "adapter_model.safetensors"))
    with open(os.path.join(src_dir, "adapter_config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(dst_dir, "adapter_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"]:
        p = os.path.join(src_dir, fname)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(dst_dir, fname))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_a", required=True)
    parser.add_argument("--adapter_b", required=True)
    parser.add_argument("--alpha_a", type=float, default=0.5)
    parser.add_argument("--alpha_b", type=float, default=0.5)
    parser.add_argument("--alpha_attn_a", type=float, default=None)
    parser.add_argument("--alpha_attn_b", type=float, default=None)
    parser.add_argument("--alpha_mlp_a", type=float, default=None)
    parser.add_argument("--alpha_mlp_b", type=float, default=None)
    parser.add_argument("--export_dir", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    tmp_a = os.path.join(args.export_dir, "_tmp_adapter_a")
    tmp_b = os.path.join(args.export_dir, "_tmp_adapter_b")
    _scale_adapter(args.adapter_a, tmp_a, args.alpha_a, args.alpha_attn_a, args.alpha_mlp_a)
    _scale_adapter(args.adapter_b, tmp_b, args.alpha_b, args.alpha_attn_b, args.alpha_mlp_b)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(args.dtype, torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch_dtype)

    model = PeftModel.from_pretrained(model, tmp_a)
    model = model.merge_and_unload()
    model = PeftModel.from_pretrained(model, tmp_b)
    model = model.merge_and_unload()

    os.makedirs(args.export_dir, exist_ok=True)
    model.save_pretrained(args.export_dir)
    tok.save_pretrained(args.export_dir)

if __name__ == "__main__":
    main()

