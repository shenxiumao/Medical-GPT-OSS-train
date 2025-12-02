import argparse
import json
import os
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def _is_attn(key: str) -> bool:
    return any(s in key for s in ["q_proj", "k_proj", "v_proj", "o_proj"])

def _is_mlp(key: str) -> bool:
    return any(s in key for s in ["up_proj", "down_proj", "gate_proj", "gate_up_proj"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_a", required=True)
    parser.add_argument("--adapter_b", required=True)
    parser.add_argument("--alpha_a", type=float, default=0.5)
    parser.add_argument("--alpha_b", type=float, default=0.5)
    parser.add_argument("--alpha_attn_a", type=float, default=None)
    parser.add_argument("--alpha_attn_b", type=float, default=None)
    parser.add_argument("--alpha_mlp_a", type=float, default=None)
    parser.add_argument("--alpha_mlp_b", type=float, default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--export_merged_dir", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    wa = load_file(os.path.join(args.adapter_a, "adapter_model.safetensors"))
    wb = load_file(os.path.join(args.adapter_b, "adapter_model.safetensors"))

    if wa.keys() != wb.keys():
        missing_a = wb.keys() - wa.keys()
        missing_b = wa.keys() - wb.keys()
        raise ValueError(f"Adapter tensors mismatch. Missing in A: {missing_a}, missing in B: {missing_b}")

    blend = {}
    for k in wa.keys():
        aa, ab = args.alpha_a, args.alpha_b
        if _is_attn(k) and args.alpha_attn_a is not None and args.alpha_attn_b is not None:
            aa, ab = args.alpha_attn_a, args.alpha_attn_b
        elif _is_mlp(k) and args.alpha_mlp_a is not None and args.alpha_mlp_b is not None:
            aa, ab = args.alpha_mlp_a, args.alpha_mlp_b
        blend[k] = aa * wa[k] + ab * wb[k]

    save_file(blend, os.path.join(args.out_dir, "adapter_model.safetensors"))

    with open(os.path.join(args.adapter_b, "adapter_config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(args.out_dir, "adapter_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    if args.base_model and args.export_merged_dir:
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(args.dtype, torch.bfloat16)
        tok = AutoTokenizer.from_pretrained(args.base_model)
        model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch_dtype)
        model = PeftModel.from_pretrained(model, args.out_dir)
        model = model.merge_and_unload()
        os.makedirs(args.export_merged_dir, exist_ok=True)
        model.save_pretrained(args.export_merged_dir)
        tok.save_pretrained(args.export_merged_dir)

if __name__ == "__main__":
    main()
