import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--pt_adapter", required=True)
    parser.add_argument("--sft_adapter", required=True)
    parser.add_argument("--export_dir", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(args.dtype, torch.bfloat16)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch_dtype)

    model = PeftModel.from_pretrained(model, args.pt_adapter)
    model = model.merge_and_unload()

    model = PeftModel.from_pretrained(model, args.sft_adapter)
    model = model.merge_and_unload()

    model.save_pretrained(args.export_dir)
    tok.save_pretrained(args.export_dir)

if __name__ == "__main__":
    main()

