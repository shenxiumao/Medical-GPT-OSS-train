# Medical-GPT-OSS-train

面向医疗场景的 GPT-OSS-20B 训练与融合示例工程，基于 NVIDIA ModelOpt QAT 与 Hugging Face 生态。包含：
- SFT 全参/LoRA 配置与脚本（支持 QAT）
- 两种适配器融合方式（顺序合并、加权缩放合并）
- MXFP4/NVFP4 权重转换脚本

## 目录结构
```
Medical-GPT-OSS-train/
├─ gpt-oss/
│  ├─ configs/            # SFT 配置：sft_full.yaml / sft_lora.yaml / zero3.yaml
│  ├─ sft.py              # QAT SFT 训练入口（accelerate + ModelOpt QATSFTTrainer）
│  ├─ utils.py            # 数据加载、PEFT 配置（默认 MoE 目标，需按模型改）
│  ├─ convert_oai_mxfp4_weight_only.py  # MXFP4/NVFP4 转换示例脚本
│  └─ README.md
├─ merge_adapters_seq.py      # 适配器顺序合并（PT→SFT）写入基座
├─ merge_adapters_scaled.py   # 适配器加权缩放后合并（支持分层权重）
├─ blend_adapters_linear.py   # 逐张量线性融合（需 rank/结构一致，不推荐）
└─ README.md
```

## 环境依赖
- Python ≥ 3.10
- `transformers` `peft` `accelerate` `datasets` `safetensors`
- ModelOpt: `modelopt.torch`

## 适配器融合
### 顺序合并（PT→SFT）
在非量化基座上依次合并 PT 与 SFT 适配器：
```bash
python merge_adapters_seq.py \
  --base_model openai/gpt-oss-20b \
  --pt_adapter /path/to/PT_adapter \
  --sft_adapter /path/to/SFT_adapter \
  --export_dir zjydiary/medical-gpt-oss-20b \
  --dtype bfloat16
```

### 加权缩放合并（推荐）
避免不同 LoRA rank/结构冲突，先分别缩放，再顺序合并：
```bash
python merge_adapters_scaled.py \
  --base_model openai/gpt-oss-20b \
  --adapter_a /path/to/PT_adapter \
  --adapter_b /path/to/SFT_adapter \
  --alpha_a 0.3 --alpha_b 0.7 \
  --alpha_attn_a 0.2 --alpha_attn_b 0.8 \
  --alpha_mlp_a 0.3 --alpha_mlp_b 0.7 \
  --export_dir zjydiary/medical-gpt-oss-20b \
  --dtype bfloat16
```
提示：如还需融合 DPO/ORPO 适配器，建议其系数较小（如 0.2），防止回答风格过度偏好。

## MXFP4/NVFP4 导出
合并完成后可转换为 MXFP4/NVFP4 推理格式：
```bash
python gpt-oss/convert_oai_mxfp4_weight_only.py \
  --src /root/workspace/model/zjydiary/medical-gpt-oss-20b \
  --dst /root/workspace/model/zjydiary/medical-gpt-oss-20b-mxfp4 \
  --quant_cfg MXFP4_MLP_WEIGHT_ONLY_CFG
```

## 训练建议
- LoRA 学习率：`1e-5–3e-5`，过高（如 `2e-4`）易不稳
- 目标模块：明确到注意力/MLP线性层；避免 MoE 目标对非 MoE 模型
- 序列长度：医疗长文本可用 `1024–2048`，显存紧张降低
- 评估：偏好对齐时适度加入 SFT 锚定（`pref_ftx 0.1–0.3`），减少风格过拟合

## 注意事项
- 量化权重上仅可加载一个适配器；融合需在非量化基座完成，再导出量化
- 适配器结构需一致或使用加权缩放融合脚本，避免形状冲突
- 医疗应用需注意隐私与安全，回答不替代专业医嘱
