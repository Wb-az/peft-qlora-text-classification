import os
import numpy as np
from tqdm import tqdm
from utilities.emotions_dataset import EmotionsDataset

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchao.quantization.quant_api import Int8WeightOnlyConfig, quantize_
from transformers import (AutoModelForSequenceClassification,
                          AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)
                          
from transformers import DataCollatorWithPadding
from peft import LoraConfig, PeftModel, PeftConfig, TaskType
from safetensors.torch import load_file


HEAD_PREFIXES = (
    "classifier", "score", "lm_head", "head",
    "qa_outputs", "classification_head", "pre_classifier"
)

if torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
else:
    dtype = torch.float16
    

class CastOutputToFloat(nn.Sequential):
    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        return out.to(torch.float32) if isinstance(out, torch.Tensor) else out

def model_quant(model: nn.Module, device=None, head_prefixes=HEAD_PREFIXES) -> nn.Module:
    
    general_config = Int8WeightOnlyConfig(version=2)

    def filter_fn(mod: nn.Module, fqn: str) -> bool:
        # Check if it's a Linear layer and NOT a head prefix
        is_linear = isinstance(mod, nn.Linear)
        is_head = any(fqn == p or fqn.startswith(p + ".") for p in head_prefixes)
        return is_linear and not is_head

    quantize_(model, general_config, filter_fn, device=device)
    return model


def seq_class_init(check_point, num_labels, id2label, label2id, device, quantized=False):
    model = AutoModelForSequenceClassification.from_pretrained(
        check_point,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        attn_implementation="sdpa",
        torch_dtype="auto",
        use_safetensors=True
    ).to(device)

    if quantized:
        model = model_quant(model, device=device)

    return model
    
    
def quantization_report(model: nn.Module) -> None:
    from torchao.quantization import Int8Tensor
    
    # Initialize our counters
    counts = {'all_modules': 0, 'total_linear': 0,
              'base_int8': 0, 'lora_layers': 0, 'head_layers': 0}
                     
    total_params_bytes = 0

    for name, m in model.named_modules():
        counts["all_modules"] += 1
        
        if isinstance(m, nn.Linear):
            counts["total_linear"] += 1
            num_elements = m.weight.nelement()
            total_params_bytes += num_elements * m.weight.element_size()
            
            if isinstance(m.weight, Int8Tensor):
                counts["base_int8"] += 1
            elif "lora_" in name.lower():
                counts['lora_layers'] += 1
            else:
                counts['head_layers'] += 1

    # Print the clean summary
    print(f"\n{'='*35}")
    print(f"📊 Model Quantization Report")
    print(f"{'='*35}")
    print(f"Total Modules:          {counts['all_modules']}")
    print(f"Linear Layers:          {counts['total_linear']}")
    print(f"   ├─  Int8 Base:       {counts['base_int8']}")
    print(f"   ├─  LoRA:            {counts['lora_layers']}")
    print(f"   └─  Head:            {counts['head_layers']}")
    print(f" Est. VRAM (Weights):   {total_params_bytes / (1024**2):.2f} MiB")
    print(f"{'='*35}\n")


def model_postprocessing(model: torch.nn.Module, target_head_name: str = 'None') -> torch.nn.Module:
    """
     Improves efficiency before calling LoRA
    :param model: model to be post-processed
    :param target_head_name: name of the head to be cast to float32
    :return: model post-processing
    """

    # Freeze layer for PEFT
    for p in model.parameters():
        p.requires_grad_(False)
        
    # Cast LayerNorm to float32 for training stability
    for name, module in model.named_modules():
        if "LayerNorm" in module.__class__.__name__ or "norm" in name.lower():
            module.to(torch.float32)

    # Memory and gradients
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    # Ensure at least one leaf in the graph requires grad
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(_module, _input, output):
            if isinstance(output, torch.Tensor):
                output.requires_grad_(True)

        input_emb = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        if isinstance(input_emb, nn.Module):
            input_emb.register_forward_hook(make_inputs_require_grad)
     
    # Cast out heads to improve precision        
    if target_head_name:
        head = getattr(model, target_head_name, None)
        if isinstance(head, nn.Module) and not isinstance(head, CastOutputToFloat):
            setattr(model, target_head_name, CastOutputToFloat(head))
            
    else:
        potential_heads = ["classifier", "score", "lm_head", "head"]
        for name in potential_heads:
            head = getattr(model, name, None)
            if isinstance(head, nn.Module) and not isinstance(head, CastOutputToFloat):
                setattr(model, name, CastOutputToFloat(head))

    return model


def build_tokenizer(model_name_or_path: str):

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return ensure_special_tokens(tokenizer)


def ensure_special_tokens(tokenizer: PreTrainedTokenizerBase):
    # Prefer tokenizer-defined defaults first
    if tokenizer.pad_token is None:
        # Decoder-only models often use eos as pad
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        elif tokenizer.cls_token is not None:
            tokenizer.pad_token = tokenizer.cls_token

    # Only set bos/eos if missing, using existing cls/sep when available (BERT-like)
    if tokenizer.bos_token is None and tokenizer.cls_token is not None:
        tokenizer.bos_token = tokenizer.cls_token
    if tokenizer.eos_token is None and tokenizer.sep_token is not None:
        tokenizer.eos_token = tokenizer.sep_token

    return tokenizer


def collate_func(tokenizer):
    data_collator = DataCollatorWithPadding(tokenizer)
    return data_collator


def lora_peft(task_type=TaskType.SEQ_CLS, target_modules=None):

    config = LoraConfig(task_type=task_type,
                        inference_mode=False,
                        r=8,
                        lora_alpha=16,
                        lora_dropout=0.1,
                        target_modules=target_modules)
    return config


def load_seqcls_with_adapter(adapter_dir, num_labels, id2label, label2id, device, quantized=False):
    peft_cfg = PeftConfig.from_pretrained(adapter_dir)
    base_ckpt = peft_cfg.base_model_name_or_path

    base = seq_class_init(
        base_ckpt, num_labels, id2label, label2id, device,
        quantized=quantized,
    )
    
    # Basic stability (Only for OPT)
    for name, module in base.named_modules():
        if isinstance(module, nn.Linear):
            module.to(dtype)
    model = PeftModel.from_pretrained(base, adapter_dir)
    
    tokenizer = build_tokenizer(adapter_dir)

    return model, tokenizer, base_ckpt


def predict( model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             device: torch.device) -> tuple[np.ndarray, np.ndarray]:

    model = model.to(device)
    model.eval()
    all_preds, all_labels = [], []

    with torch.inference_mode():
        for batch in tqdm(dataloader):
            labels = batch.get("labels", None)

            inputs = {k: v.to(device, non_blocking=True)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor) and k != "labels"}

            outputs = model(**inputs)

            if isinstance(outputs, dict):
                logits = outputs["logits"]
            elif isinstance(outputs, (tuple, list)):
                logits = outputs[0]
            else:
                logits = outputs.logits

            predictions = torch.argmax(logits, dim=-1)
            all_preds.extend(predictions.detach().cpu().numpy())

            if labels is not None:
                all_labels.extend(labels.detach().cpu().numpy())

    lbs = np.array(all_labels) if all_labels else np.array([])
    preds = np.array(all_preds) if all_preds else np.array([])

    return lbs, preds


def inference_dataloader(tokenizer, max_length, text, labels, batch):

    data_collator = collate_func(tokenizer)
    test_dataset = EmotionsDataset(text, labels, tokenizer, max_length)
    eval_dataloader = DataLoader(test_dataset, shuffle=False, 
                            collate_fn=data_collator, batch_size=batch,
                            pin_memory=True, num_workers=2)

    return eval_dataloader


def inf_predictions(adapter_dir, text, labels, **kwargs):
    
    model, tokenizer, model_name = load_seqcls_with_adapter(adapter_dir,
                                                 num_labels=kwargs['num_labels'],
                                                 id2label=kwargs['id2label'],
                                                 label2id=kwargs['label2id'],
                                                 device = kwargs['device'],
                                                 quantized=kwargs['quantized'])
   
    model.to(dtype)
    test_dataloader = inference_dataloader(tokenizer, max_length=kwargs['max_length'],
                                        text=text, labels=labels, batch=kwargs['batch'])
    out = predict(model, test_dataloader, device=kwargs['device'])

    return out, model, model_name
