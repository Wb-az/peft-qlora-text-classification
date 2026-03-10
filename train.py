import torch
import pandas as pd
from pathlib import Path

from llm_lora_emotion_analysis import emotion_to_label
from utilities.emotions_dataset import EmotionsDataset
from utilities.hf_pipeline import (build_tokenizer, collate_func, seq_class_init,
                                   model_postprocessing, lora_peft)
import peft
from peft import get_peft_model
from peft.utils import constants
from functools import partial
from utilities.weighted_loss import create_weights, weighted_ce_loss
from utilities.hf_pipeline import quantization_report
from utilities.eval_metrics import class_metrics
from utilities.eval_metrics import plot_loss

from transformers import (Trainer, TrainingArguments)


def main(trainer_args, **kwargs):

    name = kwargs['check_point'].split('/')[-1].lower()

    trainer_args.output_dir = f'{kwargs['output_dir']}/peft-{name}'
    trainer_args.logging_dir = f'{kwargs['output_dir']}/peft-{name}/logs'

    tokenizer = build_tokenizer(kwargs['check_point'])

    train_dataset = EmotionsDataset(kwargs['x_train'], kwargs['y_train'], tokenizer, kwargs['max_length'])
    val_dataset = EmotionsDataset(kwargs['x_val'], kwargs['y_val'], tokenizer, kwargs['max_length'])

    data_collator = collate_func(tokenizer=tokenizer)

    model = seq_class_init(kwargs['check_point'],
                           num_labels=kwargs['num_labels'],
                           id2label=kwargs['id2label'],
                           label2id=kwargs['label2id'], device=kwargs['device'],
                           quantized=kwargs['quantized'])

    model = model_postprocessing(model)

    if 'modernbert' in name:
        target_modules = ['Wqkv', 'Wo', 'Wi']
    else:
        target_modules_map = constants.TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
        target_modules = target_modules_map[name.split('-')[0]]

    peft_config = lora_peft(target_modules=target_modules)
    peft_config.inference_mode = False

    peft_model = get_peft_model(model, peft_config)

    peft_model.config.use_cache = False

    llm_trainer = Trainer(
    model=peft_model,
    args=trainer_args,
    compute_loss_func=partial(weighted_ce_loss, weights=kwargs['weights']),
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=class_metrics)

    print("")

    if kwargs.get('quantized'):
        quantization_report(llm_trainer.model)
        peft_type = 'PEFT-QLoRA'
        print("")

    else:
        peft_type = 'PEFT-QLoRA'

    print(f"{peft_type} {name.capitalize()} Model:")

    llm_trainer.model.print_trainable_parameters()

    return llm_trainer

if __name__ == '__main__':

    # Auto-detect the best precision for using GPU
    use_bf16 = torch.cuda.is_bf16_supported()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    training_args = TrainingArguments(
        output_dir=None,
        learning_rate=1e-4,
        eval_strategy='steps',
        eval_steps= 1400,
        num_train_epochs=1,
        bf16=use_bf16,
        fp16=not use_bf16,
        use_cpu=True if device.type == 'cpu' else False,
        dataloader_num_workers=4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        train_sampling_strategy = 'group_by_length',
        logging_steps = 100,
        weight_decay=0.05,
        save_strategy='steps',
        save_steps = 1400,
        save_total_limit=1,
        metric_for_best_model='f1',
        greater_is_better=True,
        load_best_model_at_end=True,
        report_to='none' # comet_ml, clearml, mlflow, swanlab, tensorboard, and wandb
    )

    train = pd.read_csv('dataset/train_dataset.csv')
    validation = pd.read_csv("dataset/val_dataset.csv")

    plot_dir = Path('results/plots')
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Map label to emotion
    label_to_emotion = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger',
                        4: 'fear', 5: 'surprise'}

    emotion_to_label = {v: k for k, v in label_to_emotion.items()}

    # Created weights
    weights = create_weights(labels=train.label.values)

    # Check your GPU availability before training more than one model
    check_points = ['roberta-base', 'facebook/opt-350m', 'answerdotai/ModernBERT-base']

    params = {'check_point': check_points, 'quantized': True,
              'num_labels': len(emotion_to_label), 'x_train': train.text.values, 'y_train': train.label.values,
              'x_val': validation.text.values, 'y_val': validation.label.values, 'max_length': 128,
              'id2label': label_to_emotion, 'label2id': emotion_to_label, 'weights': weights,
              'device': device, 'output_dir': 'results/weights/q8intlora'}

    for check_point in check_points:

        params['check_point'] = check_point

        trainer = main(training_args, **params)

        # Start training
        trainer.train()

        # Access model training history
        history = trainer.state.log_history
        print(f"Total training time: {history[-1]['train_runtime']} seconds")

        # Plot training loss
        model_name = check_point.split('/')[-1].lower()
        plot_loss(history, f'qlora_{model_name}', plot_dir)
        print(f'Best checkpoint {trainer.state.best_model_checkpoint}')

        # Deploy the weights to Hugging Face
        trainer.push_to_hub()

        del trainer
        torch.cuda.empty_cache()

