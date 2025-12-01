import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from config.base_config import Config


def build_trainer(
    config: "Config",
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer | PreTrainedTokenizerFast",
    train_dataset: "Dataset",
    eval_dataset: "Dataset",
    report_to: list[str],
    run_name: str | None,
) -> Trainer:
    """
    Build and configure the SFTTrainer with LoRA.
    
    Args:
        config: Training configuration object
        model: The pretrained model to fine-tune
        tokenizer: The tokenizer
        train_dataset: Training dataset with "text" column
        eval_dataset: Evaluation dataset with "text" column
        report_to: List of reporting backends (e.g., ["wandb"] or [])
        run_name: The run name for logging (e.g., wandb run name)
        
    Returns:
        Configured SFTTrainer instance ready for training
    """
    print("Configuring LoRA and SFTTrainer...")
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        target_modules=config.lora_target_modules,
        task_type=config.lora_task_type,
    )
    
    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    if config.bf16 and not config.load_in_4bit:
        model = model.to(dtype=torch.bfloat16)

    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        evaluation_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        warmup_ratio=config.warmup_ratio,
        dataloader_num_workers=config.dataloader_num_workers,
        group_by_length=config.group_by_length,
        seed=config.seed,
        data_seed=config.seed,
        logging_dir=config.logging_dir,
        report_to=report_to,
        run_name=run_name,
    )

    model.config.use_cache = False

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    return trainer
