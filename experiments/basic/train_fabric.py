"""
A training example for a 20M parameter Llama style transformer on the TinyStories dataset
- using PyTorch Lightning Fabric.
"""

import json
import os
import time
from typing import Any, Dict

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from jaxtyping import Float, Int
from logger import TrainingLogger
from torch.utils.data import DataLoader

import wandb
from omni.architectures.llama import LlamaConfig
from omni.modules.transformer import Transformer
from omni.utils.lr_schedule import CosineWarmupScheduler
from omni.utils.setup import (
    create_checkpoint_folder,
    parse_args,
    validate_model_initialization,
)
from omni.utils.system import auto_device

torch.set_float32_matmul_precision(precision="high")

llama_config = LlamaConfig(
    vocab_size=50258,
    seq_len=512,
    d_model=768,
    num_heads=8,
    num_kv_heads=8,
    num_layers=6,
    activation_fn="silu",
    mlp_bias=False,
    mlp_dropout=0.1,
    attention_bias=False,
    attention_dropout=0.1,
    weight_tying=False,
    pos_encoding_type="rope",
    mlp="mlp_swiglu",
    normalization="rmsnorm",
    attention="mha",
)

training_config = {
    # "dataset_dir": "data/pretokenized_fineweb-edu-2BT",  # pretokenized - run preprocess.py first
    "dataset_dir": "data/pretokenized_roneneldan_TinyStories",
    "batch_size": 32,
    "learning_rate": 5e-4,
    "min_lr": 5e-5,
    "num_epochs": 8,
    "eval_every": 2000,
    "warmup_steps": 1000,
    "tot_steps": 1e6,
    "gradient_clip_norm": 1.0,
    "gradient_acc_steps": 4,
    "seed": 42,
    "num_workers": 4,
    "device": "cpu",  # auto-detect
    "num_devices": 1,
    "strategy": "auto",
    "precision": "16-mixed",
}


def setup_wandb(config: Dict[str, Any]) -> None:
    wandb.init(
        project="Llama",
        config=config,
        notes="LLaMA architecture training on TinyStories",
        tags=["llama", "tinystories", "pre-training"],
        mode="online",
    )


def validate(
    val_dataloader: DataLoader,
    model: nn.Module,
    logger: TrainingLogger,
    ignore_index: Int = -1,
) -> Dict[str, float]:
    """
    Validate the model, logging loss and perplexity.

    Args:
        val_dataloader (DataLoader): DataLoader for validation data.
        model (L.LightningModule): Model to be validated.
        total_tokens (int): Total number of tokens processed.
        logger (TrainingLogger): Logger for tracking validation metrics.
        ignore_index (int): Index to ignore in the loss calculation.

    Returns:
        Dict[str, Any]: Validation metrics.
    """
    model.eval()
    total_loss = 0
    val_tokens = 0

    logger.start_validation(len(val_dataloader))

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"][:, :-1]
            target_ids = batch["input_ids"][:, 1:]
            attention_mask = batch["attention_mask"][:, :-1]

            logits = model(input_ids, attention_mask)
            loss = cross_entropy_loss(logits, target_ids, ignore_index, reduction="sum")

            total_loss += loss.item()
            val_tokens += attention_mask.sum().item()

            logger.advance_validation()

    avg_loss = total_loss / val_tokens
    clipped_loss = min(avg_loss, 100)
    perplexity = torch.exp(torch.tensor(clipped_loss))

    metrics = {
        "val/loss": avg_loss,
        "val/perplexity": perplexity.item(),
    }

    return metrics


def train(
    fabric: L.Fabric,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    model: L.LightningModule,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_epochs: Int,
    gradient_clip_norm: Float,
    gradient_acc_steps: Int = 1,
    eval_every: Int = 500,
    ignore_index: Int = -1,
) -> None:
    """
    Trains the model for 'num_epochs'.

    Args:
        fabric (L.Fabric): Fabric object for distributed training.
        train_dataloader (DataLoader): DataLoader for training data.
        val_dataloader (DataLoader): DataLoader for validation data.
        model (L.LightningModule): Model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        num_epochs (int): Number of epochs to train.
        gradient_clip_norm (float): Maximum norm for gradient clipping.
        eval_every (int, optional): Frequency of evaluation during training. Defaults to 100.
        ignore_index (int, optional): Index to ignore in the loss calculation. Defaults to -1.
    """
    model.train()
    total_steps = 0
    total_tokens = 0
    start_time = time.perf_counter()

    optimizer.zero_grad()

    with TrainingLogger(num_epochs, len(train_dataloader)) as logger:
        logger.start_training(fabric.device)
        for epoch in range(num_epochs):
            logger.start_epoch(epoch, total_steps)

            for step, batch in enumerate(train_dataloader, start=1):
                input_ids = batch["input_ids"][:, :-1]
                target_ids = batch["input_ids"][:, 1:]
                attention_mask = batch["attention_mask"][:, :-1]

                is_accumulating = step % gradient_acc_steps != 0 and step != len(
                    train_dataloader
                )

                # skip .backward() synchronization during gradient accumulation
                with fabric.no_backward_sync(model, enabled=is_accumulating):
                    logits = model(input_ids, attention_mask)

                    loss = (
                        cross_entropy_loss(
                            logits, target_ids, ignore_index, reduction="mean"
                        )
                        / gradient_acc_steps
                    )
                    fabric.backward(loss)

                # gradient accumulation
                if not is_accumulating:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=gradient_clip_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                total_steps += 1
                batch_tokens = attention_mask.sum().item()
                total_tokens += batch_tokens

                # eval and log
                if total_steps % eval_every == 0:
                    elapsed_time = time.perf_counter() - start_time
                    tokens_per_second = total_tokens / elapsed_time
                    train_metrics = {
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "utils/tokens_per_second": tokens_per_second,
                    }
                    logger.log_training_step(train_metrics, total_steps)

                    model.eval()
                    validation_metrics = validate(
                        val_dataloader,
                        model,
                        logger,
                        ignore_index,
                    )
                    logger.log_validation_step(validation_metrics, total_steps)
                    model.train()

                train_metrics = {
                    "train/loss": (loss * gradient_acc_steps).item(),
                    "Tokens": total_tokens,
                }
                logger.log_training_step(train_metrics, total_steps)

                logger.advance_train()

            logger.end_epoch(epoch)
        logger.end_training()

    return model


def cross_entropy_loss(logits, target_ids, ignore_index, reduction):
    batch_size, seq_len, vocab_size = logits.size()
    logits = logits.reshape(batch_size * seq_len, vocab_size)
    target_ids = target_ids.reshape(batch_size * seq_len)
    return F.cross_entropy(
        logits, target_ids, ignore_index=ignore_index, reduction=reduction
    )


def extract_metadata(dataset_dir: str) -> Dict[str, Any]:
    metadata_path = os.path.join(dataset_dir, "preprocessing_metadata.json")
    with open(metadata_path, "r") as f:
        return json.load(f)


def save_checkpoint(
    checkpoint_dir: str, filename: str, model: nn.Module, fabric: L.Fabric
):
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    state = {"model": model}
    fabric.save(checkpoint_path, state)


def main():
    cli_args = parse_args(training_config)
    training_config.update(cli_args)

    device = auto_device(training_config["device"])

    fabric = L.Fabric(
        accelerator=device.type,
        devices=training_config["num_devices"],
        precision=training_config["precision"],
        strategy=training_config["strategy"],
    )
    fabric.launch()
    fabric.seed_everything(training_config["seed"])

    setup_wandb({**llama_config.__dict__, **training_config})

    ### DATA ###
    data_metadata = extract_metadata(training_config["dataset_dir"])
    max_seq_length = data_metadata["preprocessing_params"]["max_seq_length"]
    pad_token_id = data_metadata["pad_token_id"]
    assert max_seq_length >= llama_config.seq_len

    dataset = load_from_disk(str(training_config["dataset_dir"]))
    train_size = int(0.99 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    def init_dataloader(dataset):
        return DataLoader(
            dataset,
            batch_size=training_config["batch_size"],
            shuffle=True,
            num_workers=training_config["num_workers"],
            pin_memory=True,
            drop_last=True,  # avoid torch model recompilation
        )

    train_dataloader = init_dataloader(train_dataset)
    val_dataloader = init_dataloader(val_dataset)

    ### MODEL ###
    model = Transformer(llama_config)
    if device.type == "cuda":
        model = torch.compile(model, fullgraph=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=training_config["warmup_steps"],
        total_steps=training_config["tot_steps"],
        min_lr=training_config["min_lr"],
    )

    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        train_dataloader, val_dataloader
    )

    validate_model_initialization(dataset, model, device, ignore_index=pad_token_id)

    checkpoint_dir = create_checkpoint_folder("llama-30M")
    save_checkpoint(checkpoint_dir, "init.ckpt", model, fabric)
    exit()
    model = train(
        fabric=fabric,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_clip_norm=training_config["gradient_clip_norm"],
        gradient_acc_steps=training_config["gradient_acc_steps"],
        num_epochs=training_config["num_epochs"],
        eval_every=training_config["eval_every"],
        ignore_index=pad_token_id,
    )

    save_checkpoint(checkpoint_dir, "final.ckpt", model, fabric)

    wandb.finish()


if __name__ == "__main__":
    main()
