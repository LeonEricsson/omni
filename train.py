import json
import os
from pathlib import Path
from typing import Any
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader

import wandb
from logger import TrainingLogger
from omni.architectures.llama import LlamaConfig
from omni.modules.transformer import Transformer
from omni.utils.lr_schedule import CosineWarmupScheduler
from omni.utils.tools import auto_device
from omni.utils.tools import get_gpu_memory
from omni.utils.tools import get_system_stats

model_config = LlamaConfig(
    vocab_size=50258,
    seq_len=512,
    d_model=256,
    hidden_dim=512,
    num_heads=8,
    num_kv_heads=4,
    num_layers=6,
    rope_theta=0.1,
    norm_eps=1e-6,
    activation_fn="silu",
    mlp_bias=False,
    mlp_dropout=0.0,
    attention_bias=False,
    attention_dropout=0.0,
    pos_encoding_type="rope",
    mlp="mlp_swiglu",
    normalization="rmsnorm",
    attention="gqa",
)

training_config = {
    "batch_size": 32,
    "learning_rate": 5e-5,
    "num_epochs": 10,
    "eval_every": 5000,
    "warmup_steps": 1000,
    "total_steps": 10000,
    "gradient_clip_norm": 1.0,
    "seed": 42,
}

dataset_dir = Path(
    "data/pretokenized_roneneldan_TinyStories"
)  # pretokenized - run preprocess.py first


device = auto_device()
amp_available = torch.amp.autocast_mode.is_autocast_available(device.type)


def setup_wandb(config: Dict[str, Any]) -> None:
    wandb.init(
        project="Llama",
        config=config,
        notes="LLaMA architecture training on TinyStories",
        tags=["llama", "tinystories", "pre-training"],
        mode="online",
    )


def validate(
    test_dataloader: DataLoader,
    model: nn.Module,
    total_tokens: int,
    device: torch.device,
    logger: TrainingLogger,
    ignore_index: int = -1,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0
    val_tokens = 0

    logger.start_validation(len(test_dataloader))

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"][:, :-1].to(device)
            target_ids = batch["input_ids"][:, 1:].to(device)
            attention_mask = batch["attention_mask"][:, :-1].to(device)

            with torch.autocast(device.type, enabled=amp_available):
                logits = model(input_ids, attention_mask)
                batch_size, seq_len, vocab_size = logits.size()

                logits = logits.reshape(batch_size * seq_len, vocab_size)
                target_ids = target_ids.reshape(batch_size * seq_len)

                loss = F.cross_entropy(
                    logits, target_ids, ignore_index=ignore_index, reduction="sum"
                )

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

    logger.end_validation(metrics, total_tokens)
    return metrics


def train(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    gradient_clip_norm: float,
    eval_every: int = 100,
    ignore_index: int = -1,
) -> None:
    model.train()
    total_steps = 0
    total_tokens = 0

    scaler = torch.amp.GradScaler()

    with TrainingLogger(num_epochs, len(train_dataloader)) as logger:
        logger.start_training(device)
        for epoch in range(num_epochs):
            logger.start_epoch(epoch, total_tokens)

            for batch in train_dataloader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"][:, :-1].to(device)
                target_ids = batch["input_ids"][:, 1:].to(device)
                attention_mask = batch["attention_mask"][:, :-1].to(device)

                with torch.autocast(device.type, enabled=amp_available):
                    logits = model(input_ids, attention_mask)

                    batch, seq, vocab_size = logits.size()
                    logits = logits.reshape(batch * seq, vocab_size)
                    target_ids = target_ids.reshape(batch * seq)
                    loss = F.cross_entropy(
                        logits, target_ids, ignore_index=ignore_index
                    )

                scaler.scale(loss).backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=gradient_clip_norm
                )

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                total_steps += 1
                batch_tokens = attention_mask.sum().item()
                total_tokens += batch_tokens

                if total_steps % eval_every == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    # metrics = {
                    #     "train/loss": loss.item(),
                    #     "train/learning_rate": current_lr,
                    #     "train/epoch": epoch + 1,
                    #     "train/step": total_steps,
                    #     "train/total_tokens": total_tokens,
                    #     **get_gpu_memory(),
                    #     **get_system_stats(),
                    # }
                    metrics = {
                        "train/loss": loss.item(),
                        "train/learning_rate": current_lr,
                        "Epoch": epoch + 1,
                        "Tokens": total_tokens,
                    }
                    logger.log_training_step(metrics, total_steps)

                    # Run validation
                    model.eval()
                    validate(
                        val_dataloader,
                        model,
                        total_tokens,
                        device,
                        logger,
                        ignore_index,
                    )
                    model.train()

                logger.advance_train()

            logger.end_epoch(epoch, total_tokens)
        logger.end_training()


def extract_metadata(dataset_dir: str) -> Dict[str, Any]:
    metadata_path = os.path.join(dataset_dir, "preprocessing_metadata.json")
    with open(metadata_path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    torch.manual_seed(training_config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_config["seed"])

    setup_wandb({**model_config.__dict__, **training_config})

    # load data
    data_metadata = extract_metadata(str(dataset_dir))
    max_seq_length = data_metadata["preprocessing_params"]["max_seq_length"]
    pad_token_id = data_metadata["pad_token_id"]
    assert max_seq_length >= model_config.seq_len

    # create dataloaders
    dataset = load_from_disk(str(dataset_dir))
    train_size = int(0.999 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = Transformer(model_config).to(device)
    if torch.cuda.is_available():
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=training_config["warmup_steps"],
        total_steps=training_config["total_steps"],
    )

    train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_clip_norm=training_config["gradient_clip_norm"],
        num_epochs=training_config["num_epochs"],
        eval_every=training_config["eval_every"],
        ignore_index=pad_token_id,
    )
    wandb.finish()
