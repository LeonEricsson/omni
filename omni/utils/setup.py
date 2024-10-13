def validate_model_initialization(dataset, model, device, ignore_index=-1):
    """
    Verifies model initialization by checking if loss from a small data
    sample matches expected random predictions: log(vocab_size).

    Args:
        dataset: Dataset containing input_ids and attention_mask
        model: Language model to validate
        ignore_index: Token index to ignore in loss calculation (usually padding token)
    """
    from random import sample

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset

    # Validate inputs
    assert hasattr(dataset, "__len__"), "Dataset must support len()"
    assert len(dataset) > 0, "Dataset cannot be empty"
    assert all(
        col in dataset.features for col in ["input_ids", "attention_mask"]
    ), "Dataset must contain input_ids and attention_mask columns"

    model.eval()

    subset_indices = sample(range(len(dataset)), min(32, len(dataset)))
    subset = Subset(dataset, subset_indices)
    dataloader = DataLoader(subset, batch_size=32, shuffle=True)

    batch = next(iter(dataloader))
    input_ids = batch["input_ids"][:, :-1].to(device)
    target_ids = batch["input_ids"][:, 1:].to(device)
    attention_mask = batch["attention_mask"][:, :-1].to(device)

    assert input_ids.size(0) == attention_mask.size(0), "Batch sizes don't match"
    assert input_ids.size(1) == attention_mask.size(1), "Sequence lengths don't match"

    with torch.no_grad():
        logits, _ = model(input_ids, attention_mask)

        batch_size, seq_len, vocab_size = logits.size()

        assert logits.size(0) == input_ids.size(0), "Model output batch size mismatch"
        assert logits.size(1) == input_ids.size(
            1
        ), "Model output sequence length mismatch"

        logits = logits.reshape(batch_size * seq_len, vocab_size)
        target_ids = target_ids.reshape(batch_size * seq_len)

        loss = F.cross_entropy(logits, target_ids, ignore_index=ignore_index)

        expected_loss = torch.log(torch.tensor(vocab_size))
        assert torch.isclose(loss, expected_loss, atol=1), (
            f"Model is not initialized correctly. "
            f"Expected loss to be close to {expected_loss} but got {loss}"
        )


def parse_args(config):
    """Parse CLI arguments to override training config."""
    import argparse

    parser = argparse.ArgumentParser()

    # Add all training_config options as CLI arguments
    for key, value in config.items():
        arg_type = type(value)
        # Handle scientific notation for floats
        if isinstance(value, float):
            parser.add_argument(f"--{key}", type=float, default=value)
        else:
            parser.add_argument(f"--{key}", type=arg_type, default=value)

    args = parser.parse_args()
    return vars(args)


def create_checkpoint_folder(base_name: str) -> str:
    """Create a unique folder for saving checkpoints based on a base name and timestamp."""
    import os
    import time

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join("checkpoints", f"{base_name}_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir
