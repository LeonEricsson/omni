""" Utility functions - primarily for training scripts. """


def auto_device(device: str = None):
    """
    Utility function to validate or determine a torch.device.

    Args:
        device (str, optional): The device string (e.g., "cuda", "cpu"). Defaults to None.

    Returns:
        torch.device: A validated or determined torch.device instance.
    """
    import torch

    if device:
        try:
            validated_device = torch.device(device)
            if validated_device.type == "cuda" and not torch.cuda.is_available():
                raise ValueError(
                    f"CUDA is not available, but 'cuda' device was requested."
                )
            return validated_device
        except Exception as e:
            raise ValueError(f"Invalid device string provided: {device}. Error: {e}")
    else:
        # Determine an available device
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_built():
            return torch.device("mps")
        else:
            return torch.device("cpu")


def get_gpu_memory():
    import GPUtil

    try:
        gpu = GPUtil.getGPUs()[0]  # Assuming single GPU setup
        return {
            "gpu_memory_used": gpu.memoryUsed,
            "gpu_memory_total": gpu.memoryTotal,
            "gpu_memory_util": gpu.memoryUtil * 100,
        }
    except Exception:
        return {}


def get_system_stats():
    import psutil

    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
    }


def num_params(model, include_embeddings: bool = False):
    num_trainable_params = sum(
        p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad
        and (
            include_embeddings
            or not any(kw in name for kw in ["token_emb", "vocab_proj"])
        )
    )
    print(f"Number of trainable parameters: {num_trainable_params:,}")
