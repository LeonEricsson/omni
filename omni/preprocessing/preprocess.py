import json
from pathlib import Path

import tokenizers.processors as processors
import torch.nn.functional as F
from datasets import (
    Dataset,
    DownloadMode,
    Sequence,
    Value,
    load_dataset,
    load_from_disk,
)
from jaxtyping import Bool, Int
from torch import Tensor

from omni.preprocessing.tokenizer import AutoTokenizer


def prepare_dataset(
    dataset_name: str,
    tokenizer: AutoTokenizer,
    min_seq_length: Int,
    max_seq_length: Int,
    name: str | None = None,
    split_long_sequences: Bool = True,
    num_proc: Int = 16,
    split: str | None = None,
    revision: str | None = None,
    data_dir: str | None = None,
    data_files: str | None = None,
    output_dir: Path = Path("./data"),
    cache_dir: Path = Path("./hf_cache"),
    push: bool = False,
    hf_username: str | None = None,
) -> None:
    """
    Preprocesses a text dataset from HuggingFace Hub and store it locally.

    Tokenizes text using provided tokenizer, handles sequences based on length constraints, and saves
    the processed dataset with metadata. Can optionally push to HuggingFace Hub.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g., 'organization/dataset')
        tokenizer: Tokenizer instance used for text tokenization
        name: Dataset configuration name if applicable
        min_seq_length: Minimum sequence length (shorter sequences are discarded)
        max_seq_length: Maximum sequence length (longer sequences are split or truncated)
        split_long_sequences: If True, splits sequences > max_length into chunks resulting in more total samples. If False, truncates
        num_proc: Number of processes for parallel preprocessing
        split: Dataset split to process ('train', 'validation'). If None, will return dict of all splits
        revision: Dataset version/revision
        data_dir: Defining the data_dir of the dataset configuring on HF.
        data_files: Path(s) to source data file(s) on HF.
        output_dir: Root directory for saving processed datasets
        cache_dir: Directory for HuggingFace cache
        push: Whether to push processed dataset to HuggingFace Hub
        hf_username: Required if push=True, username for HuggingFace Hub
    """
    assert min_seq_length <= max_seq_length
    assert tokenizer.pad_token_id is not None
    assert tokenizer.bos_token_id is not None

    dataset = _download_dataset(
        dataset_name=dataset_name,
        name=name,
        split=split,
        revision=revision,
        data_dir=data_dir,
        data_files=data_files,
        cache_dir=cache_dir,
        num_proc=num_proc,
    )

    # max seq length + 1 to account for bos token. input and target both trim
    # 1 token off the sequence from left and right respectively.
    dataset = _preprocess_and_tokenize_dataset(
        dataset,
        tokenizer.tokenizer,
        max_seq_length + 1,
        split_long_sequences,
        num_proc,
    )

    print("Final dataset: {dataset}")

    metadata = {
        "dataset_name": dataset_name,
        "tokenizer_name": tokenizer.name_or_path,
        "pad_token_id": tokenizer.pad_token_id,
        "preprocessing_params": {
            "min_seq_length": min_seq_length,
            "max_seq_length": max_seq_length,
            "split_long_sequences": split_long_sequences,
            "num_proc": num_proc,
            "split": split,
            "revision": revision,
        },
    }

    _save(dataset, metadata, output_dir)

    if push:
        assert hf_username is not None
        id = f"{hf_username}/pretokenized_{name}_{tokenizer.name_or_path}_ctx{max_seq_length}"
        dataset.push_to_hub(id, split=split)


def _download_dataset(
    dataset_name: str,
    name: str,
    split: str | None,
    revision: str | None,
    data_dir: str | None,
    data_files: str | None,
    cache_dir: Path,
    num_proc: Int,
) -> Dataset:

    local_path = cache_dir / dataset_name
    if local_path.is_dir():
        print(f"Loading dataset from local path: {local_path}")
        ds = load_from_disk(local_path)
    else:
        print("Downloading dataset from HF Hub...")
        ds = load_dataset(
            dataset_name,
            name=name,
            split=split,
            revision=revision,
            data_dir=data_dir,
            data_files=data_files,
            download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS,
            cache_dir=cache_dir,
            num_proc=num_proc,
        )
    return ds.select_columns("text")


def _add_bos_token(tokenizer: AutoTokenizer):
    tokenizer._tokenizer.post_processor = processors.Sequence(
        [
            processors.ByteLevel(trim_offsets=False),
            processors.TemplateProcessing(
                single=f"{tokenizer.bos_token}:0 $A:0",
                pair=f"{tokenizer.bos_token}:0 $A:0 {tokenizer.bos_token}:1 $B:1",
                special_tokens=[
                    (tokenizer.bos_token, tokenizer.bos_token_id),
                ],
            ),
        ]
    )
    return tokenizer


def _preprocess_and_tokenize_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_seq_length: Int,
    split_long_sequences: Bool,
    num_proc: Int,
):
    """
    Tokenizes and preprocesses a dataset for sequence modeling tasks.

    This function adds a beginning-of-sequence (BOS) token to the tokenizer and tokenizes
    the dataset's text, with optional truncation and support for handling sequences that
    exceed the specified `max_seq_length`. All sequences are padded to `max_seq_length`.
    """
    print("Tokenizing and preprocessing dataset...")
    tokenizer = _add_bos_token(tokenizer)
    dataset = dataset.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_overflowing_tokens=split_long_sequences,
        ),
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
    )
    dataset = dataset.select_columns(["input_ids", "attention_mask"])
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    dataset = dataset.cast_column(
        "attention_mask", Sequence(feature=Value(dtype="int8"))
    )
    return dataset


def _tokenize(dataset: Dataset, tokenizer: AutoTokenizer, num_proc: Int) -> Dataset:
    """Deprecated tokenize function that was used together with _process_sequences."""
    print("Tokenizing dataset...")
    tokenizer = _add_bos_token(tokenizer)
    dataset = dataset.map(
        lambda x: tokenizer(x["text"]),
        batched=True,
        num_proc=num_proc,
        remove_columns=["text"],
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return dataset


def _process_sequences(
    dataset: Dataset,
    min_seq_length: Int,
    max_seq_length: Int,
    padding_token: Int,
) -> Dataset:
    """Deprecated function that was used to split and pad sequences. Now handleded
    by the 'return_overflowing_tokens' argument in the tokenizer."""
    print("Preprocessing sequences...")
    dataset = dataset.filter(lambda x: len(x["input_ids"]) >= min_seq_length)

    def split_and_pad(example):
        """
        Splits a sequence into fixed-length chunks, discarding the last chunk if smaller than
        `min_seq_length`, and pads all chunks to `max_seq_length`. The input is padded only as
        needed to ensure divisibility, and the result is reshaped into chunks of uniform size.
        """
        input_ids: Int[Tensor, "batch len"] = example["input_ids"]
        attention_mask: Int[Tensor, "batch len"] = example["attention_mask"]
        total_length = input_ids.shape[1]

        num_full_chunks = total_length // max_seq_length
        last_chunk_length = total_length % max_seq_length

        # decide whether to keep or discard last chunk
        if last_chunk_length > 0 and last_chunk_length < min_seq_length:
            total_length = num_full_chunks * max_seq_length
            last_chunk_length = 0

        padded_length = (
            total_length
            if last_chunk_length == 0
            else total_length + (max_seq_length - last_chunk_length)
        )
        padded_input_ids = F.pad(
            input_ids,
            (0, padded_length - input_ids.shape[1]),
            value=padding_token,
        )
        padded_attention_mask = F.pad(
            attention_mask,
            (0, padded_length - attention_mask.shape[1]),
            value=0,
        )

        chunked_input_ids = padded_input_ids.view(-1, max_seq_length)
        chunked_attention_mask = padded_attention_mask.view(-1, max_seq_length)

        return {
            "input_ids": chunked_input_ids,
            "attention_mask": chunked_attention_mask,
        }

    return dataset.map(
        split_and_pad,
        remove_columns=dataset.column_names,
        batched=True,
        batch_size=1,
    )


def _save(dataset: Dataset, metadata: dict, output_dir: Path) -> None:
    print("Saving dataset...")
    dataset_dir_name = f"pretokenized_{metadata['dataset_name'].replace('/', '_')}"
    dataset_dir = output_dir / dataset_dir_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset.save_to_disk(dataset_dir)

    with open(dataset_dir / "preprocessing_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
