import asyncio
import torch
from transformers import AutoTokenizer
from dataset import DatasetLoader as DatasetLoaderOld
from dl import DatasetLoader as DatasetLoaderNew

def inspect_token_range(b1, b2, tokenizer, name="inputs", max_tokens=10):
    """Inspect why token ranges differ — checks unique tokens, special tokens, and offsets."""
    uniq1, uniq2 = set(b1.unique().tolist()), set(b2.unique().tolist())

    missing_in_2 = sorted(list(uniq1 - uniq2))
    missing_in_1 = sorted(list(uniq2 - uniq1))

    print(f"--- Inspecting {name} vocabulary difference ---")
    print(f"Loader1 unique tokens: {len(uniq1)}, range {min(uniq1)}–{max(uniq1)}")
    print(f"Loader2 unique tokens: {len(uniq2)}, range {min(uniq2)}–{max(uniq2)}")

    if missing_in_2:
        decoded = [tokenizer.decode([t]) for t in missing_in_2[:max_tokens]]
        print(f"Tokens only in Loader1 ({len(missing_in_2)}): {missing_in_2[:max_tokens]}  →  {decoded}")
    if missing_in_1:
        decoded = [tokenizer.decode([t]) for t in missing_in_1[:max_tokens]]
        print(f"Tokens only in Loader2 ({len(missing_in_1)}): {missing_in_1[:max_tokens]}  →  {decoded}")

    # Heuristic: detect offset shift (e.g., 0–50256 vs 1–50256)
    if min(uniq1) != min(uniq2):
        offset = min(uniq2) - min(uniq1)
        print(f"⚠️  Possible offset difference of {offset} between vocabularies\n")

    # Show if special tokens are the likely cause
    specials = {k: v for k, v in tokenizer.special_tokens_map_extended.items()}
    specials_found = [(k, v) for k, v in specials.items() if v in uniq1 or v in uniq2]
    if specials_found:
        print("\nSpecial tokens present:")
        for k, v in specials_found:
            in1, in2 = v in uniq1, v in uniq2
            print(f"  {k}: id={v} → in Loader1={in1}, in Loader2={in2}")

def compare_batch_properties(batch1, batch2, tokenizer, name="inputs"):
    """Compare shape, dtype, and token range between two batches, and report EOS tokens."""
    b1, b2 = batch1, batch2

    shape_match = b1.shape == b2.shape
    dtype_match = b1.dtype == b2.dtype
    token_range_match = (b1.min().item() == b2.min().item() and b1.max().item() == b2.max().item())

    if not token_range_match:
        inspect_token_range(b1, b2, tokenizer, name)

    status = "✅" if shape_match and dtype_match and token_range_match else "❌"

    eos1 = (b1 == tokenizer.eos_token_id).sum().item()
    eos2 = (b2 == tokenizer.eos_token_id).sum().item()

    print(f"--- {name.capitalize()} comparison {status} ---")
    print(f"Shape: Loader1 {b1.shape}, Loader2 {b2.shape}")
    print(f"Dtype: Loader1 {b1.dtype}, Loader2 {b2.dtype}")
    print(f"Token range: Loader1 {b1.min().item()}-{b1.max().item()}, "
          f"Loader2 {b2.min().item()}-{b2.max().item()}")
    print(f"EOS tokens: Loader1 {eos1}, Loader2 {eos2}\n")


async def compare_loaders():
    """Compare two loaders with same seed to verify reproducibility"""
    print("\n=== DATA LOADER COMPARISON ===\n")

    batch_size = 2
    sequence_length = 512
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    loader1 = await DatasetLoaderOld.create(
        batch_size=batch_size,
        sequence_length=sequence_length,
        pages_info=await DatasetLoaderOld.next_pages(offset=0, n_pages=1, seed=42),
        tokenizer=tokenizer,
        pack_samples=False
    )

    loader2 = DatasetLoaderNew(
        tokenizer=tokenizer,
        sequence_length=sequence_length
    )
    loader2.load_bucket_configs()
    await loader2.load_bucket_data_to_buffer(
        max_configs=3,
        max_shards=3,
        max_row_groups=2,
        max_rows_per_group=2
    )
    loader2.prepare_batches(
        batch_size=batch_size,
        sequence_length=sequence_length
    )

    print(f"Loader 1 buffer length: {len(loader1.buffer)}")
    print(f"Loader 2 buffer length: {len(loader2.buffer)}\n")

    batch1 = next(iter(loader1))
    batch2 = next(iter(loader2))

    compare_batch_properties(batch1[0], batch2[0], tokenizer, name="inputs")
    # compare_batch_properties(batch1[1], batch2[1], tokenizer, name="labels")

    # sample_input1 = batch1[0][0][:30]
    # sample_input2 = batch2[0][0][:30]
    # print(f"Loader1: Sample input  {sample_input1.tolist()}")
    # print(f"Loader1: Sample decoded  {tokenizer.decode(sample_input1, skip_special_tokens=False)}")
    # print(f"Loader2: Sample input  {sample_input2.tolist()}")
    # print(f"Loader2: Sample decoded  {tokenizer.decode(sample_input2, skip_special_tokens=False)}\n")

# async def test_edge_cases():
#     """Compare loaders with small sequence/batch sizes"""
#     print("\n=== EDGE CASE COMPARISON ===\n")
#     batch_size = 2
#     sequence_length = 50
#     tokenizer = AutoTokenizer.from_pretrained("gpt2")
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     loader1 = await DatasetLoaderOld.create(
#         batch_size=batch_size,
#         sequence_length=sequence_length,
#         pages_info=await DatasetLoaderOld.next_pages(offset=0, n_pages=1, seed=123),
#         tokenizer=tokenizer,
#         pack_samples=False
#     )

#     loader2 = DatasetLoaderNew(tokenizer=tokenizer, sequence_length=sequence_length)
#     loader2.load_bucket_configs()
#     await loader2.load_bucket_data_to_buffer(max_configs=1, max_shards=1, max_row_groups=1, max_rows_per_group=1)
#     loader2.prepare_batches(batch_size=batch_size, sequence_length=sequence_length)

#     batch1 = next(iter(loader1))
#     batch2 = next(iter(loader2))

#     for name, b1, b2 in [("inputs", batch1[0], batch2[0]), ("labels", batch1[1], batch2[1])]:
#         print(f"{name} shape comparison: Loader1 {b1.shape}, Loader2 {b2.shape}")

async def main():
    await compare_loaders()
    # await test_edge_cases()


if __name__ == "__main__":
    asyncio.run(main())
