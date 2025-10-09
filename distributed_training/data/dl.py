import os
import json
import yaml
import time
import asyncio
import torch
import s3fs
import pyarrow.parquet as pq
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from transformers import AutoTokenizer
import bittensor as bt


class BatchLoader:
    def __init__(self, tokenizer=None, batch_size=None, sequence_length=None):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        self.buffer = []
        self._data = None
        self._num_batches = 0
        self._batch_idx = 0

    def prepare_batches(self, batch_size=None, sequence_length=None, device="cpu"):
        """Convert token buffer to tensor batches."""
        batch_size = batch_size or self.batch_size
        sequence_length = sequence_length or self.sequence_length

        token_buffer = self.buffer
        total_tokens = len(token_buffer)
        num_sequences = total_tokens // sequence_length
        trimmed_tokens = token_buffer[: num_sequences * sequence_length]

        data = torch.tensor(trimmed_tokens, dtype=torch.long, device=device)
        data = data.view(num_sequences, sequence_length)
        num_batches = num_sequences // batch_size

        self._data = data
        self._num_batches = num_batches
        self._batch_idx = 0
        self.batch_size = batch_size
        self.sequence_length = sequence_length

    def __iter__(self):
        if self._data is None:
            raise RuntimeError("Call prepare_batches() before iterating.")
        self._batch_idx = 0
        return self

    def __next__(self):
        if self._batch_idx >= self._num_batches:
            raise StopIteration
        i = self._batch_idx
        batch = self._data[i * self.batch_size : (i + 1) * self.batch_size]
        self._batch_idx += 1
        return batch, batch.clone()

    def __len__(self):
        return self._num_batches

    def get_batch(self, batch_size=2):
        """Yield raw token slices directly from the buffer (without tensorizing)."""
        for i in range(0, len(self.buffer), batch_size):
            yield self.buffer[i : i + batch_size]


class DatasetLoader(BatchLoader):
    def __init__(
        self,
        max_configs=None,
        max_shards=None,
        max_row_groups=None,
        max_rows_per_group=None,
        tokenizer=None,
        batch_size=None,
        sequence_length=None,
    ):
        super().__init__(tokenizer=tokenizer, batch_size=batch_size, sequence_length=sequence_length)

        self.logger = bt.logging

        load_dotenv(find_dotenv())

        self.max_configs = max_configs
        self.max_shards = max_shards
        self.max_row_groups = max_row_groups
        self.max_rows_per_group = max_rows_per_group

        self.BUCKET = os.getenv("R2_BUCKET") or (_ for _ in ()).throw(ValueError("R2_BUCKET env var not set"))
        self.ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID") or (_ for _ in ()).throw(ValueError("R2_ACCOUNT_ID env var not set"))
        self.ACCESS_KEY = os.getenv("R2_ACCESS_KEY") or (_ for _ in ()).throw(ValueError("R2_ACCESS_KEY env var not set"))
        self.SECRET_KEY = os.getenv("R2_SECRET_KEY") or (_ for _ in ()).throw(ValueError("R2_SECRET_KEY env var not set"))

        self.DATASET = "HuggingFaceFW_fineweb-edu-score-2"
        self.META_NAME = "_metadata.yaml"
        self.SHARD_NAME = "_shard_sizes.json"

        self.CACHE_DIR = ".cache"
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        self.meta_cache_path = os.path.join(self.CACHE_DIR, self.META_NAME)
        self.shard_cache_path = os.path.join(self.CACHE_DIR, self.SHARD_NAME)

        self.fs = s3fs.S3FileSystem(
            key=self.ACCESS_KEY,
            secret=self.SECRET_KEY,
            client_kwargs={"endpoint_url": f"https://{self.ACCOUNT_ID}.r2.cloudflarestorage.com"},
        )

        self.metadata = {}
        self.shard_sizes = {}

        self.logger.info(f"WELCOME TO THE NEW DATALOADER! {self}")

        self.configs = []
        self.shards = []
        self.total_row_groups_loaded = 0
        self.total_rows_loaded = 0


    def download_config(self, remote_path, local_path):
        """Download file from S3 if missing locally."""
        if os.path.exists(local_path):
            return
        data = self.fs.cat(remote_path)
        with open(local_path, "wb") as dst:
            dst.write(data)

    def load_bucket_configs(self):
        """Load metadata and shard size files."""
        self.download_config(f"{self.BUCKET}/{self.DATASET}/{self.META_NAME}", self.meta_cache_path)
        self.download_config(f"{self.BUCKET}/{self.DATASET}/{self.SHARD_NAME}", self.shard_cache_path)

        with open(self.meta_cache_path, "r") as f:
            self.metadata = yaml.safe_load(f)

        with open(self.shard_cache_path, "r") as f:
            self.shard_sizes = json.load(f)

    def list_shard_files(self, config):
        """Return shard file paths for a given config."""
        config_info = self.shard_sizes.get(config, {})
        shards = config_info.get("shards", [])
        return [shard["path"] for shard in shards]

    async def get_configs_async(self):
        """Fetch all available configs from metadata (checking existence)."""
        all_configs = [c.get("config_name") for c in self.metadata.get("configs", []) if c.get("config_name")]
        async def check_config(config):
            config_path = f"{self.BUCKET}/{self.DATASET}/{config}"
            exists = await asyncio.to_thread(self.fs.exists, config_path)
            return config if exists else None

        results = await asyncio.gather(*(check_config(c) for c in all_configs))
        return [r for r in results if r]

    async def tokenize_texts(self, texts):
        """Tokenize multiple texts asynchronously using run_in_executor."""
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(
                None,
                lambda t=text: self.tokenizer.encode(
                    t,
                    truncation=True,
                    max_length=self.sequence_length
                )
            )
            for text in texts
        ]
        encoded = await asyncio.gather(*tasks)
        return encoded

    async def load_shard(self, shard_path, max_row_groups=2, max_rows_per_group=1):
        """Load and tokenize text from a single shard (async)."""
        buffer = []
        row_groups_loaded = 0
        rows_loaded = 0
        try:
            reader = await asyncio.to_thread(pq.ParquetFile, f"s3://{shard_path}", filesystem=self.fs)
        except Exception as e:
            print(f"Failed to open shard {shard_path}: {e}")
            return buffer

        for rg_idx in range(min(reader.num_row_groups, max_row_groups)):
            row_group = await asyncio.to_thread(reader.read_row_group, rg_idx, columns=["text"])
            df = await asyncio.to_thread(row_group.to_pandas)
            df = df.head(max_rows_per_group)

            row_groups_loaded += 1
            rows_loaded += len(df)

            encodings = await self.tokenize_texts(df["text"].tolist())
            for ids in encodings:
                ids.append(self.tokenizer.eos_token_id)
                buffer.extend(ids)

        self.total_row_groups_loaded += row_groups_loaded
        self.total_rows_loaded += rows_loaded

        return buffer

    async def gather_configs_and_shards(self, configs=None, max_configs=3, max_shards=3):
        """Collect shard paths from multiple configs."""
        if configs is None:
            configs = await self.get_configs_async()
        configs = configs[:max_configs]

        async def get_shards(config):
            return await asyncio.to_thread(self.list_shard_files, config)

        shard_lists = await asyncio.gather(*(get_shards(c) for c in configs))
        all_shards = []
        for config, shards in zip(configs, shard_lists):
            all_shards.extend(shards[:max_shards])

        self.configs = configs
        self.shards = all_shards

        return all_shards

    async def fetch_shard_data(self, shard_paths, max_row_groups=2, max_rows_per_group=1):
        """Download and tokenize multiple shards asynchronously."""
        semaphore = asyncio.Semaphore(10)
        async def load_with_limit(shard):
            async with semaphore:
                return await self.load_shard(shard, max_row_groups, max_rows_per_group)
        results = await asyncio.gather(*(load_with_limit(p) for p in shard_paths))
        return [token for shard_buffer in results for token in shard_buffer]

    async def load_bucket_data_to_buffer(self, configs=None, max_configs=3, max_shards=3, max_row_groups=2, max_rows_per_group=1):
        """High-level pipeline: gather shards → load data → fill buffer."""
        all_shards = await self.gather_configs_and_shards(configs, max_configs, max_shards)
        self.buffer = await self.fetch_shard_data(all_shards, max_row_groups, max_rows_per_group)
        return self.buffer


if __name__ == "__main__":
    max_configs = 3
    max_shards = 2
    max_row_groups = 2
    max_rows_per_group = 2
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    batch_size = 8
    sequence_length = 1024

    loader = DatasetLoader(
        tokenizer=tokenizer,
        sequence_length=sequence_length,
    )

    loader.load_bucket_configs()

    start_time = time.perf_counter()

    asyncio.run(loader.load_bucket_data_to_buffer(
        max_configs=max_configs,
        max_shards=max_shards,
        max_row_groups=max_row_groups,
        max_rows_per_group=max_rows_per_group
    ))

    end_time = time.perf_counter()
    print(f"load_bucket_data_to_buffer took {end_time - start_time:.2f}s")

    print(f"- len(loader.configs): {len(loader.configs)} max_configs={max_configs}")
    print(f"- len(loader.shards): {len(loader.shards)} max_shards={max_shards}")
    print(f"- Row groups loaded: {loader.total_row_groups_loaded} max_row_groups={max_row_groups}")
    print(f"- Rows loaded: {loader.total_rows_loaded} max_rows_per_group={max_rows_per_group}\n")

    print(f"len(loader.buffer): {len(loader.buffer)}\n")

    loader.prepare_batches(batch_size=batch_size, sequence_length=sequence_length)
    print(f"Batches: {len(loader)}")

    for i, (inputs, labels) in enumerate(loader):
        print(f"Batch {i}: input_ids shape {inputs.shape}")
        print(f"Batch {i}: labels shape {labels.shape}")
        if i >= 1:
            break
