import os
import json
import yaml
import time
import asyncio
import torch
import s3fs
import pyarrow.parquet as pq
from dotenv import load_dotenv, find_dotenv
from transformers import AutoTokenizer
import bittensor as bt
import random


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

class DatasetLoader(BatchLoader):
    def __init__(
        self,
        max_configs=None,
        max_shards=3,
        max_row_groups=4,
        max_rows_per_group=None,
        tokenizer=None,
        batch_size=None,
        sequence_length=None,
        debug=None,
        randomness=None
    ):
        super().__init__(tokenizer=tokenizer, batch_size=batch_size, sequence_length=sequence_length)

        self.logger = bt.logging

        load_dotenv(find_dotenv())

        self.max_configs = max_configs
        self.max_shards = max_shards
        self.max_row_groups = max_row_groups
        self.max_rows_per_group = max_rows_per_group

        self.debug = debug
        self.randomness = randomness

        self.BUCKET = os.getenv("R2_BUCKET_NAME") or (_ for _ in ()).throw(ValueError("R2_BUCKET_NAME env var not set"))
        self.ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID") or (_ for _ in ()).throw(ValueError("R2_ACCOUNT_ID env var not set"))
        self.ACCESS_KEY = os.getenv("R2_ADMIN_ACCESS_KEY_ID") or (_ for _ in ()).throw(ValueError("R2_ADMIN_ACCESS_KEY_ID env var not set"))
        self.SECRET_KEY = os.getenv("R2_ADMIN_SECRET_ACCESS_KEY") or (_ for _ in ()).throw(ValueError("R2_ADMIN_SECRET_ACCESS_KEY env var not set"))

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

    async def load_shard(self, shard_path, max_rows_per_group=2, random_row_groups=False, random_rows=False):
        """Load and tokenize text from a single shard (async) with timing."""
        buffer = []
        row_groups_loaded = 0
        rows_loaded = 0
        try:
            reader = await asyncio.to_thread(pq.ParquetFile, f"s3://{shard_path}", filesystem=self.fs)
        except Exception as e:
            print(f"Failed to open shard {shard_path}: {e}")
            return buffer

        num_row_groups = reader.num_row_groups

        if self.randomness or random_row_groups:
            start_idx = random.randint(0, num_row_groups - self.max_row_groups)
            rg_indices = list(range(start_idx, start_idx + self.max_row_groups)) # eg. [482, 483]
        else:
            rg_indices = list(range(min(self.max_row_groups, num_row_groups)))

        for rg_idx in rg_indices: 
            row_group = await asyncio.to_thread(reader.read_row_group, rg_idx, columns=["text"], use_threads=True)
            num_rows = len(row_group)

            start_idx = random.randint(0, num_rows - max_rows_per_group)
            end_idx = min(start_idx + max_rows_per_group, num_rows)

            if self.randomness or random_rows:
                start_idx = random.randint(0, num_rows - max_rows_per_group)
                end_idx = min(start_idx + max_rows_per_group, num_rows)
                rows = row_group.slice(offset=start_idx, length=end_idx - start_idx)
            else:
                rows = row_group.slice(offset=0, length=max_rows_per_group)

            row_groups_loaded += 1
            rows_loaded += len(rows)

            encodings = await self.tokenize_texts(rows["text"].to_pylist())
            for ids in encodings:
                ids.append(self.tokenizer.eos_token_id)
                buffer.extend(ids)

        self.total_row_groups_loaded += row_groups_loaded
        self.total_rows_loaded += rows_loaded

        return buffer

    async def get_shards_from_configs(self, max_configs=3, random_configs=False, random_shards=False):
        """Collect shard paths from multiple configs."""

        configs = await self.get_configs_async()
        
        if self.randomness or random_configs:
            configs = random.sample(configs, min(len(configs), max_configs))
        else:
            configs = configs[:max_configs]

        shard_lists = await asyncio.gather(
            *(asyncio.to_thread(self.list_shard_files, c) for c in configs)
        )

        if self.randomness or random_shards:
            all_shards = [
                *(
                    random.sample(shards, min(len(shards), self.max_shards))
                    for shards in shard_lists
                )
            ]
            all_shards = [shard for sublist in all_shards for shard in sublist]
        else:
            all_shards = [shard for shards in shard_lists for shard in shards[:self.max_shards]]

        if self.debug:
            print(f"Configs: {configs}")
            print(f"All_shards: {all_shards}\n")
            print(f"- configs: {len(configs)} (max_configs:{max_configs})")
            print(f"- Shards: {len(all_shards)} (self.max_shards:{self.max_shards})")

        return all_shards

    async def fetch_data_for_shards(self, shard_paths, max_rows_per_group=2):
        """Download and tokenize multiple shards asynchronously."""

        semaphore = asyncio.Semaphore(10)
        async def load_with_limit(shard):
            async with semaphore:
                return await self.load_shard(
                    shard_path=shard,
                    max_rows_per_group=max_rows_per_group
                )
        results = await asyncio.gather(*(load_with_limit(p) for p in shard_paths))

        if self.debug:
            print(f"- Row groups: {self.total_row_groups_loaded} (self.max_row_groups:{self.max_row_groups})")
            print(f"- Rows: {self.total_rows_loaded} (max_rows_per_group:{max_rows_per_group})\n")

        return [token for shard_buffer in results for token in shard_buffer]

    async def load_bucket_data_to_buffer(self, max_configs=3, max_rows_per_group=2):
        """High-level pipeline: gather shards → load data → fill buffer."""
        if not self.metadata or not self.shard_sizes:
            self.load_bucket_configs()

        all_shards = await self.get_shards_from_configs(max_configs=max_configs)
        
        start_time = time.perf_counter()

        self.buffer = await self.fetch_data_for_shards(
            shard_paths=all_shards, 
            max_rows_per_group=max_rows_per_group
            )
        
        end_time = time.perf_counter()
        
        if self.debug:
                print(f"Buffer length: {len(self.buffer)}\n")
                print(f"load_bucket_data_to_buffer took {end_time - start_time:.2f}s\n")
        return self.buffer


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("dstrbtd/llama-1b", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    debug = True
    randomness = True
    sequence_length = 1024

    max_configs = 3
    max_rows_per_group = 100

    batch_size = 4

    loader = DatasetLoader(
        debug=debug,
        randomness=randomness,
        sequence_length=sequence_length,
        tokenizer=tokenizer,        
    )

    asyncio.run(loader.load_bucket_data_to_buffer(
        max_configs=max_configs,
        max_rows_per_group=max_rows_per_group
    ))

    loader.prepare_batches(batch_size=batch_size)
    
    print(f"Batches: {len(loader)}")

    for i, (inputs, labels) in enumerate(loader):
        print(f"Batch {i}: input_ids shape {inputs.shape}")
        print(f"Batch {i}: labels shape {labels.shape}")
        if i >= 1:
            break
