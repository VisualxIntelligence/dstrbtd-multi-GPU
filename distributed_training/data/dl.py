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
import hashlib


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
        uid: int,
        current_block: int = 0,
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

        self.uid = uid
        self.current_block = current_block
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

        self.debug and print(f"[DEBUG] DatasetLoader initialized with UID={self.uid}, block={self.current_block}")

    def generate_rng(self, context: str = "") -> random.Random:
        """
        Returns a reproducible RNG based on the stored UID and current block.
        """
        seed_str = f"{self.uid}-{context}-{self.current_block}"
        # self.debug and print(f"seed_str: {seed_str}")
        seed = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**32)
        return random.Random(seed)
    
    def select_configs(self, configs, max_configs):
        rng = self.generate_rng("config_selection")
        n = min(len(configs), max_configs)
        indexes = rng.sample(range(len(configs)), n)
        # self.debug and print(f"Config idxs chosen: {indexes}")
        return [configs[i] for i in indexes]

    def select_shards(self, shards, max_shards, context="shard_selection"):
        rng = self.generate_rng(context)
        n = min(len(shards), max_shards)
        indexes = rng.sample(range(len(shards)), n)
        # self.debug and print(f"Shard idxs chosen: {indexes}")
        return [shards[i] for i in indexes]

    def select_row_groups(self, num_row_groups, max_row_groups, context="row_group"):
        rng = self.generate_rng(context)
        start_idx = rng.randint(0, num_row_groups - max_row_groups) if num_row_groups > max_row_groups else 0
        rg_indices = list(range(start_idx, start_idx + max_row_groups))
        # self.debug and print(f"row_group idxs chosen: {rg_indices} out of {num_row_groups}")
        return rg_indices

    def select_rows(self, num_rows, max_rows_per_group, context="row"):
        rng = self.generate_rng(context)
        start_idx = rng.randint(0, num_rows - max_rows_per_group) if num_rows > max_rows_per_group else 0
        end_idx = min(start_idx + max_rows_per_group, num_rows)
        # self.debug and print(f"row idxs chosen: {list(range(start_idx, end_idx))} out of {num_rows}")
        return start_idx, end_idx

    async def load_bucket_data_to_buffer(self, max_configs=3, max_rows_per_group=2):
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
            print(f"Buffer length: {len(self.buffer)}")
            print(f"load_bucket_data_to_buffer took {end_time - start_time:.2f}s\n")

        return self.buffer

    def load_bucket_configs(self):
        self.download_config(f"{self.BUCKET}/{self.DATASET}/{self.META_NAME}", self.meta_cache_path)
        self.download_config(f"{self.BUCKET}/{self.DATASET}/{self.SHARD_NAME}", self.shard_cache_path)

        with open(self.meta_cache_path, "r") as f:
            self.metadata = yaml.safe_load(f)

        with open(self.shard_cache_path, "r") as f:
            self.shard_sizes = json.load(f)

    def download_config(self, remote_path, local_path):
        if os.path.exists(local_path):
            return
        data = self.fs.cat(remote_path)
        with open(local_path, "wb") as dst:
            dst.write(data)            

    async def get_shards_from_configs(self, max_configs=3):
        configs = await self.get_configs()
        configs = self.select_configs(configs, max_configs)

        shard_lists = await asyncio.gather(
            *(asyncio.to_thread(self.list_shard_files, c) for c in configs)
        )

        all_shards = []
        for shards in shard_lists:
            selected = self.select_shards(shards, self.max_shards, context=f"shard_{shards[0] if shards else ''}")
            all_shards.extend(selected)

        self.debug and print(f"All_shards: {all_shards}\n")
        return all_shards          

    async def get_configs(self):
        all_configs = [c.get("config_name") for c in self.metadata.get("configs", []) if c.get("config_name")]
        async def check_config(config):
            config_path = f"{self.BUCKET}/{self.DATASET}/{config}"
            exists = await asyncio.to_thread(self.fs.exists, config_path)
            return config if exists else None
        results = await asyncio.gather(*(check_config(c) for c in all_configs))
        return [r for r in results if r]

    def list_shard_files(self, config):
        config_info = self.shard_sizes.get(config, {})
        shards = config_info.get("shards", [])
        return [shard["path"] for shard in shards]
            
    async def fetch_data_for_shards(self, shard_paths, max_rows_per_group=2):
        semaphore = asyncio.Semaphore(10)
        async def load_with_limit(shard):
            async with semaphore:
                return await self.load_shard(shard_path=shard, max_rows_per_group=max_rows_per_group)
        results = await asyncio.gather(*(load_with_limit(p) for p in shard_paths))
        return [token for shard_buffer in results for token in shard_buffer]
    
    async def load_shard(self, shard_path, max_rows_per_group=2):
        buffer = []
        try:
            reader = await asyncio.to_thread(pq.ParquetFile, f"s3://{shard_path}", filesystem=self.fs)
        except Exception as e:
            print(f"Failed to open shard {shard_path}: {e}")
            return buffer

        num_row_groups = reader.num_row_groups
        rg_indices = self.select_row_groups(num_row_groups, self.max_row_groups, context=f"row_group_{shard_path}")

        for rg_idx in rg_indices:
            row_group = await asyncio.to_thread(reader.read_row_group, rg_idx, columns=["text"], use_threads=True)
            num_rows = len(row_group)
            start_idx, end_idx = self.select_rows(num_rows, max_rows_per_group, context=f"row_{shard_path}_rg{rg_idx}")
            rows = row_group.slice(offset=start_idx, length=end_idx - start_idx)

            encodings = await self.tokenize_texts(rows["text"].to_pylist())
            for ids in encodings:
                ids.append(self.tokenizer.eos_token_id)
                buffer.extend(ids)

            self.total_row_groups_loaded += 1
            self.total_rows_loaded += len(rows)

        return buffer
    
    async def tokenize_texts(self, texts):
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

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("dstrbtd/llama-1b", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    debug = True
    randomness = True

    miner_uid = 20
    current_block = 5597807    

    max_configs = 3
    max_rows_per_group = 100
    sequence_length = 1024

    batch_size = 4

    loader = DatasetLoader(
        debug=debug,
        randomness=randomness,
        sequence_length=sequence_length,
        tokenizer=tokenizer,
        uid=miner_uid,
        current_block=current_block,
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
