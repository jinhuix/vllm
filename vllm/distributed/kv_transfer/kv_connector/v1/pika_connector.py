# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Minimal PikaConnector for vLLM v1 KV Cache transfer via PikiwiDB.

This is a basic implementation that supports:
- Block-level KV cache storage in PikiwiDB
- Simple opportunistic loading (GPU miss -> DB pull -> recompute if not found)
- No request metadata, no prompt indexing, no reference counting
- Single-GPU setup only
"""

import hashlib
import logging
import time
from typing import TYPE_CHECKING, Any, Optional

import redis
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)


class PikaConnectorMetadata(KVConnectorMetadata):
    """Minimal metadata for PikaConnector."""
    
    def __init__(self):
        # Dict of request_id -> list of (layer_idx, list of (block_id, block_hash)) to load
        self.blocks_to_load: dict[str, list[tuple[int, list[tuple[int, str]]]]] = {}
        # Dict of request_id -> list of (layer_idx, list of (block_id, block_hash)) to save  
        self.blocks_to_save: dict[str, list[tuple[int, list[tuple[int, str]]]]] = {}


class PikaConnector(KVConnectorBase_V1):
    """
    Minimal PikiwiDB KV Cache Connector for vLLM v1.
    
    Features:
    - Block-level storage with keys: kvblock:{ns}:{layer_id}:{block_hash}:{k_type}
    - Opportunistic loading: try to load from DB, recompute if missing
    - No async operations (blocking I/O for simplicity)
    - Single-GPU setup
    """

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        
        # PikiwiDB connection config
        kv_transfer_config = vllm_config.kv_transfer_config
        self.redis_host = kv_transfer_config.kv_connector_extra_config.get("host", "localhost")
        self.redis_port = kv_transfer_config.kv_connector_extra_config.get("port", 6379)
        self.redis_db = kv_transfer_config.kv_connector_extra_config.get("db", 0)
        
        # Performance tuning parameters
        self.mget_batch_size = kv_transfer_config.kv_connector_extra_config.get("mget_batch_size", 512)
        self.socket_timeout = kv_transfer_config.kv_connector_extra_config.get("socket_timeout", 5.0)
        
        # Generate namespace from cache_salt or model config
        self.namespace = self._generate_namespace(vllm_config)
        
        # Model config for block layout
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.parallel_config = vllm_config.parallel_config
        self.block_size = self.cache_config.block_size
        
        # Redis client (created lazily)
        self._redis_client: Optional[redis.Redis] = None
        
        # Worker-side: registered KV caches
        self.kv_caches: dict[str, torch.Tensor] = {}
        
        self._pending_load_layers: dict[str, int] = {}  # request_id -> 剩余待加载层数
        self._finished_recving: set[str] = set()        # 已完成加载的 request_id
        self._save_map_by_req: dict[str, dict[str, int]] = {}  # request_id -> {hash: physical_block_id}

        # Scheduler-side: track blocks to load/save
        if role == KVConnectorRole.SCHEDULER:
            self._blocks_to_load: dict[str, list[tuple[int, list[tuple[int, str]]]]] = {}
            self._blocks_to_save: dict[str, list[tuple[int, list[tuple[int, str]]]]] = {}
        
        # Initialize logger
        self.logger = logging.getLogger(f"{__name__}.{role.name}")
        self.logger.info(f"PikaConnector initialized: namespace={self.namespace}, "
                        f"mget_batch_size={self.mget_batch_size}, socket_timeout={self.socket_timeout}s")

    def _generate_namespace(self, vllm_config: VllmConfig) -> str:
        """Generate namespace from vLLM config."""
        # 使用vllm原生的cache_salt
        cache_config = vllm_config.cache_config
        if hasattr(cache_config, 'cache_salt') and cache_config.cache_salt:
            return cache_config.cache_salt
        
        # 如果没有cache_salt，回退到基于模型配置的命名空间
        model_config = vllm_config.model_config
        # config_str = f"{model_config.model}_{cache_config.block_size}"
        # return hashlib.md5(config_str.encode()).hexdigest()[:16]
        return model_config.model

    def _get_redis_client(self) -> redis.Redis:
        """Get Redis client (lazy initialization)."""
        if self._redis_client is None:
            self._redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_timeout,
                decode_responses=False  # Keep binary data as bytes
            )
        return self._redis_client

    def _build_block_key(self, layer_idx: int, block_hash: str, k_type: int) -> str:
        """Build PikiwiDB key for a KV block using content hash."""
        return f"kvblock:{self.namespace}:{layer_idx}:{block_hash}:{k_type}"

    # ==============================
    # Worker-side methods
    # ==============================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register KV caches for worker-side operations."""
        self.kv_caches = kv_caches.copy()
        self.logger.debug(f"Registered {len(kv_caches)} KV cache layers")

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """Start loading KV blocks from PikiwiDB using vLLM-native block hashes."""
        if not isinstance(self._connector_metadata, PikaConnectorMetadata):
            return
        
        metadata = self._connector_metadata
        if not metadata.blocks_to_load:
            return
            
        redis_client = self._get_redis_client()
        
        for request_id, layer_block_list in metadata.blocks_to_load.items():
            # 该 request 需要加载的层总数
            self._pending_load_layers[request_id] = len(layer_block_list)
            for layer_idx, block_id_hash_list in layer_block_list:
                # Find the corresponding KV cache layer
                layer_kv_cache = None
                for layer_name, kv_cache in self.kv_caches.items():
                    if f"layers.{layer_idx}." in layer_name or f".{layer_idx}." in layer_name:
                        layer_kv_cache = kv_cache
                        break
                
                if layer_kv_cache is None:
                    # 该层没找到缓存也算“完成一层”，防止永远不归零
                    self._pending_load_layers[request_id] -= 1
                    if self._pending_load_layers[request_id] == 0:
                        self._finished_recving.add(request_id)
                        del self._pending_load_layers[request_id]
                    continue
                
                # 同步（阻塞式）执行本层加载
                self._load_blocks_for_layer(redis_client, layer_idx, block_id_hash_list, layer_kv_cache, request_id)
                # 该层完成，递减计数
                self._pending_load_layers[request_id] -= 1
                if self._pending_load_layers[request_id] == 0:
                    self._finished_recving.add(request_id)
                    del self._pending_load_layers[request_id]

    def _load_blocks_for_layer(self, redis_client: redis.Redis, layer_idx: int, 
                              block_id_hash_list: list[tuple[int, str]], kv_cache: torch.Tensor, request_id: str):
        """Load blocks for a specific layer using MGET batching for better performance."""
        kv_shape = kv_cache.shape
        
        # Validate KV cache shape and determine layout
        if len(kv_shape) == 5:  # [2, num_blocks, block_size, num_heads, head_dim]
            kv_type_dim, num_blocks_dim, block_size_dim, num_heads_dim, head_dim = kv_shape
            block_shape = [block_size_dim, num_heads_dim, head_dim]
            layout_type = "standard_5d"
        elif len(kv_shape) == 4:  # [2, num_blocks, num_heads, head_dim] - assume block_size=1
            kv_type_dim, num_blocks_dim, num_heads_dim, head_dim = kv_shape
            block_shape = [num_heads_dim, head_dim]
            layout_type = "legacy_4d"
        else:
            return
        
        expected_size = torch.tensor(block_shape).prod().item() * kv_cache.element_size()
        
        # Performance counters
        total_loaded = 0
        total_missing = 0
        total_decode_time = 0.0
        total_gpu_copy_time = 0.0
        valid_blocks_total = 0
        
        # Log start only for layer 0
        if layer_idx == 0:
            self.logger.info(f"[LOAD] Request {request_id} Layer {layer_idx}: Loading {len(block_id_hash_list)} blocks "
                           f"in batches of {self.mget_batch_size}")
        
        # Process blocks in batches using MGET
        for batch_start in range(0, len(block_id_hash_list), self.mget_batch_size):
            batch_end = min(batch_start + self.mget_batch_size, len(block_id_hash_list))
            batch = block_id_hash_list[batch_start:batch_end]
            
            # Build all keys for this batch
            k_keys = []
            v_keys = []
            valid_blocks = []
            
            for block_id, block_hash_str in batch:
                if block_id >= num_blocks_dim:
                    continue
                    
                k_key = self._build_block_key(layer_idx, block_hash_str, 0)
                v_key = self._build_block_key(layer_idx, block_hash_str, 1)
                k_keys.append(k_key)
                v_keys.append(v_key)
                valid_blocks.append((block_id, block_hash_str))
            
            if not valid_blocks:
                continue
            valid_blocks_total += len(valid_blocks)
            
            # MGET batch timing
            mget_start = time.time()
            try:
                # Use pipeline for better performance
                pipe = redis_client.pipeline()
                pipe.mget(k_keys)
                pipe.mget(v_keys)
                results = pipe.execute()
                k_results = results[0] if results else []
                v_results = results[1] if len(results) > 1 else []
                
                mget_time = time.time() - mget_start
                
                if layer_idx == 0:
                    avg_per_key = mget_time / (len(k_keys) * 2) * 1000 if k_keys else 0
                    self.logger.debug(f"[LOAD] Batch {batch_start//self.mget_batch_size + 1}: "
                                    f"MGET {len(k_keys)*2} keys in {mget_time*1000:.2f}ms "
                                    f"({avg_per_key:.3f}ms/key)")
                
                # Process batch results
                decode_start = time.time()
                batch_loaded = 0
                batch_missing = 0
                
                for i, (block_id, block_hash_str) in enumerate(valid_blocks):
                    if i >= len(k_results) or i >= len(v_results):
                        batch_missing += 1
                        continue
                        
                    k_data = k_results[i]
                    v_data = v_results[i]
                    
                    if k_data is not None and v_data is not None:
                        # Validate data size
                        if len(k_data) != expected_size or len(v_data) != expected_size:
                            batch_missing += 1
                            if layer_idx == 0:
                                self.logger.warning(f"[LOAD] Size mismatch for block {block_id}: "
                                                  f"expected {expected_size}, got K={len(k_data)}, V={len(v_data)}")
                            continue
                        
                        # Decode tensors (clone to avoid read-only buffer warning)
                        k_tensor = torch.frombuffer(k_data, dtype=kv_cache.dtype).reshape(block_shape).clone()
                        v_tensor = torch.frombuffer(v_data, dtype=kv_cache.dtype).reshape(block_shape).clone()
                        
                        # GPU copy timing
                        gpu_copy_start = time.time()
                        if layout_type == "standard_5d":
                            kv_cache[0, block_id].copy_(k_tensor)
                            kv_cache[1, block_id].copy_(v_tensor)
                        elif layout_type == "legacy_4d":
                            kv_cache[0, block_id].copy_(k_tensor)
                            kv_cache[1, block_id].copy_(v_tensor)
                        gpu_copy_time = time.time() - gpu_copy_start
                        
                        batch_loaded += 1
                        total_gpu_copy_time += gpu_copy_time
                        
                        if layer_idx == 0:
                            self.logger.debug(f"[LOAD] ✅ block_id={block_id}, hash={block_hash_str[:16]}...")
                    else:
                        batch_missing += 1
                        if layer_idx == 0:
                            self.logger.warning(f"[LOAD] MISS block_id={block_id} hash={block_hash_str}")
                
                decode_time = time.time() - decode_start
                total_decode_time += decode_time
                total_loaded += batch_loaded
                total_missing += batch_missing
                
                if layer_idx == 0:
                    avg_decode = decode_time / len(valid_blocks) * 1000 if valid_blocks else 0
                    avg_gpu_copy = (gpu_copy_time / batch_loaded * 1000) if batch_loaded > 0 else 0
                    self.logger.debug(f"[LOAD] Batch decode: {decode_time*1000:.2f}ms total "
                                    f"({avg_decode:.3f}ms/block), GPU copy: {avg_gpu_copy:.3f}ms/block")
                
            except Exception as e:
                if layer_idx == 0:
                    self.logger.error(f"[LOAD] Batch {batch_start//self.mget_batch_size + 1} failed: {e}")
                total_missing += len(valid_blocks)
        
        # Log summary（仅 layer 0 打汇总，减少噪音）
        if layer_idx == 0:
            total_time = total_decode_time + total_gpu_copy_time
            avg_decode_per_block = (total_decode_time / total_loaded * 1000) if total_loaded > 0 else 0
            avg_gpu_copy_per_block = (total_gpu_copy_time / total_loaded * 1000) if total_loaded > 0 else 0
            
            self.logger.info(f"[LOAD] Layer {layer_idx} Summary: {total_loaded}/{valid_blocks_total} loaded, "
                           f"{total_missing} missing. Decode: {avg_decode_per_block:.3f}ms/block, "
                           f"GPU copy: {avg_gpu_copy_per_block:.3f}ms/block, Total: {total_time*1000:.2f}ms")

    def wait_for_layer_load(self, layer_name: str) -> None:
        """No-op for blocking implementation."""
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """Save KV layer to PikiwiDB using vLLM-native block hashes."""
        if not isinstance(self._connector_metadata, PikaConnectorMetadata):
            return
        
        # Extract layer index from layer name
        layer_idx = None
        try:
            parts = layer_name.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    layer_idx = int(parts[i + 1])
                    break
            if layer_idx is None:
                return
        except (IndexError, ValueError):
            return
        
        metadata = self._connector_metadata
        if not metadata.blocks_to_save:
            return
            
        redis_client = self._get_redis_client()
        
        # Find blocks to save for this layer
        for request_id, layer_block_list in metadata.blocks_to_save.items():
            for layer_id, block_id_hash_list in layer_block_list:
                if layer_id == layer_idx:
                    self._save_blocks_for_layer(redis_client, layer_idx, block_id_hash_list, kv_layer, request_id)

    def _save_blocks_for_layer(self, redis_client: redis.Redis, layer_idx: int,
                              block_id_hash_list: list[tuple[int, str]], kv_layer: torch.Tensor, request_id: str):
        """Save blocks for a specific layer using vLLM-native block hashes."""
        kv_shape = kv_layer.shape
        
        # Validate KV cache shape
        if len(kv_shape) == 5:  # [2, num_blocks, block_size, num_heads, head_dim]
            kv_type_dim, num_blocks_dim, block_size_dim, num_heads_dim, head_dim = kv_shape
            layout_type = "standard_5d"
        elif len(kv_shape) == 4:  # [2, num_blocks, num_heads, head_dim]
            kv_type_dim, num_blocks_dim, num_heads_dim, head_dim = kv_shape
            layout_type = "legacy_4d"
        else:
            return
        
        # Performance counters
        saved_count = 0
        error_count = 0
        total_encode_time = 0.0
        total_mset_time = 0.0
        
        if layer_idx == 0:  # 只在 layer=0 打印 save 入口
            self.logger.info(f"[SAVE] Request {request_id} Layer {layer_idx}: Saving {len(block_id_hash_list)} blocks, "
                           f"batch={self.mget_batch_size}")
        
        # Process blocks in batches using pipeline MSET
        for batch_start in range(0, len(block_id_hash_list), self.mget_batch_size):
            batch_end = min(batch_start + self.mget_batch_size, len(block_id_hash_list))
            batch = block_id_hash_list[batch_start:batch_end]
            
            # Prepare batch data
            batch_kv_pairs = {}
            valid_blocks = []
            
            encode_start = time.time()
            for placeholder_block_id, block_hash_str in batch:
                # Use placeholder as physical block ID (direct mapping assumption)
                physical_block_id = placeholder_block_id
                
                # Validate physical_block_id is within KV cache bounds
                if physical_block_id >= num_blocks_dim:
                    error_count += 1
                    continue
                
                try:
                    # Extract K and V data based on layout
                    if layout_type == "standard_5d":  # [2, num_blocks, block_size, num_heads, head_dim]
                        k_tensor = kv_layer[0, physical_block_id]  # [block_size, num_heads, head_dim]
                        v_tensor = kv_layer[1, physical_block_id]  # [block_size, num_heads, head_dim]
                    elif layout_type == "legacy_4d":  # [2, num_blocks, num_heads, head_dim]
                        k_tensor = kv_layer[0, physical_block_id]  # [num_heads, head_dim]
                        v_tensor = kv_layer[1, physical_block_id]  # [num_heads, head_dim]
                    
                    # Convert to bytes for storage
                    k_data = k_tensor.contiguous().cpu().numpy().tobytes()
                    v_data = v_tensor.contiguous().cpu().numpy().tobytes()
                    
                    # Build PikiwiDB keys using content hash
                    k_key = self._build_block_key(layer_idx, block_hash_str, 0)
                    v_key = self._build_block_key(layer_idx, block_hash_str, 1)
                    
                    batch_kv_pairs[k_key] = k_data
                    batch_kv_pairs[v_key] = v_data
                    valid_blocks.append((physical_block_id, block_hash_str))
                    
                except Exception as e:
                    error_count += 1
                    if layer_idx == 0:
                        self.logger.error(f"[SAVE] Error encoding block {physical_block_id}: {e}")
            
            encode_time = time.time() - encode_start
            total_encode_time += encode_time
            
            if not batch_kv_pairs:
                continue
            
            # Batch save using MSET
            mset_start = time.time()
            try:
                redis_client.mset(batch_kv_pairs)
                mset_time = time.time() - mset_start
                total_mset_time += mset_time
                
                batch_saved = len(valid_blocks)
                saved_count += batch_saved
                
                if layer_idx == 0:
                    avg_encode = encode_time / len(batch) * 1000 if batch else 0
                    avg_mset = mset_time / len(batch_kv_pairs) * 1000 if batch_kv_pairs else 0
                    self.logger.debug(
                        f"[SAVE] Batch {batch_start//self.mget_batch_size + 1}: "
                        f"encode={encode_time*1000:.2f}ms({avg_encode:.3f}ms/block), "
                        f"mset={mset_time*1000:.2f}ms({avg_mset:.3f}ms/key), blocks={len(valid_blocks)}"
                    )
                
            except Exception as e:
                if layer_idx == 0:
                    self.logger.error(f"[SAVE] Batch {batch_start//self.mget_batch_size + 1} MSET failed: {e}")
                error_count += len(valid_blocks)
        
        # Log summary for layer 0
        if layer_idx == 0:
            total_time = total_encode_time + total_mset_time
            avg_encode_per_block = (total_encode_time / saved_count * 1000) if saved_count > 0 else 0
            avg_mset_per_key = (total_mset_time / (saved_count * 2) * 1000) if saved_count > 0 else 0  # *2 for K+V
            self.logger.info(
                f"[SAVE] Layer {layer_idx} Summary: {saved_count}/{len(block_id_hash_list)} saved, "
                f"{error_count} errors. encode={avg_encode_per_block:.3f}ms/block, "
                f"mset={avg_mset_per_key:.3f}ms/key, total={total_time*1000:.2f}ms"
            )

    def wait_for_save(self):
        """No-op for blocking implementation."""
        pass

    def get_finished(self, finished_req_ids: set[str]) -> tuple[Optional[set[str]], Optional[set[str]]]:
        # finished_sending: 本轮所有“发送到外部存储”的请求（这里我们同步保存，返回 None）
        # finished_recving: 外部存储“接收完成/从外部加载完成”的请求（我们用上面的集合）
        if self._finished_recving:
            done = set(self._finished_recving)
            self._finished_recving.clear()
            return None, done
        return None, None

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        """Cache hit detection: check for consecutive prefix blocks using vLLM-native block hashes."""
        # Strict validation of block_hashes
        if not hasattr(request, 'block_hashes') or not isinstance(request.block_hashes, list) or not request.block_hashes:
            return 0, False
        
        total_full_blocks = request.num_tokens // self.block_size  # 只看完整块
        if total_full_blocks == 0:
            return 0, False
        
        # Calculate starting block index based on computed tokens
        start_block_idx = min(num_computed_tokens // self.block_size, total_full_blocks)
        if start_block_idx >= total_full_blocks:
            return 0, False
        
        # Check consecutive blocks starting from start_block_idx
        redis_client = self._get_redis_client()
        hit_blocks = 0
        
        for block_idx in range(start_block_idx, total_full_blocks): # 只到完整块的末尾
            block_hash_obj = request.block_hashes[block_idx]
            
            # Validate block hash object (vLLM BlockHash)
            if not block_hash_obj or not hasattr(block_hash_obj, 'hash_value'):
                break
            
            # Extract hash value as string for PikiwiDB key
            block_hash_str = str(block_hash_obj.hash_value)
            
            # Check if both K and V exist for layer 0 (representative layer)
            k_key = self._build_block_key(0, block_hash_str, 0)
            v_key = self._build_block_key(0, block_hash_str, 1)
            
            try:
                k_exists = redis_client.exists(k_key)
                v_exists = redis_client.exists(v_key)
                
                if k_exists and v_exists:
                    hit_blocks += 1
                else:
                    break  # Stop at first miss for consecutive matching
            except Exception:
                break
        
        if hit_blocks > 0:
            hit_tokens = hit_blocks * self.block_size
            self.logger.info(f"[HIT] Request {request.request_id}: Found {hit_blocks} consecutive cached blocks "
                           f"({hit_tokens} tokens) starting from block {start_block_idx}")
            return hit_tokens, True
        
        return 0, False

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        """
        Track blocks that need to be loaded/saved using vLLM-native block hashes.
        
        CRITICAL: We must correctly map physical block_ids to logical block_hashes.
        block_ids are physical GPU memory block identifiers, while block_hashes 
        represent the content-based hashes for prefix caching.
        """
        if self.role != KVConnectorRole.SCHEDULER:
            return
        
        # Get physical block IDs from the allocation
        block_ids_per_group = blocks.get_block_ids()
        if not block_ids_per_group or not block_ids_per_group[0]:
            return
        
        block_ids = block_ids_per_group[0]  # Get first group's block IDs
        
        # Strict validation of block_hashes
        if not hasattr(request, 'block_hashes') or not request.block_hashes or not isinstance(request.block_hashes, list):
            return
        
        # Calculate which blocks correspond to complete token blocks
        # In vLLM, block_hashes[i] corresponds to tokens [i*block_size : (i+1)*block_size]
        # We need to map these logical blocks to the physical block_ids that were just allocated
        
        num_computed_tokens = request.num_computed_tokens
        num_total_tokens = request.num_tokens
        
        # Calculate the range of logical blocks that need to be handled
        # These are blocks that contain tokens from num_computed_tokens onwards
        start_logical_block = num_computed_tokens // self.block_size
        end_logical_block_full = min(len(request.block_hashes), num_total_tokens // self.block_size)  # 只到完整块
        if start_logical_block >= end_logical_block_full:
            return
        
        # Build mapping of physical block_id to logical block hash
        # IMPORTANT: The physical blocks are allocated in sequence for the token range
        # that needs computation, starting from the first uncomputed token
        # 仅映射本轮需要的、且有对应物理块的那一段
        num_to_map = min(end_logical_block_full - start_logical_block, len(block_ids))
        if num_to_map <= 0:
            return

        block_id_hash_pairs = []
        for i in range(num_to_map):
            logical_idx = start_logical_block + i
            bh = request.block_hashes[logical_idx]
            if not bh or not hasattr(bh, 'hash_value'):
                continue
            physical_block_id = block_ids[i]
            block_id_hash_pairs.append((physical_block_id, str(bh.hash_value)))

        # 生成完 block_id_hash_pairs 之后，且在写 _blocks_to_load 之前，新增：
        save_map = self._save_map_by_req.setdefault(request.request_id, {})
        for physical_block_id, block_hash_str in block_id_hash_pairs:
            # 记录“这个hash目前对应哪个物理块槽位”（后写覆盖先写，保持最新）
            save_map[block_hash_str] = physical_block_id
        
        if not block_id_hash_pairs:
            return
        
        # Schedule loading for all layers
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        layer_block_list = []
        
        for layer_idx in range(num_layers):
            layer_block_list.append((layer_idx, block_id_hash_pairs.copy()))
        
        self._blocks_to_load[request.request_id] = layer_block_list

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> PikaConnectorMetadata:
        """Build metadata for this step."""
        metadata = PikaConnectorMetadata()
        
        # Move scheduled blocks to metadata
        metadata.blocks_to_load = self._blocks_to_load.copy()
        metadata.blocks_to_save = self._blocks_to_save.copy()
        
        # Clear scheduler state
        self._blocks_to_load.clear()
        self._blocks_to_save.clear()
        
        return metadata

    def request_finished(self, request: "Request", block_ids: list[int]) -> tuple[bool, Optional[dict[str, Any]]]:
        # 同步实现：返回 False 允许上层立即释放；我们自己调度保存
        if not hasattr(request, 'block_hashes') or not isinstance(request.block_hashes, list) or not request.block_hashes:
            return False, None

        full_blocks = request.num_tokens // self.block_size
        if full_blocks <= 0:
            return False, None

        # 取出在分配阶段积累的“hash -> 物理块id”映射
        save_map = self._save_map_by_req.pop(request.request_id, {})
        if not save_map:
            # 没拿到映射说明这轮没分配/没计算到完整块，直接返回
            return False, None

        # 仅保存“完整块”对应的hash；并按映射找到物理块id
        pairs: list[tuple[int, str]] = []
        miss_cnt = 0
        for i in range(min(full_blocks, len(request.block_hashes))):
            bh = request.block_hashes[i]
            if not bh or not hasattr(bh, 'hash_value'):
                continue
            h = str(bh.hash_value)
            pid = save_map.get(h)
            if pid is not None:
                pairs.append((pid, h))
            else:
                miss_cnt += 1
                # 可选：打点方便你排查
                self.logger.debug(f"[SAVE] No physical slot recorded for hash={h[:16]}..., req={request.request_id}")

        if not pairs:
            return False, None

        # 跨所有层安排保存任务（这里 pairs 的第一个元素是“真实物理块 id”）
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        self._blocks_to_save[request.request_id] = [(layer_idx, pairs.copy()) for layer_idx in range(num_layers)]
        return False, None

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> Optional[str]:
        """
        For minimal implementation, don't require specific layout.
        In production, you might want to require HND for better performance.
        """
        return None
