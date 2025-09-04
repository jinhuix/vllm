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
logging.getLogger("vllm.distributed.kv_transfer.kv_connector.v1.pika_connector").setLevel(logging.DEBUG)
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
    - Block-level storage with keys: kvblock:{ns}:{layer_id}:{block_id}:{k_type}
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
        
        # Scheduler-side: track blocks to load/save
        if role == KVConnectorRole.SCHEDULER:
            self._blocks_to_load: dict[str, list[tuple[int, list[tuple[int, str]]]]] = {}
            self._blocks_to_save: dict[str, list[tuple[int, list[tuple[int, str]]]]] = {}
        
        logger.info(f"PikaConnector initialized with namespace: {self.namespace}")

    def _generate_namespace(self, vllm_config: VllmConfig) -> str:
        """Generate namespace from vLLM config."""
        # Generate from model config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        
        # Create a unique namespace from key config parameters
        config_str = f"{model_config.model}_{model_config.dtype}_{cache_config.block_size}_{cache_config.cache_dtype}"
        return hashlib.md5(config_str.encode()).hexdigest()[:16]

    def _get_redis_client(self) -> redis.Redis:
        """Get Redis client (lazy initialization)."""
        if self._redis_client is None:
            self._redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
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
        logger.info(f"Registered {len(self.kv_caches)} KV cache layers")

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """Start loading KV blocks from PikiwiDB (blocking for simplicity)."""
        if not isinstance(self._connector_metadata, PikaConnectorMetadata):
            return
        
        metadata = self._connector_metadata
        redis_client = self._get_redis_client()
        
        for request_id, layer_block_list in metadata.blocks_to_load.items():
            logger.debug(f"Loading blocks for request {request_id}")
            
            for layer_idx, block_id_hash_list in layer_block_list:
                # Find the corresponding KV cache layer
                layer_kv_cache = None
                for layer_name, kv_cache in self.kv_caches.items():
                    if f"layers.{layer_idx}." in layer_name or f".{layer_idx}." in layer_name:
                        layer_kv_cache = kv_cache
                        break
                
                if layer_kv_cache is None:
                    logger.warning(f"No KV cache found for layer {layer_idx}")
                    continue
                
                logger.debug(f"Loading {len(block_id_hash_list)} blocks for layer {layer_idx}")
                self._load_blocks_for_layer(redis_client, layer_idx, block_id_hash_list, layer_kv_cache)

    def _load_blocks_for_layer(self, redis_client: redis.Redis, layer_idx: int, 
                              block_id_hash_list: list[tuple[int, str]], kv_cache: torch.Tensor):
        """Load blocks for a specific layer."""
        kv_shape = kv_cache.shape
        logger.debug(f"KV cache shape for layer {layer_idx}: {kv_shape}")
        
        # Support common KV cache shapes:
        # [2, num_blocks, block_size, num_heads, head_dim] - standard
        # [2, num_blocks, num_heads, head_dim] - no block_size (old format)
        # [num_blocks, 2, block_size, num_heads, head_dim] - alternative layout
        
        if len(kv_shape) == 5:  # [2, num_blocks, block_size, num_heads, head_dim]
            kv_type_dim, num_blocks_dim, block_size_dim, num_heads_dim, head_dim = kv_shape
            block_shape = [block_size_dim, num_heads_dim, head_dim]
            layout_type = "standard_5d"
        elif len(kv_shape) == 4:  # [2, num_blocks, num_heads, head_dim] - assume block_size=1
            kv_type_dim, num_blocks_dim, num_heads_dim, head_dim = kv_shape
            block_shape = [num_heads_dim, head_dim]
            layout_type = "legacy_4d"
        else:
            logger.warning(f"Unsupported KV cache shape: {kv_shape}, skipping layer {layer_idx}")
            return
        
        logger.debug(f"Using layout '{layout_type}' with block_shape {block_shape}")
        
        for block_id, block_hash in block_id_hash_list:
            # Try to load K and V blocks using content hash
            k_key = self._build_block_key(layer_idx, block_hash, 0)  # K
            v_key = self._build_block_key(layer_idx, block_hash, 1)  # V
            
            try:
                k_data = redis_client.get(k_key)
                v_data = redis_client.get(v_key)
                
                if k_data is not None and v_data is not None:
                    # Load data into KV cache
                    k_tensor = torch.frombuffer(k_data, dtype=kv_cache.dtype).reshape(block_shape)
                    v_tensor = torch.frombuffer(v_data, dtype=kv_cache.dtype).reshape(block_shape)
                    
                    # Copy to KV cache based on layout
                    if layout_type == "standard_5d":
                        kv_cache[0, block_id] = k_tensor  # K: [block_size, num_heads, head_dim]
                        kv_cache[1, block_id] = v_tensor  # V: [block_size, num_heads, head_dim]
                    elif layout_type == "legacy_4d":
                        kv_cache[0, block_id] = k_tensor  # K: [num_heads, head_dim]
                        kv_cache[1, block_id] = v_tensor  # V: [num_heads, head_dim]
                    
                    logger.debug(f"Loaded block {block_id} (hash: {block_hash[:8]}...) for layer {layer_idx}")
                else:
                    logger.debug(f"Block {block_id} (hash: {block_hash[:8]}...) not found in DB for layer {layer_idx}")
                    
            except Exception as e:
                logger.warning(f"Failed to load block {block_id} for layer {layer_idx}: {e}")

    def wait_for_layer_load(self, layer_name: str) -> None:
        """No-op for blocking implementation."""
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """Save KV layer to PikiwiDB (blocking for simplicity)."""
        if not isinstance(self._connector_metadata, PikaConnectorMetadata):
            return
        
        # Extract layer index from layer name
        # Support multiple layer name formats:
        # - "layers.{idx}.self_attn.kv_cache" 
        # - "model.decoder.layers.{idx}.self_attn.attn"
        # - "model.layers.{idx}.self_attn"
        layer_idx = None
        try:
            parts = layer_name.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    layer_idx = int(parts[i + 1])
                    break
            if layer_idx is None:
                logger.warning(f"Cannot extract layer index from {layer_name}")
                return
        except (IndexError, ValueError) as e:
            logger.warning(f"Cannot extract layer index from {layer_name}: {e}")
            return
        
        metadata = self._connector_metadata
        redis_client = self._get_redis_client()
        
        # Find blocks to save for this layer
        for request_id, layer_block_list in metadata.blocks_to_save.items():
            for layer_id, block_id_hash_list in layer_block_list:
                if layer_id == layer_idx:
                    logger.debug(f"Saving {len(block_id_hash_list)} blocks for layer {layer_idx}")
                    self._save_blocks_for_layer(redis_client, layer_idx, block_id_hash_list, kv_layer)

    def _save_blocks_for_layer(self, redis_client: redis.Redis, layer_idx: int,
                              block_id_hash_list: list[tuple[int, str]], kv_layer: torch.Tensor):
        """Save blocks for a specific layer."""
        kv_shape = kv_layer.shape
        logger.debug(f"Saving KV cache shape for layer {layer_idx}: {kv_shape}")
        
        for block_id, block_hash in block_id_hash_list:
            try:
                # Extract K and V data based on layout
                if len(kv_shape) == 5:  # [2, num_blocks, block_size, num_heads, head_dim]
                    k_data = kv_layer[0, block_id].contiguous().cpu().numpy().tobytes()
                    v_data = kv_layer[1, block_id].contiguous().cpu().numpy().tobytes()
                elif len(kv_shape) == 4:  # [2, num_blocks, num_heads, head_dim]
                    k_data = kv_layer[0, block_id].contiguous().cpu().numpy().tobytes()
                    v_data = kv_layer[1, block_id].contiguous().cpu().numpy().tobytes()
                else:
                    logger.warning(f"Unsupported KV cache shape for saving: {kv_shape}")
                    continue
                
                # Build keys using content hash
                k_key = self._build_block_key(layer_idx, block_hash, 0)
                v_key = self._build_block_key(layer_idx, block_hash, 1)
                
                # Save to PikiwiDB
                redis_client.set(k_key, k_data)
                redis_client.set(v_key, v_data)
                
                logger.debug(f"Saved block {block_id} (hash: {block_hash[:8]}...) for layer {layer_idx}")
                
            except Exception as e:
                logger.warning(f"Failed to save block {block_id} for layer {layer_idx}: {e}")

    def wait_for_save(self):
        """No-op for blocking implementation."""
        pass

    def get_finished(self, finished_req_ids: set[str]) -> tuple[Optional[set[str]], Optional[set[str]]]:
        return None, None
        
        """
        Report which requests have finished async operations.
        For blocking implementation, all operations complete immediately.
        """

        """
        # In our blocking implementation, any finished request has completed saving
        # Return the finished requests as "sending done" so blocks can be freed
        finished_sending = finished_req_ids if finished_req_ids else None
        finished_recving = None  # We don't track async receiving in this implementation
        
        if finished_sending:
            logger.debug(f"Reporting {len(finished_sending)} requests as finished sending")
        
        return finished_sending, finished_recving
        """

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        """
        Simple cache hit detection: check for consecutive prefix blocks in layer 0.
        """
        # If no block hashes available, no cache hits possible
        if not hasattr(request, 'block_hashes') or not request.block_hashes:
            return 0, False
        
        # Calculate starting block index
        start_block_idx = num_computed_tokens // self.block_size
        if start_block_idx >= len(request.block_hashes):
            return 0, False
        
        # Check consecutive blocks starting from start_block_idx
        redis_client = self._get_redis_client()
        hit_blocks = 0
        
        for block_idx in range(start_block_idx, len(request.block_hashes)):
            block_hash = request.block_hashes[block_idx]
            if not block_hash:
                break  # No hash means incomplete block
            
            # Check if both K and V exist for layer 0 (representative layer)
            k_key = self._build_block_key(0, block_hash, 0)
            v_key = self._build_block_key(0, block_hash, 1)
            
            try:
                if redis_client.exists(k_key) and redis_client.exists(v_key):
                    hit_blocks += 1
                    logger.debug(f"Cache hit for block {block_idx} (hash: {block_hash[:8]}...)")
                else:
                    logger.debug(f"Cache miss for block {block_idx} (hash: {block_hash[:8]}...)")
                    break  # Stop at first miss for consecutive matching
            except Exception as e:
                logger.warning(f"Error checking cache for block {block_idx}: {e}")
                break
        
        if hit_blocks > 0:
            hit_tokens = hit_blocks * self.block_size
            logger.info(f"Found {hit_blocks} consecutive cached blocks ({hit_tokens} tokens) for request {request.request_id}")
            return hit_tokens, True
        
        return 0, False

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        """
        Track blocks that need to be loaded/saved using block hashes.
        """
        if self.role != KVConnectorRole.SCHEDULER:
            return
        
        block_ids = blocks.get_block_ids()[0]  # Get first device's block IDs
        if not block_ids:
            return
        
        # Get block hashes from request
        if not hasattr(request, 'block_hashes') or not request.block_hashes:
            logger.debug(f"No block hashes available for request {request.request_id}")
            return
        
        # Match block_ids with block_hashes
        block_id_hash_pairs = []
        for i, block_id in enumerate(block_ids):
            if i < len(request.block_hashes):
                block_hash = request.block_hashes[i]
                if block_hash:  # Only include blocks with valid hashes
                    block_id_hash_pairs.append((block_id, block_hash))
        
        if not block_id_hash_pairs:
            logger.debug(f"No valid block hashes found for request {request.request_id}")
            return
        
        # For minimal implementation, try to load all blocks with hashes (opportunistic)
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        layer_block_list = []
        
        for layer_idx in range(num_layers):
            layer_block_list.append((layer_idx, block_id_hash_pairs.copy()))
        
        self._blocks_to_load[request.request_id] = layer_block_list
        logger.debug(f"Scheduled loading of {len(block_id_hash_pairs)} blocks with hashes for {num_layers} layers")

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
        """
        When a request finishes, save its blocks to PikiwiDB using block hashes.
        """
        if not block_ids:
            return False, None
        
        # Get block hashes for the finished blocks
        if not hasattr(request, 'block_hashes') or not request.block_hashes:
            logger.debug(f"No block hashes available for finished request {request.request_id}")
            return False, None
        
        # Match block_ids with their hashes (only save full blocks)
        block_id_hash_pairs = []
        for i, block_id in enumerate(block_ids):
            if i < len(request.block_hashes):
                block_hash = request.block_hashes[i]
                if block_hash:  # Only save blocks with valid hashes (full blocks)
                    block_id_hash_pairs.append((block_id, block_hash))
        
        if not block_id_hash_pairs:
            logger.debug(f"No valid block hashes for saving from request {request.request_id}")
            return False, None
        
        # Schedule blocks for saving
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        layer_block_list = []
        
        for layer_idx in range(num_layers):
            layer_block_list.append((layer_idx, block_id_hash_pairs.copy()))
        
        self._blocks_to_save[request.request_id] = layer_block_list
        logger.debug(f"Scheduled saving of {len(block_id_hash_pairs)} blocks with hashes for {num_layers} layers")
        
        # Return True to delay block freeing until saving is complete
        # This prevents data corruption during async saving
        return False, None

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> Optional[str]:
        """
        For minimal implementation, don't require specific layout.
        In production, you might want to require HND for better performance.
        """
        return None
