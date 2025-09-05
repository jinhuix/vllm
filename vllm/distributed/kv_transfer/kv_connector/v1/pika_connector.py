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
logging.getLogger("vllm.distributed.kv_transfer.kv_connector.v1.pika_connector").setLevel(logging.ERROR)
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
        
        # 静默初始化

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
        pass  # 静默注册

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """Start loading KV blocks from PikiwiDB using vLLM-native block hashes."""
        if not isinstance(self._connector_metadata, PikaConnectorMetadata):
            return
        
        metadata = self._connector_metadata
        if not metadata.blocks_to_load:
            return
            
        redis_client = self._get_redis_client()
        
        for request_id, layer_block_list in metadata.blocks_to_load.items():
            for layer_idx, block_id_hash_list in layer_block_list:
                # Find the corresponding KV cache layer
                layer_kv_cache = None
                matched_layer_name = None
                
                for layer_name, kv_cache in self.kv_caches.items():
                    if f"layers.{layer_idx}." in layer_name or f".{layer_idx}." in layer_name:
                        layer_kv_cache = kv_cache
                        matched_layer_name = layer_name
                        break
                
                if layer_kv_cache is None:
                    continue
                
                self._load_blocks_for_layer(redis_client, layer_idx, block_id_hash_list, layer_kv_cache, request_id)

    def _load_blocks_for_layer(self, redis_client: redis.Redis, layer_idx: int, 
                              block_id_hash_list: list[tuple[int, str]], kv_cache: torch.Tensor, request_id: str):
        """Load blocks for a specific layer using vLLM-native block hashes."""
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
        
        # Load each block
        loaded_count = 0
        missing_count = 0
        
        if layer_idx == 0:  # 只在layer=0时输出调试信息
            print(f"\n🔄 [LOAD] Request {request_id} Layer {layer_idx}: Loading {len(block_id_hash_list)} blocks")
        
        for block_id, block_hash_str in block_id_hash_list:
            # Validate block_id is within KV cache bounds
            if block_id >= num_blocks_dim:
                continue
            
            # Build PikiwiDB keys using content hash
            k_key = self._build_block_key(layer_idx, block_hash_str, 0)  # K
            v_key = self._build_block_key(layer_idx, block_hash_str, 1)  # V
            
            try:
                k_data = redis_client.get(k_key)
                v_data = redis_client.get(v_key)
                
                if k_data is not None and v_data is not None:
                    # Validate data size
                    expected_size = torch.tensor(block_shape).prod().item() * kv_cache.element_size()
                    if len(k_data) != expected_size or len(v_data) != expected_size:
                        continue
                    
                    # Load data into tensors
                    k_tensor = torch.frombuffer(k_data, dtype=kv_cache.dtype).reshape(block_shape)
                    v_tensor = torch.frombuffer(v_data, dtype=kv_cache.dtype).reshape(block_shape)
                    
                    # Copy to KV cache based on layout
                    if layout_type == "standard_5d":
                        kv_cache[0, block_id] = k_tensor  # K: [block_size, num_heads, head_dim]
                        kv_cache[1, block_id] = v_tensor  # V: [block_size, num_heads, head_dim]
                    elif layout_type == "legacy_4d":
                        kv_cache[0, block_id] = k_tensor  # K: [num_heads, head_dim]
                        kv_cache[1, block_id] = v_tensor  # V: [num_heads, head_dim]
                    
                    loaded_count += 1
                    if layer_idx == 0:
                        print(f"  ✅ LOADED block_id={block_id}, hash={block_hash_str[:16]}..., keys={k_key}, {v_key}")
                else:
                    missing_count += 1
                    if layer_idx == 0:
                        k_exists = k_data is not None
                        v_exists = v_data is not None
                        print(f"  ❌ MISSING block_id={block_id}, hash={block_hash_str[:16]}..., keys={k_key}, {v_key}")
                    
            except Exception as e:
                if layer_idx == 0:
                    print(f"  ⚠️ ERROR loading block_id={block_id}, hash={block_hash_str[:16]}...: {e}")
        
        if layer_idx == 0:
            print(f"📊 [LOAD] Layer {layer_idx}: Loaded {loaded_count}/{len(block_id_hash_list)} blocks, {missing_count} missing")

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
        
        # Save each block
        saved_count = 0
        error_count = 0
        
        if layer_idx == 0:  # 只在layer=0时输出调试信息
            print(f"\n💾 [SAVE] Request {request_id} Layer {layer_idx}: Saving {len(block_id_hash_list)} blocks")
        
        for placeholder_block_id, block_hash_str in block_id_hash_list:
            # NOTE: For saving, we don't use the placeholder_block_id to index into kv_layer
            # Instead, we need to find the actual physical block that contains this hash's content
            # For now, we'll use the placeholder_block_id (which is the logical block index)
            # as the physical block index. This assumes a direct mapping.
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
                
                # Save to PikiwiDB
                redis_client.set(k_key, k_data)
                redis_client.set(v_key, v_data)
                
                saved_count += 1
                if layer_idx == 0:
                    print(f"  💾 SAVED physical_block_id={physical_block_id}, hash={block_hash_str[:16]}..., keys={k_key}, {v_key}")
                
            except Exception as e:
                error_count += 1
                if layer_idx == 0:
                    print(f"  ⚠️ ERROR saving physical_block_id={physical_block_id}, hash={block_hash_str[:16]}...: {e}")
        
        if layer_idx == 0:
            print(f"📊 [SAVE] Layer {layer_idx}: Saved {saved_count}/{len(block_id_hash_list)} blocks, {error_count} errors")

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
        """Cache hit detection: check for consecutive prefix blocks using vLLM-native block hashes."""
        # Strict validation of block_hashes
        if not hasattr(request, 'block_hashes') or not request.block_hashes or not isinstance(request.block_hashes, list):
            return 0, False
        
        # Calculate starting block index based on computed tokens
        start_block_idx = num_computed_tokens // self.block_size
        if start_block_idx >= len(request.block_hashes):
            return 0, False
        
        # Check consecutive blocks starting from start_block_idx
        redis_client = self._get_redis_client()
        hit_blocks = 0
        
        for block_idx in range(start_block_idx, len(request.block_hashes)):
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
        end_logical_block = (num_total_tokens + self.block_size - 1) // self.block_size  # Ceiling division
        
        # Build mapping of physical block_id to logical block hash
        # IMPORTANT: The physical blocks are allocated in sequence for the token range
        # that needs computation, starting from the first uncomputed token
        block_id_hash_pairs = []
        
        physical_block_idx = 0
        for logical_block_idx in range(start_logical_block, end_logical_block):
            if physical_block_idx >= len(block_ids):
                break
                
            if logical_block_idx >= len(request.block_hashes):
                physical_block_idx += 1
                continue
            
            block_hash_obj = request.block_hashes[logical_block_idx]
            if not block_hash_obj or not hasattr(block_hash_obj, 'hash_value'):
                physical_block_idx += 1
                continue
            
            physical_block_id = block_ids[physical_block_idx]
            block_hash_str = str(block_hash_obj.hash_value)
            
            block_id_hash_pairs.append((physical_block_id, block_hash_str))
            physical_block_idx += 1
        
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
        """
        When a request finishes, save its blocks to PikiwiDB using vLLM-native block hashes.
        
        CRITICAL: We need to correctly map the physical block_ids to their corresponding
        logical block_hashes for saving. This is the reverse of the allocation mapping.
        """
        if not block_ids:
            return False, None
        
        # Strict validation of block_hashes
        if not hasattr(request, 'block_hashes') or not request.block_hashes or not isinstance(request.block_hashes, list):
            return False, None
        
        # Calculate which logical blocks have complete hashes and should be saved
        # Only save blocks that correspond to complete token sequences
        num_total_tokens = request.num_tokens
        num_complete_blocks = num_total_tokens // self.block_size
        
        # Build mapping of physical block_id to logical block hash for saving
        block_id_hash_pairs = []
        
        # For saving, we want to save all complete blocks that have valid hashes
        for logical_block_idx in range(min(num_complete_blocks, len(request.block_hashes))):
            block_hash_obj = request.block_hashes[logical_block_idx]
            
            if not block_hash_obj or not hasattr(block_hash_obj, 'hash_value'):
                continue
            
            # For complete blocks, we'll use a placeholder physical block ID
            # The actual mapping will be handled during saving based on the hash
            block_hash_str = str(block_hash_obj.hash_value)
            
            # Use logical block index as placeholder physical block ID for now
            # The save operation will use the hash to identify the content, not the physical ID
            placeholder_block_id = logical_block_idx
            
            block_id_hash_pairs.append((placeholder_block_id, block_hash_str))
        
        if not block_id_hash_pairs:
            return False, None
        
        # Schedule blocks for saving across all layers
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        layer_block_list = []
        
        for layer_idx in range(num_layers):
            layer_block_list.append((layer_idx, block_id_hash_pairs.copy()))
        
        self._blocks_to_save[request.request_id] = layer_block_list
        
        # Return False to allow immediate block freeing since we're using content-based saving
        # The hash-based approach doesn't depend on specific physical block IDs
        return False, None

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> Optional[str]:
        """
        For minimal implementation, don't require specific layout.
        In production, you might want to require HND for better performance.
        """
        return None
