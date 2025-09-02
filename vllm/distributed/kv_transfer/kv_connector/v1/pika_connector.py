# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import pickle
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
import redis

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import MLACommonMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class PikaReqMeta:
    """请求元数据，包含缓存相关信息"""
    request_id: str
    token_ids: torch.Tensor
    slot_mapping: torch.Tensor
    is_store: bool
    mm_hashes: list[str]
    cache_key_prefix: str

    @staticmethod
    def make_meta(request_id: str, token_ids: list[int], block_ids: list[int], 
                  block_size: int, is_store: bool, mm_hashes: list[str]) -> "PikaReqMeta":
        valid_num_tokens = align_to_block_size(len(token_ids), block_size)
        token_ids_tensor = torch.tensor(token_ids)[:valid_num_tokens]
        block_ids_tensor = torch.tensor(block_ids)
        num_blocks = block_ids_tensor.shape[0]
        block_offsets = torch.arange(0, block_size)
        slot_mapping = block_offsets.reshape((1, block_size)) + \
                block_ids_tensor.reshape((num_blocks, 1)) * block_size
        slot_mapping = slot_mapping.flatten()[:valid_num_tokens]
        
        # 生成缓存键前缀
        cache_key_prefix = _generate_cache_key_prefix(token_ids_tensor, mm_hashes)
        
        return PikaReqMeta(
            request_id=request_id,
            token_ids=token_ids_tensor,
            slot_mapping=slot_mapping,
            is_store=is_store,
            mm_hashes=mm_hashes,
            cache_key_prefix=cache_key_prefix,
        )


@dataclass
class PikaConnectorMetadata(KVConnectorMetadata):
    """Pika连接器元数据"""
    requests: list[PikaReqMeta]

    def __init__(self):
        self.requests = []

    def add_request(self, request_id: str, token_ids: list[int], block_ids: list[int],
                    block_size: int, is_store: bool, mm_hashes: list[str]) -> None:
        self.requests.append(
            PikaReqMeta.make_meta(request_id, token_ids, block_ids, block_size, 
                                  is_store, mm_hashes))


class PikaConnector(KVConnectorBase_V1):
    """
    PikiwiDB连接器，实现GPU-CPU-PikiwiDB三级存储架构
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        
        # 基础配置
        self._block_size = vllm_config.cache_config.block_size
        self._requests_need_load: dict[str, "Request"] = {}
        
        # PikiwiDB配置
        transfer_config = vllm_config.kv_transfer_config
        self._pikiwidb_host = transfer_config.get_from_extra_config("pikiwidb_host", "localhost")
        self._pikiwidb_port = transfer_config.get_from_extra_config("pikiwidb_port", 6379)
        self._pikiwidb_ttl = transfer_config.get_from_extra_config("pikiwidb_ttl", 3600)
        self._storage_policy = transfer_config.get_from_extra_config("storage_policy", "lru")
        
        # CPU缓存配置
        self._cpu_cache_size = transfer_config.get_from_extra_config("cpu_cache_size", 100)
        self._gpu_cache_size = transfer_config.get_from_extra_config("gpu_cache_size", 50)
        
        # 初始化Redis连接
        try:
            self._redis_client = redis.Redis(
                host=self._pikiwidb_host,
                port=self._pikiwidb_port,
                decode_responses=False,  # 处理二进制数据
                socket_timeout=5.0,
                socket_connect_timeout=5.0
            )
            # 测试连接
            self._redis_client.ping()
            logger.info(f"成功连接到PikiwiDB: {self._pikiwidb_host}:{self._pikiwidb_port}")
        except Exception as e:
            logger.warning(f"无法连接到PikiwiDB: {e}, 将使用本地缓存模式")
            self._redis_client = None
        
        # 初始化多级缓存
        self._gpu_cache: OrderedDict[str, torch.Tensor] = OrderedDict()  # GPU缓存
        self._cpu_cache: OrderedDict[str, torch.Tensor] = OrderedDict()  # CPU缓存
        
        logger.info(f"PikaConnector初始化完成 - "
                   f"CPU缓存大小: {self._cpu_cache_size}, "
                   f"GPU缓存大小: {self._gpu_cache_size}, "
                   f"TTL: {self._pikiwidb_ttl}s")

    def _generate_cache_key(self, layer_name: str, cache_key_prefix: str) -> str:
        """生成缓存键"""
        return f"vllm:kv:{cache_key_prefix}:{layer_name}"

    def _serialize_kv_cache(self, kv_cache: torch.Tensor) -> bytes:
        """序列化KV缓存数据"""
        try:
            # 移到CPU并序列化
            cpu_tensor = kv_cache.detach().cpu()
            serialized = pickle.dumps(cpu_tensor)
            # 压缩数据
            compressed = zlib.compress(serialized, level=6)
            return compressed
        except Exception as e:
            logger.error(f"序列化KV缓存失败: {e}")
            raise

    def _deserialize_kv_cache(self, data: bytes, device: torch.device) -> torch.Tensor:
        """反序列化KV缓存数据"""
        try:
            # 解压缩
            decompressed = zlib.decompress(data)
            # 反序列化
            tensor = pickle.loads(decompressed)
            # 移到指定设备
            return tensor.to(device)
        except Exception as e:
            logger.error(f"反序列化KV缓存失败: {e}")
            raise

    def _get_kv_cache(self, cache_key: str, device: torch.device) -> Optional[torch.Tensor]:
        """从三级存储中获取KV缓存"""
        # 1. 先查GPU缓存
        if cache_key in self._gpu_cache:
            kv_cache = self._gpu_cache[cache_key]
            # 更新LRU顺序
            self._gpu_cache.move_to_end(cache_key)
            logger.debug(f"GPU缓存命中: {cache_key}")
            return kv_cache.to(device)
        
        # 2. 查CPU缓存
        if cache_key in self._cpu_cache:
            kv_cache = self._cpu_cache[cache_key]
            # 更新LRU顺序
            self._cpu_cache.move_to_end(cache_key)
            # 提升到GPU缓存
            self._set_gpu_cache(cache_key, kv_cache.to(device))
            logger.debug(f"CPU缓存命中: {cache_key}")
            return kv_cache.to(device)
        
        # 3. 查PikiwiDB
        if self._redis_client:
            try:
                data = self._redis_client.get(cache_key)
                if data:
                    kv_cache = self._deserialize_kv_cache(data, device)
                    # 提升到CPU和GPU缓存
                    self._set_cpu_cache(cache_key, kv_cache.cpu())
                    self._set_gpu_cache(cache_key, kv_cache)
                    logger.debug(f"PikiwiDB缓存命中: {cache_key}")
                    return kv_cache
            except Exception as e:
                logger.warning(f"从PikiwiDB获取缓存失败: {e}")
        
        return None

    def _set_gpu_cache(self, cache_key: str, kv_cache: torch.Tensor):
        """设置GPU缓存"""
        # 如果缓存满了，删除最旧的条目
        while len(self._gpu_cache) >= self._gpu_cache_size:
            oldest_key = next(iter(self._gpu_cache))
            del self._gpu_cache[oldest_key]
        
        self._gpu_cache[cache_key] = kv_cache.detach()

    def _set_cpu_cache(self, cache_key: str, kv_cache: torch.Tensor):
        """设置CPU缓存"""
        # 如果缓存满了，删除最旧的条目
        while len(self._cpu_cache) >= self._cpu_cache_size:
            oldest_key = next(iter(self._cpu_cache))
            del self._cpu_cache[oldest_key]
        
        self._cpu_cache[cache_key] = kv_cache.detach()

    def _set_kv_cache(self, cache_key: str, kv_cache: torch.Tensor, ttl_seconds: int):
        """保存KV缓存到三级存储"""
        # 1. 保存到GPU缓存
        self._set_gpu_cache(cache_key, kv_cache)
        
        # 2. 保存到CPU缓存
        self._set_cpu_cache(cache_key, kv_cache.cpu())
        
        # 3. 异步保存到PikiwiDB
        if self._redis_client:
            try:
                serialized_data = self._serialize_kv_cache(kv_cache)
                self._redis_client.setex(cache_key, ttl_seconds, serialized_data)
                logger.debug(f"KV缓存已保存到PikiwiDB: {cache_key}")
            except Exception as e:
                logger.warning(f"保存到PikiwiDB失败: {e}")

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """开始加载KV缓存"""
        attn_metadata = forward_context.attn_metadata
        
        def inject_kv_into_layer(
            dst_kv_cache_layer: torch.Tensor,
            src_kv_cache: torch.Tensor,
            slot_mapping: torch.Tensor,
        ) -> None:
            """将KV缓存注入到层中"""
            dst_kv_cache_layer_shape = dst_kv_cache_layer.shape
            if isinstance(attn_metadata, MLACommonMetadata):
                num_pages = dst_kv_cache_layer_shape[0]
                page_size = dst_kv_cache_layer_shape[1]
                dst_kv_cache_layer = dst_kv_cache_layer.reshape(
                    num_pages * page_size, -1)
                dst_kv_cache_layer[slot_mapping, ...] = src_kv_cache
                dst_kv_cache_layer.reshape(dst_kv_cache_layer_shape)
            else:
                num_pages = dst_kv_cache_layer_shape[1]
                page_size = dst_kv_cache_layer_shape[2]
                dst_kv_cache_layer = dst_kv_cache_layer.reshape(
                    2, num_pages * page_size, -1)
                dst_kv_cache_layer[:, slot_mapping, ...] = src_kv_cache
                dst_kv_cache_layer.reshape(dst_kv_cache_layer_shape)

        # 获取元数据
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, PikaConnectorMetadata)

        if metadata is None:
            logger.warning("connector metadata为空，跳过KV加载")
            return

        if attn_metadata is None:
            logger.warning("attention metadata为空，跳过KV加载")
            return

        # 为每个请求的每个层加载KV
        for request in metadata.requests:
            if request.is_store:
                continue
                
            logger.info(f"开始注入KV缓存，token数量: {len(request.slot_mapping)}")
            
            for layer_name in forward_context.no_compile_layers:
                layer = forward_context.no_compile_layers[layer_name]
                
                # 只处理有kv_cache属性的层（注意力层）
                kv_cache_attr = getattr(layer, 'kv_cache', None)
                if kv_cache_attr is None:
                    continue
                
                kv_cache_layer = kv_cache_attr[forward_context.virtual_engine]
                cache_key = self._generate_cache_key(layer_name, request.cache_key_prefix)
                
                # 从三级存储获取KV缓存
                kv_cache = self._get_kv_cache(cache_key, kv_cache_layer.device)
                if kv_cache is not None:
                    inject_kv_into_layer(kv_cache_layer, kv_cache, request.slot_mapping)
                    logger.debug(f"成功注入层 {layer_name} 的KV缓存")

    def wait_for_layer_load(self, layer_name: str) -> None:
        """等待层加载完成（同步实现）"""
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """保存KV缓存层"""
        
        def extract_kv_from_layer(
            layer: torch.Tensor,
            slot_mapping: torch.Tensor,
        ) -> torch.Tensor:
            """从层中提取KV缓存"""
            if isinstance(attn_metadata, MLACommonMetadata):
                num_pages, page_size = layer.shape[0], layer.shape[1]
                return layer.reshape(num_pages * page_size, -1)[slot_mapping, ...]
            num_pages, page_size = layer.shape[1], layer.shape[2]
            return layer.reshape(2, num_pages * page_size, -1)[:, slot_mapping, ...]

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, PikaConnectorMetadata)
        
        for request in connector_metadata.requests:
            if request.is_store:
                cache_key = self._generate_cache_key(layer_name, request.cache_key_prefix)
                kv_cache = extract_kv_from_layer(kv_layer, request.slot_mapping)
                
                # 保存到三级存储
                self._set_kv_cache(cache_key, kv_cache.detach(), self._pikiwidb_ttl)
                logger.debug(f"已保存层 {layer_name} 的KV缓存")

    def wait_for_save(self):
        """等待保存完成（同步实现）"""
        pass

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """获取可以从外部缓存加载的新token数量"""
        if not self._found_match_for_request(request):
            return 0, False

        logger.info("外部缓存命中!")
        
        # 计算要检查的token数量（对齐到块大小）
        num_tokens_to_check = align_to_block_size(
            len(request.prompt_token_ids) - 1, self._block_size)
        
        return num_tokens_to_check - num_computed_tokens, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        """分配块后更新状态"""
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """构建连接器元数据"""
        meta = PikaConnectorMetadata()

        total_need_load = 0
        
        # 处理新请求
        for new_req in scheduler_output.scheduled_new_reqs:
            if new_req.req_id in self._requests_need_load:
                meta.add_request(request_id=new_req.req_id,
                               token_ids=new_req.prompt_token_ids,
                               block_ids=new_req.block_ids[0],
                               block_size=self._block_size,
                               is_store=False,
                               mm_hashes=new_req.mm_hashes)
                total_need_load += 1
            else:
                # 如果没有缓存命中，则需要存储
                if not self._found_match_for_request(new_req):
                    meta.add_request(request_id=new_req.req_id,
                                   token_ids=new_req.prompt_token_ids,
                                   block_ids=new_req.block_ids[0],
                                   block_size=self._block_size,
                                   is_store=True,
                                   mm_hashes=new_req.mm_hashes)

        # 处理缓存请求
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            num_computed_tokens = cached_reqs.num_computed_tokens[i]
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            new_block_ids = cached_reqs.new_block_ids[i]
            resumed_from_preemption = cached_reqs.resumed_from_preemption[i]

            if not resumed_from_preemption:
                break
                
            if req_id in self._requests_need_load:
                request = self._requests_need_load[req_id]
                total_tokens = num_computed_tokens + num_new_tokens
                token_ids = request.all_token_ids[:total_tokens]
                block_ids = new_block_ids[0]

                meta.add_request(request_id=req_id,
                               token_ids=token_ids,
                               block_ids=block_ids,
                               block_size=self._block_size,
                               is_store=False,
                               mm_hashes=request.mm_hashes)
                total_need_load += 1

        assert total_need_load == len(self._requests_need_load)
        self._requests_need_load.clear()
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """请求完成时调用"""
        return False, None

    def _found_match_for_request(self, request: "Request") -> bool:
        """检查请求是否有缓存命中"""
        num_tokens_to_check = align_to_block_size(
            len(request.prompt_token_ids) - 1, self._block_size)
        
        if num_tokens_to_check <= 0:
            return False
            
        token_ids_tensor = torch.tensor(request.prompt_token_ids)[:num_tokens_to_check]
        cache_key_prefix = _generate_cache_key_prefix(token_ids_tensor, request.mm_hashes)
        
        # 检查第一个层是否存在缓存（简化实现）
        test_layer_name = "model.layers.0.self_attn"
        cache_key = self._generate_cache_key(test_layer_name, cache_key_prefix)
        
        # 检查三级存储是否存在
        if cache_key in self._gpu_cache or cache_key in self._cpu_cache:
            return True
            
        if self._redis_client:
            try:
                return self._redis_client.exists(cache_key) > 0
            except Exception as e:
                logger.warning(f"检查PikiwiDB缓存时出错: {e}")
                
        return False


def _generate_cache_key_prefix(token_ids: torch.Tensor, mm_hashes: list[str]) -> str:
    """生成缓存键前缀"""
    token_bytes = token_ids.numpy().tobytes()
    if mm_hashes:
        mm_str = "-".join(mm_hashes)
        token_bytes += mm_str.encode('utf-8')
    return hashlib.md5(token_bytes, usedforsecurity=False).hexdigest()


def align_to_block_size(num_tokens: int, block_size: int) -> int:
    """将token数量对齐到块大小"""
    if num_tokens <= 0:
        return 0
    return (num_tokens) // block_size * block_size