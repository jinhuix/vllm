# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import pickle
import struct
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, List, Dict, Tuple

import torch
import redis
import asyncio
import concurrent.futures

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
class KVPageInfo:
    """KV Page 信息，对应 vLLM 的 PagedAttention"""
    req_id: str
    layer_idx: int
    head_idx: int  
    page_id: int
    kv_type: int  # 0=K, 1=V
    page_size: int  # tokens per page (e.g. 128)
    head_dim: int
    dtype: int  # 1=FP16, 2=FP32
    
    def build_key(self) -> str:
        """构建 PikiwiDB 存储键"""
        return f"kv:{self.req_id}:{self.layer_idx}:{self.head_idx}:{self.page_id}:{self.kv_type}"

@dataclass
class PikaReqMeta:
    """请求元数据，包含缓存相关信息 - 适配 PagedAttention (去重优化版本)"""
    request_id: str
    token_ids: torch.Tensor
    slot_mapping: torch.Tensor
    is_store: bool
    mm_hashes: list[str]
    cache_key_prefix: str
    # Page-oriented fields (去重优化)
    slot_to_page_mapping: Dict[int, Tuple[int, int, int]]  # slot_idx -> (page_id, head_idx, token_offset)
    unique_pages: Dict[Tuple[int, int, int], KVPageInfo]  # (page_id, head_idx, kv_type) -> page_info
    page_size: int  # 页大小，通常是 128 tokens

    @staticmethod
    def make_meta(request_id: str, token_ids: list[int], block_ids: list[int], 
                  block_size: int, is_store: bool, mm_hashes: list[str],
                  page_size: int = 128, head_dim: int = 128, num_heads: int = 32) -> "PikaReqMeta":
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
        
        # 构建去重的 Page 映射
        slot_to_page_mapping = {}
        unique_pages = {}
        
        for slot_idx, slot_id in enumerate(slot_mapping.tolist()):
            page_id = slot_id // page_size  # 计算页ID
            token_offset = slot_id % page_size  # 页内偏移
            
            # 每个 slot 对应一个页面位置 (保留完整的 head 信息以备将来使用)
            # 这里暂时用 head_idx=0，但保持结构完整性
            slot_to_page_mapping[slot_idx] = (page_id, 0, token_offset)
            
            # 为每个唯一的 (page_id, head_idx, kv_type) 创建一个 KVPageInfo
            for head_idx in range(num_heads):
                for kv_type in [0, 1]:  # K=0, V=1
                    page_key = (page_id, head_idx, kv_type)
                    if page_key not in unique_pages:
                        unique_pages[page_key] = KVPageInfo(
                            req_id=request_id,
                            layer_idx=0,  # 将在实际使用时设置
                            head_idx=head_idx,
                            page_id=page_id,
                            kv_type=kv_type,
                            page_size=page_size,
                            head_dim=head_dim,
                            dtype=1  # FP16
                        )
        
        return PikaReqMeta(
            request_id=request_id,
            token_ids=token_ids_tensor,
            slot_mapping=slot_mapping,
            is_store=is_store,
            mm_hashes=mm_hashes,
            cache_key_prefix=cache_key_prefix,
            slot_to_page_mapping=slot_to_page_mapping,
            unique_pages=unique_pages,
            page_size=page_size,
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
        
        # 页面缓存已经通过 PikiwiDB 直接管理，不需要额外的本地缓存
        
        logger.info(f"PikaConnector初始化完成 - "
                   f"页面大小: 128 tokens, "
                   f"TTL: {self._pikiwidb_ttl}s, "
                   f"存储策略: {self._storage_policy}")

    def _generate_cache_key(self, layer_name: str, cache_key_prefix: str) -> str:
        """生成缓存键（Legacy 方法）"""
        return f"vllm:kv:{cache_key_prefix}:{layer_name}"
    
    def _serialize_kv_page(self, kv_page: torch.Tensor, dtype: int = 1) -> bytes:
        """序列化单个 KV 页面为二进制数据 (优化版本)
        
        Args:
            kv_page: [page_size, head_dim] 的 tensor
            dtype: 数据类型 (1=FP16, 2=FP32)
        
        Returns:
            序列化的页面数据（不包含 header，header 由 PikiwiDB 处理）
        """
        try:
            # 确保 tensor 连续且在 CPU 上，避免隐式拷贝
            cpu_tensor = kv_page.detach().cpu().contiguous()
            
            # 根据 dtype 转换数据类型
            if dtype == 1:  # FP16
                cpu_tensor = cpu_tensor.to(torch.float16)
            elif dtype == 2:  # FP32
                cpu_tensor = cpu_tensor.to(torch.float32)
            
            # 确保连续性后再转换为 numpy（零拷贝）
            if not cpu_tensor.is_contiguous():
                cpu_tensor = cpu_tensor.contiguous()
            
            return cpu_tensor.numpy().tobytes()
            
        except Exception as e:
            logger.error(f"序列化KV页面失败: {e}")
            raise
    
    def _deserialize_kv_page(self, data: bytes, page_size: int, head_dim: int, 
                           dtype: int, device: torch.device, use_async: bool = True) -> torch.Tensor:
        """反序列化KV页面数据 (优化版本，支持异步传输)
        
        Args:
            data: 序列化的页面数据
            page_size: 页面大小（token数）
            head_dim: 头维度
            dtype: 数据类型 (1=FP16, 2=FP32)
            device: 目标设备
            use_async: 是否使用异步传输到 GPU
        
        Returns:
            [page_size, head_dim] 的 tensor
        """
        try:
            import numpy as np
            
            # 根据 dtype 确定 numpy 数据类型
            if dtype == 1:  # FP16
                np_dtype = np.float16
                torch_dtype = torch.float16
            elif dtype == 2:  # FP32
                np_dtype = np.float32
                torch_dtype = torch.float32
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")
            
            # 从 raw bytes 重建 numpy array (创建可写的连续数组)
            np_array = np.frombuffer(data, dtype=np_dtype).reshape(page_size, head_dim).copy()
            
            # 转换为 torch tensor
            host_tensor = torch.from_numpy(np_array)
            
            # 优化的设备传输
            if device.type == 'cuda' and use_async:
                # 使用 pinned memory 和异步传输
                try:
                    pinned_tensor = host_tensor.pin_memory()
                    cuda_tensor = pinned_tensor.to(device, non_blocking=True)
                    return cuda_tensor
                except Exception as e:
                    logger.warning(f"异步传输失败，回退到同步传输: {e}")
                    return host_tensor.to(device)
            else:
                # 同步传输
                return host_tensor.to(device)
            
        except Exception as e:
            logger.error(f"反序列化KV页面失败: {e}")
            raise
    
    def _batch_store_pages(self, page_infos: List[KVPageInfo], 
                          tensors: List[torch.Tensor]) -> bool:
        """批量存储多个页面到 PikiwiDB
        
        Args:
            page_infos: 页面信息列表
            tensors: 对应的 tensor 列表
        
        Returns:
            是否全部存储成功
        """
        if not self._redis_client or len(page_infos) != len(tensors):
            return False
        
        try:
            # 构建 KVPAGEMSET 命令参数
            cmd_args = ["KVPAGEMSET", str(len(page_infos))]
            
            for page_info, tensor in zip(page_infos, tensors):
                # 序列化 tensor 数据
                tensor_data = self._serialize_kv_page(tensor, page_info.dtype)
                
                # 添加页面参数
                cmd_args.extend([
                    page_info.req_id,
                    str(page_info.layer_idx),
                    str(page_info.head_idx),
                    str(page_info.page_id),
                    str(page_info.kv_type),
                    str(page_info.dtype),
                    str(page_info.page_size),
                    str(page_info.head_dim),
                    str(self._pikiwidb_ttl),  # TTL
                    tensor_data
                ])
            
            # 执行批量存储
            result = self._redis_client.execute_command(*cmd_args)
            success = (result == b'OK')
            
            if success:
                logger.debug(f"批量存储 {len(page_infos)} 个页面成功")
            else:
                logger.warning(f"批量存储页面失败: {result}")
                
            return success
            
        except Exception as e:
            logger.error(f"批量存储页面异常: {e}")
            return False
    
    def _batch_load_pages(self, page_infos: List[KVPageInfo], 
                         device: torch.device) -> List[Optional[torch.Tensor]]:
        """批量加载多个页面从 PikiwiDB
        
        Args:
            page_infos: 页面信息列表
            device: 目标设备
        
        Returns:
            加载的 tensor 列表（None 表示页面不存在）
        """
        if not self._redis_client:
            return [None] * len(page_infos)
        
        try:
            # 构建 KVPAGEMGET 命令参数
            cmd_args = ["KVPAGEMGET", str(len(page_infos))]
            
            for page_info in page_infos:
                cmd_args.extend([
                    page_info.req_id,
                    str(page_info.layer_idx),
                    str(page_info.head_idx),
                    str(page_info.page_id),
                    str(page_info.kv_type)
                ])
            
            # 执行批量加载
            results = self._redis_client.execute_command(*cmd_args)
            
            if not isinstance(results, list):
                logger.warning(f"批量加载返回格式异常: {type(results)}")
                return [None] * len(page_infos)
            
            # 解析结果
            tensors = []
            for i, (page_info, result) in enumerate(zip(page_infos, results)):
                if result is None or (isinstance(result, bytes) and len(result) == 0):
                    tensors.append(None)
                    continue
                
                try:
                    # 反序列化页面数据
                    tensor = self._deserialize_kv_page(
                        result, page_info.page_size, page_info.head_dim, 
                        page_info.dtype, device)
                    tensors.append(tensor)
                except Exception as e:
                    logger.warning(f"反序列化第 {i} 个页面失败: {e}")
                    tensors.append(None)
            
            logger.debug(f"批量加载 {len(page_infos)} 个页面，成功 {sum(1 for t in tensors if t is not None)} 个")
            return tensors
            
        except Exception as e:
            logger.error(f"批量加载页面异常: {e}")
            return [None] * len(page_infos)



    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """开始加载KV缓存 - Page 粒度去重优化版本"""
        attn_metadata = forward_context.attn_metadata
        
        def place_page_into_kv_cache(
            dst_kv_cache_layer: torch.Tensor,
            page_tensor: torch.Tensor,
            page_info: KVPageInfo,
        ) -> None:
            """将单个页面放入 KV cache 的正确位置 (通用适配器)"""
            dst_shape = dst_kv_cache_layer.shape
            
            try:
                if isinstance(attn_metadata, MLACommonMetadata):
                    # MLA 格式: [num_pages, page_size, num_heads, head_dim]
                    if len(dst_shape) >= 4 and page_info.page_id < dst_shape[0]:
                        dst_kv_cache_layer[page_info.page_id, :, page_info.head_idx, :] = page_tensor
                else:
                    # 标准格式: [2, num_pages, page_size, num_heads, head_dim] 或其变种
                    if len(dst_shape) >= 5 and page_info.page_id < dst_shape[1]:
                        dst_kv_cache_layer[page_info.kv_type, page_info.page_id, :, page_info.head_idx, :] = page_tensor
                    elif len(dst_shape) >= 4:
                        # 备用格式: [num_pages, page_size, num_heads, head_dim]
                        if page_info.page_id < dst_shape[0]:
                            dst_kv_cache_layer[page_info.page_id, :, page_info.head_idx, :] = page_tensor
            except Exception as e:
                logger.warning(f"页面注入失败 {page_info.page_id}:{page_info.head_idx}:{page_info.kv_type} - {e}")

        # 获取元数据
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, PikaConnectorMetadata)

        if metadata is None:
            logger.warning("connector metadata为空，跳过KV加载")
            return

        if attn_metadata is None:
            logger.warning("attention metadata为空，跳过KV加载")
            return

        # 为每个请求的每个层加载KV页面 (去重优化)
        for request in metadata.requests:
            if request.is_store:
                continue
                
            logger.info(f"开始注入KV页面缓存，唯一页面数: {len(request.unique_pages)}")
            
            for layer_idx, layer_name in enumerate(forward_context.no_compile_layers):
                layer = forward_context.no_compile_layers[layer_name]
                
                # 只处理有kv_cache属性的层（注意力层）
                kv_cache_attr = getattr(layer, 'kv_cache', None)
                if kv_cache_attr is None:
                    continue
                
                kv_cache_layer = kv_cache_attr[forward_context.virtual_engine]
                
                # 收集这一层需要加载的唯一页面 (去重，避免修改共享对象)
                layer_unique_pages = []
                for page_key, page_info in request.unique_pages.items():
                    # 克隆页面信息并设置正确的层索引
                    layer_page_info = KVPageInfo(
                        req_id=page_info.req_id,
                        layer_idx=layer_idx,  # 设置当前层索引
                        head_idx=page_info.head_idx,
                        page_id=page_info.page_id,
                        kv_type=page_info.kv_type,
                        page_size=page_info.page_size,
                        head_dim=page_info.head_dim,
                        dtype=page_info.dtype
                    )
                    layer_unique_pages.append(layer_page_info)
                
                if not layer_unique_pages:
                    continue
                
                # 批量加载唯一页面 (避免重复请求)
                page_tensors = self._batch_load_pages(layer_unique_pages, kv_cache_layer.device)
                
                # 将加载的页面注入到层中
                for page_tensor, page_info in zip(page_tensors, layer_unique_pages):
                    if page_tensor is not None:
                        place_page_into_kv_cache(kv_cache_layer, page_tensor, page_info)
                
                loaded_count = sum(1 for t in page_tensors if t is not None)
                logger.debug(f"层 {layer_name} 成功注入 {loaded_count}/{len(layer_unique_pages)} 个唯一KV页面")

    def wait_for_layer_load(self, layer_name: str) -> None:
        """等待层加载完成（同步实现）"""
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """保存KV缓存层 - Page 粒度去重优化版本"""
        
        def extract_page_from_layer(
            layer: torch.Tensor,
            page_info: KVPageInfo,
        ) -> Optional[torch.Tensor]:
            """从层中提取单个KV页面 (通用适配器)"""
            layer_shape = layer.shape
            
            try:
                if isinstance(attn_metadata, MLACommonMetadata):
                    # MLA 格式: [num_pages, page_size, num_heads, head_dim]
                    if len(layer_shape) >= 4 and page_info.page_id < layer_shape[0]:
                        return layer[page_info.page_id, :, page_info.head_idx, :].clone()
                else:
                    # 标准格式: [2, num_pages, page_size, num_heads, head_dim] 或其变种
                    if len(layer_shape) >= 5 and page_info.page_id < layer_shape[1]:
                        return layer[page_info.kv_type, page_info.page_id, :, page_info.head_idx, :].clone()
                    elif len(layer_shape) >= 4:
                        # 备用格式: [num_pages, page_size, num_heads, head_dim]
                        if page_info.page_id < layer_shape[0]:
                            return layer[page_info.page_id, :, page_info.head_idx, :].clone()
                
                return None
            except Exception as e:
                logger.warning(f"页面提取失败 {page_info.page_id}:{page_info.head_idx}:{page_info.kv_type} - {e}")
                return None

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, PikaConnectorMetadata)
        
        # 提取层索引 (简化的层名匹配)
        layer_idx = 0
        try:
            import re
            match = re.search(r'layer[._](\d+)', layer_name.lower())
            if match:
                layer_idx = int(match.group(1))
        except:
            pass
        
        for request in connector_metadata.requests:
            if not request.is_store:
                continue
            
            # 收集这一层需要存储的唯一页面 (去重，避免修改共享对象)
            layer_unique_pages = []
            for page_key, page_info in request.unique_pages.items():
                # 克隆页面信息并设置正确的层索引
                layer_page_info = KVPageInfo(
                    req_id=page_info.req_id,
                    layer_idx=layer_idx,  # 设置当前层索引
                    head_idx=page_info.head_idx,
                    page_id=page_info.page_id,
                    kv_type=page_info.kv_type,
                    page_size=page_info.page_size,
                    head_dim=page_info.head_dim,
                    dtype=page_info.dtype
                )
                layer_unique_pages.append(layer_page_info)
            
            if not layer_unique_pages:
                continue
            
            # 从层中提取唯一页面
            valid_page_infos = []
            valid_page_tensors = []
            
            for page_info in layer_unique_pages:
                page_tensor = extract_page_from_layer(kv_layer, page_info)
                if page_tensor is not None:
                    valid_page_infos.append(page_info)
                    valid_page_tensors.append(page_tensor)
            
            if valid_page_infos:
                # 批量存储页面
                success = self._batch_store_pages(valid_page_infos, valid_page_tensors)
                if success:
                    logger.debug(f"层 {layer_name} 成功保存 {len(valid_page_infos)} 个唯一KV页面")
                else:
                    logger.warning(f"层 {layer_name} 保存KV页面失败")

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

    def _found_match_for_request(self, request) -> bool:
        """检查请求是否有页面缓存命中 (基于 KVPAGEEXISTS 的检测)"""
        num_tokens_to_check = align_to_block_size(
            len(request.prompt_token_ids) - 1, self._block_size)
        
        if num_tokens_to_check <= 0:
            return False
        
        # 生成测试用的关键页面参数
        # 处理不同类型的请求对象 (Request vs NewRequestData)
        req_id = getattr(request, 'request_id', None) or getattr(request, 'req_id', None)
        test_page_id = 0  # 检查第一个页面
        test_layer_idx = 0  # 检查第一层
        test_head_idx = 0   # 检查第一个头
        test_kv_type = 0    # 检查 K cache
        
        # 检查 PikiwiDB 是否存在关键页面
        if self._redis_client:
            try:
                # 使用新的 KVPAGEEXISTS 命令检查页面是否存在
                result = self._redis_client.execute_command(
                    'KVPAGEEXISTS',
                    req_id, test_layer_idx, test_head_idx, test_page_id, test_kv_type
                )
                # KVPAGEEXISTS 返回 1 表示存在，0 表示不存在
                return result == 1
            except Exception as e:
                logger.warning(f"检查页面缓存时出错: {e}")
                
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