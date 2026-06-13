from abc import ABC, abstractmethod
import torch
import torch.multiprocessing as mp
from typing import Dict, Any, Optional, Callable
from queue import Queue, Empty
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """验证结果数据类"""
    step: int
    eval_loss: float
    eval_ppl: float
    extra_info: Optional[Dict] = None


class BaseValidator(ABC):
    """验证器基类 - 统一接口"""
    
    @abstractmethod
    def submit(self, step: int, model_state: Dict[str, torch.Tensor], 
               validation_data: Any, **kwargs) -> None:
        """
        提交验证任务（非阻塞）
        
        Args:
            step: 当前训练步数
            model_state: 模型状态字典（CPU上的权重）
            validation_data: 验证数据（如数据集路径、配置等）
            **kwargs: 验证所需的其他参数
        """
        pass
    
    @abstractmethod
    def poll(self) -> Optional[ValidationResult]:
        """
        轮询获取验证结果（非阻塞）
        
        Returns:
            ValidationResult: 如果有完成的结果
            None: 如果没有结果
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """关闭验证器，清理资源"""
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """是否有验证任务正在运行"""
        pass


class SyncValidator(BaseValidator):
    """同步验证器 - 阻塞式，最简单"""
    
    def __init__(self, validate_fn: Callable):
        """
        Args:
            validate_fn: 实际的验证函数，签名为 (step, model_state, validation_data) -> ValidationResult
        """
        self.validate_fn = validate_fn
        self._last_result: Optional[ValidationResult] = None
        self._is_running = False
    
    def submit(self, step: int, model_state: Dict[str, torch.Tensor],
               validation_data: Any, **kwargs) -> None:
        """同步执行验证（会阻塞）"""
        self._is_running = True
        try:
            # 直接调用验证函数，阻塞等待结果
            self._last_result = self.validate_fn(step, model_state, validation_data, **kwargs)
        finally:
            self._is_running = False
    
    def poll(self) -> Optional[ValidationResult]:
        """立即返回上次验证结果"""
        result = self._last_result
        self._last_result = None
        return result
    
    def shutdown(self) -> None:
        """无需清理"""
        self._is_running = False
    
    def is_running(self) -> bool:
        return self._is_running


class StreamValidator(BaseValidator):
    """CUDA Stream 异步验证器 - 单GPU并发"""
    
    def __init__(self, device: str = 'cuda'):
        """
        Args:
            device: CUDA设备
        """
        self.device = torch.device(device) if device != 'cpu' else 'cpu'
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self._result_queue: Queue = Queue(maxsize=1)
        self._current_step: Optional[int] = None
        self._is_running = False
        self._validate_fn: Optional[Callable] = None
    
    def set_validate_fn(self, validate_fn: Callable):
        """设置验证函数（需要在使用前调用）"""
        self._validate_fn = validate_fn
    
    def submit(self, step: int, model_state: Dict[str, torch.Tensor],
               validation_data: Any, **kwargs) -> None:
        """异步提交验证（不阻塞训练）"""
        if self._is_running:
            logger.warning(f"Validation at step {step} skipped - previous still running")
            return
        
        if self._validate_fn is None:
            raise RuntimeError("Validate function not set. Call set_validate_fn first.")
        
        self._is_running = True
        self._current_step = step
        
        # 使用 CUDA Stream 异步执行
        if self.stream:
            # 记录当前训练stream的事件
            current_stream = torch.cuda.current_stream()
            event = torch.cuda.Event()
            event.record(current_stream)
            
            # 在独立stream上执行验证
            with torch.cuda.stream(self.stream):
                # 等待训练stream完成
                event.wait(self.stream)
                self._run_validation_async(step, model_state, validation_data, **kwargs)
        else:
            # CPU fallback - 使用线程
            import threading
            thread = threading.Thread(
                target=self._run_validation_sync,
                args=(step, model_state, validation_data),
                kwargs=kwargs
            )
            thread.start()
    
    def _run_validation_async(self, step: int, model_state: Dict[str, torch.Tensor],
                              validation_data: Any, **kwargs):
        """在CUDA stream中执行验证"""
        try:
            # 加载模型到GPU（复用现有内存）
            # 注意：这里假设验证函数会正确处理模型状态
            result = self._validate_fn(step, model_state, validation_data, **kwargs)
            self._result_queue.put(result)
        except Exception as e:
            logger.error(f"Stream validation failed at step {step}: {e}")
            self._result_queue.put(None)
        finally:
            self._is_running = False
    
    def _run_validation_sync(self, step: int, model_state: Dict[str, torch.Tensor],
                             validation_data: Any, **kwargs):
        """CPU线程中同步执行"""
        try:
            result = self._validate_fn(step, model_state, validation_data, **kwargs)
            self._result_queue.put(result)
        except Exception as e:
            logger.error(f"Thread validation failed at step {step}: {e}")
            self._result_queue.put(None)
        finally:
            self._is_running = False
    
    def poll(self) -> Optional[ValidationResult]:
        """非阻塞获取结果"""
        try:
            result = self._result_queue.get_nowait()
            return result
        except Empty:
            return None
    
    def shutdown(self) -> None:
        """清理资源"""
        if self.stream and torch.cuda.is_available():
            self.stream.synchronize()
        self._is_running = False
    
    def is_running(self) -> bool:
        return self._is_running


class ProcessValidator(BaseValidator):
    """多进程异步验证器 - 真正并行"""
    
    def __init__(self, target_device: str = 'cuda:1', num_workers: int = 1):
        """
        Args:
            target_device: 验证进程使用的GPU设备
            num_workers: 验证进程数（通常为1）
        """
        self.target_device = target_device
        self.num_workers = num_workers
        
        self._request_queue: Queue = Queue(maxsize=1)
        self._result_queue: Queue = Queue(maxsize=1)
        self._processes = []
        self._is_running = False
        self._validate_fn: Optional[Callable] = None
        self._model_config: Optional[Dict] = None
        
    def set_validate_fn(self, validate_fn: Callable):
        """设置验证函数（需要在使用前调用）"""
        self._validate_fn = validate_fn
    
    def set_model_config(self, model_config: Dict):
        """设置模型配置（用于重建模型）"""
        self._model_config = model_config
    
    def _worker_process(self, worker_id: int, request_queue: Queue, result_queue: Queue,
                        validate_fn_serialized, device: str):
        """独立进程的工作函数"""
        import torch
        import numpy as np
        
        # 设置设备
        torch.cuda.set_device(int(device.split(':')[-1]) if 'cuda' in device else None)
        
        while True:
            try:
                request = request_queue.get(timeout=1)
                if request is None:  # 停止信号
                    break
                
                step, model_state, validation_data, kwargs = request
                
                # 重建模型（深拷贝的状态）
                # 这里需要根据你的模型类实现
                # model = rebuild_model(model_state, self._model_config)
                # model = model.to(device)
                
                # 执行验证
                # result = validate_fn(step, model, validation_data, **kwargs)
                
                # 模拟验证
                result = ValidationResult(
                    step=step,
                    eval_loss=2.5,
                    eval_ppl=12.18
                )
                
                result_queue.put(result)
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} validation failed: {e}")
                result_queue.put(None)
                break
    
    def submit(self, step: int, model_state: Dict[str, torch.Tensor],
               validation_data: Any, **kwargs) -> None:
        """异步提交验证任务"""
        if self._is_running:
            logger.warning(f"Validation at step {step} skipped - previous still running")
            return
        
        if self._validate_fn is None:
            raise RuntimeError("Validate function not set. Call set_validate_fn first.")
        
        # 启动工作进程（如果未启动）
        if not self._processes:
            self._start_workers()
        
        # 将模型状态移到CPU（减少内存占用）
        cpu_state = {k: v.cpu().clone() for k, v in model_state.items()}
        
        try:
            self._request_queue.put_nowait((step, cpu_state, validation_data, kwargs))
            self._is_running = True
        except:
            logger.warning(f"Failed to submit validation at step {step}")
    
    def _start_workers(self):
        """启动工作进程"""
        for i in range(self.num_workers):
            p = mp.Process(
                target=self._worker_process,
                args=(i, self._request_queue, self._result_queue, 
                      self._validate_fn, self.target_device),
                daemon=True
            )
            p.start()
            self._processes.append(p)
        logger.info(f"Started {self.num_workers} validation worker processes on {self.target_device}")
    
    def poll(self) -> Optional[ValidationResult]:
        """非阻塞获取结果"""
        try:
            result = self._result_queue.get_nowait()
            self._is_running = False
            return result
        except Empty:
            return None
    
    def shutdown(self) -> None:
        """关闭所有工作进程"""
        for _ in self._processes:
            try:
                self._request_queue.put_nowait(None)
            except:
                pass
        
        for p in self._processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        
        self._processes.clear()
        self._is_running = False
    
    def is_running(self) -> bool:
        return self._is_running


class ValidatorFactory:
    """验证器工厂 - 根据配置创建合适的验证器"""
    
    @staticmethod
    def create(mode: str, validate_fn: Callable, 
               device: str = 'cuda', **kwargs) -> BaseValidator:
        """
        创建验证器实例
        
        Args:
            mode: 'sync', 'stream', 'process', 'auto'
            validate_fn: 验证函数
            device: 主训练设备
            **kwargs: 额外参数
        """
        if mode == 'sync':
            validator = SyncValidator(validate_fn)
        
        elif mode == 'stream':
            validator = StreamValidator(device=device)
            validator.set_validate_fn(validate_fn)
        
        elif mode == 'process':
            target_device = kwargs.get('target_device', 'cuda:1')
            validator = ProcessValidator(target_device=target_device)
            validator.set_validate_fn(validate_fn)
            if kwargs.get('model_config'):
                validator.set_model_config(kwargs['model_config'])
        
        elif mode == 'auto':
            # 自动选择：多GPU用process，否则用stream，单卡无CUDA用sync
            if torch.cuda.device_count() >= 2:
                logger.info("Auto-selected ProcessValidator (multi-GPU)")
                validator = ProcessValidator(target_device='cuda:1', **kwargs)
                validator.set_validate_fn(validate_fn)
                if kwargs.get('model_config'):
                    validator.set_model_config(kwargs['model_config'])
            elif torch.cuda.is_available():
                logger.info("Auto-selected StreamValidator (single GPU)")
                validator = StreamValidator(device=device)
                validator.set_validate_fn(validate_fn)
            else:
                logger.info("Auto-selected SyncValidator (CPU)")
                validator = SyncValidator(validate_fn)
        
        else:
            raise ValueError(f"Unknown validator mode: {mode}")
        
        return validator