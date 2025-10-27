import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable
import logging
from collections import defaultdict
import time

class ModelMonitor:
    def __init__(self, model: nn.Module, log_interval: int = 100):
        self.model = model
        self.log_interval = log_interval
        self.step_count = 0
        
        # 存储监控数据
        self.metrics = defaultdict(list)
        self.hooks = []
        
        # 配置日志
        self.logger = logging.getLogger('ModelMonitor')
        
        # 注册前向/反向钩子
        self._register_hooks()
    
    def _register_hooks(self):
        """为各层注册前向和反向钩子"""
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # 只注册叶子模块
                # 前向钩子监控激活值
                hook_forward = module.register_forward_hook(
                    self._create_forward_hook(name)
                )
                # 反向钩子监控梯度
                hook_backward = module.register_full_backward_hook(
                    self._create_backward_hook(name)
                )
                self.hooks.extend([hook_forward, hook_backward])
    
    def _create_forward_hook(self, module_name: str) -> Callable:
        """创建前向钩子"""
        def forward_hook(module, input, output):
            if self.step_count % self.log_interval == 0:
                # 监控激活值
                if isinstance(output, torch.Tensor):
                    self._record_activation_metrics(module_name, output)
                # 监控权重
                self._record_weight_metrics(module_name, module)
        return forward_hook
    
    def _create_backward_hook(self, module_name: str) -> Callable:
        """创建反向钩子"""
        def backward_hook(module, grad_input, grad_output):
            if self.step_count % self.log_interval == 0:
                # 监控梯度
                if grad_output is not None and len(grad_output) > 0:
                    grad = grad_output[0]  # 通常取第一个梯度输出
                    if grad is not None:
                        self._record_gradient_metrics(module_name, grad)
                # 监控权重更新
                self._record_update_metrics(module_name, module)
        return backward_hook
    
    def _record_activation_metrics(self, name: str, tensor: torch.Tensor):
        """记录激活值相关指标"""
        with torch.no_grad():
            self.metrics[f'activation_norm/{name}'].append(tensor.norm().item())
            self.metrics[f'activation_mean/{name}'].append(tensor.mean().item())
            self.metrics[f'activation_std/{name}'].append(tensor.std().item())
            self.metrics[f'activation_max/{name}'].append(tensor.max().item())
    
    def _record_weight_metrics(self, name: str, module: nn.Module):
        """记录权重相关指标"""
        for param_name, param in module.named_parameters(recurse=False):
            if param.requires_grad and 'weight' in param_name:
                full_name = f"{name}.{param_name}"
                with torch.no_grad():
                    self.metrics[f'weight_norm/{full_name}'].append(param.norm().item())
                    self.metrics[f'weight_mean/{full_name}'].append(param.mean().item())
                    self.metrics[f'weight_std/{full_name}'].append(param.std().item())
    
    def _record_gradient_metrics(self, name: str, grad: torch.Tensor):
        """记录梯度相关指标"""
        with torch.no_grad():
            self.metrics[f'grad_norm/{name}'].append(grad.norm().item())
            self.metrics[f'grad_mean/{name}'].append(grad.mean().item())
            self.metrics[f'grad_std/{name}'].append(grad.std().item())
    
    def _record_update_metrics(self, name: str, module: nn.Module):
        """记录更新比率等指标"""
        for param_name, param in module.named_parameters(recurse=False):
            if param.requires_grad and param.grad is not None:
                full_name = f"{name}.{param_name}"
                with torch.no_grad():
                    weight_norm = param.norm().item()
                    grad_norm = param.grad.norm().item()
                    if weight_norm > 1e-8:  # 避免除零
                        update_ratio = grad_norm / weight_norm
                        self.metrics[f'update_ratio/{full_name}'].append(update_ratio)
    
    def step(self):
        """在训练循环的每个step后调用"""
        self.step_count += 1
    
    def get_metrics(self) -> Dict:
        """获取当前所有监控指标"""
        return dict(self.metrics)
    
    def clear_metrics(self):
        """清空监控指标"""
        self.metrics.clear()
    
    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()