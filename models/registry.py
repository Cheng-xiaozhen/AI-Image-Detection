from typing import Callable, Dict, List, TypeVar

import torch.nn as nn

ModelBuilder = Callable[..., nn.Module]
BuilderT = TypeVar("BuilderT", bound=Callable[..., nn.Module])

_MODEL_REGISTRY: Dict[str, ModelBuilder] = {}


def register_model(name: str) -> Callable[[BuilderT], BuilderT]:
    """注册模型构建器。

    Args:
        name: 模型名称（唯一）。

    Returns:
        装饰器函数，用于注册模型构建器。
    """

    def decorator(builder: BuilderT) -> BuilderT:
        if name in _MODEL_REGISTRY:
            raise KeyError(f"Model '{name}' is already registered")
        _MODEL_REGISTRY[name] = builder
        return builder

    return decorator


def create_model(name: str, **kwargs) -> nn.Module:
    """根据名称创建模型实例。"""

    if name not in _MODEL_REGISTRY:
        available = ", ".join(list_models())
        raise KeyError(f"Unknown model '{name}'. Available models: {available}")
    return _MODEL_REGISTRY[name](**kwargs)


def list_models() -> List[str]:
    """列出已注册的模型名称。"""

    return sorted(_MODEL_REGISTRY.keys())