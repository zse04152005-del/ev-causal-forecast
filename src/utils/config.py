"""极简配置系统：YAML + `_base_` 继承 + 命令行点号覆盖（不依赖 Hydra）。

用法：
    cfg = load_config("configs/default.yaml", ["model.D=16", "data.zones=40"])
    cfg.model.D  或  cfg["model"]["D"]
"""
from __future__ import annotations

import copy
import os
from typing import Any, Iterable

import yaml


class Cfg(dict):
    """支持属性访问的字典（嵌套字典自动转换）。"""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @staticmethod
    def wrap(obj: Any) -> Any:
        if isinstance(obj, dict) and not isinstance(obj, Cfg):
            return Cfg({k: Cfg.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [Cfg.wrap(v) for v in obj]
        return obj

    def to_dict(self) -> dict:
        def unwrap(o):
            if isinstance(o, dict):
                return {k: unwrap(v) for k, v in o.items()}
            if isinstance(o, list):
                return [unwrap(v) for v in o]
            return o
        return unwrap(self)


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    base = data.pop("_base_", None)
    if base:
        bases = base if isinstance(base, list) else [base]
        merged: dict = {}
        for b in bases:
            bp = b if os.path.isabs(b) else os.path.join(os.path.dirname(path), b)
            merged = deep_merge(merged, _read_yaml(bp))
        data = deep_merge(merged, data)
    return data


def apply_overrides(cfg: dict, overrides: Iterable[str] | None) -> dict:
    """覆盖项格式 a.b.c=value，value 按 YAML 解析（数字、列表、null 等）。"""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"覆盖项缺少 '='：{item}")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        if isinstance(value, str):                     # YAML 1.1 把 "5e-4" 当字符串；命令行里的科学计数法按数字处理
            try:
                value = float(value) if any(c in value.lower() for c in "e.") else value
            except ValueError:
                pass
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = value
    return cfg


def load_config(path: str, overrides: Iterable[str] | None = None) -> Cfg:
    return Cfg.wrap(apply_overrides(_read_yaml(path), overrides))


def save_config(cfg: dict, path: str) -> None:
    data = cfg.to_dict() if isinstance(cfg, Cfg) else cfg
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
