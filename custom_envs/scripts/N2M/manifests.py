"""JSON/JSONL helpers with numpy-safe serialization.

JSON 和 JSONL 的区别：

普通 JSON 通常保存一个完整对象：
{
  "schema_version": 1,
  "num_observations": 100
}
适合：
- 配置；
- 汇总；
- 数据集统计；
- 单次推理结果。

JSONL 是“一行一个 JSON 对象”：
{"candidate_id": "scene_0_edge_0_0001", "success": true}
{"candidate_id": "scene_0_edge_0_0002", "success": false}
{"candidate_id": "scene_0_edge_0_0003", "success": true}
适合：
- 候选列表；
- 多次抓取 rollout；
- 观察记录；
- O–M 配对；
- 可以持续追加的实验日志。
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np


def _json_default(value: Any):
    if is_dataclass(value):   # 把各种dataclass转换成dict
        return asdict(value)
    if isinstance(value, np.ndarray):   # 把np.array转换为list
        return value.tolist()
    if isinstance(value, np.generic):   # 把np标量转换为普通的int/float/str等
        return value.item()
    if isinstance(value, Path):   # 把路径转换为字符串
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_json(path: Path, value: Any) -> None:   # 保存一个完整 JSON 文件
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)   # 创建父目录
    temporary = path.with_name(path.name + ".tmp")   # 使用临时文件xxx.json.tmp
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)   # 替换为最终文件


def read_json(path: Path) -> Any:   # 读取整个UTF-8文件 -> json.loads()解析 -> 返回普通python对象
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[Any]) -> None:   # 一次性写入一整份 JSONL 文件
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(row, ensure_ascii=False, default=_json_default) + "\n"
        for row in rows
    )   # 每个row转换为单行json然后换行
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, row: Any) -> None:   # 向 JSONL 文件末尾追加一条记录
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
        stream.flush()


def read_jsonl(path: Path) -> List[dict]:   # 读取整个 JSONL 文件
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
