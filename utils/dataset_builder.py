from pathlib import Path
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from PIL import Image
from torch.utils.data import Dataset


# 支持的图像后缀集合，用于过滤无关文件
IMAGE_EXTENSIONS: Sequence[str] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".JPEG",
    ".PNG",
    ".jepg",
    ".tiff",
    ".tif",
    ".TIF",
    ".webp",
    ".gif",
    ".bmp",
)


class ImageSample:
    """单个样本索引信息。

    Attributes:
        path: 图像文件路径。
        label: 标签，真实图像为 0，伪造图像为 1。
        source_model: 第一级子目录名称（模型名称或 unknown）。
    """

    def __init__(self, path: Path, label: int, source_model: str) -> None:
        self.path = path
        self.label = int(label)
        self.source_model = source_model

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "label": self.label,
            "source_model": self.source_model,
        }


def build_image_index(
    root_dir: Union[str, Path],
    source_model_override: Optional[str] = None,
) -> List[ImageSample]:
    """扫描数据集数据根目录，构建图像样本索引。可以是单一模型目录或多个模型子目录。

    目录结构约定：
        root_dir/
            ├── <model_name_1>/
            │   ├── 0real/
            │   └── 1fake/
            ├── <model_name_2>/
            │   ├── 0real/
            │   └── 1fake/
            └── 0real/1fake/  # 可选，直接放在 root_dir 下，视为单一模型来源

    可能只存在 0real 或 1fake 子目录之一，也可能两者都存在。

    Args:
        root_dir: 数据集根目录路径。

    Returns:
        ImageSample 列表，每个元素对应一张图像及其标签、来源模型。
    """

    root_path = Path(root_dir)
    if not root_path.exists() or not root_path.is_dir():
        raise ValueError(f"Dataset root_dir does not exist or is not a directory: {root_dir}")

    samples: List[ImageSample] = []

    # 若 root_dir 直接包含 0real/1fake，则认为 root_dir 是单一模型目录
    direct_real = root_path / "0real"
    direct_fake = root_path / "1fake"
    if direct_real.is_dir() or direct_fake.is_dir():
        model_name = source_model_override or root_path.name
        for sub_name, label in ("0real", 0), ("1fake", 1):
            sub_dir = root_path / sub_name
            if not sub_dir.is_dir():
                continue
            for path in sub_dir.rglob("*"):
                if not path.is_file():
                    continue
                if path.suffix not in IMAGE_EXTENSIONS:
                    continue
                samples.append(ImageSample(path=path, label=label, source_model=model_name))
        return samples

    #  若root_dir 下有多个模型子目录，每个子目录作为一个来源模型
    for model_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
        model_name = source_model_override or model_dir.name

        # 真实图像和伪造图像子目录
        for sub_name, label in ("0real", 0), ("1fake", 1):
            sub_dir = model_dir / sub_name
            if not sub_dir.is_dir():
                continue

            # 遍历当前子目录下所有图像文件（允许有更深层级）
            for path in sub_dir.rglob("*"):
                if not path.is_file():
                    continue
                if path.suffix not in IMAGE_EXTENSIONS:
                    continue
                samples.append(ImageSample(path=path, label=label, source_model=model_name))

    return samples

def build_datasets_index(root_dir:Union[str,Path])->List[ImageSample]:
    """
    数据目录可能为：
    root/
    ├── modelA/
    │   ├── 0real/
    │   └── 1fake/
    ├── group1/
    │   ├── modelB/
    │   │   ├── 0real/
    │   │   └── 1fake/
    │   └── modelC/
    │       ├── 0real/
    """
    root_path = Path(root_dir)
    if not root_path.exists() or not root_path.is_dir():
        raise ValueError(f"Dataset root_dir does not exist or is not a directory: {root_dir}")

    samples: List[ImageSample] = []
    if (root_path/"0real").is_dir() and (root_path/"1fake").is_dir():
        samples.extend(build_image_index(root_path))
        return samples

    for dataset_dir in (p for p in root_path.iterdir() if p.is_dir()):
        samples.extend(build_image_index(dataset_dir))
    return samples

def split_samples(
    samples: Sequence[ImageSample],
    val_ratio: float = 0.05,
    seed: int = 42,
    stratify: bool = True,
) -> Tuple[List[ImageSample], List[ImageSample]]:
    """将样本划分为训练集与验证集。

    Args:
        samples: 样本列表。
        val_ratio: 验证集比例，范围 (0, 1)。
        seed: 随机种子，保证可复现。
        stratify: 是否按标签分层切分。

    Returns:
        (train_samples, val_samples)
    """

    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be within (0, 1)")

    rng = random.Random(seed)

    if not stratify:
        indices = list(range(len(samples)))
        rng.shuffle(indices)
        split_idx = int(len(indices) * (1.0 - val_ratio))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        return ([samples[i] for i in train_indices], [samples[i] for i in val_indices])

    # 分层切分：按 label 分组后各自划分
    label_to_samples: Dict[int, List[ImageSample]] = {}
    for sample in samples:
        label_to_samples.setdefault(sample.label, []).append(sample)

    train_samples: List[ImageSample] = []
    val_samples: List[ImageSample] = []
    for label, group in label_to_samples.items():
        group_indices = list(range(len(group)))
        rng.shuffle(group_indices)
        split_idx = int(len(group_indices) * (1.0 - val_ratio))
        train_samples.extend(group[i] for i in group_indices[:split_idx])
        val_samples.extend(group[i] for i in group_indices[split_idx:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


class ForgeryImageDataset(Dataset):
    """图像伪造检测数据集，适配 Hugging Face Trainer。

    - 目录结构与 :func:`build_image_index` 一致。
    - 支持注入 Hugging Face AutoImageProcessor / FeatureExtractor 作为 ``processor``，
      或普通的 torchvision / 自定义 ``transform``。
    - 输出字典中至少包含：
        * ``labels``: 标量标签 (0=真实, 1=伪造)
        * ``pixel_values`` 或自定义图像字段名（通过 ``image_field`` 指定）
        * 可选 ``source_model`` 字段，便于事后分析。
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        processor: Optional[Callable[..., Dict[str, Any]]] = None,
        transform: Optional[Callable[[Image.Image], Any]] = None,
        image_field: str = "pixel_values",
        return_source_model: bool = True,
        samples: Optional[Sequence[ImageSample]] = None,
    ) -> None:
        if processor is not None and transform is not None:
            raise ValueError("Only one of `processor` or `transform` should be provided.")

        self.root_dir = Path(root_dir)
        self.samples: List[ImageSample] = (
            list(samples) if samples is not None else build_image_index(self.root_dir)
        )
        self.processor = processor
        self.transform = transform
        self.image_field = image_field
        self.return_source_model = return_source_model

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # type: ignore[override]
        sample = self.samples[idx]

        try:
            # 统一图像读取为 RGB
            image = Image.open(sample.path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {sample.path}: {e}")
            # 随机取一张图片
            new_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(new_idx)

        item: Dict[str, Any]

        if self.processor is not None:
            # Hugging Face AutoImageProcessor 通常返回 batched 字典，这里取 batch_size=1 的第 0 个元素。
            processed = self.processor(images=image, return_tensors="pt")
            item = {}
            for key, value in processed.items():
                # 假设返回形状为 (1, ...)，取第 0 个样本
                try:
                    item[key] = value[0]
                except Exception:  # 保底：如果不能索引则原样返回
                    item[key] = value

            # 如果希望使用非默认键名，如 "inputs"，可通过 image_field 参数调整
            if (
                self.image_field != "pixel_values"
                and "pixel_values" in item
                and self.image_field not in item
            ):
                item[self.image_field] = item["pixel_values"]

        elif self.transform is not None:
            image_tensor = self.transform(image)
            item = {self.image_field: image_tensor}
        else:
            # 不做任何变换，直接返回 PIL.Image，便于在外部自定义处理逻辑
            item = {self.image_field: image}

        item["labels"] = sample.label

        if self.return_source_model:
            item["source_model"] = sample.source_model

        return item


def build_train_val_datasets(
    root_dir: Union[str, Path],
    val_ratio: float = 0.05,
    seed: int = 42,
    stratify: bool = True,
    processor: Optional[Callable[..., Dict[str, Any]]] = None,
    transform: Optional[Callable[[Image.Image], Any]] = None,
    image_field: str = "pixel_values",
    return_source_model: bool = True,
) -> Tuple[ForgeryImageDataset, ForgeryImageDataset]:
    """构建训练集与验证集 Dataset。

    Args:
        root_dir: 数据集根目录。
        val_ratio: 验证集比例。
        seed: 随机种子。
        stratify: 是否按标签分层切分。
        processor: Hugging Face 图像处理器。
        transform: 自定义变换。
        image_field: 图像字段名。
        return_source_model: 是否返回 source_model。

    Returns:
        (train_dataset, val_dataset)
    """

    samples = build_image_index(root_dir)
    train_samples, val_samples = split_samples(
        samples=samples, val_ratio=val_ratio, seed=seed, stratify=stratify
    )
    train_dataset = ForgeryImageDataset(
        root_dir=root_dir,
        processor=processor,
        transform=transform,
        image_field=image_field,
        return_source_model=return_source_model,
        samples=train_samples,
    )
    val_dataset = ForgeryImageDataset(
        root_dir=root_dir,
        processor=processor,
        transform=transform,
        image_field=image_field,
        return_source_model=return_source_model,
        samples=val_samples,
    )
    return train_dataset, val_dataset

if __name__ == "__main__":
    root_dir = "/home/chengxiaozhen/data/Benchmark"
    root_path = Path(root_dir)
    samples = build_datasets_index(root_path)
    print(f"Total samples found: {len(samples)}")

