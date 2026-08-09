"""Command line entry point that caches frozen backbone features for every clip.

Decoding video and running a CNN is the expensive part of training a CNN plus
LSTM baseline. With a frozen backbone those features are constant, so computing
them once turns each later training epoch into cheap tensor loading.

Example:
    python -m src.features.build_feature_cache --clips-per-video 1
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.features.feature_extractor import build_backbone, encode_clips
from src.preprocessing.dataset import ClipBatchItem, ClipDataset
from src.preprocessing.splits import VideoSample, read_manifest
from src.utils.config import DatasetConfig, ModelConfig, TrainingConfig
from src.utils.device import resolve_device
from src.utils.seeding import set_seed

logger = logging.getLogger("avas.build_feature_cache")

INDEX_COLUMNS = (
    "split",
    "label",
    "label_index",
    "group",
    "video",
    "clip",
    "frame_indices",
    "feature_path",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="split manifest (default: <processed_root>/splits.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="feature cache directory (default: feature_cache_root from training config)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="subset of splits to process (default: every split in the manifest)",
    )
    parser.add_argument(
        "--clips-per-video",
        type=int,
        default=1,
        help="clips to cache per video; values above 1 require the random strategy",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="recompute features that are already cached",
    )
    return parser.parse_args(argv)


def feature_filename(video_path: Path, dataset_root: Path, clip_index: int) -> str:
    """Build a cache file name that stays unique across classes and clips."""
    try:
        relative = video_path.relative_to(dataset_root)
    except ValueError:
        relative = Path(video_path.name)
    stem = relative.with_suffix("").as_posix().replace("/", "__")
    return f"{stem}__clip{clip_index:02d}.npy"


def cache_split(
    split_name: str,
    samples: list[VideoSample],
    *,
    dataset_config: DatasetConfig,
    backbone: torch.nn.Module,
    device: torch.device,
    output_dir: Path,
    clips_per_video: int,
    batch_size: int,
    num_workers: int,
    overwrite: bool,
) -> list[dict[str, object]]:
    """Cache features for one split and return its index rows."""
    is_random = dataset_config.sampling.strategy == "random"
    dataset = ClipDataset.from_config(samples, dataset_config, train=is_random)
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    groups = {sample.path: sample.group for sample in samples}

    rows: list[dict[str, object]] = []
    for clip_index in range(clips_per_video):
        dataset.set_epoch(clip_index)
        loader: DataLoader[ClipBatchItem] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=list,
        )
        cached = 0
        skipped = 0
        for items in loader:
            targets = [
                split_dir / feature_filename(item.video_path, dataset_config.root, clip_index)
                for item in items
            ]
            pending = [
                (item, target)
                for item, target in zip(items, targets, strict=True)
                if overwrite or not target.is_file()
            ]
            if pending:
                clips = torch.stack([item.clip for item, _ in pending])
                features = encode_clips(backbone, clips, device)
                for (_, target), clip_features in zip(pending, features, strict=True):
                    np.save(target, clip_features.numpy().astype(np.float32))
                    cached += 1
            skipped += len(items) - len(pending)

            for item, target in zip(items, targets, strict=True):
                rows.append(
                    {
                        "split": split_name,
                        "label": item.label_name,
                        "label_index": item.label,
                        "group": groups.get(item.video_path, ""),
                        "video": item.video_path.as_posix(),
                        "clip": clip_index,
                        "frame_indices": " ".join(str(index) for index in item.frame_indices),
                        "feature_path": target.as_posix(),
                    }
                )

        logger.info("%s clip %d: cached %d, reused %d", split_name, clip_index, cached, skipped)

    return rows


def write_index(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(INDEX_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    dataset_config = DatasetConfig.from_yaml(args.dataset_config)
    model_config = ModelConfig.from_yaml(args.model_config)
    training_config = TrainingConfig.from_yaml(args.training_config)

    if args.clips_per_video < 1:
        raise SystemExit("--clips-per-video must be >= 1")
    if args.clips_per_video > 1 and dataset_config.sampling.strategy != "random":
        raise SystemExit(
            "caching several clips per video only makes sense with sampling.strategy "
            f"'random'; the dataset config uses {dataset_config.sampling.strategy!r}"
        )
    if not model_config.backbone.freeze:
        raise SystemExit(
            "feature caching requires backbone.freeze to be true, otherwise the cached "
            "features stop matching the backbone once training updates it"
        )

    set_seed(training_config.seed)
    device = resolve_device(args.device or training_config.device)
    backbone, feature_dim = build_backbone(model_config.backbone)
    backbone.to(device)

    manifest_path = args.manifest or dataset_config.processed_root / "splits.csv"
    splits = read_manifest(manifest_path, relative_to=dataset_config.root)
    selected = args.splits or list(splits)
    missing = [name for name in selected if name not in splits]
    if missing:
        raise SystemExit(f"manifest {manifest_path} has no splits named {missing}")

    output_dir = args.output or training_config.feature_cache_root
    num_workers = training_config.num_workers if args.num_workers is None else args.num_workers

    rows: list[dict[str, object]] = []
    for split_name in selected:
        rows.extend(
            cache_split(
                split_name,
                splits[split_name],
                dataset_config=dataset_config,
                backbone=backbone,
                device=device,
                output_dir=output_dir,
                clips_per_video=args.clips_per_video,
                batch_size=args.batch_size,
                num_workers=num_workers,
                overwrite=args.overwrite,
            )
        )

    index_path = output_dir / "index.csv"
    write_index(rows, index_path)
    logger.info(
        "cached %d clips with feature dimension %d; index written to %s",
        len(rows),
        feature_dim,
        index_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
