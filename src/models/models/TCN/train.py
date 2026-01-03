import polars as pl
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.dataset import Dataset, Subset
from tqdm import tqdm

from src.data.load_utils import TorchDataLoader, generate_torch_dataloader
from src.models.models.TCN.config import TCNTrainConfig
from src.models.models.TCN.model import TCN


def acquire_train_data(
    train_dataframe: pl.DataFrame,
    output_mask: Tensor,
    lookback: int,
    train_config: TCNTrainConfig,
) -> TorchDataLoader:
    dataloader: TorchDataLoader = generate_torch_dataloader(
        dataframe=train_dataframe,
        output_mask=output_mask,
        batch=train_config.batch_size,
        lookback=lookback,
        horizon=1,
        slide=1,
        shuffle=True,
    )

    return dataloader


def _unpack_batch(batch: object) -> tuple[Tensor, Tensor]:
    """Extract (x, y) tensors from a dataloader batch.

    Expected:
        x: (batch, lookback, in_features)
        y: (batch, horizon, out_features)

    Args:
        batch: Batch produced by the dataloader.

    Returns:
        Tuple (x, y) with x: (batch, lookback, in_features),
        y: (batch, horizon, out_features).

    Raises:
        AssertionError: If the batch does not match expected structure.
    """
    assert isinstance(batch, (tuple, list)), "Expected batch to be a tuple/list."
    assert len(batch) >= 2, "Expected batch to contain at least (x, y)."

    x_obj: object = batch[0]
    y_obj: object = batch[1]

    assert isinstance(x_obj, Tensor), "Expected x to be a torch.Tensor."
    assert isinstance(y_obj, Tensor), "Expected y to be a torch.Tensor."

    x: Tensor = x_obj
    y: Tensor = y_obj

    assert x.ndim == 3, f"Expected x shape (batch, lookback, in_features). {x.shape}"
    assert y.ndim == 3, (
        f"Expected y shape (batch, horizon, out_features). {y.ndim} {y.shape}"
    )

    assert int(x.shape[0]) == int(y.shape[0]), "Batch sizes must match."
    assert int(y.shape[1]) >= 1, "Horizon must be >= 1."
    assert int(y.shape[2]) >= 1, "out_features must be >= 1."

    return x, y


def train_TCN(
    model: TCN,
    dataloader: TorchDataLoader,
    train_config: TCNTrainConfig,
) -> None:
    """Train a TCN model (training only, no validation).

    This runs standard supervised training over the provided dataloader, showing
    a tqdm progress bar for each epoch and printing the epoch loss after each
    epoch.

    Notes:
        The torch TCN module outputs (batch, out_features) (horizon = 1 head).
        The dataloader yields y as (batch, horizon, out_features). We train the
        model to predict the last horizon step: y[:, -1, :].

    Args:
        model: TCN model to train.
        dataloader: Training dataloader yielding (x, y) batches where
            x: (batch, lookback, in_features), y: (batch, horizon, out_features).
        train_config: Training hyperparameters and runtime options.

    Returns:
        The trained model (same object, returned for convenience).

    Raises:
        AssertionError: If inputs are invalid or dataloader is empty.
    """
    # assert isinstance(model, TCN)
    # assert isinstance(train_config, TCNTrainConfig)
    # assert train_config.epochs >= 1
    # assert train_config.batch_size >= 1

    # steps_per_epoch: int = len(dataloader)  # type: ignore[arg-type]
    # assert steps_per_epoch >= 1, "Dataloader is empty."

    # device: torch.device = (
    #     train_config.device
    #     if isinstance(train_config.device, torch.device)
    #     else torch.device(device=train_config.device)
    # )

    # model.to(device=device)
    # model.train()

    # loss_fn: torch.nn.Module = torch.nn.SmoothL1Loss(beta=1.0)

    # optimizer: torch.optim.Optimizer = torch.optim.AdamW(
    #     params=model.parameters(),
    #     lr=train_config.lr,
    #     weight_decay=train_config.weight_decay,
    #     betas=(0.9, 0.99),
    # )

    # max_lr: float = (
    #     train_config.lr if train_config.max_lr is None else train_config.max_lr
    # )
    # scheduler: torch.optim.lr_scheduler.OneCycleLR = (
    #     torch.optim.lr_scheduler.OneCycleLR(
    #         optimizer=optimizer,
    #         max_lr=max_lr,
    #         epochs=train_config.epochs,
    #         steps_per_epoch=steps_per_epoch,
    #         pct_start=train_config.pct_start,
    #         div_factor=train_config.div_factor,
    #         final_div_factor=train_config.final_div_factor,
    #         anneal_strategy="cos",
    #     )
    # )

    # for epoch_idx in range(train_config.epochs):
    #     epoch_loss_sum: float = 0.0
    #     epoch_count: int = 0

    #     pbar: tqdm[pl.Any] = tqdm(
    #         dataloader,
    #         desc=f"Epoch {epoch_idx + 1}/{train_config.epochs}",
    #         leave=True,
    #     )

    #     for batch in pbar:
    #         x: Tensor
    #         y: Tensor
    #         x, y = _unpack_batch(batch=batch)
    #         x = x.to(device=device, non_blocking=True)
    #         y = y.to(device=device, non_blocking=True)

    #         # Train against the last horizon target (works for horizon=1 too).
    #         y_target: Tensor = y[:, -1, :]  # (batch, out_features)
    #         assert y_target.ndim == 2
    #         assert int(y_target.shape[0]) == int(x.shape[0])

    #         optimizer.zero_grad(set_to_none=True)

    #         y_hat: Tensor = model(x)  # (batch, out_features)
    #         assert y_hat.ndim == 2, (
    #             f"Expected y_hat (batch, out_features). {y_hat.shape}"
    #         )
    #         assert int(y_hat.shape[0]) == int(y_target.shape[0])
    #         assert int(y_hat.shape[1]) == int(y_target.shape[1])

    #         loss_t: Tensor = loss_fn(y_hat, y_target)
    #         loss_t.backward()

    #         if train_config.grad_clip_norm is not None:
    #             clip_grad_norm_(
    #                 parameters=model.parameters(),
    #                 max_norm=float(train_config.grad_clip_norm),
    #             )

    #         optimizer.step()
    #         scheduler.step()

    #         loss_value: float = float(loss_t.detach().item())
    #         epoch_loss_sum += loss_value
    #         epoch_count += 1

    #         pbar.set_postfix(
    #             loss=f"{loss_value:.6f}",
    #             lr=f"{scheduler.get_last_lr()[0]:.2e}",
    #             horizon=int(y.shape[1]),
    #         )

    #     epoch_loss_mean: float = epoch_loss_sum / float(max(1, epoch_count))
    #     print(f"SmoothL1Loss (huber) loss : {epoch_loss_mean:.6f}")

    # return None

    assert isinstance(model, TCN)
    assert isinstance(train_config, TCNTrainConfig)
    assert train_config.epochs >= 1
    assert train_config.batch_size >= 1

    assert isinstance(train_config.eval_proportion, float)
    assert 0.0 <= train_config.eval_proportion <= 1.0

    assert hasattr(dataloader, "dataset")
    dataset: Dataset[tuple[Tensor, ...]] = dataloader.dataset
    dataset_size: int = len(dataset)
    assert dataset_size >= 1, "Dataloader dataset is empty."

    val_size: int = int(round(number=dataset_size * train_config.eval_proportion))
    val_size = max(0, min(val_size, dataset_size))
    train_size: int = dataset_size - val_size
    assert train_size >= 1, "Training split is empty (eval_proportion too high)."

    train_indices: list[int] = list(range(0, train_size))
    val_indices: list[int] = list(range(train_size, dataset_size))

    train_subset: Subset[tuple[Tensor, ...]] = torch.utils.data.Subset(
        dataset=dataset, indices=train_indices
    )
    val_subset: Subset[tuple[Tensor, ...]] = torch.utils.data.Subset(
        dataset=dataset, indices=val_indices
    )

    num_workers: int = int(getattr(dataloader, "num_workers", 0))
    pin_memory: bool = bool(getattr(dataloader, "pin_memory", False))
    persistent_workers: bool = bool(getattr(dataloader, "persistent_workers", False))
    drop_last: bool = bool(getattr(dataloader, "drop_last", False))
    collate_fn = getattr(dataloader, "collate_fn", None)

    train_loader: TorchDataLoader = torch.utils.data.DataLoader(
        dataset=train_subset,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )

    val_loader: TorchDataLoader = torch.utils.data.DataLoader(
        dataset=val_subset,
        batch_size=train_config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )

    steps_per_epoch: int = len(train_loader)  # type: ignore[arg-type]
    assert steps_per_epoch >= 1, "Training dataloader is empty."

    device: torch.device = (
        train_config.device
        if isinstance(train_config.device, torch.device)
        else torch.device(device=train_config.device)
    )

    model.to(device=device)
    model.train()

    loss_fn: torch.nn.Module = torch.nn.SmoothL1Loss(beta=1.0)

    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
        betas=(0.9, 0.99),
    )

    max_lr: float = (
        train_config.lr if train_config.max_lr is None else train_config.max_lr
    )
    scheduler: torch.optim.lr_scheduler.OneCycleLR = (
        torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=max_lr,
            epochs=train_config.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=train_config.pct_start,
            div_factor=train_config.div_factor,
            final_div_factor=train_config.final_div_factor,
            anneal_strategy="cos",
        )
    )

    for epoch_idx in range(train_config.epochs):
        model.train()

        epoch_train_loss_sum: float = 0.0
        epoch_train_count: int = 0

        pbar: tqdm[object] = tqdm(
            iterable=train_loader,
            desc=f"Epoch {epoch_idx + 1}/{train_config.epochs}",
            leave=True,
        )

        for batch in pbar:
            x: Tensor
            y: Tensor
            x, y = _unpack_batch(batch=batch)
            x = x.to(device=device, non_blocking=True)
            y = y.to(device=device, non_blocking=True)

            y_target: Tensor = y[:, -1, :]  # (batch, out_features)
            assert y_target.ndim == 2
            assert int(y_target.shape[0]) == int(x.shape[0])

            optimizer.zero_grad(set_to_none=True)

            y_hat: Tensor = model(x)  # (batch, out_features)
            assert y_hat.ndim == 2, (
                f"Expected y_hat (batch, out_features). {tuple(y_hat.shape)}"
            )
            assert int(y_hat.shape[0]) == int(y_target.shape[0])
            assert int(y_hat.shape[1]) == int(y_target.shape[1])

            loss_t: Tensor = loss_fn(y_hat, y_target)
            loss_t.backward()

            if train_config.grad_clip_norm is not None:
                clip_grad_norm_(
                    parameters=model.parameters(),
                    max_norm=float(train_config.grad_clip_norm),
                )

            optimizer.step()
            scheduler.step()

            loss_value: float = float(loss_t.detach().item())
            epoch_train_loss_sum += loss_value
            epoch_train_count += 1

            pbar.set_postfix(
                loss=f"{loss_value:.6f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
                horizon=int(y.shape[1]),
            )

        epoch_train_loss_mean: float = epoch_train_loss_sum / float(
            max(1, epoch_train_count)
        )

        model.eval()
        epoch_test_loss_sum: float = 0.0
        epoch_test_count: int = 0

        if val_size == 0:
            epoch_test_loss_mean: float = float("nan")
        else:
            with torch.no_grad():
                for batch in val_loader:
                    x_val: Tensor
                    y_val: Tensor
                    x_val, y_val = _unpack_batch(batch=batch)
                    x_val = x_val.to(device=device, non_blocking=True)
                    y_val = y_val.to(device=device, non_blocking=True)

                    y_val_target: Tensor = y_val[:, -1, :]
                    y_val_hat: Tensor = model(x_val)

                    loss_val_t: Tensor = loss_fn(y_val_hat, y_val_target)
                    epoch_test_loss_sum += float(loss_val_t.detach().item())
                    epoch_test_count += 1

            epoch_test_loss_mean = epoch_test_loss_sum / float(max(1, epoch_test_count))

        print(
            f"[SmoothL1Loss (huber) loss] "
            f"train: {epoch_train_loss_mean:.6f}, "
            f"eval: {epoch_test_loss_mean:.6f}"
        )

    return None
