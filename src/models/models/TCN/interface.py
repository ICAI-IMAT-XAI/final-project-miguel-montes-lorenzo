import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, override

import polars as pl
import torch
from torch import Tensor

from src.data.load_utils import TorchDataLoader
from src.models.abstract import ForecastingModel
from src.models.models.TCN.config import TCNModelConfig, TCNTrainConfig
from src.models.models.TCN.model import TCN as TCN_model
from src.models.models.TCN.train import acquire_train_data, train_TCN
from src.models.models.TCN.utils import receptive_field


class TCN(ForecastingModel):
    def __init__(
        self,
        feature_size: int,
        output_mask: Tensor,
        temporal_lookback: int,
        temporal_horizon: int,
    ) -> None:

        super().__init__(
            feature_size=feature_size,
            output_mask=output_mask,
            temporal_lookback=temporal_lookback,
            temporal_horizon=temporal_horizon,
        )

        self._model: TCN_model
        self._model_config: TCNModelConfig
        self._train_config: TCNTrainConfig

        return None

    @override
    def _init_model(
        self,
        feature_size: int,
        output_mask: Tensor,
        temporal_lookback: int,
        temporal_horizon: int,
    ) -> None:
        """Initialize the model parameters."""

        self._model_config = TCNModelConfig(
            in_features=feature_size,
            out_features=int(output_mask.to(dtype=int).sum().item()),
        )

        self._train_config = TCNTrainConfig()

        rf: int = receptive_field(
            kernel_size=self._model_config.kernel_size,
            dilations=self._model_config.dilations,
            blocks_per_dilation=self._model_config.blocks_per_dilation,
        )

        assert temporal_lookback >= int(rf * 0.8), (
            f"lookback too short (make it >= {int(rf * 0.8)}"
        )
        assert temporal_horizon == 1

        self._model = TCN_model(cfg=self._model_config)

        return None

    @override
    def fit(self, train_data: pl.DataFrame) -> None:
        """Train the model on the given data."""

        dataloader: TorchDataLoader = acquire_train_data(
            train_dataframe=train_data,
            output_mask=self._output_mask,
            lookback=self._temporal_lookback,
            train_config=self._train_config,
        )

        train_TCN(
            model=self._model, dataloader=dataloader, train_config=self._train_config
        )

        return None

    @override
    def _predict(self, x: Tensor) -> Tensor:
        """Predict a horizon=1 window.

        Args:
            x: Input tensor with shape (batch, lookback, feature_size).

        Returns:
            Prediction tensor with shape (batch, temporal_horizon=1, out_features).
        """
        assert isinstance(x, Tensor)
        assert x.ndim == 3, "x must have shape (batch, lookback, feature_size)"
        assert int(x.shape[1]) == self._temporal_lookback
        assert int(x.shape[2]) == self._feature_size

        assert x.dtype in (torch.float16, torch.float32, torch.float64), (
            f"Expected float tensor, got {x.dtype}."
        )

        self._model.eval()
        y_hat: Tensor = self._model(x)  # (batch, out_features)
        assert y_hat.ndim == 2
        assert int(y_hat.shape[0]) == int(x.shape[0])
        assert int(y_hat.shape[1]) == int(self._output_mask.to(dtype=int).sum().item())

        y: Tensor = y_hat.unsqueeze(dim=1)  # (batch, 1, out_features)
        return y

    @override
    def save(self, store_path: Path) -> None:
        """Save the model weights and configuration.

        Layout:
            <store_path>/
                model.pt
                config.json

        Args:
            store_path: Directory where the model will be stored.
        """
        assert isinstance(store_path, Path)

        if store_path.exists():
            assert store_path.is_dir(), "store_path must be a directory."
        else:
            store_path.mkdir(parents=True, exist_ok=True)

        weights_path: Path = store_path / "model.pt"
        config_path: Path = store_path / "config.json"

        output_mask_cpu: Tensor = self._output_mask.detach().cpu()
        assert output_mask_cpu.dtype is torch.bool
        assert output_mask_cpu.ndim == 1

        payload_torch: dict[str, Any] = {
            "feature_size": self._feature_size,
            "temporal_lookback": self._temporal_lookback,
            "temporal_horizon": self._temporal_horizon,
            "output_mask": output_mask_cpu,  # tensor is fine in torch.save
            "config": self._get_config(),
        }

        payload_json: dict[str, Any] = {
            "feature_size": int(self._feature_size),
            "temporal_lookback": int(self._temporal_lookback),
            "temporal_horizon": int(self._temporal_horizon),
            "output_mask": [bool(v) for v in output_mask_cpu.tolist()],
            "config": self._get_config(),
        }

        torch.save(
            obj={"state_dict": self._model.state_dict(), "meta": payload_torch},
            f=weights_path,
        )

        with config_path.open(mode="w", encoding="utf-8") as f:
            json.dump(obj=payload_json, fp=f, indent=2, sort_keys=True)

        return None

    @override
    def load(self, store_path: Path) -> None:
        """Load the model weights and configuration.

        Expected layout inside ``store_path``:
            model.pt
            config.json

        Args:
            store_path: Directory containing the saved model files.
        """
        assert isinstance(store_path, Path)
        assert store_path.exists(), "store_path does not exist."
        assert store_path.is_dir(), "store_path must be a directory."

        weights_path: Path = store_path / "model.pt"
        config_path: Path = store_path / "config.json"

        assert weights_path.exists() and weights_path.is_file(), (
            f"Missing weights file at {weights_path}."
        )
        assert config_path.exists() and config_path.is_file(), (
            f"Missing config file at {config_path}."
        )

        with config_path.open(mode="r", encoding="utf-8") as f:
            payload: dict[str, Any] = json.load(fp=f)

        assert isinstance(payload, dict)
        assert "feature_size" in payload
        assert "temporal_lookback" in payload
        assert "temporal_horizon" in payload
        assert "output_mask" in payload
        assert "config" in payload

        feature_size: int = int(payload["feature_size"])
        temporal_lookback: int = int(payload["temporal_lookback"])
        temporal_horizon: int = int(payload["temporal_horizon"])
        config: dict[str, Any] = payload["config"]

        assert feature_size == self._feature_size
        assert temporal_lookback == self._temporal_lookback
        assert temporal_horizon == self._temporal_horizon

        output_mask_list: list[bool] = payload["output_mask"]
        assert isinstance(output_mask_list, list)
        assert len(output_mask_list) == self._feature_size
        assert all(isinstance(v, bool) for v in output_mask_list)

        self._output_mask = torch.tensor(data=output_mask_list, dtype=torch.bool)

        checkpoint: dict[str, Any] = torch.load(
            f=weights_path,
            map_location="cpu",
        )
        assert isinstance(checkpoint, dict)
        assert "state_dict" in checkpoint

        self._set_config(config=config)

        state_dict: dict[str, Tensor] = checkpoint["state_dict"]
        self._model.load_state_dict(state_dict=state_dict)
        self._model.eval()

        return None

    def _get_config(self) -> dict[str, Any]:
        model_config: dict[str, Any] = asdict(obj=self._model_config)
        train_config: dict[str, Any] = asdict(obj=self._train_config)
        config: dict[str, Any] = {"model": model_config, "train": train_config}

        return config

    def _set_config(self, config: dict[str, Any]) -> None:
        """Set model/train configuration with strict validation.

        Accepts JSON-loaded structures (lists) for tuple fields and converts them to
        the dataclass' expected container types.
        """
        assert isinstance(config, dict)

        expected_top_keys: set[str] = {"model", "train"}
        assert set(config.keys()) == expected_top_keys

        model_cfg_in_raw: dict[str, Any] = config["model"]
        train_cfg_in_raw: dict[str, Any] = config["train"]
        assert isinstance(model_cfg_in_raw, dict)
        assert isinstance(train_cfg_in_raw, dict)

        model_cfg_current: dict[str, Any] = asdict(obj=self._model_config)
        train_cfg_current: dict[str, Any] = asdict(obj=self._train_config)

        assert set(model_cfg_in_raw.keys()) == set(model_cfg_current.keys()), (
            "Model config keys do not match TCNModelConfig."
        )
        assert set(train_cfg_in_raw.keys()) == set(train_cfg_current.keys()), (
            "Train config keys do not match TCNTrainConfig."
        )

        def _coerce_container(value: Any, expected: Any) -> Any:
            """Coerce JSON containers to expected dataclass container types."""
            # tuple fields become list after JSON roundtrip: convert list -> tuple
            if isinstance(expected, tuple) and isinstance(value, list):
                return tuple(value)
            # if you ever have list fields but you stored tuple, allow tuple -> list
            if isinstance(expected, list) and isinstance(value, tuple):
                return list(value)
            return value

        model_cfg_in: dict[str, Any] = {}
        for k, v in model_cfg_in_raw.items():
            model_cfg_in[k] = _coerce_container(value=v, expected=model_cfg_current[k])

        train_cfg_in: dict[str, Any] = {}
        for k, v in train_cfg_in_raw.items():
            train_cfg_in[k] = _coerce_container(value=v, expected=train_cfg_current[k])

        # Type checks (after coercion): ensure incoming values match current types.
        for k, v in model_cfg_in.items():
            cur_v: Any = model_cfg_current[k]
            assert isinstance(v, type(cur_v)), (
                f"Model config field '{k}' must be {type(cur_v)}, got {type(v)}."
            )

        for k, v in train_cfg_in.items():
            cur_v: Any = train_cfg_current[k]
            assert isinstance(v, type(cur_v)), (
                f"Train config field '{k}' must be {type(cur_v)}, got {type(v)}."
            )

        # Extra strictness: validate tuple/list element types match what current has.
        for k, v in model_cfg_in.items():
            cur_v = model_cfg_current[k]
            if isinstance(cur_v, tuple):
                assert isinstance(v, tuple)
                assert all(
                    isinstance(a, type(b)) for a, b in zip(v, cur_v, strict=False)
                )

        for k, v in train_cfg_in.items():
            cur_v = train_cfg_current[k]
            if isinstance(cur_v, tuple):
                assert isinstance(v, tuple)
                assert all(
                    isinstance(a, type(b)) for a, b in zip(v, cur_v, strict=False)
                )

        # Apply train config changes (safe, no need to rebuild the torch module).
        self._train_config = TCNTrainConfig(**train_cfg_in)

        # Apply model config; if changed, rebuild model.
        new_model_cfg: TCNModelConfig = TCNModelConfig(**model_cfg_in)
        model_changed: bool = new_model_cfg != self._model_config
        self._model_config = new_model_cfg

        if model_changed:
            self._model = TCN_model(cfg=self._model_config)

        return None
