"""
CANN (Combined Actuarial Neural Network) transfer learning via PyTorch.

Pre-trains a CANN on source data, then fine-tunes on the target segment.
The CANN architecture combines a GLM (skip connection) with a neural network,
giving an interpretable GLM baseline with neural residual correction.

Reference architecture: Schelldorfer and Wuthrich (2019). PyTorch implementation.

Fine-tuning strategies:
  - ``'all'``: Unfreeze all layers for target fine-tuning.
  - ``'head_only'``: Freeze the feature extraction layers, only update the
    final linear layer. Best for very thin segments (< 100 observations).
  - ``'progressive'``: Start with head-only, then progressively unfreeze
    deeper layers. More robust but slower.

PyTorch is an optional dependency. If not installed, import raises ImportError
with a clear install instruction.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


FinetuneStrategy = Literal["all", "head_only", "progressive"]


def _check_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError as exc:
        raise ImportError(
            "CANNTransfer requires PyTorch. Install it with: "
            "pip install insurance-transfer[torch]"
        ) from exc


class _CANNModule:
    """Thin wrapper so torch classes are only defined when torch is available."""

    @staticmethod
    def build(torch, nn, n_features: int, hidden_sizes: List[int], dropout: float):
        """Build a CANN module: GLM skip + MLP body."""

        class CANNNet(nn.Module):
            def __init__(self):
                super().__init__()
                # GLM skip connection: direct linear path
                self.glm_layer = nn.Linear(n_features, 1, bias=True)

                # MLP body
                layers = []
                in_size = n_features
                for h in hidden_sizes:
                    layers.append(nn.Linear(in_size, h))
                    layers.append(nn.ReLU())
                    if dropout > 0.0:
                        layers.append(nn.Dropout(dropout))
                    in_size = h
                layers.append(nn.Linear(in_size, 1, bias=False))
                self.mlp = nn.Sequential(*layers)

            def forward(self, x):
                glm_out = self.glm_layer(x)
                mlp_out = self.mlp(x)
                # CANN: combine GLM and MLP in log space
                return glm_out + mlp_out

            def freeze_body(self):
                for param in self.mlp.parameters():
                    param.requires_grad_(False)
                for param in self.glm_layer.parameters():
                    param.requires_grad_(False)

            def unfreeze_body(self):
                for param in self.mlp.parameters():
                    param.requires_grad_(True)
                for param in self.glm_layer.parameters():
                    param.requires_grad_(True)

            def freeze_except_head(self):
                """Freeze MLP body, keep only final layer and GLM trainable."""
                for param in self.mlp.parameters():
                    param.requires_grad_(False)
                # Unfreeze last MLP layer (the head)
                for param in list(self.mlp.children())[-1].parameters():
                    param.requires_grad_(True)
                for param in self.glm_layer.parameters():
                    param.requires_grad_(True)

        return CANNNet()


def _poisson_deviance_loss(torch, nn):
    """Return a Poisson deviance loss function."""

    class PoissonDevianceLoss(nn.Module):
        def forward(self, log_mu: "torch.Tensor", y: "torch.Tensor") -> "torch.Tensor":
            mu = torch.exp(torch.clamp(log_mu, -30, 30))
            # D = 2 * sum(y * log(y/mu) - (y - mu))
            # Use numerically stable form: mean(-y * log_mu + mu)
            return torch.mean(-y * log_mu + mu)

    return PoissonDevianceLoss()


class CANNTransfer(BaseEstimator, RegressorMixin):
    """CANN pre-train / fine-tune transfer for thin insurance segments.

    Pre-trains a Combined Actuarial Neural Network on source data, then
    fine-tunes on the target segment using a frozen-body or progressive
    unfreezing strategy.

    The CANN architecture (Schelldorfer & Wuthrich 2019) has a direct GLM
    skip connection plus an MLP body. This guarantees the model is at least
    as good as a GLM at target-only fitting, while giving the MLP room to
    learn non-linearities from the richer source data.

    Parameters
    ----------
    hidden_sizes:
        MLP hidden layer sizes. [32, 16] works well for most pricing tasks.
    dropout:
        Dropout rate applied after each hidden layer during training.
    finetune_strategy:
        How to fine-tune on target:
        ``'all'``: unfreeze everything.
        ``'head_only'``: freeze MLP body, update final layer + GLM only.
        ``'progressive'``: head-only for first half of epochs, then all.
    pretrain_epochs:
        Number of epochs for source pre-training.
    finetune_epochs:
        Number of epochs for target fine-tuning.
    learning_rate:
        Learning rate for both pre-training and fine-tuning.
    batch_size:
        Mini-batch size.
    scale_features:
        Whether to standardise features.
    random_state:
        Seed for reproducibility.

    Attributes
    ----------
    net_:
        Fitted PyTorch module.
    scaler_:
        Fitted feature scaler (or None).
    pretrain_losses_:
        Training loss per epoch during pre-training.
    finetune_losses_:
        Training loss per epoch during fine-tuning.

    Examples
    --------
    >>> import numpy as np
    >>> try:
    ...     from insurance_transfer import CANNTransfer
    ...     rng = np.random.default_rng(0)
    ...     X_src = rng.standard_normal((300, 5))
    ...     y_src = rng.poisson(np.exp(0.3 * X_src[:, 0]))
    ...     X_tgt = rng.standard_normal((60, 5)) + 0.5
    ...     y_tgt = rng.poisson(np.exp(0.3 * X_tgt[:, 0]))
    ...     model = CANNTransfer(hidden_sizes=[16, 8], pretrain_epochs=5,
    ...                          finetune_epochs=5, random_state=42)
    ...     model.fit_source(X_src, y_src)
    ...     model.fit(X_tgt, y_tgt)
    ...     preds = model.predict(X_tgt)
    ...     assert preds.shape == (60,)
    ... except ImportError:
    ...     pass
    """

    def __init__(
        self,
        hidden_sizes: List[int] = None,
        dropout: float = 0.0,
        finetune_strategy: FinetuneStrategy = "head_only",
        pretrain_epochs: int = 50,
        finetune_epochs: int = 30,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        scale_features: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        self.hidden_sizes = hidden_sizes if hidden_sizes is not None else [32, 16]
        self.dropout = dropout
        self.finetune_strategy = finetune_strategy
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.scale_features = scale_features
        self.random_state = random_state

    def _train_loop(
        self,
        torch,
        nn,
        net,
        X_t: "torch.Tensor",
        y_t: "torch.Tensor",
        log_exp_t: "torch.Tensor",
        n_epochs: int,
        lr: float,
    ) -> List[float]:
        loss_fn = _poisson_deviance_loss(torch, nn)
        optimiser = torch.optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()), lr=lr
        )
        n = X_t.shape[0]
        losses = []

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        for epoch in range(n_epochs):
            net.train()
            perm = torch.randperm(n)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                Xb = X_t[idx]
                yb = y_t[idx]
                log_exp_b = log_exp_t[idx]

                optimiser.zero_grad()
                log_mu = net(Xb).squeeze(-1) + log_exp_b
                loss = loss_fn(log_mu, yb)
                loss.backward()
                optimiser.step()

                epoch_loss += loss.item()
                n_batches += 1

            losses.append(epoch_loss / max(n_batches, 1))

        return losses

    def fit_source(
        self,
        X_source: NDArray,
        y_source: NDArray,
        exposure_source: Optional[NDArray] = None,
    ) -> "CANNTransfer":
        """Pre-train the CANN on source data.

        Must be called before ``fit()`` (fine-tuning step). If you skip this,
        ``fit()`` will train from scratch (no transfer).

        Parameters
        ----------
        X_source:
            Source feature matrix.
        y_source:
            Source response (claim counts for Poisson).
        exposure_source:
            Source exposure. Defaults to ones.

        Returns
        -------
        self
        """
        torch, nn = _check_torch()

        X_source = np.asarray(X_source, dtype=np.float64)
        y_source = np.asarray(y_source, dtype=np.float64)
        n_src = X_source.shape[0]

        if exposure_source is None:
            exposure_source = np.ones(n_src)
        exposure_source = np.asarray(exposure_source, dtype=np.float64)
        log_exp_src = np.log(np.maximum(exposure_source, 1e-10))

        if self.scale_features:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X_source)
        else:
            self.scaler_ = None
            X_scaled = X_source

        n_features = X_scaled.shape[1]
        self.net_ = _CANNModule.build(torch, nn, n_features, self.hidden_sizes, self.dropout)
        self.net_ = self.net_.double()

        X_t = torch.tensor(X_scaled, dtype=torch.float64)
        y_t = torch.tensor(y_source, dtype=torch.float64)
        log_exp_t = torch.tensor(log_exp_src, dtype=torch.float64)

        self.pretrain_losses_ = self._train_loop(
            torch, nn, self.net_, X_t, y_t, log_exp_t,
            self.pretrain_epochs, self.learning_rate
        )

        self._source_fitted = True
        return self

    def fit(
        self,
        X: NDArray,
        y: NDArray,
        exposure: Optional[NDArray] = None,
    ) -> "CANNTransfer":
        """Fine-tune on target data.

        Call ``fit_source()`` first. If ``fit_source()`` has not been called,
        the model trains from scratch.

        Parameters
        ----------
        X:
            Target feature matrix.
        y:
            Target response.
        exposure:
            Target exposure. Defaults to ones.

        Returns
        -------
        self
        """
        torch, nn = _check_torch()

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = X.shape[0]

        if exposure is None:
            exposure = np.ones(n)
        exposure = np.asarray(exposure, dtype=np.float64)
        log_exp = np.log(np.maximum(exposure, 1e-10))

        # If no pre-training, initialise from scratch
        if not hasattr(self, "_source_fitted"):
            if self.scale_features:
                self.scaler_ = StandardScaler()
                X_scaled = self.scaler_.fit_transform(X)
            else:
                self.scaler_ = None
                X_scaled = X
            n_features = X_scaled.shape[1]
            self.net_ = _CANNModule.build(
                torch, nn, n_features, self.hidden_sizes, self.dropout
            )
            self.net_ = self.net_.double()
            self.pretrain_losses_ = []
        else:
            if self.scaler_ is not None:
                X_scaled = self.scaler_.transform(X)
            else:
                X_scaled = X

        X_t = torch.tensor(X_scaled, dtype=torch.float64)
        y_t = torch.tensor(y, dtype=torch.float64)
        log_exp_t = torch.tensor(log_exp, dtype=torch.float64)

        # Apply fine-tuning strategy
        if self.finetune_strategy == "head_only":
            self.net_.freeze_except_head()
            self.finetune_losses_ = self._train_loop(
                torch, nn, self.net_, X_t, y_t, log_exp_t,
                self.finetune_epochs, self.learning_rate
            )

        elif self.finetune_strategy == "all":
            self.net_.unfreeze_body()
            self.finetune_losses_ = self._train_loop(
                torch, nn, self.net_, X_t, y_t, log_exp_t,
                self.finetune_epochs, self.learning_rate
            )

        elif self.finetune_strategy == "progressive":
            half = self.finetune_epochs // 2
            remainder = self.finetune_epochs - half
            self.net_.freeze_except_head()
            losses_1 = self._train_loop(
                torch, nn, self.net_, X_t, y_t, log_exp_t, half, self.learning_rate
            )
            self.net_.unfreeze_body()
            losses_2 = self._train_loop(
                torch, nn, self.net_, X_t, y_t, log_exp_t, remainder, self.learning_rate / 5
            )
            self.finetune_losses_ = losses_1 + losses_2

        else:
            raise ValueError(
                f"Unknown finetune_strategy: {self.finetune_strategy!r}. "
                "Use 'all', 'head_only', or 'progressive'."
            )

        return self

    def predict(
        self,
        X: NDArray,
        exposure: Optional[NDArray] = None,
    ) -> NDArray:
        """Predict expected values.

        Parameters
        ----------
        X:
            Feature matrix.
        exposure:
            Exposure vector. Defaults to ones.

        Returns
        -------
        Predicted values on natural scale, shape (n,).
        """
        check_is_fitted(self, ["net_"])
        torch, nn = _check_torch()

        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]

        if exposure is None:
            exposure = np.ones(n)
        exposure = np.asarray(exposure, dtype=np.float64)
        log_exp = np.log(np.maximum(exposure, 1e-10))

        if self.scaler_ is not None:
            X = self.scaler_.transform(X)

        self.net_.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float64)
            log_exp_t = torch.tensor(log_exp, dtype=torch.float64)
            log_mu = self.net_(X_t).squeeze(-1) + log_exp_t
            mu = torch.exp(torch.clamp(log_mu, -30, 30))
        return mu.numpy().astype(np.float64)
