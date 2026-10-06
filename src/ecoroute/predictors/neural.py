"""Matrix factorization (RouteLLM-style) and IRT predictors, trained with PyTorch.

Both learn from the observed (prompt, model) cells only and use soft labels, since
some sources grade with a 0..1 score rather than right/wrong.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ecoroute.predictors.base import Predictor


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _TorchPredictor(Predictor):
    def __init__(
        self,
        epochs: int = 20,
        min_steps: int = 2000,
        batch_size: int = 4096,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.epochs = epochs
        # Small datasets get more epochs so the model still sees enough updates.
        self.min_steps = min_steps
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed

    def _build(self, dim: int, n_models: int) -> nn.Module:
        raise NotImplementedError

    def fit(self, X, Y, models):
        torch.manual_seed(self.seed)
        self.models = list(models)
        dev = _device()
        self.net_ = self._build(X.shape[1], len(models)).to(dev)
        rows, cols = np.nonzero(~np.isnan(Y))
        Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
        r = torch.as_tensor(rows, device=dev)
        c = torch.as_tensor(cols, device=dev)
        y = torch.as_tensor(Y[rows, cols], dtype=torch.float32, device=dev)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        self.net_.train()
        steps_per_epoch = -(-len(y) // self.batch_size)
        epochs = max(self.epochs, -(-self.min_steps // steps_per_epoch))
        for _ in range(epochs):
            perm = torch.randperm(len(y), device=dev)
            for start in range(0, len(y), self.batch_size):
                b = perm[start : start + self.batch_size]
                logits = self.net_(Xt[r[b]], c[b])
                loss = loss_fn(logits, y[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.net_.eval()
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        dev = _device()
        Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
        n, m = len(X), len(self.models)
        rows = torch.arange(n, device=dev).repeat_interleave(m)
        cols = torch.arange(m, device=dev).repeat(n)
        return torch.sigmoid(self.net_(Xt[rows], cols)).reshape(n, m).cpu().numpy()


class _MFNet(nn.Module):
    def __init__(self, dim: int, n_models: int, hidden: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, hidden, bias=False)
        self.model_emb = nn.Embedding(n_models, hidden)
        self.out = nn.Linear(hidden, 1, bias=False)
        self.model_bias = nn.Embedding(n_models, 1)

    def forward(self, x, model_idx):
        h = self.proj(x) * self.model_emb(model_idx)
        return (self.out(h) + self.model_bias(model_idx)).squeeze(-1)


class MatrixFactorizationPredictor(_TorchPredictor):
    """logit = w . (W x  *  v_model) + b_model  (RouteLLM's strongest router)."""

    name = "matrix_factorization"

    def __init__(self, hidden: int = 128, **kw) -> None:
        super().__init__(**kw)
        self.hidden = hidden

    def _build(self, dim, n_models):
        return _MFNet(dim, n_models, self.hidden)


class _IRTNet(nn.Module):
    def __init__(self, dim: int, n_models: int, dims: int, hidden: int) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(dim, hidden), nn.GELU())
        self.disc = nn.Linear(hidden, dims)  # discrimination a_q (made positive)
        self.diff = nn.Linear(hidden, 1)  # difficulty b_q
        self.ability = nn.Embedding(n_models, dims)  # theta_m
        nn.init.normal_(self.ability.weight, std=0.1)

    def item_params(self, x):
        h = self.body(x)
        return nn.functional.softplus(self.disc(h)), self.diff(h).squeeze(-1)

    def forward(self, x, model_idx):
        a, b = self.item_params(x)
        return (a * self.ability(model_idx)).sum(-1) - b


class IRTPredictor(_TorchPredictor):
    """Item Response Theory: P(correct) = sigmoid(a_q . theta_m - b_q).

    b_q is the prompt's difficulty and theta_m the model's ability, both readable, which
    is what the explanation layer reports. With dims=1 this is the classic 2PL model.
    """

    name = "irt"

    def __init__(self, dims: int = 1, hidden: int = 256, **kw) -> None:
        super().__init__(**kw)
        self.dims = dims
        self.hidden = hidden

    def _build(self, dim, n_models):
        return _IRTNet(dim, n_models, self.dims, self.hidden)

    @torch.no_grad()
    def difficulty(self, X: np.ndarray) -> np.ndarray:
        """Per-prompt difficulty b_q (higher is harder)."""
        Xt = torch.as_tensor(X, dtype=torch.float32, device=_device())
        return self.net_.item_params(Xt)[1].cpu().numpy()

    def abilities(self) -> dict[str, np.ndarray]:
        w = self.net_.ability.weight.detach().cpu().numpy()
        return {m: w[i] for i, m in enumerate(self.models)}


class _MLPNet(nn.Module):
    def __init__(self, dim: int, n_models: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_models),
        )

    def forward(self, x, model_idx):
        return self.body(x).gather(1, model_idx[:, None]).squeeze(-1)


class MLPPredictor(_TorchPredictor):
    """Two-layer network with one output per model, trained on observed cells only.

    More flexible than MF or IRT: it can learn that a model is strong on one kind of
    prompt and weak on another, which is where the hard prompts get lost.
    """

    name = "mlp"

    def __init__(self, hidden: int = 512, dropout: float = 0.2, **kw) -> None:
        kw.setdefault("weight_decay", 1e-4)
        super().__init__(**kw)
        self.hidden = hidden
        self.dropout = dropout

    def _build(self, dim, n_models):
        return _MLPNet(dim, n_models, self.hidden, self.dropout)
