"""Paper model families and their unchanged training policy."""
from __future__ import annotations
from data import compute_physical_adjacency, haversine_distance

# ======================================================================
"""Explicit device selection and reproducible execution."""
import os
import random
import numpy as np
import torch

def resolve_device(requested="auto"):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; check the driver and CUDA PyTorch build.")
    return requested


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


# ======================================================================
"""Common training policy: physical-unit MAE, validation-only checkpoint selection."""
import numpy as np
import torch


def fit_network(model, device, X_train, y_train, X_val, y_val, *, lr,
                weight_decay, epochs, batch_size, target_scale=None, patience=10):
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size and patience must be positive")
    model.to(device)
    # These small datasets fit in device memory; avoid a host transfer each batch.
    x = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    vx = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    vy = torch.as_tensor(y_val, dtype=torch.float32, device=device)
    scale = torch.as_tensor(1.0 if target_scale is None else target_scale,
                            dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_epoch, updates, stale = float("inf"), 0, 0, 0
    best_weights = None
    for epoch in range(epochs):
        model.train()
        for indices in torch.randperm(len(x), device=device).split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(x[indices]) - y[indices]).abs() * scale).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            updates += 1
        model.eval()
        with torch.no_grad():
            val = float(((model(vx) - vy).abs() * scale).mean().item())
        if not np.isfinite(val):
            raise FloatingPointError("Nonfinite validation loss")
        if val < best:
            best, best_epoch, stale = val, epoch + 1, 0
            best_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_weights)
    model.eval()
    return {"epochs_requested": epochs, "epochs_executed": epoch + 1,
            "best_epoch": best_epoch, "optimizer_updates": updates,
            "batch_size": batch_size, "patience": patience,
            "selection_metric": "MAE_dB" if target_scale is not None else "MAE_scaled",
            "best_validation_loss": best}


# ======================================================================
"""Causal Box--Jenkins ARIMAX and multivariate VARX estimators.

For horizon H, w_s = weather[s-H]. Hence w_{t+1:t+H} is entirely observed
at origin t. The same delayed exogenous design is used by every statistical
family. ARIMAX is estimated per node by exact Gaussian state-space likelihood over
independent contiguous segments. VARX is a regularized joint linear model.
"""
import numpy as np
import warnings
from scipy.optimize import minimize
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX


class UnstableModelError(ValueError):
    """A candidate violates the declared stationary dynamics constraint."""


def _ridge_solution(design, target, alpha):
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0
    if alpha == 0:
        return np.linalg.lstsq(design, target, rcond=None)[0]
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


def _segment_equations(segment, n_targets, horizon, node):
    """Align y[t] with weather[t-H] without using future meteorology."""
    segment = np.asarray(segment, dtype=np.float64)
    return segment[horizon:, node], segment[:-horizon, n_targets:]


def _sarimax_exog(weather, order):
    """Use a level intercept only when d=0; differenced models have no drift."""
    return (np.column_stack((np.ones(len(weather)), weather))
            if order[1] == 0 else weather)


def _fit_segmented_sarimax(segments, n_targets, horizon, node, order, maxiter=200):
    """Maximize the sum of independent-segment SARIMAX log likelihoods."""
    models = []
    lengths = []
    for segment in segments:
        endog, exog = _segment_equations(segment, n_targets, horizon, node)
        if len(endog) <= sum(order) + 2:
            continue
        models.append(SARIMAX(endog, exog=_sarimax_exog(exog, order),
                              order=order, trend="n",
                              enforce_stationarity=True, enforce_invertibility=True))
        lengths.append(len(endog))
    if not models:
        raise ValueError(f"No segment supports ARIMAX{order}")
    reference = models[int(np.argmax(lengths))]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            initial = reference.untransform_params(reference.start_params)
    except (ValueError, np.linalg.LinAlgError):
        raise RuntimeError(f"ARIMAX{order} could not initialize state-space parameters")

    def objective(unconstrained):
        values = [-model.loglike(unconstrained, transformed=False) for model in models]
        total = float(np.sum(values))
        return total if np.isfinite(total) else 1e100

    optimum = minimize(objective, initial, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-8})
    if not optimum.success:
        # Expensive recovery path only for candidates whose joint optimization
        # cannot converge from the deterministic state-space starting values.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                preliminary = reference.fit(disp=False, maxiter=min(75, maxiter))
            retry_initial = reference.untransform_params(preliminary.params)
            retry = minimize(objective, retry_initial, method="L-BFGS-B",
                             options={"maxiter": maxiter, "ftol": 1e-8})
            if retry.success or retry.fun < optimum.fun:
                optimum = retry
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            pass
    if not np.isfinite(optimum.fun):
        raise RuntimeError(f"ARIMAX{order} likelihood is not finite")
    params = reference.transform_params(optimum.x)
    loglike = -objective(optimum.x)
    nobs = int(sum(lengths))
    k = int(len(params))
    aic = 2 * k - 2 * loglike
    aicc = aic + (2 * k * (k + 1) / (nobs - k - 1) if nobs > k + 1 else np.inf)
    bic = np.log(nobs) * k - 2 * loglike
    diagnostic_pvalues = []
    residual_count = 0
    for model in models:
        result = model.filter(params)
        burn = max(result.loglikelihood_burn, max(order) + 1)
        residual = np.asarray(result.resid[burn:], dtype=np.float64)
        residual = residual[np.isfinite(residual)]
        residual_count += len(residual)
        if len(residual) >= 12:
            lag = min(24, max(1, len(residual) // 5))
            diagnostic_pvalues.append(float(acorr_ljungbox(
                residual, lags=[lag], return_df=True)["lb_pvalue"].iloc[0]))
    return {
        "params": params, "order": tuple(int(value) for value in order),
        "log_likelihood": float(loglike), "aic": float(aic),
        "aicc": float(aicc), "bic": float(bic), "nobs": nobs,
        "parameters": k, "converged": bool(optimum.success),
        "optimizer_message": str(optimum.message),
        "ljung_box_min_pvalue": min(diagnostic_pvalues) if diagnostic_pvalues else None,
        "diagnostic_residuals": residual_count,
        "segment_lengths": lengths,
    }


class BoxJenkinsARIMAX:
    """One independently identified Box--Jenkins ARIMAX model per RSSI node."""
    def __init__(self, n_targets=8, pred_length=1, orders=None):
        self.n_targets = n_targets
        self.pred_length = pred_length
        self.orders = ([tuple(order) for order in orders] if orders is not None
                       else [(1, 0, 1)] * n_targets)
        if len(self.orders) != n_targets:
            raise ValueError("ARIMAX requires one (p,d,q) order per node")
        self.fits = []
        self.training_summary = {}

    def fit(self, X_train, y_train, *, training_segments):
        segments = [np.asarray(segment, dtype=np.float64) for segment in training_segments]
        self.fits = [_fit_segmented_sarimax(
            segments, self.n_targets, self.pred_length, node, self.orders[node])
            for node in range(self.n_targets)]
        self.training_summary = {
            "estimator": "exact Gaussian state-space likelihood summed over independent contiguous segments",
            "selection": "training-only AICc per node; ACF/PACF bound candidate orders",
            "joint": False, "orders": [list(order) for order in self.orders],
            "weather_lag_hours": self.pred_length,
            "diagnostics": [{key: value for key, value in fit.items() if key != "params"}
                            for fit in self.fits],
            "innovation_initialization": "stationary initialization independently per segment/window",
        }
        return self

    def predict(self, X, pred_length=None):
        horizon = self.pred_length if pred_length is None else pred_length
        if horizon != self.pred_length:
            raise ValueError("Horizon must match fitted delayed-exogenous model")
        X = np.asarray(X, dtype=np.float64)
        batch, length, _ = X.shape
        predictions = np.empty((batch, horizon, self.n_targets), dtype=np.float32)
        for sample in range(batch):
            weather = X[sample, :, self.n_targets:]
            for node, fit in enumerate(self.fits):
                endog = X[sample, horizon:, node]
                observed_exog = weather[:-horizon]
                model = SARIMAX(endog, exog=_sarimax_exog(observed_exog, fit["order"]),
                                order=fit["order"], trend="n",
                                enforce_stationarity=True, enforce_invertibility=True)
                result = model.filter(fit["params"])
                future_exog = weather[length-horizon:length]
                predictions[sample, :, node] = result.forecast(
                    horizon, exog=_sarimax_exog(future_exog, fit["order"])).astype(np.float32)
        return predictions

    def total_parameters(self):
        return sum(fit["parameters"] for fit in self.fits)


class HistoricalWeatherVARX:
    """Regularized joint VARX with dense or explicitly sparse lag sets."""
    def __init__(self, n_targets=8, pred_length=1, lags=3, alpha=1.0,
                 minimum_burn_in=0):
        lag_indices = (list(range(1, int(lags) + 1))
                       if np.isscalar(lags) else sorted({int(lag) for lag in lags}))
        if (not lag_indices or lag_indices[0] < 1 or pred_length < 1
                or alpha < 0 or minimum_burn_in < 0):
            raise ValueError("Invalid statistical configuration")
        self.n_targets, self.pred_length = n_targets, pred_length
        self.lag_indices, self.alpha = lag_indices, alpha
        self.minimum_burn_in = int(minimum_burn_in)
        self.lags = max(lag_indices)  # Backward-compatible maximum-lag attribute.
        self.joint = True
        self.coefficients = []
        self.training_summary = {}

    @property
    def burn_in(self):
        return max(self.pred_length, max(self.lag_indices), self.minimum_burn_in)

    def _equation_data(self, segment, node):
        y = np.asarray(segment[:, :self.n_targets], dtype=np.float64)
        weather = np.asarray(segment[:, self.n_targets:], dtype=np.float64)
        z = y
        times = np.arange(self.burn_in, len(y))
        channels = np.arange(self.n_targets)
        ar = np.stack([z[times-lag][:, channels] for lag in self.lag_indices], axis=1)
        design = np.column_stack([np.ones(len(times)), ar.reshape(len(times), -1),
                                  weather[times-self.pred_length]])
        return design, z[times, node]

    def fit(self, X_train, y_train, *, training_segments):
        if X_train.shape[1] < self.burn_in:
            raise ValueError(f"History must contain at least {self.burn_in} hours for "
                             f"the requested lag structure and horizon {self.pred_length}.")
        segments = [np.asarray(s, dtype=np.float64) for s in training_segments
                    if len(s) > self.burn_in]
        if not segments:
            raise ValueError("No contiguous training segments support this model")
        self.coefficients = []
        for node in range(self.n_targets):
            equations = [self._equation_data(s, node) for s in segments]
            design = np.concatenate([item[0] for item in equations])
            target = np.concatenate([item[1] for item in equations])
            self.coefficients.append(_ridge_solution(design, target, self.alpha))
        radius = self.spectral_radius()
        if radius >= 1.0:
            raise UnstableModelError(f"AR dynamics not stationary: spectral radius={radius:.6f}")
        self.training_summary = {
            "estimator": "regularized conditional sum of squares",
            "joint": True, "p_max": max(self.lag_indices),
            "lag_indices": self.lag_indices, "d": 0,
            "q": 0,
            "weather_lag_hours": self.pred_length,
            "training_segment_lengths": [len(s) for s in segments],
            "conditional_observations": sum(len(s)-self.burn_in for s in segments),
            "excluded_short_segment_rows": sum(len(s) for s in training_segments if len(s)<=self.burn_in),
            "spectral_radius": radius,
        }
        residual_blocks = []
        for segment in segments:
            columns = []
            for node in range(self.n_targets):
                design, target = self._equation_data(segment, node)
                columns.append(target - design @ self.coefficients[node])
            residual_blocks.append(np.column_stack(columns))
        residuals = np.vstack(residual_blocks)
        covariance = residuals.T @ residuals / len(residuals)
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0 or not np.isfinite(logdet):
            raise ValueError("Residual covariance is not positive definite")
        coefficient_parameters = self.total_parameters()
        covariance_parameters = self.n_targets * (self.n_targets + 1) // 2
        bic_parameters = coefficient_parameters + covariance_parameters
        self.training_summary["conditional_sse"] = float(np.square(residuals).sum())
        self.training_summary["residual_logdet"] = float(logdet)
        self.training_summary["bic_parameter_count"] = bic_parameters
        self.training_summary["bic"] = float(
            len(residuals) * logdet + bic_parameters * np.log(len(residuals)))
        return self

    def spectral_radius(self):
        n, p = self.n_targets, max(self.lag_indices)
        companion = np.zeros((n*p, n*p))
        for node, beta in enumerate(self.coefficients):
            for position, lag in enumerate(self.lag_indices):
                start = 1 + position * n
                companion[node, (lag-1)*n:lag*n] = beta[start:start+n]
        if p > 1:
            companion[n:, :-n] = np.eye(n*(p-1))
        return float(np.max(np.abs(np.linalg.eigvals(companion))))

    def predict(self, X, pred_length=None):
        horizon = self.pred_length if pred_length is None else pred_length
        if horizon != self.pred_length:
            raise ValueError("Horizon must match fitted delayed-exogenous model")
        if X.shape[1] < self.burn_in:
            raise ValueError("Insufficient observed history")
        batch, length, _ = X.shape
        y = np.asarray(X[:, :, :self.n_targets], dtype=np.float64)
        z = y.copy()
        weather = np.asarray(X[:, :, self.n_targets:], dtype=np.float64)
        history = [z[:,i].copy() for i in range(length)]
        levels = y[:,-1].copy()
        forecasts = []
        for lead in range(horizon):
            step = length + lead
            ar_all = np.stack([history[-lag] for lag in self.lag_indices], axis=1)
            next_z = np.empty((batch,self.n_targets))
            for node,beta in enumerate(self.coefficients):
                ar = ar_all.reshape(batch,-1)
                # step-H never exceeds length-1: no future weather is accessed.
                design = np.column_stack([np.ones(batch), ar, weather[:,step-horizon]])
                next_z[:,node] = design @ beta
            history.append(next_z)
            levels = next_z
            forecasts.append(levels.copy())
        return np.stack(forecasts,axis=1).astype(np.float32)

    def total_parameters(self):
        return sum(len(beta) for beta in self.coefficients)


# ======================================================================
"""
Integrated recurrent encoder-decoder with additive temporal attention.

LSTM encoder and autoregressive LSTMCell decoder, following the encoder-decoder
and additive-attention pattern of Sutskever et al. (2014) and Bahdanau et al.
(2015), adapted to continuous multi-node RSSI. Free-running training uses previous
predictions, never future observations; a last-observation residual anchors outputs.

Incorporates a direct residual link from the last observed time step (X_{t-1}),
allowing the neural network to focus exclusively on learning the dynamic
variations (Delta RSSI) rather than re-learning the static DC offset.
"""

import torch
import torch.nn as nn
import numpy as np


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        seq_len = keys.size(1)
        q_expanded = query.unsqueeze(1).repeat(1, seq_len, 1)
        combined = torch.cat((q_expanded, keys), dim=2)
        energy = torch.tanh(self.attn(combined))
        scores = self.v(energy).squeeze(2)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context


class MultiNodeSeq2SeqAttention(nn.Module):
    def __init__(
        self,
        in_features: int,
        n_targets: int,
        pred_length: int,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.in_features = in_features
        self.n_targets = n_targets
        self.pred_length = pred_length
        self.hidden_dim = hidden_dim

        # Encoder
        self.encoder = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Attention
        self.attention = TemporalAttention(hidden_dim)

        # Recurrent attention decoder, initialized with the encoder state.
        self.decoder = nn.LSTMCell(n_targets + hidden_dim, hidden_dim)
        self.decoder_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_targets)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, in_features)
        last_obs = x[:, -1:, :self.n_targets]  # Residual base: (batch_size, 1, n_targets)

        encoder_outputs, (h_n, c_n) = self.encoder(x)
        hidden, cell = h_n[-1], c_n[-1]
        anchor = last_obs[:, 0]
        previous = anchor
        predictions = []
        for _ in range(self.pred_length):
            context = self.attention(hidden, encoder_outputs)
            hidden, cell = self.decoder(torch.cat([previous, context], dim=-1),
                                        (hidden, cell))
            previous = anchor + self.decoder_head(torch.cat([hidden, context], dim=-1))
            predictions.append(previous)
        return torch.stack(predictions, dim=1)


class MultiNodeSeq2SeqTrainer:
    def __init__(
        self,
        in_features: int,
        n_targets: int,
        pred_length: int,
        hidden_dim: int = 32,
        num_layers: int = 1,
        lr: float = 1e-3,
        dropout: float = 0.1,
        weight_decay: float = 1e-4,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.model = MultiNodeSeq2SeqAttention(
            in_features=in_features,
            n_targets=n_targets,
            pred_length=pred_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers, dropout=dropout
        ).to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.training_summary = {}

    def fit(self, X_train, y_train, X_val, y_val, epochs=64,
            batch_size=32, verbose=False, target_scale=None, patience=10):
        self.training_summary = fit_network(
            self.model, self.device, X_train, y_train, X_val, y_val,
            lr=self.lr, weight_decay=self.weight_decay, epochs=epochs,
            batch_size=batch_size, target_scale=target_scale, patience=patience)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            bx = torch.tensor(X, dtype=torch.float32).to(self.device)
            return self.model(bx).cpu().numpy()

    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


# ======================================================================
"""Original channel-independent NLinear/DLinear formulations, Zeng et al. (AAAI 2023).
Reference: https://github.com/cure-lab/LTSF-Linear/tree/main/models
RSSI channels only; shared temporal weights do not mix nodes or use weather.
"""
import torch
from torch import nn


class NLinear(nn.Module):
    def __init__(self, seq_len, pred_len, in_features, n_targets, individual=False):
        super().__init__()
        self.n_targets, self.individual = n_targets, individual
        self.linear = (nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(n_targets)])
                       if individual else nn.Linear(seq_len, pred_len))

    def forward(self, x):
        x = x[:, :, :self.n_targets]
        last = x[:, -1:, :].detach()
        centered = (x - last).transpose(1, 2)
        if self.individual:
            out = torch.stack([layer(centered[:, i]) for i, layer in enumerate(self.linear)], 1)
        else:
            out = self.linear(centered)
        return out.transpose(1, 2) + last


class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, in_features, n_targets, individual=False):
        super().__init__()
        self.n_targets, self.individual = n_targets, individual
        self.average = nn.AvgPool1d(25, stride=1)
        def projection():
            return (nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(n_targets)])
                    if individual else nn.Linear(seq_len, pred_len))
        self.seasonal, self.trend = projection(), projection()

    def forward(self, x):
        x = x[:, :, :self.n_targets].transpose(1, 2)
        padded = torch.cat([x[:, :, :1].expand(-1, -1, 12), x,
                            x[:, :, -1:].expand(-1, -1, 12)], dim=2)
        trend = self.average(padded)
        seasonal = x - trend
        if self.individual:
            out = torch.stack([self.seasonal[i](seasonal[:, i]) + self.trend[i](trend[:, i])
                               for i in range(self.n_targets)], 1)
        else:
            out = self.seasonal(seasonal) + self.trend(trend)
        return out.transpose(1, 2)


class NLinearTrainer:
    architecture = NLinear

    def __init__(self, seq_len, pred_len, in_features, n_targets, lr=1e-3,
                 weight_decay=1e-3, device="cpu", individual=False):
        self.device = torch.device(device)
        self.model = self.architecture(seq_len, pred_len, in_features, n_targets,
                                       individual=individual).to(self.device)
        self.lr, self.weight_decay = lr, weight_decay
        self.training_summary = {}

    def fit(self, X_train, y_train, X_val, y_val, epochs=64, batch_size=32,
            verbose=False, target_scale=None, patience=10):
        self.training_summary = fit_network(
            self.model, self.device, X_train, y_train, X_val, y_val,
            lr=self.lr, weight_decay=self.weight_decay, epochs=epochs,
            batch_size=batch_size, target_scale=target_scale, patience=patience)
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.as_tensor(X, dtype=torch.float32,
                                              device=self.device)).cpu().numpy()

    def total_parameters(self):
        return sum(p.numel() for p in self.model.parameters())


class DLinearTrainer(NLinearTrainer):
    architecture = DLinear


# ======================================================================
"""Independent per-node copies of exactly the same neural architecture."""
import numpy as np


class DedicatedNodeTrainer:
    def __init__(self, trainer_class, n_targets, trainer_kwargs, include_weather=True):
        self.trainer_class = trainer_class
        self.n_targets = n_targets
        self.trainer_kwargs = trainer_kwargs
        self.include_weather = include_weather
        self.models = []
        self.training_summary = {}

    def features(self, X, node):
        own = X[:, :, node:node+1]
        return np.concatenate([own, X[:, :, self.n_targets:]], axis=-1) if self.include_weather else own

    def fit(self, X_train, y_train, X_val, y_val, epochs=64, batch_size=32,
            verbose=False, target_scale=None, patience=10):
        self.models = []
        summaries = []
        for node in range(self.n_targets):
            model = self.trainer_class(**self.trainer_kwargs)
            model.fit(self.features(X_train, node), y_train[:, :, node:node+1],
                      self.features(X_val, node), y_val[:, :, node:node+1],
                      epochs=epochs, batch_size=batch_size, patience=patience,
                      target_scale=None if target_scale is None else target_scale[node:node+1])
            self.models.append(model)
            summaries.append(model.training_summary)
        self.training_summary = {"per_node": summaries, "epochs_requested": epochs,
                                 "best_epoch_per_node": [s["best_epoch"] for s in summaries],
                                 "optimizer_updates": sum(s["optimizer_updates"] for s in summaries)}
        return self

    def predict(self, X):
        return np.concatenate([m.predict(self.features(X, i)) for i, m in enumerate(self.models)],axis=-1)

    def total_parameters(self):
        return sum(m.total_parameters() for m in self.models)


# ======================================================================
"""
Adaptive Spatio-Temporal Graph Neural Network (Adaptive STGNN) for LoRaWAN.

Formulation:
- Graph Topology: geographic distance prior blended with a static learned graph.
- Ablations: no cross-node edges, geographic-only, learned-only, and hybrid.
- Adjacencies represent predictive weights, not physical radio connectivity.
- Spatial Modeling: directed one-hop weighted aggregation with a separate self path.
- Temporal Modeling: gated, dilated causal convolutions; residual connections.
- Residual Link: Directly anchored to the last observed RSSI (X_{t-1}).
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class AdaptiveGraphConvolution(nn.Module):
    """
    Graph convolution with a static adjacency learned during training.
    Computes diffusion: Z = A * X * W_spatial + X * W_self
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_nodes: int,
        physical_adjacency: np.ndarray,
        node_emb_dim: int = 4,
        graph_mode: str = "hybrid",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        if graph_mode not in ("none", "physical", "adaptive", "hybrid"):
            raise ValueError("Unknown graph mode")
        if physical_adjacency.shape != (num_nodes, num_nodes):
            raise ValueError("Graph dimensions must match target node ordering")
        self.graph_mode = graph_mode
        self.register_buffer("off_diagonal", ~torch.eye(num_nodes, dtype=torch.bool))
        physical = torch.as_tensor(physical_adjacency, dtype=torch.float32).clone()
        physical.fill_diagonal_(0)
        physical = physical / physical.sum(-1, keepdim=True).clamp_min(1e-12)
        self.register_buffer("physical_adjacency", physical)
        if graph_mode in ("adaptive", "hybrid"):
            self.source_embedding = nn.Parameter(torch.randn(num_nodes, node_emb_dim)*0.1)
            self.target_embedding = nn.Parameter(torch.randn(num_nodes, node_emb_dim)*0.1)
        if graph_mode == "hybrid":
            self.adaptive_mix_logit = nn.Parameter(torch.tensor(0.0))
        self.w_spatial = (nn.Linear(in_features, out_features, bias=False)
                          if graph_mode != "none" else None)
        self.w_self = nn.Linear(in_features, out_features, bias=True)
        self.layer_norm = nn.LayerNorm(out_features)

    def get_adjacency(self) -> torch.Tensor:
        """Returns the normalized learned adjacency matrix A in [0, 1]^(N x N)."""
        if self.graph_mode == "none" or self.num_nodes == 1:
            return torch.zeros_like(self.physical_adjacency)
        if self.graph_mode == "physical":
            return self.physical_adjacency
        score = torch.mm(self.source_embedding, self.target_embedding.t())
        score = score.masked_fill(~self.off_diagonal, -torch.inf)
        adaptive = F.softmax(score, dim=-1)
        if self.graph_mode == "adaptive":
            return adaptive
        weight = torch.sigmoid(self.adaptive_mix_logit)
        return weight * adaptive + (1-weight) * self.physical_adjacency

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, num_nodes, in_features)
        Returns: (batch_size, num_nodes, out_features)
        """
        A = self.get_adjacency()  # (num_nodes, num_nodes)
        # Spatial diffusion: support = A @ x
        support = torch.einsum("nm, bmc -> bnc", A, x)
        out = self.w_self(x)
        if self.w_spatial is not None:
            out = out + self.w_spatial(support)
        return self.layer_norm(F.relu(out))


class SpatioTemporalBlock(nn.Module):
    """
    Spatio-Temporal block combining:
    1. Temporal 1D Convolution
    2. Spatial Adaptive Graph Convolution
    3. Residual connection
    """
    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int,
        physical_adjacency: np.ndarray,
        kernel_size: int = 3,
        dropout: float = 0.1,
        dilation: int = 1,
        graph_mode: str = "hybrid",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # Temporal convolution across time (kernel_size, causal padding)
        self.padding = (kernel_size - 1) * dilation
        self.temporal_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=2 * hidden_dim,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding
        )
        self.spatial_gcn = AdaptiveGraphConvolution(
            hidden_dim, hidden_dim, num_nodes, physical_adjacency, graph_mode=graph_mode
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, seq_len, num_nodes, hidden_dim)
        """
        batch_size, seq_len, num_nodes, feat_dim = x.shape

        # 1. Temporal processing per node
        # Reshape to (batch_size * num_nodes, hidden_dim, seq_len)
        x_t = x.permute(0, 2, 3, 1).contiguous().view(batch_size * num_nodes, feat_dim, seq_len)
        t_out = self.temporal_conv(x_t)
        if self.padding > 0:
            t_out = t_out[:, :, :-self.padding]
        value, gate = t_out.chunk(2, dim=1)
        t_out = torch.tanh(value) * torch.sigmoid(gate)
        t_out = t_out.view(batch_size, num_nodes, feat_dim, seq_len).permute(0, 3, 1, 2)
        # t_out shape: (batch_size, seq_len, num_nodes, hidden_dim)

        # 2. Spatial processing across nodes for each time step
        # Reshape to (batch_size * seq_len, num_nodes, hidden_dim)
        s_in = t_out.contiguous().view(batch_size * seq_len, num_nodes, feat_dim)
        s_out = self.spatial_gcn(s_in)
        s_out = s_out.view(batch_size, seq_len, num_nodes, feat_dim)

        # 3. Residual connection
        out = self.norm(self.dropout(s_out) + x)
        return out


class AdaptiveSTGNN(nn.Module):
    """
    Complete Spatio-Temporal Graph Neural Network for LoRaWAN RSSI.
    """
    def __init__(
        self,
        n_targets: int = 8,
        n_exogenous: int = 4,
        seq_length: int = 24,
        pred_length: int = 6,
        hidden_dim: int = 32,
        num_blocks: int = 2,
        dropout: float = 0.1,
        graph_mode: str = "hybrid",
        node_coordinates: Optional[np.ndarray] = None,
        gateway_coordinates: Optional[np.ndarray] = None,
        gateway_distances_m: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.n_targets = n_targets
        self.n_exogenous = n_exogenous
        self.seq_length = seq_length
        self.pred_length = pred_length

        # Per node: RSSI, weather variables, and normalized distance to the gateway.
        in_feat_per_node = 2 + n_exogenous
        self.input_proj = nn.Linear(in_feat_per_node, hidden_dim)

        if node_coordinates is None or gateway_coordinates is None:
            raise ValueError("Graph model requires explicit sensor and gateway coordinates")
        node_coordinates = np.asarray(node_coordinates, dtype=np.float64)
        gateway_coordinates = np.asarray(gateway_coordinates, dtype=np.float64)
        if node_coordinates.shape != (n_targets, 2) or gateway_coordinates.shape != (2,) or num_blocks < 1:
            raise ValueError("Invalid node count or number of graph blocks")
        physical_adjacency = compute_physical_adjacency(node_coords=node_coordinates)

        self.blocks = nn.ModuleList([
            SpatioTemporalBlock(
                num_nodes=n_targets,
                hidden_dim=hidden_dim,
                physical_adjacency=physical_adjacency,
                dropout=dropout, dilation=2**block_index, graph_mode=graph_mode,
            )
            for block_index in range(num_blocks)
        ])

        gateway_distance = np.asarray(
            ([haversine_distance(lat, lon, gateway_coordinates[0], gateway_coordinates[1])
              for lat, lon in node_coordinates]
             if gateway_distances_m is None else gateway_distances_m),
            dtype=np.float64,
        )
        if gateway_distance.shape != (n_targets,) or not np.all(gateway_distance > 0):
            raise ValueError("Invalid gateway distances")
        gateway_distance = gateway_distance / gateway_distance.max()
        self.register_buffer(
            "gateway_distance",
            torch.as_tensor(gateway_distance, dtype=torch.float32).view(1, 1, n_targets, 1),
        )

        # Temporal pooling and projection head
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_dim * seq_length, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_length)
        )

    def _prepare_spatio_temporal_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Converts flat (B, T, 8 + 4) tensor into spatio-temporal tensor (B, T, N, F=2+n_exogenous).
        """
        batch_size, seq_len, _ = x.shape
        rssi = x[:, :, :self.n_targets]         # (B, T, 8)
        exog = x[:, :, self.n_targets:]         # (B, T, 4)

        # Expand exogenous across all nodes: (B, T, 8, 4)
        exog_expanded = exog.unsqueeze(2).repeat(1, 1, self.n_targets, 1)
        rssi_expanded = rssi.unsqueeze(-1)      # (B, T, 8, 1)

        distance = self.gateway_distance.expand(batch_size, seq_len, -1, -1)
        st_tensor = torch.cat([rssi_expanded, exog_expanded, distance], dim=-1)
        return st_tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, seq_length, in_features)
        Returns: (batch_size, pred_length, n_targets)
        """
        # Residual anchor from last known RSSI: (batch_size, 1, n_targets)
        last_rssi = x[:, -1:, :self.n_targets]

        st_tensor = self._prepare_spatio_temporal_tensor(x)  # (B, T, N, F)
        h = self.input_proj(st_tensor)                       # (B, T, N, hidden_dim)

        for block in self.blocks:
            h = block(h)

        # h shape: (B, T, N, hidden_dim) -> permute to (B, N, hidden_dim * T)
        batch_size, seq_len, num_nodes, hidden_dim = h.shape
        h_flat = h.permute(0, 2, 3, 1).contiguous().view(batch_size, num_nodes, hidden_dim * seq_len)

        # Predict delta per node: (B, N, pred_length)
        delta_per_node = self.fc_head(h_flat)
        # Permute to (B, pred_length, N)
        delta = delta_per_node.permute(0, 2, 1)

        return last_rssi + delta

    def get_learned_adjacency_matrix(self) -> np.ndarray:
        """Returns the spatial adjacency matrix learned by the first ST block."""
        with torch.no_grad():
            adj = self.blocks[0].spatial_gcn.get_adjacency()
            return adj.cpu().numpy()


class AdaptiveSTGNNTrainer:
    def __init__(
        self,
        n_targets: int = 8,
        n_exogenous: int = 4,
        seq_length: int = 24,
        pred_length: int = 6,
        hidden_dim: int = 24,
        num_blocks: int = 2,
        lr: float = 1e-3,
        dropout: float = 0.1,
        weight_decay: float = 1e-4,
        device: str = "cpu",
        graph_mode: str = "hybrid",
        node_coordinates: Optional[np.ndarray] = None,
        gateway_coordinates: Optional[np.ndarray] = None,
        gateway_distances_m: Optional[np.ndarray] = None,
    ):
        self.device = torch.device(device)
        self.model = AdaptiveSTGNN(
            n_targets=n_targets,
            n_exogenous=n_exogenous,
            seq_length=seq_length,
            pred_length=pred_length,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks, dropout=dropout, graph_mode=graph_mode,
            node_coordinates=node_coordinates, gateway_coordinates=gateway_coordinates,
            gateway_distances_m=gateway_distances_m,
        ).to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.training_summary = {}

    def fit(self, X_train, y_train, X_val, y_val, epochs=64,
            batch_size=32, verbose=False, target_scale=None, patience=10):
        self.training_summary = fit_network(
            self.model, self.device, X_train, y_train, X_val, y_val,
            lr=self.lr, weight_decay=self.weight_decay, epochs=epochs,
            batch_size=batch_size, target_scale=target_scale, patience=patience)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            bx = torch.tensor(X, dtype=torch.float32).to(self.device)
            return self.model(bx).cpu().numpy()

    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_learned_adjacency(self) -> np.ndarray:
        return self.model.get_learned_adjacency_matrix()

    def graph_diagnostics(self):
        """All layers, rather than only the first adjacency plot."""
        return {
            "edge_orientation": "A[receiver, sender]",
            "learned_graph": "static after fitting; not time-varying or causal connectivity",
            "temporal_receptive_field": 1 + 2 * (2**len(self.model.blocks)-1),
            "readout": "uses all observed temporal features",
            "layers": [
                {"mode": block.spatial_gcn.graph_mode,
                 "adjacency": block.spatial_gcn.get_adjacency().detach().cpu().tolist(),
                 "adaptive_weight": float(torch.sigmoid(block.spatial_gcn.adaptive_mix_logit).item())
                    if hasattr(block.spatial_gcn, "adaptive_mix_logit") else None}
                for block in self.model.blocks
            ],
        }


# ======================================================================
"""Registry of paired statistical/neural families and graph ablations."""

STATISTICAL_MODELS = ("ARIMAX", "VARX")
NEURAL_MODELS = ("SingleNode_Seq2Seq", "MultiNode_Seq2Seq",
                 "SingleNode_NLinear", "MultiNode_NLinear",
                 "SingleNode_DLinear", "MultiNode_DLinear", "PhysicalAdaptive_STGNN")
GRAPH_ABLATIONS = {"STGNN_NoGraph": "none", "STGNN_PhysicalOnly": "physical",
                   "STGNN_AdaptiveOnly": "adaptive"}
DEFAULTS = {
    "ARIMAX": {"orders": None},
    "VARX": {"lags": 3, "alpha": 0.01},
    **{name: {"hidden_dim": 24, "num_layers": 1, "blocks": 2,
              "lr": 0.001, "weight_decay": 0.0001, "dropout": 0.1,
              "batch_size": 32} for name in NEURAL_MODELS},
}


def model_metadata(name, nodes=8):
    local = name == "ARIMAX" or name.startswith("SingleNode_")
    linear = name.endswith(("NLinear", "DLinear"))
    graph = name == "PhysicalAdaptive_STGNN" or name in GRAPH_ABLATIONS
    cross = not local and not linear and name != "STGNN_NoGraph"
    return {"model_instances": nodes if local else 1,
            "joint_multi_node_model": cross, "cross_node_inputs": cross,
            "parameter_sharing": not local and name not in STATISTICAL_MODELS,
            "uses_historical_weather": not linear,
            "uses_geometry": graph,
            "integration": "dedicated" if local else ("shared_channel_independent" if linear
                           or name == "STGNN_NoGraph" else "cross_node"),
            "forecast_strategy": "recursive" if name in STATISTICAL_MODELS or "Seq2Seq" in name else "direct"}


def make_statistical(name, cfg, data, horizon, *, allow_default_orders=False):
    nodes = len(data["target_names"])
    if name == "ARIMAX":
        orders = cfg.get("orders")
        if orders is None:
            if not allow_default_orders:
                raise ValueError("Optimized ARIMAX orders are missing; run --optimize.")
            orders = [[1, 0, 1] for _ in range(nodes)]
        if len(orders) != nodes or any(len(order) != 3 for order in orders):
            raise ValueError("ARIMAX orders must contain one (p,d,q) per selected link.")
        return BoxJenkinsARIMAX(nodes, horizon, orders)
    if name == "VARX":
        return HistoricalWeatherVARX(
            n_targets=nodes, pred_length=horizon, lags=cfg["lags"],
            alpha=cfg["alpha"], minimum_burn_in=cfg.get("minimum_burn_in", 0))
    raise ValueError(name)


def make_neural(name, cfg, data, horizon, device):
    features = data["train"][0].shape[-1]
    history = data["train"][0].shape[1]
    nodes = len(data["target_names"])
    local = name.startswith("SingleNode_")
    common = dict(lr=cfg["lr"], weight_decay=cfg["weight_decay"], device=device)
    if name.endswith(("NLinear", "DLinear")):
        cls = NLinearTrainer if name.endswith("NLinear") else DLinearTrainer
        kwargs = dict(seq_len=history, pred_len=horizon,
                      in_features=1 if local else features,
                      n_targets=1 if local else nodes, individual=False, **common)
        return (DedicatedNodeTrainer(cls, nodes, kwargs, include_weather=False)
                if local else cls(**kwargs))
    common.update(hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"],
                  n_targets=nodes, pred_length=horizon)
    if name == "PhysicalAdaptive_STGNN" or name in GRAPH_ABLATIONS:
        topology = data.get("topology", {})
        return AdaptiveSTGNNTrainer(n_exogenous=features-nodes, seq_length=history,
                                    num_blocks=cfg["blocks"],
                                    graph_mode=GRAPH_ABLATIONS.get(name, "hybrid"),
                                    node_coordinates=topology.get("node_coordinates"),
                                    gateway_coordinates=topology.get("gateway_coordinates"),
                                    gateway_distances_m=topology.get("gateway_distances_m"),
                                    **common)
    common.update(num_layers=cfg["num_layers"],
                  in_features=1+features-nodes if local else features)
    if local:
        common["n_targets"] = 1
        return DedicatedNodeTrainer(MultiNodeSeq2SeqTrainer, nodes, common)
    if name == "MultiNode_Seq2Seq":
        return MultiNodeSeq2SeqTrainer(**common)
    raise ValueError(name)
