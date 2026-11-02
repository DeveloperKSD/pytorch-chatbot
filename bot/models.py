"""Two interchangeable neural-net backends with the same interface.

* ``TorchMLP`` - the "real" implementation in PyTorch (autograd, Adam, GPU).
* ``NumpyMLP`` - the same network written from scratch: forward pass, manual
  backprop and Adam.  Slower and simpler, but it works without PyTorch and is
  the best way to *see* what ``loss.backward()`` actually does.
  ``tests/test_bot.py`` contains a numerical gradient check that proves the
  hand-written gradients are correct.

Network: input -> Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear
Loss:    cross-entropy with a little label smoothing (less over-confident).
"""
import copy

import numpy as np


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def is_better(val_acc, val_loss, best_acc, best_loss):
    """Checkpoint rule: higher validation accuracy wins; ties go to the lower loss."""
    if val_acc > best_acc + 1e-9:
        return True
    return abs(val_acc - best_acc) <= 1e-9 and val_loss < best_loss - 1e-4


def torch_available():
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
#  NumPy implementation (everything by hand)
# --------------------------------------------------------------------------- #
class NumpyMLP:
    name = "numpy"

    def __init__(self, in_dim, hidden, out_dim, dropout=0.2, seed=0, label_smoothing=0.05):
        self.dims = (in_dim, hidden, hidden, out_dim)
        self.dropout = dropout
        self.label_smoothing = label_smoothing
        self.rng = np.random.default_rng(seed)
        self.params = {}
        for i in range(3):
            fan_in, fan_out = self.dims[i], self.dims[i + 1]
            # He initialisation: keeps activations well-scaled with ReLU
            self.params[f"W{i}"] = self.rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)
            self.params[f"b{i}"] = np.zeros(fan_out)

    # ---- forward ---------------------------------------------------------- #
    def _forward(self, X, train):
        cache = []
        a = X
        for i in range(3):
            z = a @ self.params[f"W{i}"] + self.params[f"b{i}"]
            if i < 2:
                h = np.maximum(z, 0.0)  # ReLU
                mask = None
                if train and self.dropout > 0:
                    # "inverted" dropout: scale at train time so inference needs no change
                    mask = (self.rng.random(h.shape) >= self.dropout) / (1.0 - self.dropout)
                    h = h * mask
                cache.append((a, z, mask))
                a = h
            else:
                cache.append((a, z, None))
                logits = z
        return logits, cache

    # ---- backward --------------------------------------------------------- #
    def loss_and_grads(self, X, y, train=True):
        logits, cache = self._forward(X, train)
        probs = softmax(logits)
        n, k = probs.shape
        eps = self.label_smoothing
        target = np.full((n, k), eps / k)
        target[np.arange(n), y] += 1.0 - eps
        loss = -(target * np.log(probs + 1e-12)).sum(axis=1).mean()

        d = (probs - target) / n  # dLoss/dLogits for softmax + cross-entropy
        grads = {}
        for i in (2, 1, 0):
            a_in, z, mask = cache[i]
            if i < 2:  # undo dropout and ReLU on the way back
                if mask is not None:
                    d = d * mask
                d = d * (z > 0)
            grads[f"W{i}"] = a_in.T @ d
            grads[f"b{i}"] = d.sum(axis=0)
            d = d @ self.params[f"W{i}"].T
        return loss, grads

    # ---- training --------------------------------------------------------- #
    def fit(self, X, y, X_val=None, y_val=None, epochs=300, lr=3e-3, batch_size=16,
            patience=40, weight_decay=1e-4, log=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v = {k: np.zeros_like(p) for k, p in self.params.items()}
        b1, b2, eps = 0.9, 0.999, 1e-8
        t = 0
        n = len(X)
        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_acc, best_loss, best_params, bad = -1.0, np.inf, None, 0
        has_val = X_val is not None and len(X_val) > 0

        for epoch in range(1, epochs + 1):
            order = self.rng.permutation(n)
            total = 0.0
            for i in range(0, n, batch_size):
                idx = order[i:i + batch_size]
                loss, grads = self.loss_and_grads(X[idx], y[idx])
                t += 1
                for k in self.params:  # Adam update
                    g = grads[k]
                    if k.startswith("W"):
                        g = g + weight_decay * self.params[k]
                    m[k] = b1 * m[k] + (1 - b1) * g
                    v[k] = b2 * v[k] + (1 - b2) * g * g
                    m_hat = m[k] / (1 - b1 ** t)
                    v_hat = v[k] / (1 - b2 ** t)
                    self.params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)
                total += loss * len(idx)
            history["train_loss"].append(total / n)

            if has_val:
                p = self.predict_proba(X_val)
                val_loss = float(-np.log(p[np.arange(len(y_val)), y_val] + 1e-12).mean())
                val_acc = float((p.argmax(axis=1) == y_val).mean())
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                if is_better(val_acc, val_loss, best_acc, best_loss):
                    best_acc, best_loss, best_params, bad = val_acc, val_loss, copy.deepcopy(self.params), 0
                else:
                    bad += 1
                    if bad >= patience:
                        if log:
                            log(f"  early stop at epoch {epoch}")
                        break
            if log and (epoch % 50 == 0 or epoch == 1):
                msg = f"  epoch {epoch:4d}  train_loss {history['train_loss'][-1]:.4f}"
                if has_val:
                    msg += f"  val_loss {history['val_loss'][-1]:.4f}  val_acc {history['val_acc'][-1]:.2%}"
                log(msg)
        if best_params is not None:
            self.params = best_params
        return history

    def predict_proba(self, X):
        logits, _ = self._forward(np.asarray(X, dtype=np.float64), train=False)
        return softmax(logits)

    def save(self, path):
        np.savez(path, **self.params)

    def load(self, path):
        with np.load(path) as d:
            self.params = {k: d[k] for k in d.files}


# --------------------------------------------------------------------------- #
#  PyTorch implementation
# --------------------------------------------------------------------------- #
class TorchMLP:
    name = "torch"

    def __init__(self, in_dim, hidden, out_dim, dropout=0.2, seed=0, label_smoothing=0.05):
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        self.torch = torch
        self.label_smoothing = label_smoothing
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        ).to(self.device)

    def fit(self, X, y, X_val=None, y_val=None, epochs=300, lr=3e-3, batch_size=16,
            patience=40, weight_decay=1e-4, log=None):
        torch = self.torch
        Xt = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)
        yt = torch.as_tensor(np.asarray(y), dtype=torch.long, device=self.device)
        has_val = X_val is not None and len(X_val) > 0
        if has_val:
            Xv = torch.as_tensor(np.asarray(X_val, dtype=np.float32), device=self.device)
            yv = torch.as_tensor(np.asarray(y_val), dtype=torch.long, device=self.device)

        opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_acc, best_loss, best_state, bad = -1.0, float("inf"), None, 0
        n = len(Xt)

        for epoch in range(1, epochs + 1):
            self.net.train()
            perm = torch.randperm(n, device=self.device)
            total = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net(Xt[idx]), yt[idx])
                loss.backward()   # <- autograd does what NumpyMLP.loss_and_grads does by hand
                opt.step()
                total += loss.item() * len(idx)
            history["train_loss"].append(total / n)

            if has_val:
                self.net.eval()
                with torch.no_grad():
                    logits = self.net(Xv)
                    val_loss = float(torch.nn.functional.cross_entropy(logits, yv))
                    val_acc = float((logits.argmax(dim=1) == yv).float().mean())
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                if is_better(val_acc, val_loss, best_acc, best_loss):
                    best_acc, best_loss, bad = val_acc, val_loss, 0
                    best_state = copy.deepcopy(self.net.state_dict())
                else:
                    bad += 1
                    if bad >= patience:
                        if log:
                            log(f"  early stop at epoch {epoch}")
                        break
            if log and (epoch % 50 == 0 or epoch == 1):
                msg = f"  epoch {epoch:4d}  train_loss {history['train_loss'][-1]:.4f}"
                if has_val:
                    msg += f"  val_loss {history['val_loss'][-1]:.4f}  val_acc {history['val_acc'][-1]:.2%}"
                log(msg)
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return history

    def predict_proba(self, X):
        torch = self.torch
        self.net.eval()
        with torch.no_grad():
            Xt = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)
            return torch.softmax(self.net(Xt), dim=1).cpu().numpy()

    def save(self, path):
        self.torch.save(self.net.state_dict(), path)

    def load(self, path):
        state = self.torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(state)


def make_backend(kind, in_dim, hidden, out_dim, dropout=0.2, seed=0):
    if kind == "auto":
        kind = "torch" if torch_available() else "numpy"
    if kind == "torch":
        if not torch_available():
            raise RuntimeError("PyTorch is not installed. Run `pip install torch` or use --backend numpy.")
        return TorchMLP(in_dim, hidden, out_dim, dropout, seed)
    if kind == "numpy":
        return NumpyMLP(in_dim, hidden, out_dim, dropout, seed)
    raise ValueError(f"Unknown backend '{kind}' (use auto, torch or numpy)")


def weights_filename(backend_name):
    return "weights.pt" if backend_name == "torch" else "weights.npz"
