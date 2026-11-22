import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QSlider, QLabel, QGroupBox,
                               QComboBox, QTextEdit, QFileDialog, QCheckBox,
                               QSplitter, QStatusBar, QMessageBox, QFrame,
                               QDockWidget, QToolBar, QScrollArea, QSizePolicy,
                               QShortcut, QProgressBar, QDialog, QFormLayout,
                               QSpinBox, QDialogButtonBox, QToolButton, QStyle)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QSize
from PySide6.QtGui import QKeySequence, QAction, QIcon, QFont
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import csv
import pickle
from PIL import Image


# ============================================================
# 1. NEURAL ODE MODEL (Customizable Architecture)
# ============================================================
class NeuralODEFunc(nn.Module):
    def __init__(self, hidden_dim=64, n_layers=2, activation='tanh',
                 bifurcation_param=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation_name = activation

        act_map = {'tanh': nn.Tanh, 'relu': nn.ReLU,
                   'gelu': nn.GELU, 'silu': nn.SiLU}
        act_cls = act_map.get(activation, nn.Tanh)

        layers = [nn.Linear(2, hidden_dim), act_cls()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), act_cls()])
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

        self.register_buffer('bifurcation', torch.tensor(bifurcation_param))

    def forward(self, t, x):
        return self.net(x)

    def forward_with_bifurcation(self, t, x):
        return self.net(x) * self.bifurcation

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def jacobian(self, x):
        x = x.detach().requires_grad_(True)
        y = self.forward_with_bifurcation(torch.tensor(0.0), x).squeeze()
        J = torch.zeros(2, 2)
        for i in range(2):
            grad = torch.autograd.grad(y[i], x, retain_graph=True)[0]
            J[i] = grad
        return J.detach().numpy()

    def divergence(self, x):
        return np.trace(self.jacobian(x))


# ============================================================
# 2. NUMERICAL SOLVERS
# ============================================================
class Solvers:
    @staticmethod
    def euler(f, x, t, dt):
        return x + f(t, x) * dt

    @staticmethod
def midpoint(f, x, t, dt):
        k1 = f(t, x); k2 = f(t + dt/2, x + dt/2 * k1)
        return x + dt * k2

    @staticmethod
    def rk4(f, x, t, dt):
        k1 = f(t, x)
        k2 = f(t + dt/2, x + dt/2 * k1)
        k3 = f(t + dt/2, x + dt/2 * k2)
        k4 = f(t + dt, x + dt * k3)
        return x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)


# ============================================================
# 3. DYNAMICAL SYSTEM PRESETS
# ============================================================
class Presets:
    @staticmethod
    def spiral(model):
        with torch.no_grad():
            W = torch.zeros(model.hidden_dim, 2)
            W[0, 0] = -0.3; W[0, 1] = -1.0
            W[1, 0] = 1.0;  W[1, 1] = -0.3
            model.net[0].weight.copy_(W)
            model.net[0].bias.zero_()
            for layer in model.net[2::2]:
                if isinstance(layer, nn.Linear):
                    layer.weight.zero_(); layer.bias.zero_()

    @staticmethod
    def saddle(model):
        with torch.no_grad():
            W = torch.zeros(model.hidden_dim, 2)
            W[0, 0] = 1.0; W[1, 1] = -1.0
            model.net[0].weight.copy_(W)
            model.net[0].bias.zero_()
            for layer in model.net[2::2]:
                if isinstance(layer, nn.Linear):
                    layer.weight.zero_(); layer.bias.zero_()

    @staticmethod
    def limit_cycle(model):
        with torch.no_grad():
            W = torch.zeros(model.hidden_dim, 2)
            W[0, 1] = 1.0; W[1, 0] = -1.0; W[1, 1] = 0.3
            model.net[0].weight.copy_(W)
            model.net[0].bias.zero_()
            if model.n_layers >= 2:
                W2 = torch.zeros(model.hidden_dim, model.hidden_dim)
                W2[0, 0] = -0.2; W2[1, 1] = -0.2
                model.net[2].weight.copy_(W2)
                model.net[2].bias.zero_()
            if model.n_layers >= 3:
                W3 = torch.zeros(2, model.hidden_dim)
                W3[0, 0] = 1.0; W3[1, 1] = 1.0
                model.net[4].weight.copy_(W3)
                model.net[4].bias.zero_()

    @staticmethod
    def two_attractors(model):
        with torch.no_grad():
            for p in model.parameters():
                nn.init.zeros_(p)
            W = torch.zeros(model.hidden_dim, 2)
            W[0, 0] = 1.0; W[1, 1] = 1.0
            model.net[0].weight.copy_(W)
            if model.n_layers >= 2:
                W2 = torch.zeros(model.hidden_dim, model.hidden_dim)
                W2[2, 0] = -1.0; W2[3, 0] = -1.0; W2[4, 1] = -1.0
                model.net[2].weight.copy_(W2)
            if model.n_layers >= 3:
                W3 = torch.zeros(2, model.hidden_dim)
                W3[0, 2] = 1.0; W3[0, 3] = -1.0; W3[1, 4] = -1.0
                model.net[4].weight.copy_(W3)
            for p in model.parameters():
                p.mul_(0.8)

    @staticmethod
    def random(model):
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
            else:
                nn.init.zeros_(p)


# ============================================================
# 4. FIXED POINT FINDER
# ============================================================
class FixedPointFinder:
    @staticmethod
    def find(model, n_starts=20, max_iter=100, tol=1e-5):
        """Find fixed points via gradient descent on ||f(x)||^2."""
        results = []
        starts = np.random.uniform(-2, 2, (n_starts, 2))
        for s in starts:
            x = torch.tensor(s, dtype=torch.float32, requires_grad=True)
            opt = optim.Adam([x], lr=0.05)
            for _ in range(max_iter):
                opt.zero_grad()
                f_val = model.forward_with_bifurcation(torch.tensor(0.0), x.unsqueeze(0)).squeeze()
                loss = (f_val ** 2).sum()
                loss.backward()
                opt.step()
                if loss.item() < tol:
                    break
            if loss.item() < 1e-3:
                pt = x.detach().numpy()
                # Check if duplicate
                is_dup = False
                for r in results:
                    if np.linalg.norm(r['point'] - pt) < 0.1:
                        is_dup = True; break
                if not is_dup:
                    J = model.jacobian(torch.tensor(pt, dtype=torch.float32).unsqueeze(0))
                    eig = np.linalg.eigvals(J)
                    stable = all(e.real < 0 for e in eig)
                    results.append({'point': pt, 'eigenvalues': eig,
                                    'stable': stable, 'J': J})
        return results


# ============================================================
# 5. CANVAS CLASSES
# ============================================================
class ODECanvas(FigureCanvas):
    clicked = Signal(float, float)

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#0d1117')
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self._setup_ax()

    def _on_click(self, event):
        if event.inaxes == self.ax and event.xdata is not None:
            self.clicked.emit(event.xdata, event.ydata)

    def _setup_ax(self, is3d=False):
        self.fig.clear()
        if is3d:
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_facecolor('#0d1117')
            self.ax.tick_params(colors='#aaa', labelsize=8)
            for pane in [self.ax.xaxis.pane, self.ax.yaxis.pane, self.ax.zaxis.pane]:
                pane.fill = False; pane.set_edgecolor('#333')
        else:
            self.ax = self.fig.add_subplot(111)
            self.ax.set_facecolor('#0d1117')
            self.ax.set_xlim([-2.5, 2.5]); self.ax.set_ylim([-2.5, 2.5])
            self.ax.set_aspect('equal')
            self.ax.tick_params(colors='#aaa', labelsize=8)
            for spine in self.ax.spines.values():
                spine.set_color('#333')
            self.ax.grid(True, alpha=0.15, color='#555')

    def _compute_field(self, model, grid_size):
        x = np.linspace(-2.5, 2.5, grid_size)
        y = np.linspace(-2.5, 2.5, grid_size)
        X, Y = np.meshgrid(x, y)
        grid = np.vstack([X.ravel(), Y.ravel()]).T
        with torch.no_grad():
            dX_dY = model.forward_with_bifurcation(
                torch.tensor(0.0), torch.tensor(grid, dtype=torch.float32)).numpy()
        U = dX_dY[:, 0].reshape(X.shape)
        V = dX_dY[:, 1].reshape(X.shape)
        speed = np.sqrt(U**2 + V**2)
        return X, Y, U, V, speed

    def draw_streamlines(self, model):
        self._setup_ax()
        x = np.linspace(-2.5, 2.5, 25); y = np.linspace(-2.5, 2.5, 25)
        X, Y = np.meshgrid(x, y)
        grid = np.vstack([X.ravel(), Y.ravel()]).T
        with torch.no_grad():
            dX_dY = model.forward_with_bifurcation(
                torch.tensor(0.0), torch.tensor(grid, dtype=torch.float32)).numpy()
        U = dX_dY[:, 0].reshape(X.shape); V = dX_dY[:, 1].reshape(X.shape)
        speed = np.sqrt(U**2 + V**2)
        self.ax.streamplot(X, Y, U, V, color=speed, cmap='plasma',
                           density=1.4, linewidth=1.2, arrowsize=0.8)
        self.ax.set_title("Streamlines", color='white', fontsize=12, pad=10)
        self.draw()

    def draw_vector_field(self, model):
        self._setup_ax()
        X, Y, U, V, speed = self._compute_field(model, 18)
        self.ax.quiver(X, Y, U/(speed+1e-8), V/(speed+1e-8), speed,
                       cmap='plasma', alpha=0.9, scale=25, width=0.004)
        self.ax.set_title("Vector Field", color='white', fontsize=12, pad=10)
        self.draw()

    def draw_divergence(self, model):
        self._setup_ax()
        gs = 50
        x = np.linspace(-2.5, 2.5, gs); y = np.linspace(-2.5, 2.5, gs)
        X, Y = np.meshgrid(x, y)
        div = np.zeros_like(X)
        for i in range(gs):
            for j in range(gs):
                pt = torch.tensor([[X[i,j], Y[i,j]]], dtype=torch.float32)
                div[i, j] = model.divergence(pt)
        vmax = max(np.percentile(np.abs(div), 95), 0.1)
        im = self.ax.imshow(div, extent=[-2.5, 2.5, -2.5, 2.5],
                            origin='lower', cmap='RdBu_r',
                            vmin=-vmax, vmax=vmax, alpha=0.85)
        self.fig.colorbar(im, ax=self.ax, label='∇·f', shrink=0.8)
        self.ax.set_title("Divergence  ∇·f", color='white', fontsize=12, pad=10)
        self.draw()

    def draw_density(self, model, trajectories, step):
        self._setup_ax()
        X, Y, U, V, speed = self._compute_field(model, 14)
        self.ax.quiver(X, Y, U/(speed+1e-8), V/(speed+1e-8), speed,
                       cmap='Greys', alpha=0.25, scale=30, width=0.003)
        n = trajectories.shape[0]
        side = int(np.sqrt(n))
        if side * side == n:
            current = trajectories[:, step, :]
            init = trajectories[:, 0, :]
            self.ax.scatter(init[:, 0], init[:, 1], c='#444', s=20,
                           alpha=0.4, edgecolors='#666', zorder=1)
            cmap = plt.get_cmap('viridis')
            colors = [cmap(i/max(n-1,1)) for i in range(n)]
            self.ax.scatter(current[:, 0], current[:, 1], c=colors, s=60,
                           edgecolors='white', linewidths=0.8, zorder=2)
            for i in range(side):
                for j in range(side):
                    idx = i*side + j
                    if j < side-1:
                        p1, p2 = current[idx], current[idx+1]
                        self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                                    color='white', alpha=0.3, linewidth=0.8, zorder=1)
                    if i < side-1:
                        p1, p2 = current[idx], current[idx+side]
                        self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                                    color='white', alpha=0.3, linewidth=0.8, zorder=1)
        self.ax.set_title("Density Evolution", color='white', fontsize=12, pad=10)
        self.draw()

    def draw_particles(self, model, particles, step):
        self._setup_ax()
        x = np.linspace(-2.5, 2.5, 20); y = np.linspace(-2.5, 2.5, 20)
        X, Y = np.meshgrid(x, y)
        grid = np.vstack([X.ravel(), Y.ravel()]).T
        with torch.no_grad():
            dX_dY = model.forward_with_bifurcation(
                torch.tensor(0.0), torch.tensor(grid, dtype=torch.float32)).numpy()
        U = dX_dY[:, 0].reshape(X.shape); V = dX_dY[:, 1].reshape(X.shape)
        self.ax.streamplot(X, Y, U, V, color='#333', density=1.0,
                           linewidth=0.6, arrowsize=0.5)
        n_p = particles.shape[0]
        trail_len = min(30, step+1)
        for i in range(n_p):
            color = plt.get_cmap('hsv')(i/n_p)
            trail = particles[i, max(0, step-trail_len+1):step+1, :]
            if len(trail) > 1:
                for k in range(len(trail)-1):
                    alpha = 0.1 + 0.6*(k/len(trail))
                    self.ax.plot(trail[k:k+2, 0], trail[k:k+2, 1],
                                color=color, alpha=alpha, linewidth=1.0)
            head = particles[i, step, :]
            self.ax.scatter([head[0]], [head[1]], color=color, s=25,
                           edgecolors='white', linewidths=0.5, zorder=3)
        self.ax.set_title(f"Particle Flow ({n_p})", color='white', fontsize=12, pad=10)
        self.draw()

    def draw_trajectories(self, trajectories, step, velocity_color=False):
        self._setup_ax()
        n = trajectories.shape[0]
        for i in range(n):
            path = trajectories[i, :step+1, :]
            if len(path) < 2:
                continue
            if velocity_color:
                # Color by speed
                speeds = np.linalg.norm(np.diff(path, axis=0), axis=1)
                speeds = np.concatenate([[0], speeds])
                points = path.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap='plasma',
                                    norm=Normalize(vmin=0, vmax=speeds.max()+1e-6),
                                    alpha=0.85, linewidths=1.5)
                lc.set_array(speeds[:-1])
                self.ax.add_collection(lc)
            else:
                hue = plt.get_cmap('hsv')(i/max(n,1))[:3]
                points = path.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, colors=[hue+(0.8,)]*len(segments),
                                    linewidths=1.5, alpha=0.85)
                self.ax.add_collection(lc)
            # Glowing head
            head = path[-1]
            hue = plt.get_cmap('hsv')(i/max(n,1))[:3]
            for r, a in [(12, 0.15), (8, 0.3), (5, 0.6)]:
                self.ax.scatter([head[0]], [head[1]], color=hue+(1.0,),
                               s=r*10, alpha=a, edgecolors='none', zorder=3)
            self.ax.scatter([head[0]], [head[1]], color='white', s=20,
                           edgecolors=hue+(1.0,), linewidths=1.2, zorder=4)
        self.ax.set_title("Trajectories", color='white', fontsize=12, pad=10)
        self.draw()

    def draw_basin(self, model):
        self._setup_ax()
        gs = 40
        x = np.linspace(-2.5, 2.5, gs); y = np.linspace(-2.5, 2.5, gs)
        X, Y = np.meshgrid(x, y)
        grid = np.vstack([X.ravel(), Y.ravel()]).T
        pts = torch.tensor(grid, dtype=torch.float32)
        f = lambda t, x: model.forward_with_bifurcation(t, x)
        with torch.no_grad():
            for _ in range(200):
                pts = Solvers.rk4(f, pts, torch.tensor(0.0), 0.05)
                pts = torch.clamp(pts, -4.0, 4.0)
        final = pts.numpy()
        unique = np.unique(np.round(final, 1), axis=0)
        if len(unique) < 2:
            unique = np.array([[0.0, 0.0], [1.0, 1.0]])
        from scipy.spatial import cKDTree
        tree = cKDTree(unique)
        _, labels = tree.query(final)
        basin = labels.reshape(X.shape)
        self.ax.imshow(basin, extent=[-2.5, 2.5, -2.5, 2.5],
                       origin='lower', cmap='tab10', alpha=0.7,
                       vmin=0, vmax=max(len(unique)-1, 1))
        grid2 = np.vstack([X.ravel(), Y.ravel()]).T
        with torch.no_grad():
            dX_dY = model.forward_with_bifurcation(
                torch.tensor(0.0), torch.tensor(grid2, dtype=torch.float32)).numpy()
        U = dX_dY[:, 0].reshape(X.shape); V = dX_dY[:, 1].reshape(X.shape)
        self.ax.streamplot(X, Y, U, V, color='white', density=1.2,
                           linewidth=0.6, arrowsize=0.6, alpha=0.5)
        self.ax.set_title("Basins of Attraction", color='white', fontsize=12, pad=10)
        self.draw()

    def draw_poincare(self, trajectories, period_steps):
        """Poincaré section — sample trajectory every N steps."""
        self._setup_ax()
        # Background field
        x = np.linspace(-2.5, 2.5, 15); y = np.linspace(-2.5, 2.5, 15)
        X, Y = np.meshgrid(x, y)
        grid = np.vstack([X.ravel(), Y.ravel()]).T
        with torch.no_grad():
            dX_dY = trajectories.__class__(grid)  # placeholder
        # Just plot sampled points
        n = trajectories.shape[0]
        for i in range(n):
            color = plt.get_cmap('hsv')(i/max(n,1))
            path = trajectories[i]
            samples = path[::max(1, period_steps)]
            self.ax.scatter(samples[:, 0], samples[:, 1], c=[color]*len(samples),
                           s=30, edgecolors='white', linewidths=0.5,
                           alpha=0.8, zorder=2)
            # Connect sequential samples
            if len(samples) > 1:
                for k in range(len(samples)-1):
                    self.ax.plot([samples[k,0], samples[k+1,0]],
                                [samples[k,1], samples[k+1,1]],
                                color=color, alpha=0.3, linewidth=0.8, zorder=1)
        self.ax.set_title(f"Poincaré Section (period={period_steps})",
                          color='white', fontsize=12, pad=10)
        self.draw()

    def draw_with_fixed_points(self, model, trajectories, step, fixed_points):
        """Draw trajectories with fixed points overlaid."""
        self.draw_trajectories(trajectories, step)
        for fp in fixed_points:
            pt = fp['point']
            color = '#00ff88' if fp['stable'] else '#ff4444'
            # Marker: star for stable, X for unstable
            marker = '*' if fp['stable'] else 'X'
            self.ax.scatter([pt[0]], [pt[1]], color=color, s=200,
                           marker=marker, edgecolors='white',
                           linewidths=1.5, zorder=10)
        self.draw()


class LossCanvas(FigureCanvas):
    def __init__(self, parent=None, width=4, height=1.5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#0d1117')
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

    def update(self, losses):
        self.ax.clear()
        self.ax.set_facecolor('#0d1117')
        self.ax.tick_params(colors='#aaa', labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color('#333')
        if losses:
            self.ax.semilogy(losses, color='#00ff88', linewidth=1.5)
            self.ax.fill_between(range(len(losses)), losses,
                                color='#00ff88', alpha=0.15)
            self.ax.set_title(f"Loss: {losses[-1]:.3e}",
                              color='white', fontsize=10)
        else:
            self.ax.set_title("Training Loss", color='white', fontsize=10)
        self.ax.set_xlabel("Step", color='white', fontsize=9)
        self.ax.set_ylabel("MSE", color='white', fontsize=9)
        self.ax.grid(True, alpha=0.15, color='#555')
        self.draw()


class BifurcationCanvas(FigureCanvas):
    def __init__(self, parent=None, width=4, height=3, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#0d1117')
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot(self, lambdas, fixed_points_by_lambda):
        self.ax.clear()
        self.ax.set_facecolor('#0d1117')
        self.ax.tick_params(colors='#aaa', labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color('#333')

        stable_x, stable_y = [], []
        unstable_x, unstable_y = [], []
        for lam, fps in zip(lambdas, fixed_points_by_lambda):
            for fp in fps:
                if fp['stable']:
                    stable_x.append(lam); stable_y.append(fp['point'][1])
                else:
                    unstable_x.append(lam); unstable_y.append(fp['point'][1])

        if stable_x:
            self.ax.scatter(stable_x, stable_y, color='#00ff88',
                           s=15, label='Stable', alpha=0.8)
        if unstable_x:
            self.ax.scatter(unstable_x, unstable_y, color='#ff4444',
                           s=15, label='Unstable', alpha=0.8, marker='x')

        self.ax.set_xlabel('λ (bifurcation)', color='white', fontsize=9)
        self.ax.set_ylabel('y* (fixed point)', color='white', fontsize=9)
        self.ax.set_title('Bifurcation Diagram', color='white', fontsize=11)
        self.ax.legend(loc='best', fontsize=8)
        self.ax.grid(True, alpha=0.15, color='#555')
        self.draw()


# ============================================================
# 6. TRAINING WORKER
# ============================================================
class TrainingWorker(QObject):
    loss_update = Signal(list)
    finished = Signal()
    progress = Signal(int, int)

    def __init__(self, model, target_func, steps=500, lr=1e-2):
        super().__init__()
        self.model = model
        self.target_func = target_func
        self.steps = steps
        self.lr = lr
        self.losses = []
        self.running = False
        self.current_step = 0

    def start(self):
        self.running = True
        self.current_step = 0
        self.losses = []
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def step(self):
        if not self.running or self.current_step >= self.steps:
            self.running = False
            self.finished.emit()
            return
        t_vals = torch.linspace(0, 5, 50)
        target = torch.stack([self.target_func(t) for t in t_vals])
        x = target[0:1].clone().detach()
        pred = [x.squeeze().detach()]
        dt = 5.0 / 49
        f = lambda t, x: self.model.forward_with_bifurcation(t, x)
        with torch.no_grad():
            for t in t_vals[1:]:
                x = Solvers.rk4(f, x, t-dt, dt)
                pred.append(x.squeeze().detach())
        pred = torch.stack(pred)
        loss = nn.MSELoss()(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())
        self.current_step += 1
        self.loss_update.emit(self.losses)
        self.progress.emit(self.current_step, self.steps)


# ============================================================
# 7. MODEL EDITOR DIALOG
# ============================================================
class ModelEditorDialog(QDialog):
    def __init__(self, current_config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Architecture")
        self.resize(350, 250)
        self.config = current_config.copy()

        layout = QFormLayout(self)

        self.hidden_spin = QSpinBox()
        self.hidden_spin.setRange(8, 256)
        self.hidden_spin.setValue(current_config['hidden_dim'])
        self.hidden_spin.setSingleStep(8)
        layout.addRow("Hidden dimension:", self.hidden_spin)

        self.layers_spin = QSpinBox()
        self.layers_spin.setRange(1, 5)
        self.layers_spin.setValue(current_config['n_layers'])
        layout.addRow("Hidden layers:", self.layers_spin)

        self.act_combo = QComboBox()
        self.act_combo.addItems(['tanh', 'relu', 'gelu', 'silu'])
        self.act_combo.setCurrentText(current_config['activation'])
        layout.addRow("Activation:", self.act_combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_config(self):
        return {
            'hidden_dim': self.hidden_spin.value(),
            'n_layers': self.layers_spin.value(),
            'activation': self.act_combo.currentText(),
        }


# ============================================================
# 8. COLLAPSIBLE GROUP BOX
# ============================================================
class CollapsibleBox(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.layout_main = QVBoxLayout(self)
        self.layout_main.setContentsMargins(0, 0, 0, 0)
        self.layout_main.setSpacing(2)

        self.toggle_btn = QToolButton()
        self.toggle_btn.setText(title)
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(True)
        self.toggle_btn.setStyleSheet("""
            QToolButton {
                border: 1px solid #444;
                border-radius: 4px;
                padding: 6px;
                background-color: #2a3a4a;
                color: #00ff88;
                font-weight: bold;
                text-align: left;
            }
            QToolButton:hover { background-color: #3a4a5a; }
        """)
        self.toggle_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_btn.setArrowType(Qt.DownArrow)
        self.toggle_btn.toggled.connect(self._toggle)
        self.layout_main.addWidget(self.toggle_btn)

        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(8, 4, 8, 4)
        self.layout_main.addWidget(self.content)

    def _toggle(self, checked):
        self.content.setVisible(checked)
        self.toggle_btn.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

    def addWidget(self, w):
        self.content_layout.addWidget(w)

    def addLayout(self, l):
        self.content_layout.addLayout(l)


# ============================================================
# 9. MAIN WINDOW
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural ODE Visualizer — Professional Edition")
        self.resize(1500, 900)

        # --- Model Config ---
        self.model_config = {'hidden_dim': 64, 'n_layers': 2, 'activation': 'tanh'}
        self.model = NeuralODEFunc(**self.model_config)
        Presets.spiral(self.model)

        # --- Simulation State ---
        self.dt = 0.05
        self.max_time = 8.0
        self.current_time = 0.0
        self.init_states = self._generate_ring(20)
        self.trajectories = None
        self.particles = None
        self.solver_name = "RK4"
        self.viz_mode = "Streamlines"
        self.stroboscopic = False
        self.reverse_time = False
        self.bifurcation = 1.0
        self.velocity_color = False
        self.show_fixed_points = False
        self.fixed_points = []
        self.comparison_mode = False

        # --- Training ---
        self.training_worker = None
        self.target_func = self._target_spiral

        # --- Recording ---
        self.recording = False
        self.recorded_frames = []

        self._compute_all()
        self._build_ui()
        self._setup_shortcuts()

        # --- Timers ---
        self.timer = QTimer()
        self.timer.timeout.connect(self._animation_step)
        self.timer.start(50)
        self.train_timer = QTimer()
        self.train_timer.timeout.connect(self._train_step)

        self._refresh_viz()
        self._update_info()
        self._update_stats()

    # ---------- Initial Conditions ----------
    def _generate_ring(self, n, radius=1.8):
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        return np.stack([radius*np.cos(angles), radius*np.sin(angles)], axis=1)

    def _generate_grid(self, side=7):
        x = np.linspace(-1.8, 1.8, side); y = np.linspace(-1.8, 1.8, side)
        X, Y = np.meshgrid(x, y)
        return np.vstack([X.ravel(), Y.ravel()]).T

    def _generate_particles(self, n=400):
        return np.random.uniform(-2.2, 2.2, (n, 2))

    # ---------- Simulation ----------
    def _solve_step(self, x, t, dt):
        f = lambda t, x: self.model.forward_with_bifurcation(t, x)
        if self.reverse_time:
            f = lambda t, x: -self.model.forward_with_bifurcation(t, x)
        if self.solver_name == "Euler":
            return Solvers.euler(f, x, t, dt)
        elif self.solver_name == "Midpoint":
            return Solvers.midpoint(f, x, t, dt)
        return Solvers.rk4(f, x, t, dt)

    def _compute_trajectories(self):
        steps = int(self.max_time / self.dt)
        x = torch.tensor(self.init_states, dtype=torch.float32)
        traj = [x.numpy().copy()]
        with torch.no_grad():
            for i in range(steps):
                x = self._solve_step(x, i*self.dt, self.dt)
                x = torch.clamp(x, -4.0, 4.0)
                traj.append(x.numpy().copy())
        self.trajectories = np.array(traj).transpose(1, 0, 2)

    def _compute_particles(self):
        steps = int(self.max_time / self.dt)
        x = torch.tensor(self.particles, dtype=torch.float32)
        traj = [x.numpy().copy()]
        with torch.no_grad():
            for i in range(steps):
                x = self._solve_step(x, i*self.dt, self.dt)
                x = torch.clamp(x, -4.0, 4.0)
                traj.append(x.numpy().copy())
        self.particle_traj = np.array(traj).transpose(1, 0, 2)

    def _compute_all(self):
        self._compute_trajectories()
        if self.particles is None:
            self.particles = self._generate_particles(400)
        self._compute_particles()
        self.current_time = 0.0

    def _reset_sim(self):
        self.current_time = 0.0
        self._compute_all()
        if self.show_fixed_points:
            self.fixed_points = FixedPointFinder.find(self.model)
        self._refresh_viz()
        self._update_stats()

    # ---------- UI Construction ----------
    def _build_ui(self):
        # --- Toolbar ---
        toolbar = QToolBar("Main")
        toolbar.setIconSize(QSize(20, 20))
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.act_play = QAction("▶ Play/Pause", self)
        self.actPlay.setShortcut(QKeySequence("Space"))
        self.actPlay.triggered.connect(self._toggle_play)
        toolbar.addAction(self.actPlay)

        self.actReset = QAction("⟲ Reset", self)
        self.actReset.setShortcut(QKeySequence("R"))
        self.actReset.triggered.connect(self._reset_sim)
        toolbar.addAction(self.actReset)

        toolbar.addSeparator()

        self.actModel = QAction("⚙ Model", self)
        self.actModel.triggered.connect(self._edit_model)
        toolbar.addAction(self.actModel)

        self.actFindFP = QAction("✱ Find Fixed Points", self)
        self.actFindFP.triggered.connect(self._find_fixed_points)
        toolbar.addAction(self.actFindFP)

        self.actCompare = QAction("⇔ Compare", self)
        self.actCompare.setCheckable(True)
        self.actCompare.triggered.connect(self._toggle_comparison)
        toolbar.addAction(self.actCompare)

        self.actBifur = QAction("📈 Bifurcation", self)
        self.actBifur.triggered.connect(self._show_bifurcation)
        toolbar.addAction(self.actBifur)

        toolbar.addSeparator()

        self.actRecord = QAction("⏺ Record GIF", self)
        self.actRecord.setCheckable(True)
        self.actRecord.triggered.connect(self._toggle_recording)
        toolbar.addAction(self.actRecord)

        self.actSave = QAction("💾 Save PNG", self)
        self.actSave.triggered.connect(self._save_png)
        toolbar.addAction(self.actSave)

        # --- Stats Bar ---
        self.stats_bar = QFrame()
        self.stats_bar.setStyleSheet("""
            QFrame { background-color: #1a1a25; border-bottom: 1px solid #333; }
            QLabel { color: #aaa; padding: 4px 10px; font-size: 11px; }
        """)
        stats_layout = QHBoxLayout(self.stats_bar)
        stats_layout.setContentsMargins(5, 2, 5, 2)
        self.stat_time = QLabel("t = 0.00")
        self.stat_speed = QLabel("avg |f| = 0.00")
        self.stat_div = QLabel("∇·f(0) = 0.00")
        self.stat_params = QLabel("params = 0")
        self.stat_traj = QLabel("trajectories = 0")
        for lbl in [self.stat_time, self.stat_speed, self.stat_div,
                    self.stat_params, self.stat_traj]:
            stats_layout.addWidget(lbl)
            stats_layout.addWidget(self._separator())
        stats_layout.addStretch()

        # --- Central Widget ---
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.stats_bar)

        main_splitter = QSplitter(Qt.Horizontal)

        # Left dock: Controls
        left_dock = QDockWidget("Controls", self)
        left_dock.setFeatures(QDockWidget.DockWidgetMovable |
                              QDockWidget.DockWidgetFloatable)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_content = QWidget()
        self.left_layout = QVBoxLayout(left_content)
        self.left_layout.setSpacing(4)
        self._build_controls()
        left_scroll.setWidget(left_content)
        left_dock.setWidget(left_scroll)
        self.addDockWidget(Qt.LeftDockWidgetArea, left_dock)

        # Center: Canvas area
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas_splitter = QSplitter(Qt.Horizontal)
        self.canvas = ODECanvas(self, width=8, height=5)
        self.canvas.clicked.connect(self._add_trajectory_at)
        self.canvas_splitter.addWidget(self.canvas)

        self.canvas2 = ODECanvas(self, width=8, height=5)
        self.canvas2.setVisible(False)
        self.canvas_splitter.addWidget(self.canvas2)

        self.canvas_splitter.setStretchFactor(0, 1)
        self.canvas_splitter.setStretchFactor(1, 1)
        center_layout.addWidget(self.canvas_splitter, stretch=3)

        self.loss_canvas = LossCanvas(self, width=8, height=1.5)
        center_layout.addWidget(self.loss_canvas, stretch=1)

        main_splitter.addWidget(center_widget)

        # Right dock: Info
        right_dock = QDockWidget("Info & Analysis", self)
        right_dock.setFeatures(QDockWidget.DockWidgetMovable |
                               QDockWidget.DockWidgetFloatable)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.NoFrame)
        right_content = QWidget()
        self.right_layout = QVBoxLayout(right_content)
        self._build_info_panels()
        right_scroll.setWidget(right_content)
        right_dock.setWidget(right_scroll)
        self.addDockWidget(Qt.RightDockWidgetArea, right_dock)

        root.addWidget(main_splitter)

        # --- Status Bar with Progress ---
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.status.addPermanentWidget(self.progress_bar)
        self.status.showMessage("Ready | Space=play/pause | R=reset | 1-7=viz modes")

    def _separator(self):
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: #333;")
        return sep

    def _build_controls(self):
        # --- Visualization ---
        viz_box = CollapsibleBox("🎨 Visualization")
        self.viz_combo = QComboBox()
        self.viz_combo.addItems(["Streamlines", "Vector Field", "Divergence Heatmap",
                                 "Density Evolution", "Particle Flow", "Trajectories",
                                 "Basin of Attraction", "Poincaré Section"])
        self.viz_combo.setToolTip("Choose visualization mode (shortcuts 1-8)")
        self.viz_combo.currentTextChanged.connect(self._change_viz_mode)
        viz_box.addWidget(self.viz_combo)

        self.chk_vel_color = QCheckBox("Color by velocity magnitude")
        self.chk_vel_color.setToolTip("Color trajectories by speed (plasma colormap)")
        self.chk_vel_color.toggled.connect(self._toggle_velocity_color)
        viz_box.addWidget(self.chk_vel_color)

        self.chk_fp_overlay = QCheckBox("Overlay fixed points")
        self.chk_fp_overlay.setToolTip("Show equilibria: green *=stable, red X=unstable")
        self.chk_fp_overlay.toggled.connect(self._toggle_fp_overlay)
        viz_box.addWidget(self.chk_fp_overlay)
        self.left_layout.addWidget(viz_box)

        # --- Animation ---
        anim_box = CollapsibleBox("⏯ Animation")
        btn_row = QHBoxLayout()
        self.btn_play = QPushButton("⏸ Pause")
        self.btn_play.setToolTip("Play/Pause animation (Space)")
        self.btn_play.clicked.connect(self._toggle_play)
        btn_row.addWidget(self.btn_play)
        btn_reset = QPushButton("⟲ Reset")
        btn_reset.setToolTip("Reset trajectories (R)")
        btn_reset.clicked.connect(self._reset_sim)
        btn_row.addWidget(btn_reset)
        anim_box.addLayout(btn_row)

        anim_box.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 20); self.speed_slider.setValue(5)
        self.speed_slider.setToolTip("Animation speed multiplier")
        anim_box.addWidget(self.speed_slider)

        self.chk_reverse = QCheckBox("⏪ Time Reversal")
        self.chk_reverse.setToolTip("Integrate backward in time")
        self.chk_reverse.toggled.connect(self._toggle_reverse)
        anim_box.addWidget(self.chk_reverse)

        anim_box.addWidget(QLabel("Poincaré period:"))
        self.poincare_slider = QSlider(Qt.Horizontal)
        self.poincare_slider.setRange(5, 100); self.poincare_slider.setValue(20)
        self.poincare_slider.setToolTip("Sample interval for Poincaré section")
        anim_box.addWidget(self.poincare_slider)
        self.left_layout.addWidget(anim_box)

        # --- Initial Conditions ---
        ic_box = CollapsibleBox("📍 Initial Conditions")
        btn_row = QHBoxLayout()
        for name, fn in [("Ring", lambda: self._set_init(self._generate_ring(25))),
                         ("Grid 7×7", lambda: self._set_init(self._generate_grid(7))),
                         ("Random", lambda: self._set_init(np.random.uniform(-2,2,(25,2))))]:
            b = QPushButton(name)
            b.clicked.connect(fn)
            btn_row.addWidget(b)
        ic_box.addLayout(btn_row)
        lbl = QLabel("💡 Click on canvas to add points")
        lbl.setStyleSheet("color: #888; font-size: 10px;")
        ic_box.addWidget(lbl)
        self.left_layout.addWidget(ic_box)

        # --- Dynamics ---
        dyn_box = CollapsibleBox("⚙️ Dynamics")
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Spiral", "Saddle", "Limit Cycle",
                                    "Two Attractors", "Random"])
        self.preset_combo.setToolTip("Preset vector fields")
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        dyn_box.addWidget(self.preset_combo)

        self.solver_combo = QComboBox()
        self.solver_combo.addItems(["Euler", "Midpoint", "RK4"])
        self.solver_combo.setCurrentText("RK4")
        self.solver_combo.setToolTip("Numerical integration method")
        self.solver_combo.currentTextChanged.connect(self._change_solver)
        dyn_box.addWidget(QLabel("Solver:"))
        dyn_box.addWidget(self.solver_combo)

        dyn_box.addWidget(QLabel("Bifurcation λ:"))
        self.bifur_slider = QSlider(Qt.Horizontal)
        self.bifur_slider.setRange(-200, 200); self.bifur_slider.setValue(100)
        self.bifur_slider.setToolTip("Scale nonlinear term — morphs dynamics")
        self.bifur_slider.valueChanged.connect(self._change_bifurcation)
        dyn_box.addWidget(self.bifur_slider)
        self.bifur_label = QLabel("λ = 1.00")
        self.bifur_label.setAlignment(Qt.AlignCenter)
        self.bifur_label.setStyleSheet("color: #ffaa00; font-weight: bold;")
        dyn_box.addWidget(self.bifur_label)
        self.left_layout.addWidget(dyn_box)

        # --- Training ---
        train_box = CollapsibleBox("🎯 Training")
        self.target_combo = QComboBox()
        self.target_combo.addItems(["Spiral", "Figure-8", "Circle", "Lissajous"])
        self.target_combo.currentTextChanged.connect(self._change_target)
        train_box.addWidget(QLabel("Target:"))
        train_box.addWidget(self.target_combo)

        train_box.addWidget(QLabel("Learning rate:"))
        self.lr_slider = QSlider(Qt.Horizontal)
        self.lr_slider.setRange(-5, -1); self.lr_slider.setValue(-2)
        self.lr_label = QLabel("1e-2")
        self.lr_slider.valueChanged.connect(lambda v: self.lr_label.setText(f"1e{v}"))
        train_box.addWidget(self.lr_slider)
        train_box.addWidget(self.lr_label)

        self.btn_train = QPushButton("▶ Start Training")
        self.btn_train.clicked.connect(self._toggle_training)
        train_box.addWidget(self.btn_train)

        btn_reset_model = QPushButton("Reset Weights")
        btn_reset_model.clicked.connect(self._reset_model)
        train_box.addWidget(btn_reset_model)
        self.left_layout.addWidget(train_box)

        # --- Export ---
        export_box = CollapsibleBox("💾 Export")
        btn_png = QPushButton("📷 Save PNG")
        btn_png.clicked.connect(self._save_png)
        export_box.addWidget(btn_png)
        btn_csv = QPushButton("📊 Export CSV")
        btn_csv.clicked.connect(self._save_csv)
        export_box.addWidget(btn_csv)
        btn_save = QPushButton("💾 Save Model")
        btn_save.clicked.connect(self._save_model)
        export_box.addWidget(btn_save)
        btn_load = QPushButton("📂 Load Model")
        btn_load.clicked.connect(self._load_model)
        export_box.addWidget(btn_load)
        self.left_layout.addWidget(export_box)

        self.left_layout.addStretch()

    def _build_info_panels(self):
        # --- Model Info ---
        info_box = CollapsibleBox("📋 Model Info")
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(180)
        self.info_text.setStyleSheet("font-size: 10px; font-family: monospace;")
        info_box.addWidget(self.info_text)
        self.right_layout.addWidget(info_box)

        # --- Jacobian ---
        jac_box = CollapsibleBox("🔬 Local Stability")
        self.jacobian_label = QLabel("Analyzing...")
        self.jacobian_label.setWordWrap(True)
        self.jacobian_label.setStyleSheet("font-size: 10px;")
        jac_box.addWidget(self.jacobian_label)
        btn_jac = QPushButton("Analyze at Origin")
        btn_jac.clicked.connect(lambda: self._analyze_jacobian(np.array([0.0, 0.0])))
        jac_box.addWidget(btn_jac)
        self.right_layout.addWidget(jac_box)

        # --- Fixed Points ---
        fp_box = CollapsibleBox("✱ Fixed Points")
        self.fp_text = QTextEdit()
        self.fp_text.setReadOnly(True)
        self.fp_text.setMaximumHeight(150)
        self.fp_text.setStyleSheet("font-size: 10px; font-family: monospace;")
        self.fp_text.setHtml("<i>Click 'Find Fixed Points' in toolbar</i>")
        fp_box.addWidget(self.fp_text)
        btn_find = QPushButton("🔍 Find Fixed Points")
        btn_find.clicked.connect(self._find_fixed_points)
        fp_box.addWidget(btn_find)
        self.right_layout.addWidget(fp_box)

        # --- Trajectory Stats ---
        traj_box = CollapsibleBox("📊 Trajectory Stats")
        self.traj_stats_label = QLabel("No data")
        self.traj_stats_label.setWordWrap(True)
        self.traj_stats_label.setStyleSheet("font-size: 10px;")
        traj_box.addWidget(self.traj_stats_label)
        self.right_layout.addWidget(traj_box)

        # --- Keyboard Shortcuts ---
        keys_box = CollapsibleBox("⌨️ Shortcuts")
        keys_text = QTextEdit()
        keys_text.setReadOnly(True)
        keys_text.setMaximumHeight(180)
        keys_text.setHtml("""
        <b>Space</b> — Play/Pause<br>
        <b>R</b> — Reset<br>
        <b>1-8</b> — Viz modes<br>
        <b>←/→</b> — Time step<br>
        <b>C</b> — Toggle compare<br>
        <b>F</b> — Find fixed points<br>
        <b>G</b> — Record GIF<br>
        <b>Click canvas</b> — Add point
        """)
        keys_text.setStyleSheet("font-size: 10px;")
        keys_box.addWidget(keys_text)
        self.right_layout.addWidget(keys_box)

        self.right_layout.addStretch()

    def _setup_shortcuts(self):
        for i, mode in enumerate(["Streamlines", "Vector Field", "Divergence Heatmap",
                                   "Density Evolution", "Particle Flow", "Trajectories",
                                   "Basin of Attraction", "Poincaré Section"], 1):
            sc = QShortcut(QKeySequence(str(i)), self)
            sc.activated.connect(lambda m=mode: self.viz_combo.setCurrentText(m))

        QShortcut(QKeySequence("Left"), self).activated.connect(self._step_back)
        QShortcut(QKeySequence("Right"), self).activated.connect(self._step_forward)
        QShortcut(QKeySequence("C"), self).activated.connect(self._toggle_comparison)
        QShortcut(QKeySequence("F"), self).activated.connect(self._find_fixed_points)
        QShortcut(QKeySequence("G"), self).activated.connect(self._toggle_recording)

    # ---------- Callbacks ----------
    def _set_init(self, states):
        self.init_states = states
        self._reset_sim()

    def _add_trajectory_at(self, x, y):
        new_pt = np.array([[x, y]])
        self.init_states = np.vstack([self.init_states, new_pt])
        self._compute_trajectories()
        self._refresh_viz()
        self._update_stats()

    def _change_viz_mode(self, mode):
        self.viz_mode = mode
        self._refresh_viz()

    def _toggle_velocity_color(self, checked):
        self.velocity_color = checked
        self._refresh_viz()

    def _toggle_fp_overlay(self, checked):
        self.show_fixed_points = checked
        if checked and not self.fixed_points:
            self.fixed_points = FixedPointFinder.find(self.model)
            self._update_fp_display()
        self._refresh_viz()

    def _toggle_reverse(self, checked):
        self.reverse_time = checked
        self._reset_sim()

    def _change_bifurcation(self, v):
        self.bifurcation = v / 100.0
        self.model.bifurcation.fill_(self.bifurcation)
        self.bifur_label.setText(f"λ = {self.bifurcation:+.2f}")
        self._reset_sim()

    def _apply_preset(self, name):
        mapping = {"Spiral": Presets.spiral, "Saddle": Presets.saddle,
                   "Limit Cycle": Presets.limit_cycle,
                   "Two Attractors": Presets.two_attractors,
                   "Random": Presets.random}
        mapping[name](self.model)
        self._reset_sim()
        self._update_info()

    def _change_solver(self, name):
        self.solver_name = name
        self._reset_sim()

    def _toggle_play(self):
        if self.timer.isActive():
            self.timer.stop(); self.btn_play.setText("▶ Play")
        else:
            self.timer.start(50); self.btn_play.setText("⏸ Pause")

    def _step_back(self):
        self.current_time = max(0, self.current_time - self.dt * 5)
        self._refresh_viz()

    def _step_forward(self):
        self.current_time = min(self.max_time, self.current_time + self.dt * 5)
        self._refresh_viz()

    def _edit_model(self):
        dlg = ModelEditorDialog(self.model_config, self)
        if dlg.exec():
            new_cfg = dlg.get_config()
            if new_cfg != self.model_config:
                self.model_config = new_cfg
                self.model = NeuralODEFunc(**new_cfg)
                Presets.spiral(self.model)
                self.model.bifurcation.fill_(self.bifurcation)
                self._reset_sim()
                self._update_info()

    def _find_fixed_points(self):
        self.status.showMessage("Finding fixed points...")
        QApplication.processEvents()
        self.fixed_points = FixedPointFinder.find(self.model, n_starts=30)
        self.show_fixed_points = True
        self.chk_fp_overlay.setChecked(True)
        self._update_fp_display()
        self._refresh_viz()
        self.status.showMessage(f"Found {len(self.fixed_points)} fixed point(s)", 3000)

    def _update_fp_display(self):
        if not self.fixed_points:
            self.fp_text.setHtml("<i>No fixed points found</i>")
            return
        html = ""
        for i, fp in enumerate(self.fixed_points):
            pt = fp['point']
            eig = fp['eigenvalues']
            kind = "STABLE" if fp['stable'] else "UNSTABLE"
            color = "#00ff88" if fp['stable'] else "#ff4444"
            html += (f"<b style='color:{color}'>FP {i+1} ({kind})</b><br>"
                    f"  pos = ({pt[0]:+.3f}, {pt[1]:+.3f})<br>"
                    f"  λ₁ = {eig[0].real:+.3f}"
                    f"{'  ' if abs(eig[0].imag)<1e-3 else f' {eig[0].imag:+.3f}i'}<br>"
                    f"  λ₂ = {eig[1].real:+.3f}"
                    f"{'  ' if abs(eig[1].imag)<1e-3 else f' {eig[1].imag:+.3f}i'}<br>"
                    f"  ∇·f = {np.trace(fp['J']):+.3f}<br><br>")
        self.fp_text.setHtml(html)

    def _toggle_comparison(self, checked):
        self.comparison_mode = checked
        self.canvas2.setVisible(checked)
        if checked:
            # Canvas2 shows the current model state at start of training
            if not hasattr(self, 'comparison_model'):
                self.comparison_model = NeuralODEFunc(**self.model_config)
                Presets.spiral(self.comparison_model)
            self._draw_comparison()
        self._refresh_viz()

    def _draw_comparison(self):
        if not hasattr(self, 'comparison_model'):
            return
        step = int(self.current_time / self.dt)
        if self.viz_mode == "Trajectories":
            self.canvas2.draw_trajectories(self.trajectories, step,
                                           velocity_color=self.velocity_color)
        elif self.viz_mode == "Streamlines":
            self.canvas2.draw_streamlines(self.comparison_model)
        else:
            self.canvas2.draw_vector_field(self.comparison_model)

    def _show_bifurcation(self):
        """Compute and show bifurcation diagram."""
        self.status.showMessage("Computing bifurcation diagram...")
        QApplication.processEvents()
        lambdas = np.linspace(-2.0, 2.0, 30)
        fps_by_lam = []
        orig_bif = self.model.bifurcation.item()
        for lam in lambdas:
            self.model.bifurcation.fill_(lam)
            fps = FixedPointFinder.find(self.model, n_starts=15, max_iter=80)
            fps_by_lam.append(fps)
        self.model.bifurcation.fill_(orig_bif)

        # Show in dialog
        dlg = QDialog(self)
        dlg.setWindowTitle("Bifurcation Diagram")
        dlg.resize(600, 400)
        layout = QVBoxLayout(dlg)
        canvas = BifurcationCanvas(dlg, width=6, height=4)
        layout.addWidget(canvas)
        canvas.plot(lambdas, fps_by_lam)
        dlg.exec()
        self.status.showMessage("Bifurcation diagram complete", 3000)

    def _toggle_recording(self, checked):
        if checked:
            self.recording = True
            self.recorded_frames = []
            self.actRecord.setText("⏹ Stop Recording")
            self.status.showMessage("Recording... press G or button to stop")
        else:
            self.recording = False
            self.actRecord.setText("⏺ Record GIF")
            if self.recorded_frames:
                self._save_gif()

    def _save_gif(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save GIF", "neural_ode.gif",
                                              "GIF (*.gif)")
        if path:
            images = [Image.fromarray(frame) for frame in self.recorded_frames]
            images[0].save(path, save_all=True, append_images=images[1:],
                          duration=50, loop=0)
            self.status.showMessage(f"Saved GIF: {path} ({len(images)} frames)", 5000)

    # ---------- Visualization Refresh ----------
    def _refresh_viz(self):
        step = int(self.current_time / self.dt)
        step = min(step, self.trajectories.shape[1] - 1)

        if self.viz_mode == "Streamlines":
            self.canvas.draw_streamlines(self.model)
        elif self.viz_mode == "Vector Field":
            self.canvas.draw_vector_field(self.model)
        elif self.viz_mode == "Divergence Heatmap":
            self.canvas.draw_divergence(self.model)
        elif self.viz_mode == "Density Evolution":
            if self.trajectories.shape[0] != 49:
                self.init_states = self._generate_grid(7)
                self._compute_trajectories()
            self.canvas.draw_density(self.model, self.trajectories, step)
        elif self.viz_mode == "Particle Flow":
            self.canvas.draw_particles(self.model, self.particle_traj, step)
        elif self.viz_mode == "Trajectories":
            self.canvas.draw_trajectories(self.trajectories, step,
                                          velocity_color=self.velocity_color)
            if self.show_fixed_points and self.fixed_points:
                self.canvas.draw_with_fixed_points(self.model, self.trajectories,
                                                   step, self.fixed_points)
        elif self.viz_mode == "Basin of Attraction":
            self.canvas.draw_basin(self.model)
        elif self.viz_mode == "Poincaré Section":
            self.canvas.draw_poincare(self.trajectories,
                                      self.poincare_slider.value())

        # Recording
        if self.recording:
            self.canvas.draw()
            self.canvas.repaint()
            QApplication.processEvents()
            img = np.array(self.canvas.fig.canvas.renderer.buffer_rgba())
            self.recorded_frames.append(img[:, :, :3])

        if self.comparison_mode:
            self._draw_comparison()

    def _animation_step(self):
        step_inc = self.speed_slider.value()
        current_step = int(self.current_time / self.dt)
        max_steps = self.trajectories.shape[1] - 1
        current_step = min(current_step + step_inc, max_steps)
        self.current_time = current_step * self.dt

        if self.viz_mode in ["Density Evolution", "Particle Flow",
                              "Trajectories", "Poincaré Section"]:
            self._refresh_viz()

        if current_step >= max_steps:
            self.current_time = 0.0

        self._update_stats()
        if self.trajectories.shape[0] > 0:
            head = self.trajectories[0, current_step]
            self._analyze_jacobian(head, silent=True)

    def _analyze_jacobian(self, point, silent=False):
        x = torch.tensor(point, dtype=torch.float32).unsqueeze(0)
        J = self.model.jacobian(x)
        eig = np.linalg.eigvals(J)
        stable = "STABLE" if all(e.real < 0 for e in eig) else "UNSTABLE"
        text = (f"<b>At ({point[0]:.2f}, {point[1]:.2f})</b><br>"
                f"J = [[{J[0,0]:+.2f}, {J[0,1]:+.2f}]<br>"
                f"&nbsp;&nbsp;&nbsp;&nbsp;[{J[1,0]:+.2f}, {J[1,1]:+.2f}]]<br>"
                f"λ₁ = {eig[0].real:+.2f}{'' if abs(eig[0].imag)<1e-3 else f' {eig[0].imag:+.2f}i'}<br>"
                f"λ₂ = {eig[1].real:+.2f}{'' if abs(eig[1].imag)<1e-3 else f' {eig[1].imag:+.2f}i'}<br>"
                f"∇·f = {np.trace(J):+.3f}<br>"
                f"<b>{stable}</b>")
        self.jacobian_label.setText(text)

    # ---------- Stats & Info ----------
    def _update_stats(self):
        step = int(self.current_time / self.dt)
        step = min(step, self.trajectories.shape[1] - 1)
        self.stat_time.setText(f"t = {self.current_time:.2f}")

        # Average speed at current step
        if self.trajectories.shape[0] > 0:
            current = self.trajectories[:, step, :]
            with torch.no_grad():
                f_val = self.model.forward_with_bifurcation(
                    torch.tensor(0.0),
                    torch.tensor(current, dtype=torch.float32)).numpy()
            avg_speed = np.linalg.norm(f_val, axis=1).mean()
            self.stat_speed.setText(f"avg |f| = {avg_speed:.3f}")

        origin = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        self.stat_div.setText(f"∇·f(0) = {self.model.divergence(origin):+.3f}")
        self.stat_params.setText(f"params = {self.model.count_parameters():,}")
        self.stat_traj.setText(f"trajectories = {self.trajectories.shape[0]}")

        # Trajectory stats
        if self.trajectories.shape[0] > 0:
            total_steps = self.trajectories.shape[1]
            lengths = np.linalg.norm(np.diff(self.trajectories, axis=1), axis=2).sum(axis=1)
            distances = np.linalg.norm(
                self.trajectories[:, -1] - self.trajectories[:, 0], axis=1)
            html = (f"<b>Avg trajectory length:</b> {lengths.mean():.3f}<br>"
                    f"<b>Avg displacement:</b> {distances.mean():.3f}<br>"
                    f"<b>Total steps:</b> {total_steps}<br>"
                    f"<b>Max length:</b> {lengths.max():.3f}<br>"
                    f"<b>Min length:</b> {lengths.min():.3f}")
            self.traj_stats_label.setText(html)

    def _update_info(self):
        cfg = self.model_config
        n_params = self.model.count_parameters()
        per_layer = 2*2*cfg['hidden_dim'] + cfg['hidden_dim']
        equiv_layers = n_params // per_layer if per_layer else 0
        origin = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        div0 = self.model.divergence(origin)
        html = f"""
        <b>Architecture:</b><br>
        Input(2) → Linear({cfg['hidden_dim']}) → {cfg['activation'].upper()}<br>
        """
        for _ in range(cfg['n_layers'] - 1):
            html += f"→ Linear({cfg['hidden_dim']}) → {cfg['activation'].upper()}<br>"
        html += f"""→ Linear(2)<br><br>
        <b>Parameters:</b> {n_params:,}<br>
        <b>ResNet equiv:</b> ~{equiv_layers} layers<br>
        <b>∇·f at origin:</b> {div0:+.3f}<br>
        <b>Bifurcation λ:</b> {self.bifurcation:+.2f}<br>
        <b>Solver:</b> {self.solver_name}<br>
        <b>dt:</b> {self.dt} | <b>T:</b> {self.max_time}
        """
        self.info_text.setHtml(html)

    # ---------- Training ----------
    def _target_spiral(self, t):
        r = 1.5 * torch.exp(-0.1 * t)
        return torch.stack([r*torch.cos(t), r*torch.sin(t)])
    def _target_figure8(self, t):
        return torch.stack([torch.sin(t), torch.sin(2*t)*0.8])
    def _target_circle(self, t):
        return torch.stack([torch.cos(t), torch.sin(t)])
    def _target_lissajous(self, t):
        return torch.stack([torch.sin(2*t), torch.sin(3*t)])

    def _change_target(self, name):
        mapping = {"Spiral": self._target_spiral, "Figure-8": self._target_figure8,
                   "Circle": self._target_circle, "Lissajous": self._target_lissajous}
        self.target_func = mapping[name]

    def _reset_model(self):
        for p in self.model.parameters():
            if p.dim() > 1: nn.init.xavier_normal_(p)
            else: nn.init.zeros_(p)
        self.loss_canvas.update([])
        self._reset_sim()
        self._update_info()

    def _toggle_training(self):
        if self.train_timer.isActive():
            self.train_timer.stop()
            self.btn_train.setText("▶ Start Training")
            self.progress_bar.setVisible(False)
        else:
            lr = 10 ** self.lr_slider.value()
            self.training_worker = TrainingWorker(
                self.model, self.target_func, steps=2000, lr=lr)
            self.training_worker.loss_update.connect(self._on_loss_update)
            self.training_worker.finished.connect(self._on_training_finished)
            self.training_worker.progress.connect(self._on_train_progress)
            self.training_worker.start()
            self.train_timer.start(30)
            self.btn_train.setText("⏹ Stop Training")
            self.progress_bar.setVisible(True)
            if self.comparison_mode:
                self.comparison_model = NeuralODEFunc(**self.model_config)
                for p_src, p_dst in zip(self.model.parameters(),
                                        self.comparison_model.parameters()):
                    p_dst.data.copy_(p_src.data)

    def _train_step(self):
        if self.training_worker and self.training_worker.running:
            for _ in range(3):
                self.training_worker.step()
                if not self.training_worker.running: break
            if self.training_worker.current_step % 10 == 0:
                self._reset_sim()

    def _on_loss_update(self, losses):
        self.loss_canvas.update(losses)

    def _on_train_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def _on_training_finished(self):
        self.train_timer.stop()
        self.btn_train.setText("▶ Start Training")
        self.progress_bar.setVisible(False)
        self._reset_sim()
        self._update_info()
        self.status.showMessage("Training complete!", 3000)

    # ---------- Export ----------
    def _save_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save PNG", "neural_ode.png",
                                              "PNG (*.png)")
        if path:
            self.canvas.fig.savefig(path, dpi=180, bbox_inches='tight',
                                    facecolor=self.canvas.fig.get_facecolor())
            self.status.showMessage(f"Saved: {path}", 3000)

    def _save_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "trajectories.csv",
                                              "CSV (*.csv)")
        if path:
            with open(path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(["trajectory_id", "time_step", "x", "y"])
                for i in range(self.trajectories.shape[0]):
                    for t in range(self.trajectories.shape[1]):
                        w.writerow([i, t*self.dt,
                                    self.trajectories[i,t,0],
                                    self.trajectories[i,t,1]])
            self.status.showMessage(f"Saved: {path}", 3000)

    def _save_model(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Model", "model.pkl",
                                              "Pickle (*.pkl)")
        if path:
            with open(path, 'wb') as f:
                pickle.dump({'state': self.model.state_dict(),
                             'config': self.model_config}, f)
            self.status.showMessage(f"Saved: {path}", 3000)

    def _load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Model", "",
                                              "Pickle (*.pkl)")
        if path:
            try:
                data = pickle.load(open(path, 'rb'))
                if isinstance(data, dict) and 'config' in data:
                    self.model_config = data['config']
                    self.model = NeuralODEFunc(**data['config'])
                else:
                    self.model.load_state_dict(data)
                self.model.bifurcation.fill_(self.bifurcation)
                self._reset_sim()
                self._update_info()
                self.status.showMessage(f"Loaded: {path}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load: {e}")


# ============================================================
# 10. ENTRY POINT
# ============================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = app.palette()
    from PySide6.QtGui import QColor
    palette.setColor(palette.Window, QColor(30, 30, 40))
    palette.setColor(palette.WindowText, Qt.white)
    palette.setColor(palette.Base, QColor(20, 20, 30))
    palette.setColor(palette.Text, Qt.white)
    palette.setColor(palette.Button, QColor(50, 50, 65))
    palette.setColor(palette.ButtonText, Qt.white)
    palette.setColor(palette.Highlight, QColor(0, 200, 150))
    palette.setColor(palette.HighlightedText, Qt.white)
    app.setPalette(palette)

    app.setStyleSheet("""
        QToolBar { background-color: #1a1a25; border-bottom: 1px solid #333; spacing: 6px; padding: 4px; }
        QToolBar QToolButton { background-color: #2a3a4a; border: 1px solid #444;
            border-radius: 4px; padding: 5px 10px; color: white; }
        QToolBar QToolButton:hover { background-color: #3a4a5a; border-color: #00ff88; }
        QToolBar QToolButton:checked { background-color: #00aa66; color: black; }
        QDockWidget { color: white; font-weight: bold; }
        QDockWidget::title { background-color: #1a1a25; padding: 6px;
            border: 1px solid #333; text-align: left; }
        QScrollArea { border: none; background-color: #15151f; }
        QGroupBox { font-weight: bold; border: 1px solid #444; border-radius: 5px;
            margin-top: 10px; padding-top: 10px; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px;
            padding: 0 5px; color: #00ff88; }
        QPushButton { background-color: #2a3a4a; border: 1px solid #444;
            border-radius: 4px; padding: 6px 12px; color: white; }
        QPushButton:hover { background-color: #3a4a5a; border-color: #00ff88; }
        QPushButton:pressed { background-color: #00aa66; }
        QComboBox, QSpinBox { background-color: #2a3a4a; border: 1px solid #444;
            border-radius: 3px; padding: 4px; color: white; }
        QComboBox:hover, QSpinBox:hover { border-color: #00ff88; }
        QComboBox::drop-down { border: none; }
        QComboBox QAbstractItemView { background-color: #2a3a4a; color: white;
            selection-background-color: #00aa66; }
        QSlider::groove:horizontal { height: 6px; background: #333; border-radius: 3px; }
        QSlider::handle:horizontal { background: #00ff88; width: 14px;
            margin: -4px 0; border-radius: 7px; }
        QSlider::handle:horizontal:hover { background: #00ffaa; }
        QCheckBox { color: white; spacing: 6px; }
        QCheckBox::indicator { width: 16px; height: 16px;
            border: 1px solid #444; border-radius: 3px; background: #2a3a4a; }
        QCheckBox::indicator:checked { background: #00aa66; border-color: #00ff88; }
        QProgressBar { border: 1px solid #444; border-radius: 3px;
            background-color: #1a1a25; text-align: center; color: white; }
        QProgressBar::chunk { background-color: #00aa66; }
        QTextEdit { background-color: #0d1117; color: #ddd;
            border: 1px solid #333; border-radius: 3px; }
        QStatusBar { background-color: #1a1a25; color: #aaa; border-top: 1px solid #333; }
        QSplitter::handle { background-color: #333; }
        QSplitter::handle:horizontal { width: 2px; }
        QSplitter::handle:vertical { height: 2px; }
    """)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())