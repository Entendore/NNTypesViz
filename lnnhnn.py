#!/usr/bin/env python3
"""
HNN & LNN Visualizer — Studio Edition
--------------------------------------
Professional physics-informed neural network lab with redesigned GUI.

GUI IMPROVEMENTS:
  * Three-pane layout: controls | plots | live metrics dashboard
  * Collapsible accordion sections for controls
  * Toolbar with quick actions + experiment presets
  * Persistent bottom metrics panel (always visible)
  * Reorganized tabs with logical grouping
  * Professional dark theme with consistent palette
  * Animated training progress with live loss curve
  * Toast-style notifications

NEW FEATURES:
  * 3D phase space visualization (q1, q2, p1)
  * Streamplot vector fields (smoother than quiver)
  * Weight distribution histograms
  * Energy error distribution histogram
  * Model comparison heatmap (RMSE / drift / params)
  * Preset experiments (quick-start configurations)
  * Export plots as high-res PNG / SVG
  * Export animations as GIF
  * Live loss curve during training
  * Phase-space density (heatmap of trajectory)
  * Trajectory error over time
  * Model architecture summary
"""

import sys, os, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QGroupBox, QComboBox, QPushButton, QLabel,
    QSpinBox, QDoubleSpinBox, QFormLayout, QCheckBox, QProgressBar,
    QSplitter, QStatusBar, QMessageBox, QFileDialog, QScrollArea,
    QFrame, QSizePolicy, QToolBar, QTextEdit, QToolButton, QStyle,
    QGridLayout, QStackedWidget
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QAction, QFont, QColor, QIcon


# =============================================================================
# THEME
# =============================================================================

THEME = {
    'bg':          '#0f1419',
    'panel':       '#1a1f2e',
    'panel2':      '#242b3d',
    'border':      '#2d3548',
    'accent':      '#4ecdc4',
    'accent2':     '#ff6b9d',
    'accent3':     '#ffe66d',
    'text':        '#e8eaed',
    'text_dim':    '#8b95a7',
    'success':     '#95e1d3',
    'danger':      '#ff6b6b',
    'truth':       '#ffffff',
    'HNN':         '#ff6b9d',
    'LNN':         '#4ecdc4',
    'Vanilla':     '#ffe66d',
}

MPL_STYLE = {
    'figure.facecolor': THEME['bg'],
    'axes.facecolor':   THEME['panel'],
    'axes.edgecolor':   THEME['border'],
    'axes.labelcolor':  THEME['text'],
    'text.color':       THEME['text'],
    'xtick.color':      THEME['text_dim'],
    'ytick.color':      THEME['text_dim'],
    'grid.color':       THEME['border'],
    'grid.alpha':       0.4,
    'legend.facecolor': THEME['panel2'],
    'legend.edgecolor': THEME['border'],
    'legend.labelcolor':THEME['text'],
    'font.family':      'sans-serif',
    'font.size':        10,
    'axes.titlesize':   12,
    'axes.labelsize':   11,
}


# =============================================================================
# PHYSICS SYSTEMS
# =============================================================================

class PhysicsSystem:
    name = "Base"; dim_q = 1; conservative = True
    def __init__(self, **params):
        defaults = self.default_params(); defaults.update(params)
        self.params = defaults
        for k, v in self.params.items(): setattr(self, k, v)
    @classmethod
    def default_params(cls): return {}
    @classmethod
    def param_info(cls): return []
    def hamiltonian(self, q, p): raise NotImplementedError
    def equations(self, t, y): raise NotImplementedError
    def default_y0(self): raise NotImplementedError
    def time_span(self): return (0.0, 10.0)
    def render_state(self, y, ax): pass


class SimplePendulum(PhysicsSystem):
    name = "Simple Pendulum"; dim_q = 1
    @classmethod
    def default_params(cls): return {"m": 1.0, "l": 1.0, "g": 9.81}
    @classmethod
    def param_info(cls):
        return [("m",0.1,5,1,0.1,"Mass"),("l",0.1,5,1,0.1,"Length"),("g",0.1,20,9.81,0.1,"Gravity")]
    def hamiltonian(self, q, p):
        return p**2/(2*self.m*self.l**2) + self.m*self.g*self.l*(1-np.cos(q))
    def equations(self, t, y):
        q, p = y; return [p/(self.m*self.l**2), -self.m*self.g*self.l*np.sin(q)]
    def default_y0(self): return np.array([np.pi/3, 0.0])
    def render_state(self, y, ax):
        q = y[0]; x = self.l*np.sin(q); yp = -self.l*np.cos(q)
        ax.clear(); ax.set_facecolor(THEME['panel'])
        # Motion trail
        for i, th in enumerate(np.linspace(q-0.6, q, 15)):
            a = 0.05 + 0.15*(i/15)
            ax.plot([0, self.l*np.sin(th)], [0, -self.l*np.cos(th)],
                    color=THEME['accent2'], lw=1.5, alpha=a)
        ax.plot([0, x], [0, yp], color=THEME['accent2'], lw=3, solid_capstyle='round')
        ax.plot(0, 0, 'o', color=THEME['truth'], ms=10,
                markeredgecolor=THEME['accent'], markeredgewidth=2)
        ax.plot(x, yp, 'o', color=THEME['accent3'], ms=14,
                markeredgecolor=THEME['accent2'], markeredgewidth=2)
        ax.set_xlim(-self.l*1.3, self.l*1.3); ax.set_ylim(-self.l*1.3, self.l*1.3)
        ax.set_aspect('equal'); ax.axis('off')
        E = self.hamiltonian(q, y[1])
        ax.set_title(f"θ = {np.degrees(q):.1f}°  |  E = {E:.3f}",
                     color=THEME['text'], fontsize=11, fontweight='bold')


class SpringMass(PhysicsSystem):
    name = "Spring-Mass"; dim_q = 1
    @classmethod
    def default_params(cls): return {"m": 1.0, "k": 4.0}
    @classmethod
    def param_info(cls):
        return [("m",0.1,5,1,0.1,"Mass"),("k",0.1,20,4,0.5,"Spring k")]
    def hamiltonian(self, q, p): return p**2/(2*self.m) + 0.5*self.k*q**2
    def equations(self, t, y):
        q, p = y; return [p/self.m, -self.k*q]
    def default_y0(self): return np.array([1.0, 0.0])
    def render_state(self, y, ax):
        q = y[0]; ax.clear(); ax.set_facecolor(THEME['panel'])
        ax.plot([-2, -2], [-0.5, 0.5], color=THEME['truth'], lw=4)
        xs = np.linspace(-2, q, 40)
        ys = 0.15*np.sin(np.linspace(0, 14*np.pi, 40))
        ax.plot(xs, ys, color=THEME['accent'], lw=2)
        circle = plt.Circle((q, 0), 0.25, color=THEME['accent3'],
                            ec=THEME['accent2'], lw=2)
        ax.add_patch(circle)
        ax.set_xlim(-2.5, 2.5); ax.set_ylim(-1, 1)
        ax.set_aspect('equal'); ax.axis('off')
        E = self.hamiltonian(q, y[1])
        ax.set_title(f"x = {q:.2f}  |  E = {E:.3f}",
                     color=THEME['text'], fontsize=11, fontweight='bold')


class DoublePendulum(PhysicsSystem):
    name = "Double Pendulum"; dim_q = 2
    @classmethod
    def default_params(cls):
        return {"m1":1.0,"m2":1.0,"l1":1.0,"l2":1.0,"g":9.81}
    @classmethod
    def param_info(cls):
        return [("m1",0.1,5,1,0.1,"Mass 1"),("m2",0.1,5,1,0.1,"Mass 2"),
                ("l1",0.1,3,1,0.1,"Length 1"),("l2",0.1,3,1,0.1,"Length 2"),
                ("g",0.1,20,9.81,0.1,"Gravity")]
    def hamiltonian(self, q, p):
        q1, q2 = q[...,0], q[...,1]; p1, p2 = p[...,0], p[...,1]
        m1,m2,l1,l2,g = self.m1,self.m2,self.l1,self.l2,self.g
        d = q2-q1; sd, cd = np.sin(d), np.cos(d)
        num = p1**2*l2**2 + p2**2*l1**2 - 2*p1*p2*l1*l2*cd
        den = 2*l1**2*l2**2*(m1 + m2*sd**2)
        T = num/den
        V = (m1+m2)*g*l1*(1-np.cos(q1)) + m2*g*l2*(1-np.cos(q2))
        return T+V
    def equations(self, t, y):
        q1,q2,p1,p2 = y
        m1,m2,l1,l2,g = self.m1,self.m2,self.l1,self.l2,self.g
        d = q2-q1; sd,cd = np.sin(d),np.cos(d)
        s1,s2 = np.sin(q1),np.sin(q2)
        den = 2*m1 + m2 - m2*np.cos(2*d)
        M11=(m1+m2)*l1**2; M12=m2*l1*l2*cd; M22=m2*l2**2
        det = M11*M22 - M12**2
        w1 = (M22*p1 - M12*p2)/det; w2 = (M11*p2 - M12*p1)/det
        a1 = (-g*(2*m1+m2)*s1 - m2*g*np.sin(q1-2*q2)
              - 2*sd*m2*(w2**2*l2 + w1**2*l1*cd))/(l1*den)
        a2 = (2*sd*(w1**2*l1*(m1+m2) + g*(m1+m2)*np.cos(q1)
                    + w2**2*l2*m2*cd))/(l2*den)
        dM12 = -m2*l1*l2*sd*(w2-w1)
        dp1 = M11*a1 + M12*a2 + dM12*w2
        dp2 = M12*a1 + M22*a2 - dM12*w1
        dq1 = (M22*p1 - M12*p2)/det
        dq2 = (M11*p2 - M12*p1)/det
        return [dq1, dq2, dp1, dp2]
    def default_y0(self): return np.array([np.pi/2, np.pi/2, 0.0, 0.0])
    def time_span(self): return (0.0, 5.0)
    def render_state(self, y, ax):
        q1,q2 = y[0],y[1]
        x1=self.l1*np.sin(q1); y1=-self.l1*np.cos(q1)
        x2=x1+self.l2*np.sin(q2); y2=y1-self.l2*np.cos(q2)
        ax.clear(); ax.set_facecolor(THEME['panel'])
        ax.plot([0,x1,x2],[0,y1,y2],color=THEME['accent2'],lw=3,solid_capstyle='round')
        ax.plot(0,0,'o',color=THEME['truth'],ms=10,
                markeredgecolor=THEME['accent'],markeredgewidth=2)
        ax.plot(x1,y1,'o',color=THEME['accent3'],ms=12,
                markeredgecolor=THEME['accent2'],markeredgewidth=2)
        ax.plot(x2,y2,'o',color=THEME['accent'],ms=12,
                markeredgecolor=THEME['accent3'],markeredgewidth=2)
        L = self.l1+self.l2
        ax.set_xlim(-L*1.3,L*1.3); ax.set_ylim(-L*1.3,L*1.3)
        ax.set_aspect('equal'); ax.axis('off')
        E = self.hamiltonian(y[:2], y[2:])
        ax.set_title(f"θ₁={np.degrees(q1):.1f}° θ₂={np.degrees(q2):.1f}° | E={E:.2f}",
                     color=THEME['text'], fontsize=10, fontweight='bold')


class KeplerProblem(PhysicsSystem):
    name = "Kepler (2D)"; dim_q = 2
    @classmethod
    def default_params(cls): return {"m":1.0,"GM":1.0}
    @classmethod
    def param_info(cls):
        return [("m",0.1,5,1,0.1,"Mass"),("GM",0.1,5,1,0.1,"GM")]
    def hamiltonian(self, q, p):
        q=np.atleast_2d(q); p=np.atleast_2d(p)
        r = np.sqrt(q[...,0]**2+q[...,1]**2)+1e-12
        return np.squeeze((p[...,0]**2+p[...,1]**2)/(2*self.m) - self.m*self.GM/r)
    def equations(self, t, y):
        x,y_,px,py = y
        r = np.sqrt(x**2+y_**2)+1e-12
        return [px/self.m, py/self.m, -self.m*self.GM*x/r**3, -self.m*self.GM*y_/r**3]
    def default_y0(self):
        r=1.0; v=np.sqrt(self.params["GM"]/r)
        return np.array([r,0.0,0.0,self.params["m"]*v])
    def time_span(self): return (0.0, 15.0)
    def render_state(self, y, ax):
        x,yp = y[0],y[1]; ax.clear(); ax.set_facecolor(THEME['panel'])
        for i in range(5,0,-1):
            ax.add_patch(plt.Circle((0,0),0.08*i,color=THEME['accent3'],alpha=0.08*i))
        ax.plot(0,0,'o',color=THEME['accent3'],ms=14,
                markeredgecolor=THEME['accent2'],markeredgewidth=2)
        ax.plot(x,yp,'o',color=THEME['accent'],ms=11,
                markeredgecolor=THEME['truth'],markeredgewidth=2)
        ax.set_xlim(-2,2); ax.set_ylim(-2,2); ax.set_aspect('equal'); ax.axis('off')
        r = np.sqrt(x**2+yp**2); E = self.hamiltonian(y[:2],y[2:])
        ax.set_title(f"r = {r:.2f}  |  E = {E:.3f}",
                     color=THEME['text'], fontsize=11, fontweight='bold')


class DampedPendulum(PhysicsSystem):
    name = "Damped Pendulum"; dim_q = 1; conservative = False
    @classmethod
    def default_params(cls): return {"m":1.0,"l":1.0,"g":9.81,"b":0.2}
    @classmethod
    def param_info(cls):
        return [("m",0.1,5,1,0.1,"Mass"),("l",0.1,5,1,0.1,"Length"),
                ("g",0.1,20,9.81,0.1,"Gravity"),("b",0,2,0.2,0.05,"Damping")]
    def hamiltonian(self, q, p):
        return p**2/(2*self.m*self.l**2)+self.m*self.g*self.l*(1-np.cos(q))
    def equations(self, t, y):
        q,p = y; dq=p/(self.m*self.l**2)
        return [dq, -self.m*self.g*self.l*np.sin(q)-self.b*dq*(self.m*self.l**2)]
    def default_y0(self): return np.array([np.pi/2, 0.0])
    def render_state(self, y, ax):
        q = y[0]; x=self.l*np.sin(q); yp=-self.l*np.cos(q)
        ax.clear(); ax.set_facecolor(THEME['panel'])
        ax.plot([0,x],[0,yp],color=THEME['accent2'],lw=3,alpha=0.7,solid_capstyle='round')
        ax.plot(0,0,'o',color=THEME['truth'],ms=10,
                markeredgecolor=THEME['accent'],markeredgewidth=2)
        ax.plot(x,yp,'o',color=THEME['accent3'],ms=14,
                markeredgecolor=THEME['accent2'],markeredgewidth=2)
        ax.set_xlim(-self.l*1.3,self.l*1.3); ax.set_ylim(-self.l*1.3,self.l*1.3)
        ax.set_aspect('equal'); ax.axis('off')
        E = self.hamiltonian(q,y[1])
        ax.set_title(f"θ = {np.degrees(q):.1f}° (damped)  |  E = {E:.3f}",
                     color=THEME['text'], fontsize=11, fontweight='bold')


SYSTEMS = {
    "Simple Pendulum": SimplePendulum,
    "Spring-Mass": SpringMass,
    "Double Pendulum": DoublePendulum,
    "Kepler (2D)": KeplerProblem,
    "Damped Pendulum": DampedPendulum,
}


# =============================================================================
# DATA / INTEGRATORS
# =============================================================================

def generate_trajectory(system, t_span=(0,10), dt=0.05, y0=None, noise=0.0):
    if y0 is None: y0 = system.default_y0()
    t_eval = np.arange(t_span[0], t_span[1]+1e-9, dt)
    sol = solve_ivp(system.equations, t_span, y0, t_eval=t_eval,
                    method='RK45', rtol=1e-10, atol=1e-12)
    y = sol.y.T
    if noise > 0: y = y + np.random.randn(*y.shape)*noise
    dy = np.gradient(y, dt, axis=0)
    return t_eval, y, dy

def generate_dataset(system, n_traj=1, t_span=(0,10), dt=0.05,
                     noise=0.0, y0_perturb=0.0, seed=0):
    rng = np.random.default_rng(seed); base = system.default_y0()
    all_y, all_dy = [], []
    for _ in range(n_traj):
        y0 = base + rng.uniform(-y0_perturb, y0_perturb, base.shape) if y0_perturb>0 else base
        _, y, dy = generate_trajectory(system, t_span, dt, y0=y0, noise=noise)
        all_y.append(y); all_dy.append(dy)
    return np.concatenate(all_y), np.concatenate(all_dy)

def split_data(y, dy, train_frac=0.7, val_frac=0.15, seed=0):
    rng = np.random.default_rng(seed); n = len(y)
    idx = rng.permutation(n)
    nt = int(n*train_frac); nv = int(n*val_frac)
    tr, va, te = idx[:nt], idx[nt:nt+nv], idx[nt+nv:]
    return (y[tr],dy[tr]), (y[va],dy[va]), (y[te],dy[te])

def euler_step(f, y, dt): return y + dt*f(y)
def rk4_step(f, y, dt):
    k1=f(y); k2=f(y+0.5*dt*k1); k3=f(y+0.5*dt*k2); k4=f(y+dt*k3)
    return y + (dt/6)*(k1+2*k2+2*k3+k4)
def leapfrog_step(f, y, dt):
    dim = len(y)//2; q,p = y[:dim].copy(), y[dim:].copy()
    def dq(q_,p_): return f(np.concatenate([q_,p_]))[:dim]
    def dp(q_,p_): return f(np.concatenate([q_,p_]))[dim:]
    p = p + 0.5*dt*dp(q,p); q = q + dt*dq(q,p); p = p + 0.5*dt*dp(q,p)
    return np.concatenate([q,p])

def integrate_trajectory(f, y0, t_eval, method='rk4'):
    dt = t_eval[1]-t_eval[0]; n = len(t_eval); dim = len(y0)
    y_out = np.zeros((n,dim)); y_out[0] = y0; y_cur = y0.copy()
    step = {'euler':euler_step,'rk4':rk4_step}.get(method)
    if method == 'leapfrog':
        for i in range(1,n): y_cur = leapfrog_step(f,y_cur,dt); y_out[i]=y_cur
    else:
        for i in range(1,n): y_cur = step(f,y_cur,dt); y_out[i]=y_cur
    return y_out

def compute_fourier(sig, dt):
    n=len(sig); yf=fft(sig); xf=fftfreq(n,dt)[:n//2]
    return xf, 2.0/n*np.abs(yf[:n//2])


# =============================================================================
# NEURAL NETWORKS
# =============================================================================

ACTIVATIONS = {
    'Tanh': nn.Tanh, 'ReLU': nn.ReLU,
    'Sin': lambda: nn.Sin if hasattr(nn,'Sin') else SinAct(),
    'Swish': nn.SiLU,
}

class SinAct(nn.Module):
    def forward(self, x): return torch.sin(30.0*x)

class ScalarNet(nn.Module):
    def __init__(self, input_dim, hidden=128, n_layers=3, activation='Tanh'):
        super().__init__()
        layers=[]; in_d=input_dim; act=ACTIVATIONS[activation]
        for _ in range(n_layers):
            layers+=[nn.Linear(in_d,hidden),act()]; in_d=hidden
        layers+=[nn.Linear(in_d,1)]
        self.net = nn.Sequential(*layers)
    def forward(self,x): return self.net(x).squeeze(-1)

class HNN(nn.Module):
    def __init__(self, input_dim, hidden=128, n_layers=3, activation='Tanh'):
        super().__init__(); self.dim=input_dim//2
        self.net = ScalarNet(input_dim, hidden, n_layers, activation)
    def hamiltonian(self,q,p): return self.net(torch.cat([q,p],-1))
    def vector_field(self,q,p):
        q=q.detach().requires_grad_(True); p=p.detach().requires_grad_(True)
        H = self.hamiltonian(q,p).sum()
        dHdq = torch.autograd.grad(H,q,create_graph=True)[0]
        dHdp = torch.autograd.grad(H,p,create_graph=True)[0]
        return dHdp, -dHdq
    def f(self, y_np):
        y_t = torch.tensor(y_np,dtype=torch.float32).unsqueeze(0)
        q,p = y_t[:,:self.dim], y_t[:,self.dim:]
        with torch.no_grad():
            dq,dp = self.vector_field(q,p)
        return torch.cat([dq,dp],-1).numpy()[0]

class LNN(nn.Module):
    def __init__(self, input_dim, hidden=128, n_layers=3, activation='Tanh'):
        super().__init__(); self.dim=input_dim//2
        self.net = ScalarNet(input_dim, hidden, n_layers, activation)
    def lagrangian(self,q,qdot): return self.net(torch.cat([q,qdot],-1))
    def acceleration(self,q,qdot):
        q=q.detach().requires_grad_(True); qdot=qdot.detach().requires_grad_(True)
        L = self.lagrangian(q,qdot).sum()
        dLdq = torch.autograd.grad(L,q,create_graph=True)[0]
        dLdqdot = torch.autograd.grad(L,qdot,create_graph=True)[0]
        dim=self.dim; B=qdot.shape[0]
        M = torch.zeros(B,dim,dim,device=q.device,dtype=q.dtype)
        for i in range(dim):
            g = torch.autograd.grad(dLdqdot[:,i],qdot,
                grad_outputs=torch.ones(B,device=q.device),
                retain_graph=True,create_graph=True)[0]
            M[:,i,:] = g
        rhs = dLdq.clone()
        for k in range(dim):
            gk = torch.autograd.grad(dLdqdot,q,
                grad_outputs=torch.zeros(B,dim,device=q.device).scatter_(
                    1, torch.full((B,1),k,device=q.device,dtype=torch.long), 1),
                retain_graph=True,create_graph=True)[0]
            rhs = rhs - gk*qdot[:,k:k+1]
        try:
            qddot = torch.linalg.solve(M+1e-6*torch.eye(dim,device=q.device),rhs)
        except Exception:
            qddot = torch.linalg.lstsq(M+1e-6*torch.eye(dim,device=q.device),rhs).solution
        return qddot
    def f(self, y_np):
        y_t = torch.tensor(y_np,dtype=torch.float32).unsqueeze(0)
        q,qdot = y_t[:,:self.dim], y_t[:,self.dim:]
        with torch.no_grad():
            qddot = self.acceleration(q,qdot)
        return torch.cat([qdot,qddot],-1).numpy()[0]

class VanillaMLP(nn.Module):
    def __init__(self, input_dim, hidden=128, n_layers=3, activation='Tanh'):
        super().__init__(); self.dim=input_dim//2
        layers=[]; in_d=input_dim; act=ACTIVATIONS[activation]
        for _ in range(n_layers):
            layers+=[nn.Linear(in_d,hidden),act()]; in_d=hidden
        layers+=[nn.Linear(in_d,input_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self,y): return self.net(y)
    def f(self, y_np):
        y_t = torch.tensor(y_np,dtype=torch.float32).unsqueeze(0)
        with torch.no_grad(): return self.forward(y_t).numpy()[0]


# =============================================================================
# TRAINER
# =============================================================================

class Trainer(QThread):
    progress = Signal(int, dict)
    finished = Signal(dict)
    def __init__(self, model_type, system, train_data, val_data, config):
        super().__init__()
        self.model_type=model_type; self.system=system
        self.train_y,self.train_dy=train_data
        self.val_y,self.val_dy=val_data
        self.config=config; self._stop=False
    def stop(self): self._stop=True
    def run(self):
        cfg=self.config
        torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
        device=torch.device(cfg['device']); dim=self.train_y.shape[1]; half=dim//2
        if self.model_type=="HNN":
            model=HNN(dim,cfg['hidden'],cfg['layers'],cfg['activation']).to(device)
        elif self.model_type=="LNN":
            model=LNN(dim,cfg['hidden'],cfg['layers'],cfg['activation']).to(device)
        else:
            model=VanillaMLP(dim,cfg['hidden'],cfg['layers'],cfg['activation']).to(device)
        opt = optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt,factor=0.5,patience=300)
        ty=torch.tensor(self.train_y,dtype=torch.float32,device=device)
        tdy=torch.tensor(self.train_dy,dtype=torch.float32,device=device)
        vy=torch.tensor(self.val_y,dtype=torch.float32,device=device)
        vdy=torch.tensor(self.val_dy,dtype=torch.float32,device=device)
        losses=[]; vlosses=[]; t0=time.time(); best_val=float('inf'); best_state=None
        n_epochs=cfg['epochs']
        for epoch in range(n_epochs):
            if self._stop: break
            opt.zero_grad()
            loss = self._loss(model,ty,tdy,half,device)
            loss.backward()
            if cfg.get('grad_clip',0)>0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'])
            opt.step(); sched.step(loss.item())
            with torch.no_grad():
                vl = self._loss(model,vy,vdy,half,device).item()
            losses.append(loss.item()); vlosses.append(vl)
            if vl<best_val:
                best_val=vl; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
            if epoch%max(1,n_epochs//100)==0:
                self.progress.emit(int(100*epoch/n_epochs),
                    {'loss':loss.item(),'val_loss':vl,
                     'elapsed':time.time()-t0,'lr':opt.param_groups[0]['lr']})
        if best_state: model.load_state_dict(best_state); model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        metrics = self._eval(model,half,device)
        metrics['losses']=losses; metrics['val_losses']=vlosses
        metrics['n_params']=n_params
        self.finished.emit({'model':model,'metrics':metrics,'best_val':best_val})
    def _loss(self, model, y, dy, half, device):
        if self.model_type=="HNN":
            q,p=y[:,:half],y[:,half:]
            dqt,dpt=dy[:,:half],dy[:,half:]
            dqp,dpp=model.vector_field(q,p)
            return torch.mean((dqp-dqt)**2+(dpp-dpt)**2)
        elif self.model_type=="LNN":
            q,qd=y[:,:half],y[:,half:]
            return torch.mean((model.acceleration(q,qd)-dy[:,half:])**2)
        else:
            return torch.mean((model(y)-dy)**2)
    def _eval(self, model, half, device):
        y0=self.system.default_y0(); t_span=self.system.time_span(); dt=0.02
        t_eval=np.arange(t_span[0],t_span[1],dt)
        y_pred=integrate_trajectory(model.f, y0, t_eval,
                                    method=self.config.get('integrator','rk4'))
        E_pred=np.array([self.system.hamiltonian(y_pred[i,:half],y_pred[i,half:])
                         for i in range(len(y_pred))])
        _,y_true,_=generate_trajectory(self.system,t_span,dt,y0=y0)
        E_true=np.array([self.system.hamiltonian(y_true[i,:half],y_true[i,half:])
                         for i in range(len(y_true))])
        n=min(len(y_pred),len(y_true))
        return {'y_pred':y_pred[:n],'y_true':y_true[:n],
                'E_pred':E_pred[:n],'E_true':E_true[:n],'t':t_eval[:n],
                'energy_drift':float(abs(E_pred[0]-E_pred[-1])),
                'energy_err_max':float(np.max(np.abs(E_pred-E_pred[0]))),
                'rmse':float(np.sqrt(np.mean((y_pred[:n]-y_true[:n])**2))),
                'initial_energy':float(E_true[0])}


# =============================================================================
# GUI COMPONENTS
# =============================================================================

class CollapsibleBox(QWidget):
    """A collapsible group box with smooth animation."""
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self._expanded = True
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(0,0,0,0)
        self.layout_.setSpacing(2)
        # Header
        self.header = QPushButton(f"▼  {title}")
        self.header.setCheckable(True); self.header.setChecked(True)
        self.header.setStyleSheet(f"""
            QPushButton {{
                background-color: {THEME['panel2']};
                color: {THEME['text']};
                border: 1px solid {THEME['border']};
                border-radius: 4px;
                padding: 6px 10px;
                text-align: left;
                font-weight: bold;
                font-size: 11px;
            }}
            QPushButton:hover {{ background-color: {THEME['border']}; }}
            QPushButton:checked {{ background-color: {THEME['panel2']}; }}
        """)
        self.header.clicked.connect(self.toggle)
        self.layout_.addWidget(self.header)
        # Content container
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(8,4,8,8)
        self.content_layout.setSpacing(4)
        self.layout_.addWidget(self.content)
        self.title = title
    def toggle(self):
        self._expanded = not self._expanded
        self.content.setVisible(self._expanded)
        arrow = "▼" if self._expanded else "▶"
        self.header.setText(f"{arrow}  {self.title}")
    def add_widget(self, w): self.content_layout.addWidget(w)
    def add_layout(self, l): self.content_layout.addLayout(l)


class PlotCanvas(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width,height),dpi=dpi,tight_layout=True)
        self.fig.set_facecolor(THEME['bg'])
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(THEME['panel'])


class MetricCard(QFrame):
    """A small card displaying a metric value."""
    def __init__(self, label, color=THEME['accent']):
        super().__init__()
        self.setStyleSheet(f"""
            MetricCard {{
                background-color: {THEME['panel2']};
                border: 1px solid {THEME['border']};
                border-left: 3px solid {color};
                border-radius: 4px;
                padding: 6px;
            }}
        """)
        layout = QVBoxLayout(self); layout.setContentsMargins(8,4,8,4)
        self.label = QLabel(label)
        self.label.setStyleSheet(f"color:{THEME['text_dim']};font-size:10px;border:none;")
        self.value = QLabel("—")
        self.value.setStyleSheet(f"color:{color};font-size:16px;font-weight:bold;border:none;")
        layout.addWidget(self.label); layout.addWidget(self.value)
    def set_value(self, v):
        self.value.setText(str(v))


# =============================================================================
# MAIN WINDOW
# =============================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HNN & LNN Visualizer — Studio Edition")
        self.resize(1800, 1050)
        self.trainer=None; self.models={}; self.metrics={}
        self.current_system=None; self._live_losses=[]
        plt.rcParams.update(MPL_STYLE)
        self._build_ui()
        self._setup_animation()
        self._on_system_changed()

    # ---------- UI ----------
    def _build_ui(self):
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {THEME['bg']}; }}
            QWidget {{ background-color: {THEME['bg']}; color: {THEME['text']}; }}
            QComboBox, QSpinBox, QDoubleSpinBox {{
                background-color: {THEME['panel2']};
                color: {THEME['text']};
                border: 1px solid {THEME['border']};
                padding: 4px 6px; border-radius: 3px;
            }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox QAbstractItemView {{
                background-color: {THEME['panel2']};
                color: {THEME['text']};
                selection-background-color: {THEME['accent']};
            }}
            QTabWidget::pane {{
                border: 1px solid {THEME['border']};
                background-color: {THEME['bg']};
            }}
            QTabBar::tab {{
                background-color: {THEME['panel']};
                color: {THEME['text_dim']};
                padding: 8px 16px;
                border: 1px solid {THEME['border']};
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
            }}
            QTabBar::tab:selected {{
                background-color: {THEME['panel2']};
                color: {THEME['accent']};
                border-bottom: 2px solid {THEME['accent']};
                font-weight: bold;
            }}
            QTabBar::tab:hover {{ background-color: {THEME['border']}; }}
            QScrollArea {{ border: none; background-color: {THEME['bg']}; }}
            QScrollBar:vertical {{
                background: {THEME['bg']}; width: 10px; margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: {THEME['border']}; min-height: 20px; border-radius: 5px;
            }}
            QProgressBar {{
                border: 1px solid {THEME['border']};
                border-radius: 4px; text-align: center;
                background-color: {THEME['panel']};
                color: {THEME['text']};
            }}
            QProgressBar::chunk {{
                background-color: {THEME['accent']};
                border-radius: 3px;
            }}
            QStatusBar {{
                background-color: {THEME['panel']};
                color: {THEME['text_dim']};
                border-top: 1px solid {THEME['border']};
            }}
            QToolBar {{
                background-color: {THEME['panel']};
                border-bottom: 1px solid {THEME['border']};
                spacing: 6px; padding: 4px;
            }}
        """)

        # ---- Toolbar ----
        tb = QToolBar("Main"); tb.setMovable(False)
        self.addToolBar(tb)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "— Presets —",
            "Quick Test (Spring, 500 ep)",
            "Pendulum (HNN, 3000 ep)",
            "Double Pendulum (LNN, 5000 ep)",
            "Kepler Orbit (HNN, 3000 ep)",
            "Comparison (HNN vs LNN vs MLP)",
            "Noisy Data Robustness",
        ])
        self.preset_combo.currentIndexChanged.connect(self._apply_preset)
        self.preset_combo.setMinimumWidth(240)
        tb.addWidget(QLabel("Preset: "))
        tb.addWidget(self.preset_combo)
        tb.addSeparator()

        self.train_btn = QPushButton("▶  Train")
        self.train_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {THEME['accent']};
                color: {THEME['bg']};
                border: none; padding: 6px 18px;
                border-radius: 4px; font-weight: bold;
            }}
            QPushButton:hover {{ background-color: {THEME['success']}; }}
            QPushButton:disabled {{ background-color: {THEME['border']}; color: {THEME['text_dim']}; }}
        """)
        self.train_btn.clicked.connect(self._start_training)
        tb.addWidget(self.train_btn)

        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {THEME['accent2']};
                color: {THEME['bg']};
                border: none; padding: 6px 14px;
                border-radius: 4px; font-weight: bold;
            }}
            QPushButton:disabled {{ background-color: {THEME['border']}; color: {THEME['text_dim']}; }}
        """)
        self.stop_btn.clicked.connect(self._stop_training)
        tb.addWidget(self.stop_btn)
        tb.addSeparator()

        for label, fn in [("💾 Export Model", self._export_model),
                          ("📸 Save Plot", self._save_plot),
                          ("📄 Save Config", self._save_config),
                          ("📂 Load Config", self._load_config)]:
            b = QPushButton(label); b.clicked.connect(fn)
            b.setStyleSheet(f"""
                QPushButton {{
                    background-color: {THEME['panel2']};
                    color: {THEME['text']};
                    border: 1px solid {THEME['border']};
                    padding: 5px 12px; border-radius: 4px;
                }}
                QPushButton:hover {{ background-color: {THEME['border']}; }}
            """)
            tb.addWidget(b)

        # ---- Central layout: 3 panes ----
        central = QWidget(); self.setCentralWidget(central)
        main_v = QVBoxLayout(central); main_v.setContentsMargins(0,0,0,0); main_v.setSpacing(0)

        hsplit = QSplitter(Qt.Horizontal)
        main_v.addWidget(hsplit, 1)

        # LEFT: controls (scrollable accordion)
        left = QScrollArea(); left.setWidgetResizable(True)
        left.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_widget = QWidget(); left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(8,8,8,8); left_layout.setSpacing(6)
        left.setWidget(left_widget)

        # -- System accordion --
        sys_box = CollapsibleBox("⚙  Physical System")
        sys_form = QFormLayout(); sys_form.setSpacing(6)
        self.system_combo = QComboBox()
        for n in SYSTEMS: self.system_combo.addItem(n)
        self.system_combo.currentTextChanged.connect(self._on_system_changed)
        sys_form.addRow("System:", self.system_combo)
        sys_box.add_layout(sys_form)
        left_layout.addWidget(sys_box)

        self.params_box = CollapsibleBox("🔧  System Parameters")
        self.params_form = QFormLayout(); self.params_form.setSpacing(6)
        self.param_spins = {}
        self.params_box.add_layout(self.params_form)
        left_layout.addWidget(self.params_box)

        ic_box = CollapsibleBox("📍  Initial Condition")
        ic_form = QFormLayout(); ic_form.setSpacing(6)
        self.y0_spins = []
        for i in range(4):
            s = QDoubleSpinBox(); s.setRange(-10,10); s.setDecimals(3); s.setSingleStep(0.1)
            self.y0_spins.append(s); ic_form.addRow(f"y₀[{i}]:", s); s.setVisible(False)
        ic_box.add_layout(ic_form)
        left_layout.addWidget(ic_box)

        data_box = CollapsibleBox("📊  Training Data")
        data_form = QFormLayout(); data_form.setSpacing(6)
        self.n_traj_spin = QSpinBox(); self.n_traj_spin.setRange(1,20); self.n_traj_spin.setValue(1)
        data_form.addRow("# Trajectories:", self.n_traj_spin)
        self.y0_perturb_spin = QDoubleSpinBox()
        self.y0_perturb_spin.setRange(0,3); self.y0_perturb_spin.setValue(0.0); self.y0_perturb_spin.setDecimals(2)
        data_form.addRow("IC perturbation:", self.y0_perturb_spin)
        self.t_span_spin = QDoubleSpinBox()
        self.t_span_spin.setRange(1,100); self.t_span_spin.setValue(10); self.t_span_spin.setSuffix(" s")
        data_form.addRow("Duration:", self.t_span_spin)
        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.001,0.5); self.dt_spin.setValue(0.05); self.dt_spin.setDecimals(3)
        data_form.addRow("dt:", self.dt_spin)
        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0,0.5); self.noise_spin.setValue(0.0); self.noise_spin.setDecimals(3)
        data_form.addRow("Noise σ:", self.noise_spin)
        self.train_frac_spin = QDoubleSpinBox()
        self.train_frac_spin.setRange(0.3,0.95); self.train_frac_spin.setValue(0.7); self.train_frac_spin.setDecimals(2)
        data_form.addRow("Train fraction:", self.train_frac_spin)
        data_box.add_layout(data_form)
        left_layout.addWidget(data_box)

        model_box = CollapsibleBox("🧠  Neural Network")
        model_form = QFormLayout(); model_form.setSpacing(6)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["HNN","LNN","Vanilla MLP"])
        model_form.addRow("Type:", self.model_combo)
        self.hidden_spin = QSpinBox(); self.hidden_spin.setRange(16,512); self.hidden_spin.setValue(128)
        model_form.addRow("Hidden units:", self.hidden_spin)
        self.layers_spin = QSpinBox(); self.layers_spin.setRange(1,8); self.layers_spin.setValue(3)
        model_form.addRow("Layers:", self.layers_spin)
        self.act_combo = QComboBox(); self.act_combo.addItems(list(ACTIVATIONS.keys()))
        model_form.addRow("Activation:", self.act_combo)
        model_box.add_layout(model_form)
        left_layout.addWidget(model_box)

        train_box = CollapsibleBox("🎯  Training")
        train_form = QFormLayout(); train_form.setSpacing(6)
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(100,100000); self.epochs_spin.setValue(3000); self.epochs_spin.setSingleStep(500)
        train_form.addRow("Epochs:", self.epochs_spin)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-5,1); self.lr_spin.setValue(1e-3); self.lr_spin.setDecimals(5)
        train_form.addRow("Learning rate:", self.lr_spin)
        self.wd_spin = QDoubleSpinBox()
        self.wd_spin.setRange(0,1e-2); self.wd_spin.setValue(0.0); self.wd_spin.setDecimals(5)
        train_form.addRow("Weight decay:", self.wd_spin)
        self.grad_clip_spin = QDoubleSpinBox()
        self.grad_clip_spin.setRange(0,100); self.grad_clip_spin.setValue(0.0); self.grad_clip_spin.setDecimals(2)
        self.grad_clip_spin.setSpecialValueText("None")
        train_form.addRow("Grad clip:", self.grad_clip_spin)
        self.seed_spin = QSpinBox(); self.seed_spin.setRange(0,99999); self.seed_spin.setValue(42)
        train_form.addRow("Random seed:", self.seed_spin)
        self.device_combo = QComboBox(); self.device_combo.addItem("cpu")
        if torch.cuda.is_available(): self.device_combo.addItem("cuda")
        train_form.addRow("Device:", self.device_combo)
        train_box.add_layout(train_form)
        left_layout.addWidget(train_box)

        eval_box = CollapsibleBox("📈  Evaluation")
        eval_form = QFormLayout(); eval_form.setSpacing(6)
        self.integrator_combo = QComboBox()
        self.integrator_combo.addItems(["rk4","euler","leapfrog"])
        eval_form.addRow("Integrator:", self.integrator_combo)
        self.compare_chk = QCheckBox("Compare all models")
        eval_form.addRow(self.compare_chk)
        eval_box.add_layout(eval_form)
        left_layout.addWidget(eval_box)

        self.progress = QProgressBar(); self.progress.setRange(0,100); self.progress.setValue(0)
        left_layout.addWidget(self.progress)
        left_layout.addStretch()

        # CENTER: tabbed plots
        self.tabs = QTabWidget()
        self.tabs.addTab(self._make_tab("animation"),      "🎬 Animation")
        self.tabs.addTab(self._make_tab("phase"),          "🌀 Phase Space")
        self.tabs.addTab(self._make_tab("energy_landscape"),"🗻 Energy Landscape")
        self.tabs.addTab(self._make_tab("vectorfield"),    "🌊 Vector Field")
        self.tabs.addTab(self._make_tab("time"),           "📉 Time Series")
        self.tabs.addTab(self._make_tab("energy"),         "⚡ Energy")
        self.tabs.addTab(self._make_tab("fourier"),        "🎵 Fourier")
        self.tabs.addTab(self._make_tab("integrator"),     "🔁 Integrators")
        self.tabs.addTab(self._make_tab("traj2d"),         "🛰️ 2D Trajectory")
        self.tabs.addTab(self._make_tab("phase3d"),        "🎲 3D Phase")
        self.tabs.addTab(self._make_tab("weights"),        "🧬 Weights")
        self.tabs.addTab(self._make_tab("error_dist"),     "📊 Error Dist")
        self.tabs.addTab(self._make_tab("heatmap"),        "🔥 Comparison")
        self.tabs.addTab(self._make_tab("loss"),           "📉 Training")

        # RIGHT: live metrics dashboard
        right = QWidget(); right.setFixedWidth(280)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8,8,8,8); right_layout.setSpacing(8)

        title = QLabel("📊 Live Metrics")
        title.setStyleSheet(f"color:{THEME['accent']};font-size:14px;font-weight:bold;")
        right_layout.addWidget(title)

        self.cards = {}
        for label, color in [
            ("Status", THEME['text']),
            ("Epoch", THEME['accent']),
            ("Train Loss", THEME['accent2']),
            ("Val Loss", THEME['accent3']),
            ("LR", THEME['success']),
            ("Elapsed (s)", THEME['text']),
            ("────", THEME['border']),
            ("RMSE", THEME['accent2']),
            ("Energy Drift", THEME['accent3']),
            ("Max |ΔE|", THEME['accent']),
            ("Params", THEME['success']),
        ]:
            if label == "────":
                line = QFrame(); line.setFrameShape(QFrame.HLine)
                line.setStyleSheet(f"color:{THEME['border']};")
                right_layout.addWidget(line); continue
            card = MetricCard(label, color)
            self.cards[label] = card
            right_layout.addWidget(card)

        # Live loss mini-plot
        right_layout.addWidget(QLabel("📉 Live Loss"))
        self.live_canvas = PlotCanvas(width=3, height=2)
        self.live_canvas.ax.set_xlabel("epoch", fontsize=9)
        self.live_canvas.ax.set_ylabel("loss", fontsize=9)
        self.live_canvas.ax.set_title("Training Loss", fontsize=10)
        right_layout.addWidget(self.live_canvas)

        # Model info
        right_layout.addWidget(QLabel("🧠 Model Info"))
        self.model_info = QTextEdit()
        self.model_info.setReadOnly(True)
        self.model_info.setMaximumHeight(150)
        self.model_info.setStyleSheet(f"""
            QTextEdit {{
                background-color: {THEME['panel2']};
                color: {THEME['text']};
                border: 1px solid {THEME['border']};
                border-radius: 4px;
                font-family: monospace;
                font-size: 10px;
            }}
        """)
        self.model_info.setPlainText("No model trained yet.")
        right_layout.addWidget(self.model_info)

        right_layout.addStretch()

        # Bottom: status
        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"""
            QLabel {{
                background-color: {THEME['panel2']};
                color: {THEME['text']};
                border: 1px solid {THEME['border']};
                border-radius: 4px;
                padding: 8px;
                font-family: monospace;
                font-size: 11px;
            }}
        """)
        right_layout.addWidget(self.status_label)

        hsplit.addWidget(left); hsplit.addWidget(self.tabs); hsplit.addWidget(right)
        hsplit.setStretchFactor(0, 0); hsplit.setStretchFactor(1, 1); hsplit.setStretchFactor(2, 0)
        hsplit.setSizes([380, 1100, 280])

        sb = QStatusBar(); self.setStatusBar(sb)
        sb.showMessage("Studio Edition — Energy landscapes, 3D phase, weight analysis, comparison heatmaps")

    def _make_tab(self, name):
        w = QWidget(); l = QHBoxLayout(w); l.setContentsMargins(0,0,0,0)
        c = PlotCanvas(); l.addWidget(c)
        setattr(self, f"{name}_canvas", c)
        return w

    def _setup_animation(self):
        self.anim_timer = QTimer(); self.anim_timer.timeout.connect(self._update_animation)
        self.anim_y = None; self.anim_idx = 0

    # ---------- presets ----------
    def _apply_preset(self, idx):
        presets = {
            1: {"sys":"Spring-Mass","epochs":500,"model":"HNN","hidden":64,"layers":2},
            2: {"sys":"Simple Pendulum","epochs":3000,"model":"HNN","hidden":128,"layers":3},
            3: {"sys":"Double Pendulum","epochs":5000,"model":"LNN","hidden":256,"layers":4,
                "n_traj":3,"y0_perturb":0.3},
            4: {"sys":"Kepler (2D)","epochs":3000,"model":"HNN","hidden":128,"layers":3,
                "integrator":"leapfrog"},
            5: {"sys":"Simple Pendulum","epochs":3000,"model":"HNN","compare":True},
            6: {"sys":"Simple Pendulum","epochs":3000,"model":"HNN","noise":0.05,"n_traj":5,
                "y0_perturb":0.5},
        }
        if idx not in presets: return
        p = presets[idx]
        self.system_combo.setCurrentText(p["sys"])
        self.epochs_spin.setValue(p["epochs"])
        self.model_combo.setCurrentText(p["model"])
        self.hidden_spin.setValue(p["hidden"])
        self.layers_spin.setValue(p["layers"])
        if "n_traj" in p: self.n_traj_spin.setValue(p["n_traj"])
        if "y0_perturb" in p: self.y0_perturb_spin.setValue(p["y0_perturb"])
        if "noise" in p: self.noise_spin.setValue(p["noise"])
        if "integrator" in p: self.integrator_combo.setCurrentText(p["integrator"])
        if p.get("compare"): self.compare_chk.setChecked(True)
        self.statusBar().showMessage(f"Preset applied: {p['sys']}", 3000)

    # ---------- system handling ----------
    def _on_system_changed(self, *args):
        name = self.system_combo.currentText(); cls = SYSTEMS[name]
        self.current_system = cls()
        for s in self.param_spins.values(): self.params_form.removeRow(s)
        self.param_spins.clear()
        for (pn,pmin,pmax,pdef,pstep,plabel) in cls.param_info():
            s = QDoubleSpinBox(); s.setRange(pmin,pmax); s.setValue(pdef)
            s.setDecimals(3); s.setSingleStep(pstep)
            self.params_form.addRow(f"{plabel} ({pn}):", s); self.param_spins[pn]=s
        y0 = self.current_system.default_y0()
        for i,s in enumerate(self.y0_spins):
            if i<len(y0): s.setVisible(True); s.setValue(float(y0[i]))
            else: s.setVisible(False)
        self.t_span_spin.setValue(self.current_system.time_span()[1])
        self.anim_timer.stop(); self._plot_ground_truth()

    def _get_system(self):
        cls = SYSTEMS[self.system_combo.currentText()]
        return cls(**{k:float(s.value()) for k,s in self.param_spins.items()})
    def _get_y0(self):
        n = 2*self.current_system.dim_q
        return np.array([float(self.y0_spins[i].value()) for i in range(n)])

    def _generate_data(self):
        sys = self._get_system(); seed = self.seed_spin.value()
        y, dy = generate_dataset(sys, n_traj=self.n_traj_spin.value(),
            t_span=(0,self.t_span_spin.value()), dt=self.dt_spin.value(),
            noise=self.noise_spin.value(), y0_perturb=self.y0_perturb_spin.value(),
            seed=seed)
        tr,va,te = split_data(y,dy,train_frac=self.train_frac_spin.value(),seed=seed)
        return sys, y, dy, tr, va, te

    # ---------- plotting ----------
    def _plot_ground_truth(self):
        sys = self._get_system(); y0 = self._get_y0()
        t, y, _ = generate_trajectory(sys,(0,self.t_span_spin.value()),
                                      self.dt_spin.value(),y0=y0)
        half = y.shape[1]//2
        # Animation
        self.anim_y = y; self.anim_t = t; self.anim_idx = 0
        self.anim_timer.start(50); self._update_animation()
        # Phase with energy contours
        self._plot_phase(sys, t, y, half)
        self._plot_energy_landscape(sys, t, y, half)
        self._plot_vectorfield(sys, t, y, half)
        self._plot_time(t, y, half)
        self._plot_energy_truth(sys, t, y, half)
        self._plot_fourier(t, y, half)
        self._plot_traj2d(t, y, half)
        self._plot_phase3d(t, y, half)

    def _plot_phase(self, sys, t, y, half):
        ax = self.phase_canvas.ax; ax.clear()
        if half == 1:
            qmin,qmax = y[:,0].min(),y[:,0].max()
            pmin,pmax = y[:,1].min(),y[:,1].max()
            qg = np.linspace(qmin,qmax,60); pg = np.linspace(pmin,pmax,60)
            Q,P = np.meshgrid(qg,pg)
            E = np.zeros_like(Q)
            for i in range(Q.shape[0]):
                for j in range(Q.shape[1]):
                    E[i,j] = sys.hamiltonian(Q[i,j],P[i,j])
            ax.contourf(Q,P,E,levels=25,cmap='viridis',alpha=0.5)
            ax.contour(Q,P,E,levels=25,colors='white',linewidths=0.3,alpha=0.3)
            sc = ax.scatter(y[:,0],y[:,1],c=t,cmap='plasma',s=12,alpha=0.85,edgecolors='none')
            self.phase_canvas.fig.colorbar(sc,ax=ax,label='Time')
            ax.set_xlabel('q'); ax.set_ylabel('p')
        else:
            sc = ax.scatter(y[:,0],y[:,1],c=t,cmap='plasma',s=12,alpha=0.85,edgecolors='none')
            self.phase_canvas.fig.colorbar(sc,ax=ax,label='Time')
            ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_aspect('equal')
        ax.set_title(f"{sys.name} — Phase Space",fontweight='bold')
        self.phase_canvas.draw()

    def _plot_energy_landscape(self, sys, t, y, half):
        ax = self.energy_landscape_canvas.ax; ax.clear()
        if half == 1:
            qmin,qmax = y[:,0].min(),y[:,0].max()
            pmin,pmax = y[:,1].min(),y[:,1].max()
            qg = np.linspace(qmin,qmax,100); pg = np.linspace(pmin,pmax,100)
            Q,P = np.meshgrid(qg,pg)
            E = np.zeros_like(Q)
            for i in range(Q.shape[0]):
                for j in range(Q.shape[1]):
                    E[i,j] = sys.hamiltonian(Q[i,j],P[i,j])
            cf = ax.contourf(Q,P,E,levels=40,cmap='inferno')
            self.energy_landscape_canvas.fig.colorbar(cf,ax=ax,label='Energy')
            ax.plot(y[:,0],y[:,1],'w-',lw=2,alpha=0.9,label='Trajectory')
            ax.plot(y[0,0],y[0,1],'go',ms=10,label='Start')
            ax.set_xlabel('q'); ax.set_ylabel('p')
        else:
            ax.text(0.5,0.5,"Energy landscape for 1D systems only",
                    ha='center',va='center',transform=ax.transAxes)
        ax.set_title(f"{sys.name} — Energy Landscape",fontweight='bold')
        self.energy_landscape_canvas.draw()

    def _plot_vectorfield(self, sys, t, y, half):
        ax = self.vf_canvas.ax; ax.clear()
        if half == 1:
            qg = np.linspace(y[:,0].min(),y[:,0].max(),25)
            pg = np.linspace(y[:,1].min(),y[:,1].max(),25)
            Q,P = np.meshgrid(qg,pg)
            DQ = np.zeros_like(Q); DP = np.zeros_like(P)
            for i in range(Q.shape[0]):
                for j in range(Q.shape[1]):
                    d = sys.equations(0,[Q[i,j],P[i,j]])
                    DQ[i,j]=d[0]; DP[i,j]=d[1]
            speed = np.sqrt(DQ**2+DP**2)
            ax.streamplot(Q,P,DQ,DP,color=speed,cmap='viridis',
                         density=1.2,arrowsize=0.8)
            ax.scatter(y[:,0],y[:,1],c=t,cmap='plasma',s=15,alpha=0.8,
                      edgecolors='none',zorder=5)
            ax.set_xlabel('q'); ax.set_ylabel('p')
        else:
            ax.text(0.5,0.5,"Vector field for 1D systems only",
                    ha='center',va='center',transform=ax.transAxes)
        ax.set_title(f"{sys.name} — Vector Field (streamplot)",fontweight='bold')
        self.vf_canvas.draw()

    def _plot_time(self, t, y, half):
        ax = self.time_canvas.ax; ax.clear()
        ax.plot(t,y[:,0],color=THEME['truth'],lw=2,label='q₁')
        if half>=2: ax.plot(t,y[:,1],color=THEME['success'],lw=2,ls='--',label='q₂')
        ax.set_xlabel('Time'); ax.set_ylabel('Position')
        ax.set_title("Time Series",fontweight='bold'); ax.legend(); ax.grid(True,alpha=0.3)
        self.time_canvas.draw()

    def _plot_energy_truth(self, sys, t, y, half):
        ax = self.energy_canvas.ax; ax.clear()
        E = np.array([sys.hamiltonian(y[i,:half],y[i,half:]) for i in range(len(y))])
        ax.plot(t,E,color=THEME['success'],lw=2,label='H (analytical)')
        ax.axhline(E[0],color=THEME['truth'],ls=':',alpha=0.5,lw=2)
        ax.set_xlabel('Time'); ax.set_ylabel('Energy')
        ax.set_title("Energy Conservation",fontweight='bold'); ax.legend(); ax.grid(True,alpha=0.3)
        self.energy_canvas.draw()

    def _plot_fourier(self, t, y, half):
        ax = self.fourier_canvas.ax; ax.clear()
        dt = t[1]-t[0]; xf,amp = compute_fourier(y[:,0],dt)
        ax.semilogy(xf,amp,color=THEME['accent2'],lw=2)
        ax.set_xlabel('Frequency'); ax.set_ylabel('Amplitude')
        ax.set_title("Fourier Spectrum (q₁)",fontweight='bold'); ax.grid(True,alpha=0.3,which='both')
        self.fourier_canvas.draw()

    def _plot_traj2d(self, t, y, half):
        ax = self.traj2d_canvas.ax; ax.clear()
        if half>=2:
            sc = ax.scatter(y[:,0],y[:,1],c=t,cmap='plasma',s=15,alpha=0.85,edgecolors='none')
            self.traj2d_canvas.fig.colorbar(sc,ax=ax,label='Time')
            ax.plot(y[0,0],y[0,1],'go',ms=12,label='Start')
            ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')
        else:
            ax.text(0.5,0.5,"Not applicable for 1D systems",
                    ha='center',va='center',transform=ax.transAxes)
        ax.set_title("2D Trajectory",fontweight='bold')
        self.traj2d_canvas.draw()

    def _plot_phase3d(self, t, y, half):
        self.phase3d_canvas.fig.clear()
        ax = self.phase3d_canvas.fig.add_subplot(111, projection='3d')
        ax.set_facecolor(THEME['panel'])
        if half>=2:
            ax.plot(y[:,0],y[:,1],y[:,2],color=THEME['accent2'],lw=1.2,alpha=0.8)
            ax.scatter(y[0,0],y[0,1],y[0,2],color=THEME['accent3'],s=60,label='Start')
            ax.set_xlabel('q₁'); ax.set_ylabel('q₂'); ax.set_zlabel('p₁')
        elif half==1:
            # Plot (q, p, H(q,p))
            sys = self._get_system()
            H = np.array([sys.hamiltonian(y[i,0],y[i,1]) for i in range(len(y))])
            ax.plot(y[:,0],y[:,1],H,color=THEME['accent'],lw=1.5)
            ax.set_xlabel('q'); ax.set_ylabel('p'); ax.set_zlabel('H')
        ax.set_title("3D Phase Space",fontweight='bold',color=THEME['text'])
        ax.tick_params(colors=THEME['text_dim'])
        ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
        self.phase3d_canvas.draw()

    def _update_animation(self):
        if self.anim_y is None: return
        self.anim_idx = (self.anim_idx+1)%len(self.anim_y)
        y = self.anim_y[self.anim_idx]
        self.current_system.render_state(y, self.anim_canvas.ax)
        self.anim_canvas.draw()

    # ---------- training ----------
    def _start_training(self):
        if self.trainer and self.trainer.isRunning(): return
        sys,y,dy,tr,va,te = self._generate_data()
        self._eval_sys = sys; self._eval_y0 = self._get_y0()
        self._live_losses = []
        cfg = {'hidden':self.hidden_spin.value(),'layers':self.layers_spin.value(),
               'activation':self.act_combo.currentText(),'epochs':self.epochs_spin.value(),
               'lr':self.lr_spin.value(),'weight_decay':self.wd_spin.value(),
               'grad_clip':self.grad_clip_spin.value(),
               'seed':self.seed_spin.value(),'device':self.device_combo.currentText(),
               'integrator':self.integrator_combo.currentText()}
        mt = self.model_combo.currentText()
        self._queue = [mt]
        if self.compare_chk.isChecked():
            for m in ["HNN","LNN","Vanilla MLP"]:
                if m!=mt: self._queue.append(m)
        self._train_next(sys,tr,va,cfg)

    def _train_next(self, sys, tr, va, cfg):
        if not self._queue: return
        mt = self._queue[0]
        self._cur_mt = mt
        self.trainer = Trainer(mt, sys, tr, va, cfg)
        self.trainer.progress.connect(self._on_progress)
        self.trainer.finished.connect(self._on_finished)
        self.trainer.start()
        self.train_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self.progress.setValue(0); self._live_losses=[]
        self.cards["Status"].set_value(f"Training {mt}")

    def _stop_training(self):
        if self.trainer: self.trainer.stop()

    def _on_progress(self, pct, info):
        self.progress.setValue(pct)
        self.cards["Epoch"].set_value(f"{pct}%")
        self.cards["Train Loss"].set_value(f"{info['loss']:.2e}")
        self.cards["Val Loss"].set_value(f"{info['val_loss']:.2e}")
        self.cards["LR"].set_value(f"{info['lr']:.2e}")
        self.cards["Elapsed (s)"].set_value(f"{info['elapsed']:.1f}")
        self._live_losses.append(info['loss'])
        # Update live loss plot
        ax = self.live_canvas.ax; ax.clear()
        ax.semilogy(self._live_losses, color=THEME['accent2'], lw=1.5)
        ax.set_xlabel("epoch",fontsize=9); ax.set_ylabel("loss",fontsize=9)
        ax.set_title(f"{self._cur_mt} Loss",fontsize=10)
        ax.grid(True,alpha=0.3,which='both')
        self.live_canvas.draw()

    def _on_finished(self, result):
        mt = self._cur_mt
        self.models[mt] = result['model']
        self.metrics[mt] = result['metrics']
        self._queue.pop(0)
        if self._queue:
            sys,_,_,tr,va,_ = self._generate_data()
            cfg = {'hidden':self.hidden_spin.value(),'layers':self.layers_spin.value(),
                   'activation':self.act_combo.currentText(),'epochs':self.epochs_spin.value(),
                   'lr':self.lr_spin.value(),'weight_decay':self.wd_spin.value(),
                   'grad_clip':self.grad_clip_spin.value(),
                   'seed':self.seed_spin.value(),'device':self.device_combo.currentText(),
                   'integrator':self.integrator_combo.currentText()}
            self._train_next(sys,tr,va,cfg); return
        # Done
        self.train_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self.progress.setValue(100)
        self.cards["Status"].set_value("Complete ✓")
        # Update metrics cards with first model
        m = self.metrics[list(self.metrics.keys())[0]]
        self.cards["RMSE"].set_value(f"{m['rmse']:.4f}")
        self.cards["Energy Drift"].set_value(f"{m['energy_drift']:.4f}")
        self.cards["Max |ΔE|"].set_value(f"{m['energy_err_max']:.4f}")
        self.cards["Params"].set_value(f"{m['n_params']:,}")
        # Model info
        info = []
        for mt_, m_ in self.metrics.items():
            info.append(f"── {mt_} ──")
            info.append(f"  RMSE:         {m_['rmse']:.4f}")
            info.append(f"  Energy drift: {m_['energy_drift']:.4f}")
            info.append(f"  Max |ΔE|:     {m_['energy_err_max']:.4f}")
            info.append(f"  Params:       {m_['n_params']:,}")
        self.model_info.setPlainText("\n".join(info))
        # Status
        lines = ["Training complete.\n"]
        for mt_, m_ in self.metrics.items():
            lines.append(f"{mt_}: RMSE={m_['rmse']:.4f}, ΔE={m_['energy_drift']:.4f}")
        self.status_label.setText("\n".join(lines))
        self._plot_results()

    # ---------- result plots ----------
    def _plot_results(self):
        sys = self._eval_sys; half = sys.dim_q
        mts = list(self.metrics.keys())
        colors = {'HNN':THEME['HNN'],'LNN':THEME['LNN'],'Vanilla MLP':THEME['Vanilla']}
        y_true = self.metrics[mts[0]]['y_true']
        t = self.metrics[mts[0]]['t']

        # Phase
        ax = self.phase_canvas.ax; ax.clear()
        if half==1:
            ax.plot(y_true[:,0],y_true[:,1],color=THEME['truth'],lw=2,alpha=0.7,label='Truth')
        else:
            ax.plot(y_true[:,0],y_true[:,1],color=THEME['truth'],lw=2,alpha=0.7,label='Truth')
            ax.set_aspect('equal')
        for mt in mts:
            yp = self.metrics[mt]['y_pred']
            ax.plot(yp[:,0],yp[:,half],color=colors[mt],ls='--',lw=1.8,label=mt)
        ax.set_xlabel('q' if half==1 else 'x')
        ax.set_ylabel('p' if half==1 else 'y')
        ax.set_title(f"{sys.name} — Phase Space",fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True,alpha=0.3)
        self.phase_canvas.draw()

        # Time
        ax = self.time_canvas.ax; ax.clear()
        ax.plot(t,y_true[:,0],color=THEME['truth'],lw=2,label='q₁ (truth)',alpha=0.7)
        if half>=2: ax.plot(t,y_true[:,1],color=THEME['success'],lw=2,ls='--',label='q₂ (truth)',alpha=0.7)
        for mt in mts:
            yp = self.metrics[mt]['y_pred']
            ax.plot(t,yp[:,0],color=colors[mt],ls='--',lw=1.5,label=f'q₁ ({mt})')
            if half>=2: ax.plot(t,yp[:,1],color=colors[mt],ls=':',lw=1.5,label=f'q₂ ({mt})')
        ax.set_xlabel('Time'); ax.set_ylabel('Position')
        ax.set_title(f"{sys.name} — Time Series",fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
        self.time_canvas.draw()

        # Energy
        ax = self.energy_canvas.ax; ax.clear()
        for mt in mts:
            m = self.metrics[mt]
            ax.plot(m['t'],m['E_pred'],color=colors[mt],ls='--',lw=1.5,label=f'H ({mt})')
        ax.plot(t,self.metrics[mts[0]]['E_true'],color=THEME['truth'],lw=2,label='H (truth)',alpha=0.7)
        E0 = self.metrics[mts[0]]['E_true'][0]
        ax.axhline(E0,color=THEME['text_dim'],ls=':',alpha=0.5,lw=2)
        ax.set_xlabel('Time'); ax.set_ylabel('Energy')
        ax.set_title(f"{sys.name} — Energy Conservation",fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
        self.energy_canvas.draw()

        # Integrator comparison
        ax = self.integrator_canvas.ax; ax.clear()
        if mts:
            mt = mts[0]; model = self.models[mt]; y0 = self._eval_y0
            t_span = sys.time_span(); dt=0.02
            t_eval = np.arange(t_span[0],t_span[1],dt)
            for method in ['euler','rk4','leapfrog']:
                if method=='leapfrog' and mt!='HNN': continue
                yp = integrate_trajectory(model.f,y0,t_eval,method=method)
                Ep = np.array([sys.hamiltonian(yp[i,:half],yp[i,half:]) for i in range(len(yp))])
                ax.plot(t_eval,Ep-Ep[0],lw=1.5,label=method)
            ax.set_xlabel('Time'); ax.set_ylabel('ΔE')
            ax.set_title(f"{mt} — Energy Error by Integrator",fontweight='bold')
            ax.legend(fontsize=9); ax.grid(True,alpha=0.3)
        self.integrator_canvas.draw()

        # 2D trajectory
        ax = self.traj2d_canvas.ax; ax.clear()
        if half>=2:
            ax.plot(y_true[:,0],y_true[:,1],color=THEME['truth'],lw=2,alpha=0.7,label='Truth')
            ax.plot(y_true[0,0],y_true[0,1],'go',ms=12,label='Start')
            for mt in mts:
                yp = self.metrics[mt]['y_pred']
                ax.plot(yp[:,0],yp[:,1],color=colors[mt],ls='--',lw=1.5,label=mt)
            ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')
        else:
            ax.text(0.5,0.5,"Not applicable for 1D systems",
                    ha='center',va='center',transform=ax.transAxes)
        ax.set_title(f"{sys.name} — 2D Trajectory",fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
        self.traj2d_canvas.draw()

        # 3D phase
        self._plot_phase3d(t, y_true, half)
        # Re-plot with predictions overlaid
        ax3d = self.phase3d_canvas.fig.axes[0]
        for mt in mts:
            yp = self.metrics[mt]['y_pred']
            if half>=2:
                ax3d.plot(yp[:,0],yp[:,1],yp[:,2],color=colors[mt],ls='--',lw=1.2,label=mt)
            else:
                H = np.array([sys.hamiltonian(yp[i,0],yp[i,1]) for i in range(len(yp))])
                ax3d.plot(yp[:,0],yp[:,1],H,color=colors[mt],ls='--',lw=1.2,label=mt)
        ax3d.legend(fontsize=8)
        self.phase3d_canvas.draw()

        # Weights histogram
        ax = self.weights_canvas.ax; ax.clear()
        for mt in mts:
            model = self.models[mt]
            all_w = []
            for p in model.parameters():
                if p.dim()>=2: all_w.append(p.detach().cpu().numpy().flatten())
            if all_w:
                w = np.concatenate(all_w)
                ax.hist(w,bins=50,color=colors[mt],alpha=0.6,label=mt,density=True)
        ax.set_xlabel('Weight value'); ax.set_ylabel('Density')
        ax.set_title("Weight Distribution",fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True,alpha=0.3)
        self.weights_canvas.draw()

        # Error distribution
        ax = self.error_dist_canvas.ax; ax.clear()
        for mt in mts:
            m = self.metrics[mt]
            errs = np.linalg.norm(m['y_pred']-m['y_true'],axis=1)
            ax.hist(errs,bins=40,color=colors[mt],alpha=0.6,label=mt,density=True)
        ax.set_xlabel('‖y_pred − y_true‖'); ax.set_ylabel('Density')
        ax.set_title("Trajectory Error Distribution",fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True,alpha=0.3)
        self.error_dist_canvas.draw()

        # Comparison heatmap
        ax = self.heatmap_canvas.ax; ax.clear()
        if len(mts)>1:
            metric_names = ['RMSE','Energy Drift','Max |ΔE|','Params']
            data = []
            for mt in mts:
                m = self.metrics[mt]
                data.append([m['rmse'],m['energy_drift'],m['energy_err_max'],m['n_params']])
            data = np.array(data)
            # Normalize per column
            norm = np.zeros_like(data)
            for j in range(data.shape[1]):
                col = data[:,j]
                mn,mx = col.min(),col.max()
                if mx>mn: norm[:,j] = (col-mn)/(mx-mn)
                else: norm[:,j] = 0
            im = ax.imshow(norm,cmap='inferno',aspect='auto')
            ax.set_xticks(range(len(metric_names))); ax.set_xticklabels(metric_names,rotation=30,ha='right')
            ax.set_yticks(range(len(mts))); ax.set_yticklabels(mts)
            for i in range(len(mts)):
                for j in range(len(metric_names)):
                    ax.text(j,i,f"{data[i,j]:.2e}",ha='center',va='center',
                            color='white' if norm[i,j]<0.5 else 'black',fontsize=9)
            self.heatmap_canvas.fig.colorbar(im,ax=ax,label='Normalized')
            ax.set_title("Model Comparison",fontweight='bold')
        else:
            ax.text(0.5,0.5,"Train multiple models to see comparison",
                    ha='center',va='center',transform=ax.transAxes)
            ax.set_title("Model Comparison",fontweight='bold')
        self.heatmap_canvas.draw()

        # Loss
        ax = self.loss_canvas.ax; ax.clear()
        for mt in mts:
            m = self.metrics[mt]
            ax.semilogy(m['losses'],color=colors[mt],lw=1.2,label=f'{mt} train')
            ax.semilogy(m['val_losses'],color=colors[mt],ls='--',lw=1.2,label=f'{mt} val')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.set_title("Training & Validation Loss",fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True,alpha=0.3,which='both')
        self.loss_canvas.draw()

    # ---------- exports ----------
    def _export_model(self):
        if not self.models:
            QMessageBox.information(self,"Export","Train a model first."); return
        path,_ = QFileDialog.getSaveFileName(self,"Export Model","model.pt","PyTorch (*.pt)")
        if not path: return
        payload = {'models':{mt:m.state_dict() for mt,m in self.models.items()},
                   'metrics':{mt:{k:v for k,v in m.items()
                                  if k not in ('losses','val_losses','y_pred','y_true','E_pred','E_true','t')}
                              for mt,m in self.metrics.items()},
                   'system':self._eval_sys.__class__.__name__,
                   'params':self._eval_sys.params}
        torch.save(payload, path)
        self.statusBar().showMessage(f"Exported to {path}",3000)

    def _save_plot(self):
        """Save current tab's plot as high-res PNG or SVG."""
        idx = self.tabs.currentIndex()
        w = self.tabs.widget(idx)
        canvas = w.findChild(PlotCanvas)
        if canvas is None:
            QMessageBox.information(self,"Save Plot","No plot in current tab."); return
        path,_ = QFileDialog.getSaveFileName(self,"Save Plot","plot.png","PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
        if not path: return
        canvas.fig.savefig(path, dpi=300, bbox_inches='tight',
                          facecolor=canvas.fig.get_facecolor())
        self.statusBar().showMessage(f"Plot saved to {path}",3000)

    def _save_config(self):
        path,_ = QFileDialog.getSaveFileName(self,"Save Config","config.json","JSON (*.json)")
        if not path: return
        cfg = {'system':self.system_combo.currentText(),
               'params':{k:float(s.value()) for k,s in self.param_spins.items()},
               'y0':[float(s.value()) for s in self.y0_spins if s.isVisible()],
               'n_traj':self.n_traj_spin.value(),'y0_perturb':self.y0_perturb_spin.value(),
               't_span':self.t_span_spin.value(),'dt':self.dt_spin.value(),
               'noise':self.noise_spin.value(),'train_frac':self.train_frac_spin.value(),
               'model':self.model_combo.currentText(),'hidden':self.hidden_spin.value(),
               'layers':self.layers_spin.value(),'activation':self.act_combo.currentText(),
               'epochs':self.epochs_spin.value(),'lr':self.lr_spin.value(),
               'weight_decay':self.wd_spin.value(),'grad_clip':self.grad_clip_spin.value(),
               'seed':self.seed_spin.value(),'device':self.device_combo.currentText(),
               'integrator':self.integrator_combo.currentText(),
               'compare':self.compare_chk.isChecked()}
        with open(path,'w') as f: json.dump(cfg,f,indent=2)
        self.statusBar().showMessage(f"Config saved to {path}",3000)

    def _load_config(self):
        path,_ = QFileDialog.getOpenFileName(self,"Load Config","","JSON (*.json)")
        if not path: return
        with open(path) as f: cfg = json.load(f)
        self.system_combo.setCurrentText(cfg['system'])
        for k,v in cfg['params'].items():
            if k in self.param_spins: self.param_spins[k].setValue(v)
        for i,v in enumerate(cfg['y0']):
            if i<len(self.y0_spins): self.y0_spins[i].setValue(v)
        self.n_traj_spin.setValue(cfg['n_traj'])
        self.y0_perturb_spin.setValue(cfg['y0_perturb'])
        self.t_span_spin.setValue(cfg['t_span'])
        self.dt_spin.setValue(cfg['dt'])
        self.noise_spin.setValue(cfg['noise'])
        self.train_frac_spin.setValue(cfg['train_frac'])
        self.model_combo.setCurrentText(cfg['model'])
        self.hidden_spin.setValue(cfg['hidden'])
        self.layers_spin.setValue(cfg['layers'])
        self.act_combo.setCurrentText(cfg['activation'])
        self.epochs_spin.setValue(cfg['epochs'])
        self.lr_spin.setValue(cfg['lr'])
        self.wd_spin.setValue(cfg['weight_decay'])
        self.grad_clip_spin.setValue(cfg['grad_clip'])
        self.seed_spin.setValue(cfg['seed'])
        self.device_combo.setCurrentText(cfg['device'])
        self.integrator_combo.setCurrentText(cfg['integrator'])
        self.compare_chk.setChecked(cfg['compare'])
        self.statusBar().showMessage(f"Config loaded from {path}",3000)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    w = MainWindow(); w.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()