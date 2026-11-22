import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import json

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QGroupBox, QSlider, QLabel, QPushButton, 
                               QComboBox, QSplitter, QSpinBox, QStatusBar, QFileDialog,
                               QScrollArea, QTabWidget, QGridLayout, QToolTip, QMessageBox,
                               QCheckBox, QDoubleSpinBox)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QPoint
from PySide6.QtGui import QKeySequence, QShortcut

# ==========================================
# 1. STYLING (Light & Dark Modes)
# ==========================================
LIGHT_QSS = """
    QMainWindow, QWidget { background-color: #f4f6f9; color: #2c3e50; }
    QGroupBox { font-weight: bold; border: 1px solid #dcdde1; border-radius: 8px; margin-top: 12px; padding-top: 14px; background-color: #ffffff; }
    QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #2c3e50; background-color: #ffffff; }
    QPushButton { padding: 8px 12px; border-radius: 6px; background-color: #3498db; color: white; border: none; font-weight: bold; }
    QPushButton:hover { background-color: #2980b9; }
    QPushButton:checked { background-color: #e74c3c; }
    QPushButton:checked:hover { background-color: #c0392b; }
    QSpinBox, QDoubleSpinBox, QComboBox { padding: 6px; border: 1px solid #dcdde1; border-radius: 6px; background: white; }
    QTabWidget::pane { border: 1px solid #dcdde1; border-radius: 8px; background: white; }
    QTabBar::tab { padding: 10px 20px; font-weight: bold; background: #ecf0f1; border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 4px; }
    QTabBar::tab:selected { background: #3498db; color: white; }
    QScrollArea { border: none; }
    QCheckBox { spacing: 8px; }
"""

DARK_QSS = """
    QMainWindow, QWidget { background-color: #1e1e1e; color: #ecf0f1; }
    QGroupBox { font-weight: bold; border: 1px solid #444; border-radius: 8px; margin-top: 12px; padding-top: 14px; background-color: #2d2d2d; }
    QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #ecf0f1; background-color: #2d2d2d; }
    QPushButton { padding: 8px 12px; border-radius: 6px; background-color: #2980b9; color: white; border: none; font-weight: bold; }
    QPushButton:hover { background-color: #3498db; }
    QPushButton:checked { background-color: #c0392b; }
    QPushButton:checked:hover { background-color: #e74c3c; }
    QSpinBox, QDoubleSpinBox, QComboBox { padding: 6px; border: 1px solid #555; border-radius: 6px; background-color: #333; color: white; }
    QTabWidget::pane { border: 1px solid #444; border-radius: 8px; background: #2d2d2d; }
    QTabBar::tab { padding: 10px 20px; font-weight: bold; background: #333; color: #aaa; border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 4px; }
    QTabBar::tab:selected { background: #2980b9; color: white; }
    QScrollArea { border: none; background: #1e1e1e; }
    QCheckBox { spacing: 8px; }
"""

# ==========================================
# 2. MOCK DEEPONET & PHYSICS
# ==========================================
class MockDeepONet:
    def __init__(self, operator_type="Heat Equation", n_sensors=30, latent_dim=15, n_query=100):
        self.alpha = 0.05  
        self.c_wave = 0.5  
        self.update_operator(operator_type)
        self.update_params(n_sensors, latent_dim, n_query)

    def update_operator(self, operator_type):
        self.operator_type = operator_type
        self.t_grid = np.linspace(0, 2.0, 60)

    def update_params(self, n_sensors, latent_dim, n_query):
        self.n_sensors = n_sensors
        self.latent_dim = latent_dim
        self.n_query = n_query
        
        self.x_sensors = np.linspace(0, 1, n_sensors)
        self.y_query = np.linspace(0, 1, n_query)
        
        if self.operator_type == "Anti-derivative":
            self.trunk_basis = np.array([(1 - np.cos(k * np.pi * self.y_query)) / (k * np.pi) for k in range(1, latent_dim + 1)])
        else:
            self.trunk_basis = np.array([np.sin(k * np.pi * self.y_query) for k in range(1, latent_dim + 1)])
            
        if self.operator_type == "Heat Equation":
            self.rates = self.alpha * (np.arange(1, latent_dim + 1) * np.pi)**2
        elif self.operator_type == "Wave Equation":
            self.rates = self.c_wave * (np.arange(1, latent_dim + 1) * np.pi)
        else:
            self.rates = np.zeros(latent_dim)

    def branch_network(self, u_sensors):
        b = np.zeros(self.latent_dim)
        for k in range(1, self.latent_dim + 1):
            basis_at_sensors = np.sin(k * np.pi * self.x_sensors)
            b[k-1] = np.trapz(u_sensors * basis_at_sensors, self.x_sensors) * 2
        return b

    def forward_spatiotemporal(self, u_sensors):
        b_init = self.branch_network(u_sensors)
        V = np.zeros((self.n_query, len(self.t_grid)))
        B_t = np.zeros((self.latent_dim, len(self.t_grid)))
        
        for i, t in enumerate(self.t_grid):
            if self.operator_type == "Heat Equation":
                b_t = b_init * np.exp(-self.rates * t)
            elif self.operator_type == "Wave Equation":
                b_t = b_init * np.cos(self.rates * t)
            else:
                b_t = b_init * t
            V[:, i] = np.dot(b_t, self.trunk_basis)
            B_t[:, i] = b_t
        return b_init, V, B_t

    def get_ground_truth(self, u_sensors_clean, t_idx):
        b_gt = np.zeros(100)
        for k in range(1, 101):
            basis = np.sin(k * np.pi * self.x_sensors)
            b_gt[k-1] = np.trapz(u_sensors_clean * basis, self.x_sensors) * 2
            
        t = self.t_grid[t_idx]
        if self.operator_type == "Heat Equation":
            rates_gt = self.alpha * (np.arange(1, 101) * np.pi)**2
            b_t = b_gt * np.exp(-rates_gt * t)
        elif self.operator_type == "Wave Equation":
            rates_gt = self.c_wave * (np.arange(1, 101) * np.pi)
            b_t = b_gt * np.cos(rates_gt * t)
        else:
            b_t = b_gt * t
            
        trunk_gt = np.array([np.sin(k * np.pi * self.y_query) for k in range(1, 101)])
        if self.operator_type == "Anti-derivative":
            trunk_gt = np.array([(1 - np.cos(k * np.pi * self.y_query)) / (k * np.pi) for k in range(1, 101)])
            
        return np.dot(b_t, trunk_gt)

# ==========================================
# 3. BACKGROUND WORKER
# ==========================================
class InferenceWorker(QThread):
    finished = Signal(object, object, object, object)

    def __init__(self, model, u_sensors):
        super().__init__()
        self.model = model
        self.u_sensors = u_sensors

    def run(self):
        QThread.msleep(15) 
        b_init, V, B_t = self.model.forward_spatiotemporal(self.u_sensors)
        self.finished.emit(self.u_sensors, b_init, V, B_t)

# ==========================================
# 4. MAIN UI
# ==========================================
class DeepONetVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepONet Pro Visualizer - Advanced Research Tool")
        self.resize(1600, 1000)
        
        self.model = MockDeepONet()
        self.current_t_idx = 0
        self.is_playing = False
        self.worker = None
        self.is_dark_mode = False
        self.last_data = None
        self.custom_input = None
        self.is_drawing = False
        self.comparison_data = []
        
        self.init_ui()
        self.setup_shortcuts()
        self.trigger_update()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # --- LEFT PANEL: CONTROLS (Scrollable) ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumWidth(360)
        scroll_widget = QWidget()
        control_layout = QVBoxLayout(scroll_widget)
        
        # 1. Presets
        grp_pre = QGroupBox("⚡ Quick Presets")
        grp_pre.setToolTip("Load pre-configured scenarios for common use cases")
        lay_pre = QVBoxLayout()
        self.combo_presets = QComboBox()
        self.combo_presets.addItems(["Smooth Heat Diffusion", "High Freq Wave", "Noisy Anti-derivative", "Custom Drawing", "Custom"])
        self.combo_presets.currentTextChanged.connect(self.apply_preset)
        lay_pre.addWidget(self.combo_presets)
        grp_pre.setLayout(lay_pre)
        control_layout.addWidget(grp_pre)
        
        # 2. Operator
        grp_op = QGroupBox("🌡️ Operator Selection")
        grp_op.setToolTip("Select the physical operator to visualize")
        lay_op = QVBoxLayout()
        self.combo_op = QComboBox()
        self.combo_op.addItems(["Heat Equation", "Wave Equation", "Anti-derivative"])
        self.combo_op.currentTextChanged.connect(self.change_operator)
        lay_op.addWidget(self.combo_op)
        grp_op.setLayout(lay_op)
        control_layout.addWidget(grp_op)
        
        # 3. Input
        grp_in = QGroupBox("📈 Input Function")
        grp_in.setToolTip("Configure the input function u(x) fed to the Branch network")
        lay_in = QVBoxLayout()
        self.combo_func = QComboBox()
        self.combo_func.addItems(["Sine Wave", "Square Wave", "Sawtooth", "Double Peak", "Custom Drawing"])
        self.combo_func.currentIndexChanged.connect(self.trigger_update)
        
        self.slider_amp = self._create_slider("Amplitude:", 1, 30, 10, self.trigger_update)
        self.slider_freq = self._create_slider("Frequency:", 1, 20, 4, self.trigger_update)
        self.slider_noise = self._create_slider("Noise:", 0, 50, 0, self.trigger_update)
        
        lay_in.addWidget(self.combo_func)
        lay_in.addWidget(self.slider_amp['layout'])
        lay_in.addWidget(self.slider_freq['layout'])
        lay_in.addWidget(self.slider_noise['layout'])
        grp_in.setLayout(lay_in)
        control_layout.addWidget(grp_in)
        
        # 4. Architecture
        grp_arch = QGroupBox("🧠 Architecture")
        grp_arch.setToolTip("Adjust network hyperparameters")
        lay_arch = QGridLayout()
        self.spin_sensors = QSpinBox(); self.spin_sensors.setRange(10, 100); self.spin_sensors.setValue(30)
        self.spin_sensors.setToolTip("Number of sensor points m")
        self.spin_sensors.valueChanged.connect(self.update_model_params)
        self.spin_latent = QSpinBox(); self.spin_latent.setRange(3, 50); self.spin_latent.setValue(15)
        self.spin_latent.setToolTip("Latent dimension p (number of basis functions)")
        self.spin_latent.valueChanged.connect(self.update_model_params)
        
        lay_arch.addWidget(QLabel("Sensors (m):"), 0, 0); lay_arch.addWidget(self.spin_sensors, 0, 1)
        lay_arch.addWidget(QLabel("Latent Dim (p):"), 1, 0); lay_arch.addWidget(self.spin_latent, 1, 1)
        grp_arch.setLayout(lay_arch)
        control_layout.addWidget(grp_arch)

        # 5. Time
        grp_time = QGroupBox("⏱️ Time Evolution")
        grp_time.setToolTip("Control temporal evolution of the operator output")
        lay_time = QVBoxLayout()
        self.slider_time = QSlider(Qt.Horizontal)
        self.slider_time.setRange(0, len(self.model.t_grid)-1)
        self.slider_time.setValue(0)
        self.slider_time.setToolTip("Drag to set time step (or click on heatmap)")
        self.slider_time.valueChanged.connect(self.update_time)
        self.lbl_time = QLabel("Time (t): 0.00")
        self.btn_play = QPushButton("▶ Play"); self.btn_play.setCheckable(True); self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setToolTip("Play/Pause animation (Space)")
        
        lay_time.addWidget(self.lbl_time); lay_time.addWidget(self.slider_time); lay_time.addWidget(self.btn_play)
        grp_time.setLayout(lay_time)
        control_layout.addWidget(grp_time)

        # 6. Comparison
        grp_comp = QGroupBox("🔍 Comparison Mode")
        grp_comp.setToolTip("Overlay multiple predictions for comparison")
        lay_comp = QVBoxLayout()
        self.btn_add_comparison = QPushButton("➕ Add Current to Comparison")
        self.btn_add_comparison.clicked.connect(self.add_to_comparison)
        self.btn_clear_comparison = QPushButton("🗑️ Clear Comparison")
        self.btn_clear_comparison.clicked.connect(self.clear_comparison)
        self.lbl_comp_count = QLabel("0 configurations stored")
        
        lay_comp.addWidget(self.btn_add_comparison)
        lay_comp.addWidget(self.btn_clear_comparison)
        lay_comp.addWidget(self.lbl_comp_count)
        grp_comp.setLayout(lay_comp)
        control_layout.addWidget(grp_comp)

        # 7. Actions
        grp_act = QGroupBox("⚙️ Actions")
        lay_act = QVBoxLayout()
        self.btn_theme = QPushButton("🌗 Toggle Dark Mode"); self.btn_theme.clicked.connect(self.toggle_theme)
        self.btn_export = QPushButton("💾 Export Data (CSV)"); self.btn_export.clicked.connect(self.export_data)
        self.btn_export_fig = QPushButton("📸 Export Figure (PNG/SVG)"); self.btn_export_fig.clicked.connect(self.export_figure)
        self.btn_save_config = QPushButton("💿 Save Configuration"); self.btn_save_config.clicked.connect(self.save_config)
        self.btn_load_config = QPushButton("📂 Load Configuration"); self.btn_load_config.clicked.connect(self.load_config)
        
        lay_act.addWidget(self.btn_theme)
        lay_act.addWidget(self.btn_export)
        lay_act.addWidget(self.btn_export_fig)
        lay_act.addWidget(self.btn_save_config)
        lay_act.addWidget(self.btn_load_config)
        grp_act.setLayout(lay_act)
        control_layout.addWidget(grp_act)

        control_layout.addStretch()
        scroll_area.setWidget(scroll_widget)
        
        # --- RIGHT PANEL: TABBED VISUALIZATIONS ---
        self.tab_widget = QTabWidget()
        self.tab_widget.currentChanged.connect(self.refresh_active_tab)
        
        # Tab 1: 1D Profiles & Architecture
        self.fig1 = Figure(figsize=(8, 6), dpi=100, facecolor='#ffffff')
        self.canvas1 = FigureCanvas(self.fig1)
        self.canvas1.mpl_connect('button_press_event', self.on_canvas_click)
        tab1_widget = QWidget(); lay1 = QVBoxLayout(tab1_widget)
        lay1.addWidget(NavigationToolbar(self.canvas1, self)); lay1.addWidget(self.canvas1)
        self.tab_widget.addTab(tab1_widget, "📐 1D Profiles & Architecture")
        
        # Tab 2: Spatiotemporal Evolution
        self.fig2 = Figure(figsize=(8, 6), dpi=100, facecolor='#ffffff')
        self.canvas2 = FigureCanvas(self.fig2)
        self.canvas2.mpl_connect('button_press_event', self.on_heatmap_click)
        tab2_widget = QWidget(); lay2 = QVBoxLayout(tab2_widget)
        lay2.addWidget(NavigationToolbar(self.canvas2, self)); lay2.addWidget(self.canvas2)
        self.tab_widget.addTab(tab2_widget, "🌊 Spatiotemporal Evolution")
        
        # Tab 3: Error Analysis
        self.fig3 = Figure(figsize=(8, 6), dpi=100, facecolor='#ffffff')
        self.canvas3 = FigureCanvas(self.fig3)
        tab3_widget = QWidget(); lay3 = QVBoxLayout(tab3_widget)
        lay3.addWidget(NavigationToolbar(self.canvas3, self)); lay3.addWidget(self.canvas3)
        self.tab_widget.addTab(tab3_widget, "📊 Error Analysis")
        
        # Tab 4: Latent Space Dynamics
        self.fig4 = Figure(figsize=(8, 6), dpi=100, facecolor='#ffffff')
        self.canvas4 = FigureCanvas(self.fig4)
        tab4_widget = QWidget(); lay4 = QVBoxLayout(tab4_widget)
        lay4.addWidget(NavigationToolbar(self.canvas4, self)); lay4.addWidget(self.canvas4)
        self.tab_widget.addTab(tab4_widget, "🎯 Latent Space Dynamics")
        
        # Tab 5: Frequency Analysis
        self.fig5 = Figure(figsize=(8, 6), dpi=100, facecolor='#ffffff')
        self.canvas5 = FigureCanvas(self.fig5)
        tab5_widget = QWidget(); lay5 = QVBoxLayout(tab5_widget)
        lay5.addWidget(NavigationToolbar(self.canvas5, self)); lay5.addWidget(self.canvas5)
        self.tab_widget.addTab(tab5_widget, "📊 Frequency Analysis")
        
        # Layout assembly
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(scroll_area)
        splitter.addWidget(self.tab_widget)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)
        
        # Status Bar
        self.statusBar().showMessage("Ready | Shortcuts: Space=Play/Pause, ←→=Step Time, D=Dark Mode")
        self.lbl_metrics = QLabel("Relative L2 Error: 0.00% | MSE: 0.0000")
        self.lbl_metrics.setStyleSheet("font-weight: bold; color: #d35400;")
        self.statusBar().addPermanentWidget(self.lbl_metrics)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.step_time)
        
        self.setStyleSheet(LIGHT_QSS)

    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        QShortcut(QKeySequence(Qt.Key_Space), self, self.toggle_play_shortcut)
        QShortcut(QKeySequence(Qt.Key_Left), self, self.step_time_backward)
        QShortcut(QKeySequence(Qt.Key_Right), self, self.step_time_forward)
        QShortcut(QKeySequence("D"), self, self.toggle_theme)
        QShortcut(QKeySequence("Ctrl+S"), self, self.save_config)
        QShortcut(QKeySequence("Ctrl+O"), self, self.load_config)

    # --- Helper Methods ---
    def _create_slider(self, label, min_val, max_val, default, callback):
        lbl = QLabel(f"{label} {default/10.0 if label != 'Noise:' else default/100.0:.2f}")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val); slider.setValue(default)
        slider.valueChanged.connect(lambda: self._update_slider_label(lbl, label, slider))
        slider.valueChanged.connect(callback)
        layout = QHBoxLayout(); layout.addWidget(lbl); layout.addWidget(slider)
        container = QWidget(); container.setLayout(layout)
        return {'label': lbl, 'slider': slider, 'layout': container}

    def _update_slider_label(self, lbl, prefix, slider):
        divisor = 100.0 if "Noise" in prefix else 10.0
        lbl.setText(f"{prefix} {slider.value() / divisor:.2f}")

    def generate_input_function(self, noise=True):
        if self.custom_input is not None and self.combo_func.currentText() == "Custom Drawing":
            return self.custom_input
            
        amp = self.slider_amp['slider'].value() / 10.0
        freq = self.slider_freq['slider'].value() / 2.0
        noise_level = self.slider_noise['slider'].value() / 100.0 if noise else 0.0
        func_type = self.combo_func.currentText()
        x = self.model.x_sensors
        
        if func_type == "Sine Wave": u_cont = amp * np.sin(freq * np.pi * x)
        elif func_type == "Square Wave": u_cont = amp * np.sign(np.sin(freq * np.pi * x))
        elif func_type == "Sawtooth": u_cont = amp * (2 * (freq * x % 1) - 1)
        else: u_cont = amp * (np.exp(-50 * (x - 0.3)**2) + np.exp(-50 * (x - 0.7)**2))

        return u_cont + np.random.randn(len(x)) * noise_level

    # --- Control Callbacks ---
    def apply_preset(self, preset_name):
        if preset_name == "Smooth Heat Diffusion":
            self.combo_op.setCurrentText("Heat Equation"); self.combo_func.setCurrentText("Double Peak")
            self.slider_amp['slider'].setValue(10); self.slider_freq['slider'].setValue(4); self.slider_noise['slider'].setValue(0)
        elif preset_name == "High Freq Wave":
            self.combo_op.setCurrentText("Wave Equation"); self.combo_func.setCurrentText("Sine Wave")
            self.slider_amp['slider'].setValue(10); self.slider_freq['slider'].setValue(16); self.slider_noise['slider'].setValue(0)
        elif preset_name == "Noisy Anti-derivative":
            self.combo_op.setCurrentText("Anti-derivative"); self.combo_func.setCurrentText("Square Wave")
            self.slider_amp['slider'].setValue(15); self.slider_freq['slider'].setValue(6); self.slider_noise['slider'].setValue(20)
        elif preset_name == "Custom Drawing":
            self.combo_func.setCurrentText("Custom Drawing")
            self.custom_input = None
        self.trigger_update()

    def change_operator(self, op_type):
        self.model.update_operator(op_type)
        self.slider_time.setMaximum(len(self.model.t_grid)-1)
        self.trigger_update()

    def update_model_params(self):
        self.model.update_params(self.spin_sensors.value(), self.spin_latent.value(), self.model.n_query)
        self.trigger_update()

    def update_time(self):
        self.current_t_idx = self.slider_time.value()
        self.lbl_time.setText(f"Time (t): {self.model.t_grid[self.current_t_idx]:.2f}")
        self.trigger_update()

    def toggle_play(self, checked):
        self.is_playing = checked
        self.btn_play.setText("⏸ Pause" if checked else "▶ Play")
        if checked: self.timer.start(50)
        else: self.timer.stop()

    def toggle_play_shortcut(self):
        self.btn_play.setChecked(not self.btn_play.isChecked())
        self.toggle_play(self.btn_play.isChecked())

    def step_time(self):
        val = self.slider_time.value() + 1
        if val > self.slider_time.maximum(): self.slider_time.setValue(0)
        else: self.slider_time.setValue(val)

    def step_time_forward(self):
        val = min(self.slider_time.value() + 1, self.slider_time.maximum())
        self.slider_time.setValue(val)

    def step_time_backward(self):
        val = max(self.slider_time.value() - 1, 0)
        self.slider_time.setValue(val)

    def toggle_theme(self):
        self.is_dark_mode = not self.is_dark_mode
        self.setStyleSheet(DARK_QSS if self.is_dark_mode else LIGHT_QSS)
        self.trigger_update()

    def add_to_comparison(self):
        if self.last_data:
            u, b, V, B_t = self.last_data
            config = {
                'operator': self.combo_op.currentText(),
                'func': self.combo_func.currentText(),
                'amp': self.slider_amp['slider'].value(),
                'freq': self.slider_freq['slider'].value(),
                'noise': self.slider_noise['slider'].value(),
                'latent': self.spin_latent.value(),
                'V': V.copy(),
                'label': f"{self.combo_op.currentText()} p={self.spin_latent.value()}"
            }
            self.comparison_data.append(config)
            self.lbl_comp_count.setText(f"{len(self.comparison_data)} configurations stored")
            self.statusBar().showMessage(f"Added configuration {len(self.comparison_data)} to comparison", 2000)

    def clear_comparison(self):
        self.comparison_data = []
        self.lbl_comp_count.setText("0 configurations stored")
        self.statusBar().showMessage("Comparison cleared", 2000)

    def export_data(self):
        if not self.last_data: return
        path, _ = QFileDialog.getSaveFileName(self, "Save Data", "", "CSV Files (*.csv)")
        if path:
            u, b, V, B_t = self.last_data
            data = np.column_stack([self.model.y_query, V[:, self.current_t_idx]])
            np.savetxt(path, data, header="y, v_pred", delimiter=",", comments="")
            self.statusBar().showMessage(f"Exported to {path}", 3000)

    def export_figure(self):
        current_tab = self.tab_widget.currentIndex()
        if current_tab == 0: fig = self.fig1
        elif current_tab == 1: fig = self.fig2
        elif current_tab == 2: fig = self.fig3
        elif current_tab == 3: fig = self.fig4
        else: fig = self.fig5
        
        path, _ = QFileDialog.getSaveFileName(self, "Save Figure", "", "PNG Files (*.png);;SVG Files (*.svg);;PDF Files (*.pdf)")
        if path:
            fig.savefig(path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
            self.statusBar().showMessage(f"Figure exported to {path}", 3000)

    def save_config(self):
        config = {
            'operator': self.combo_op.currentText(),
            'func': self.combo_func.currentText(),
            'amp': self.slider_amp['slider'].value(),
            'freq': self.slider_freq['slider'].value(),
            'noise': self.slider_noise['slider'].value(),
            'sensors': self.spin_sensors.value(),
            'latent': self.spin_latent.value(),
            'time_idx': self.current_t_idx
        }
        path, _ = QFileDialog.getSaveFileName(self, "Save Configuration", "", "JSON Files (*.json)")
        if path:
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
            self.statusBar().showMessage(f"Configuration saved to {path}", 3000)

    def load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "", "JSON Files (*.json)")
        if path:
            with open(path, 'r') as f:
                config = json.load(f)
            self.combo_op.setCurrentText(config['operator'])
            self.combo_func.setCurrentText(config['func'])
            self.slider_amp['slider'].setValue(config['amp'])
            self.slider_freq['slider'].setValue(config['freq'])
            self.slider_noise['slider'].setValue(config['noise'])
            self.spin_sensors.setValue(config['sensors'])
            self.spin_latent.setValue(config['latent'])
            self.slider_time.setValue(config['time_idx'])
            self.statusBar().showMessage(f"Configuration loaded from {path}", 3000)

    def on_canvas_click(self, event):
        """Handle clicks on Tab 1 for custom input drawing"""
        if self.combo_func.currentText() != "Custom Drawing":
            return
        if event.inaxes is None or event.inaxes.get_title() != "Input Function $u(x)$":
            return
            
        if event.button == 1:  # Left click - start drawing
            self.is_drawing = True
            self.custom_input = np.zeros_like(self.model.x_sensors)
            self.update_custom_point(event)
        elif event.button == 3:  # Right click - clear
            self.custom_input = None
            self.trigger_update()

    def on_heatmap_click(self, event):
        """Handle clicks on heatmap to set time"""
        if event.inaxes is None or event.inaxes.get_title() != "2D Spatiotemporal Heatmap":
            return
        if event.xdata is not None:
            t_click = event.xdata
            t_idx = np.argmin(np.abs(self.model.t_grid - t_click))
            self.slider_time.setValue(t_idx)

    def update_custom_point(self, event):
        if not self.is_drawing or event.xdata is None:
            return
        x_idx = np.argmin(np.abs(self.model.x_sensors - event.xdata))
        self.custom_input[x_idx] = event.ydata
        self.trigger_update()

    def trigger_update(self):
        u_sensors = self.generate_input_function()
        if self.worker and self.worker.isRunning(): self.worker.quit()
        self.worker = InferenceWorker(self.model, u_sensors)
        self.worker.finished.connect(self.update_plots)
        self.worker.start()

    def refresh_active_tab(self):
        if self.last_data:
            self.update_plots(*self.last_data)

    # --- Plotting Logic ---
    def update_plots(self, u_sensors, b_init, V, B_t):
        self.last_data = (u_sensors, b_init, V, B_t)
        current_tab = self.tab_widget.currentIndex()
        
        text_color = 'white' if self.is_dark_mode else 'black'
        bg_color = '#1e1e1e' if self.is_dark_mode else '#ffffff'
        
        # Calculate Ground Truth & Metrics
        u_clean = self.generate_input_function(noise=False)
        v_gt = self.model.get_ground_truth(u_clean, self.current_t_idx)
        v_pred = V[:, self.current_t_idx]
        mse = np.mean((v_pred - v_gt)**2)
        l2 = np.linalg.norm(v_pred - v_gt) / (np.linalg.norm(v_gt) + 1e-8) * 100
        self.lbl_metrics.setText(f"Relative L2 Error: {l2:.2f}% | MSE: {mse:.2e}")
        self.lbl_metrics.setStyleSheet(f"font-weight: bold; color: {'#f39c12' if self.is_dark_mode else '#d35400'};")

        if current_tab == 0:
            self.draw_tab1(u_sensors, b_init, V, text_color, bg_color)
            self.canvas1.draw()
        elif current_tab == 1:
            self.draw_tab2(V, text_color, bg_color)
            self.canvas2.draw()
        elif current_tab == 2:
            self.draw_tab3(v_pred, v_gt, text_color, bg_color)
            self.canvas3.draw()
        elif current_tab == 3:
            self.draw_tab4(B_t, text_color, bg_color)
            self.canvas4.draw()
        elif current_tab == 4:
            self.draw_tab5(u_sensors, v_pred, text_color, bg_color)
            self.canvas5.draw()

    def draw_tab1(self, u_sensors, b_init, V, text_color, bg_color):
        self.fig1.clear(); self.fig1.set_facecolor(bg_color)
        
        # 1. Architecture
        ax_arch = self.fig1.add_subplot(231)
        self.draw_architecture(ax_arch, text_color, bg_color)
        
        # 2. Input
        ax_in = self.fig1.add_subplot(232)
        ax_in.plot(self.model.x_sensors, u_sensors, 'o', color='#3498db', markersize=5, label='Sensors')
        if self.combo_func.currentText() == "Custom Drawing":
            ax_in.set_title("Input Function $u(x)$ [CUSTOM - Click to draw, Right-click to clear]", color=text_color, fontweight='bold')
        else:
            ax_in.set_title("Input Function $u(x)$", color=text_color, fontweight='bold')
        ax_in.set_ylabel("u(x)", color=text_color); ax_in.tick_params(colors=text_color)
        ax_in.grid(True, linestyle='--', alpha=0.3, color=text_color)
        
        # 3. Output with comparison
        ax_out = self.fig1.add_subplot(233)
        current_t = self.model.t_grid[self.current_t_idx]
        
        # Plot comparison configurations
        for i, comp in enumerate(self.comparison_data):
            ax_out.plot(self.model.y_query, comp['V'][:, self.current_t_idx], 
                       '--', linewidth=1.5, alpha=0.6, label=comp['label'])
        
        ax_out.plot(self.model.y_query, V[:, self.current_t_idx], 'r-', linewidth=2.5, 
                   label=f'Current (t={current_t:.2f})')
        ax_out.set_title("Operator Output $v(y)$", color=text_color, fontweight='bold')
        ax_out.set_ylabel("v(y)", color=text_color); ax_out.tick_params(colors=text_color)
        if len(self.comparison_data) > 0:
            ax_out.legend(facecolor=bg_color, edgecolor=text_color, labelcolor=text_color, fontsize=8)
        ax_out.grid(True, linestyle='--', alpha=0.3, color=text_color)
        
        # 4. Branch
        ax_br = self.fig1.add_subplot(234)
        ax_br.bar(range(1, self.model.latent_dim + 1), b_init, color='#e74c3c', alpha=0.8)
        ax_br.set_title("Branch Latent Weights $b_k$", color=text_color, fontweight='bold')
        ax_br.set_ylabel("Weight", color=text_color); ax_br.tick_params(colors=text_color)
        ax_br.grid(True, axis='y', linestyle='--', alpha=0.3, color=text_color)
        
        # 5. Trunk
        ax_tr = self.fig1.add_subplot(235)
        cmap = plt.cm.viridis
        for i in range(self.model.latent_dim):
            ax_tr.plot(self.model.y_query, self.model.trunk_basis[i], color=cmap(i/self.model.latent_dim), alpha=0.8)
        ax_tr.set_title("Trunk Basis Functions $t_k(y)$", color=text_color, fontweight='bold')
        ax_tr.set_ylabel("Basis", color=text_color); ax_tr.tick_params(colors=text_color)
        ax_tr.grid(True, linestyle='--', alpha=0.3, color=text_color)
        
        # 6. Metrics Text
        ax_met = self.fig1.add_subplot(236)
        ax_met.axis('off')
        ax_met.set_facecolor(bg_color)
        metrics_text = f"Current Time: {self.model.t_grid[self.current_t_idx]:.2f}\n\nRelative L2 Error: {self.lbl_metrics.text().split('|')[0].split(': ')[1]}\nMSE: {self.lbl_metrics.text().split(': ')[-1]}"
        ax_met.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=14, fontweight='bold', color=text_color, transform=ax_met.transAxes)
        
        self.fig1.tight_layout()

    def draw_tab2(self, V, text_color, bg_color):
        self.fig2.clear(); self.fig2.set_facecolor(bg_color)
        
        # 1. Heatmap
        ax_heat = self.fig2.add_subplot(121)
        cax = ax_heat.pcolormesh(self.model.t_grid, self.model.y_query, V, cmap='inferno', shading='auto')
        current_t = self.model.t_grid[self.current_t_idx]
        ax_heat.axvline(current_t, color='white', linestyle='--', linewidth=2)
        ax_heat.set_title("2D Spatiotemporal Heatmap [Click to set time]", color=text_color, fontweight='bold')
        ax_heat.set_xlabel("Time $t$", color=text_color); ax_heat.set_ylabel("Space $y$", color=text_color)
        ax_heat.tick_params(colors=text_color)
        cbar = self.fig2.colorbar(cax, ax=ax_heat, pad=0.02)
        cbar.ax.yaxis.set_tick_params(color=text_color)
        cbar.set_label('Amplitude', color=text_color)
        
        # 2. 3D Surface
        ax3d = self.fig2.add_subplot(122, projection='3d')
        ax3d.set_facecolor(bg_color)
        Y, T = np.meshgrid(self.model.y_query, self.model.t_grid)
        surf = ax3d.plot_surface(Y, T, V.T, cmap='viridis', edgecolor='none', alpha=0.9)
        ax3d.set_title("3D Surface Plot", color=text_color, fontweight='bold')
        ax3d.set_xlabel("Time $t$", color=text_color); ax3d.set_ylabel("Space $y$", color=text_color); ax3d.set_zlabel("v(y,t)", color=text_color)
        ax3d.tick_params(colors=text_color)
        ax3d.xaxis.pane.fill = False; ax3d.yaxis.pane.fill = False; ax3d.zaxis.pane.fill = False
        
        self.fig2.tight_layout()

    def draw_tab3(self, v_pred, v_gt, text_color, bg_color):
        self.fig3.clear(); self.fig3.set_facecolor(bg_color)
        
        abs_err = np.abs(v_pred - v_gt)
        
        # 1. Absolute Error over Space
        ax_err = self.fig3.add_subplot(211)
        ax_err.plot(self.model.y_query, abs_err, 'r-', linewidth=2)
        ax_err.fill_between(self.model.y_query, abs_err, color='red', alpha=0.2)
        ax_err.set_title(f"Absolute Error Distribution | $v_{{pred}} - v_{{true}}$ | (t={self.model.t_grid[self.current_t_idx]:.2f})", color=text_color, fontweight='bold')
        ax_err.set_ylabel("Absolute Error", color=text_color); ax_err.tick_params(colors=text_color)
        ax_err.grid(True, linestyle='--', alpha=0.3, color=text_color)
        
        # 2. Error Histogram
        ax_hist = self.fig3.add_subplot(212)
        ax_hist.hist(abs_err, bins=30, color='orange', alpha=0.7, edgecolor='black')
        ax_hist.set_title("Error Histogram", color=text_color, fontweight='bold')
        ax_hist.set_xlabel("Error Magnitude", color=text_color); ax_hist.set_ylabel("Frequency", color=text_color)
        ax_hist.tick_params(colors=text_color)
        ax_hist.grid(True, axis='y', linestyle='--', alpha=0.3, color=text_color)
        
        # Add text box with stats
        stats = f"Max Error: {np.max(abs_err):.2e}\nMean Error: {np.mean(abs_err):.2e}"
        ax_hist.text(0.95, 0.95, stats, transform=ax_hist.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.fig3.tight_layout()

    def draw_tab4(self, B_t, text_color, bg_color):
        """Draw latent space dynamics over time"""
        self.fig4.clear(); self.fig4.set_facecolor(bg_color)
        
        # 1. Latent weights evolution (heatmap)
        ax_lat = self.fig4.add_subplot(211)
        cax = ax_lat.pcolormesh(self.model.t_grid, range(1, self.model.latent_dim+1), B_t, 
                                cmap='RdBu_r', shading='auto')
        ax_lat.axvline(self.model.t_grid[self.current_t_idx], color='white', linestyle='--', linewidth=2)
        ax_lat.set_title("Latent Weights $b_k(t)$ Evolution", color=text_color, fontweight='bold')
        ax_lat.set_ylabel("Latent Dimension $k$", color=text_color)
        ax_lat.set_xlabel("Time $t$", color=text_color)
        ax_lat.tick_params(colors=text_color)
        cbar = self.fig4.colorbar(cax, ax=ax_lat, pad=0.02)
        cbar.ax.yaxis.set_tick_params(color=text_color)
        cbar.set_label('Weight magnitude', color=text_color)
        
        # 2. Individual latent trajectories
        ax_traj = self.fig4.add_subplot(212)
        cmap = plt.cm.tab20
        for k in range(min(10, self.model.latent_dim)):  # Show first 10
            ax_traj.plot(self.model.t_grid, B_t[k, :], linewidth=2, 
                        color=cmap(k/10), label=f'$b_{{{k+1}}}$')
        ax_traj.axvline(self.model.t_grid[self.current_t_idx], color='gray', linestyle='--', linewidth=2, alpha=0.5)
        ax_traj.set_title("Latent Weight Trajectories (First 10 dimensions)", color=text_color, fontweight='bold')
        ax_traj.set_xlabel("Time $t$", color=text_color)
        ax_traj.set_ylabel("Weight value", color=text_color)
        ax_traj.tick_params(colors=text_color)
        ax_traj.legend(ncol=5, fontsize=8, facecolor=bg_color, edgecolor=text_color, labelcolor=text_color)
        ax_traj.grid(True, linestyle='--', alpha=0.3, color=text_color)
        
        self.fig4.tight_layout()

    def draw_tab5(self, u_sensors, v_pred, text_color, bg_color):
        """Draw frequency analysis"""
        self.fig5.clear(); self.fig5.set_facecolor(bg_color)
        
        # Compute FFT
        u_fft = np.abs(np.fft.fft(u_sensors))[:len(u_sensors)//2]
        v_fft = np.abs(np.fft.fft(v_pred))[:len(v_pred)//2]
        freqs = np.fft.fftfreq(len(u_sensors), d=1.0/len(u_sensors))[:len(u_sensors)//2]
        
        # 1. Input spectrum
        ax_in = self.fig5.add_subplot(211)
        ax_in.plot(freqs, u_fft, 'b-', linewidth=2)
        ax_in.set_title("Input Function Power Spectrum", color=text_color, fontweight='bold')
        ax_in.set_ylabel("|FFT(u)|", color=text_color)
        ax_in.tick_params(colors=text_color)
        ax_in.grid(True, linestyle='--', alpha=0.3, color=text_color)
        ax_in.set_xlim(0, max(freqs))
        
        # 2. Output spectrum
        ax_out = self.fig5.add_subplot(212)
        ax_out.plot(freqs[:len(v_fft)], v_fft, 'r-', linewidth=2)
        ax_out.set_title("Output Function Power Spectrum", color=text_color, fontweight='bold')
        ax_out.set_xlabel("Frequency", color=text_color)
        ax_out.set_ylabel("|FFT(v)|", color=text_color)
        ax_out.tick_params(colors=text_color)
        ax_out.grid(True, linestyle='--', alpha=0.3, color=text_color)
        ax_out.set_xlim(0, max(freqs))
        
        self.fig5.tight_layout()

    def draw_architecture(self, ax, text_color, bg_color):
        ax.clear(); ax.set_xlim(-1, 11); ax.set_ylim(-1, 7); ax.axis('off')
        ax.set_facecolor(bg_color)
        ax.set_title("Network Architecture", fontsize=12, fontweight='bold', color=text_color, pad=10)
        
        c_in, c_hid, c_out = '#3498db', '#95a5a6', '#e74c3c'
        
        # Branch Net
        b_in_x, b_hid_x, b_out_x = 1, 3, 5
        b_in_y = np.linspace(1, 5, 5); b_hid_y = np.linspace(2, 4, 3); b_out_y = np.linspace(2.5, 3.5, 3)
        for y1 in b_in_y:
            for y2 in b_hid_y: ax.plot([b_in_x, b_hid_x], [y1, y2], color='gray', alpha=0.3, lw=0.5)
        for y1 in b_hid_y:
            for y2 in b_out_y: ax.plot([b_hid_x, b_out_x], [y1, y2], color='gray', alpha=0.3, lw=0.5)
        for y in b_in_y: ax.add_patch(plt.Circle((b_in_x, y), 0.2, color=c_in))
        for y in b_hid_y: ax.add_patch(plt.Circle((b_hid_x, y), 0.2, color=c_hid))
        for i, y in enumerate(b_out_y): 
            ax.add_patch(plt.Circle((b_out_x, y), 0.2, color=c_out))
            ax.text(b_out_x + 0.4, y, f'$b_{i+1}$', va='center', fontsize=10, color=text_color)
        ax.text(3, 0.2, 'Branch Net', ha='center', fontsize=11, fontweight='bold', color=text_color)
        ax.text(1, 5.8, '$u(x_i)$', ha='center', fontsize=10, color=text_color)

        # Trunk Net
        t_in_x, t_hid_x, t_out_x = 6, 8, 10
        t_in_y = [3]; t_hid_y = np.linspace(2, 4, 3); t_out_y = np.linspace(2.5, 3.5, 3)
        for y1 in t_in_y:
            for y2 in t_hid_y: ax.plot([t_in_x, t_hid_x], [y1, y2], color='gray', alpha=0.3, lw=0.5)
        for y1 in t_hid_y:
            for y2 in t_out_y: ax.plot([t_hid_x, t_out_x], [y1, y2], color='gray', alpha=0.3, lw=0.5)
        ax.add_patch(plt.Circle((t_in_x, 3), 0.2, color=c_in))
        for y in t_hid_y: ax.add_patch(plt.Circle((t_hid_x, y), 0.2, color=c_hid))
        for i, y in enumerate(t_out_y): 
            ax.add_patch(plt.Circle((t_out_x, y), 0.2, color=c_out))
            ax.text(t_out_x + 0.4, y, f'$t_{i+1}$', va='center', fontsize=10, color=text_color)
        ax.text(8, 0.2, 'Trunk Net', ha='center', fontsize=11, fontweight='bold', color=text_color)
        ax.text(6, 2.2, '$y$', ha='center', fontsize=10, color=text_color)
        ax.text(5.5, 3, r'$\sum$', ha='center', va='center', fontsize=24, fontweight='bold', color=text_color)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.quit(); self.worker.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = DeepONetVisualizer()
    window.show()
    sys.exit(app.exec())