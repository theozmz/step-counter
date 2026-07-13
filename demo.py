import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QScrollArea, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QFileDialog, QDialog, QLabel, QLineEdit,
    QSpinBox, QDoubleSpinBox, QFormLayout, QDialogButtonBox, QMessageBox,
    QGroupBox, QGridLayout, QSplitter, QCheckBox
)
from PyQt5.QtCore import QThread, pyqtSignal, QObject, Qt, QTimer
from step_counter import DataPreprocessor, StepCounter, Analysis
import os
import glob


class OfflineParamDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Offline Step Counter")
        self.setMinimumWidth(450)

        layout = QVBoxLayout()

        # File selection
        file_group = QGroupBox("Input Files")
        file_layout = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setReadOnly(True)
        self.file_btn = QPushButton("Browse...")
        self.file_btn.clicked.connect(self.browse_files)
        file_layout.addWidget(self.file_edit)
        file_layout.addWidget(self.file_btn)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # LSTM Model Group
        lstm_group = QGroupBox("LSTM Model")
        lstm_layout = QVBoxLayout()
        
        self.lstm_checkbox = QCheckBox("Enable LSTM Model (not recommended)")
        self.lstm_checkbox.setChecked(False)
        self.lstm_checkbox.stateChanged.connect(self.toggle_model_file)
        lstm_layout.addWidget(self.lstm_checkbox)
        
        model_file_layout = QHBoxLayout()
        self.model_file_edit = QLineEdit()
        self.model_file_edit.setReadOnly(True)
        self.model_file_edit.setEnabled(False)
        self.model_file_btn = QPushButton("Browse Model...")
        self.model_file_btn.setEnabled(False)
        self.model_file_btn.clicked.connect(self.browse_model_file)
        model_file_layout.addWidget(self.model_file_edit)
        model_file_layout.addWidget(self.model_file_btn)
        lstm_layout.addLayout(model_file_layout)
        
        lstm_group.setLayout(lstm_layout)
        layout.addWidget(lstm_group)

        # Parameters - Left side with scroll area
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout()
        
        # Create scroll area for parameters
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(280)
        scroll_area.setMaximumWidth(350)
        
        params_widget = QWidget()
        params_form = QFormLayout(params_widget)
        
        # Window and basic parameters
        self.window_sec = QDoubleSpinBox()
        self.window_sec.setRange(0.5, 20.0)
        self.window_sec.setValue(2.0)
        params_form.addRow("window_sec:", self.window_sec)
        
        self.threshold_factor = QDoubleSpinBox()
        self.threshold_factor.setRange(0.5, 5.0)
        self.threshold_factor.setValue(0.8)
        params_form.addRow("threshold_factor:", self.threshold_factor)
        
        self.refractory_sec = QDoubleSpinBox()
        self.refractory_sec.setRange(0.1, 1.0)
        self.refractory_sec.setValue(0.75)
        params_form.addRow("refractory_sec:", self.refractory_sec)
        
        self.filter_len = QSpinBox()
        self.filter_len.setRange(1, 50)
        self.filter_len.setValue(5)
        params_form.addRow("filter_len:", self.filter_len)
        
        self.cutoff_low = QDoubleSpinBox()
        self.cutoff_low.setRange(0.1, 2.0)
        self.cutoff_low.setValue(0.5)
        params_form.addRow("cutoff_low:", self.cutoff_low)
        
        self.cutoff_high = QDoubleSpinBox()
        self.cutoff_high.setRange(1.0, 10.0)
        self.cutoff_high.setValue(2.0)
        params_form.addRow("cutoff_high:", self.cutoff_high)
        
        self.fs = QDoubleSpinBox()
        self.fs.setRange(20.0, 200.0)
        self.fs.setValue(50.0)
        params_form.addRow("fs:", self.fs)
        
        self.order = QSpinBox()
        self.order.setRange(2, 8)
        self.order.setValue(3)
        params_form.addRow("order:", self.order)
        
        self.n_sigma = QDoubleSpinBox()
        self.n_sigma.setRange(1.0, 5.0)
        self.n_sigma.setValue(3.0)
        params_form.addRow("n_sigma:", self.n_sigma)
        
        self.polyorder = QSpinBox()
        self.polyorder.setRange(1, 5)
        self.polyorder.setValue(3)
        params_form.addRow("polyorder:", self.polyorder)
        
        self.gyro_weight = QDoubleSpinBox()
        self.gyro_weight.setRange(0.0, 1.0)
        self.gyro_weight.setValue(0.3)
        params_form.addRow("gyro_weight:", self.gyro_weight)
        
        self.noise_threshold = QDoubleSpinBox()
        self.noise_threshold.setRange(0.0, 5.0)
        self.noise_threshold.setValue(0.2)
        params_form.addRow("noise_threshold:", self.noise_threshold)
        
        self.threshold_absolute = QDoubleSpinBox()
        self.threshold_absolute.setRange(0.0, 5.0)
        self.threshold_absolute.setValue(0.6)
        params_form.addRow("threshold_absolute:", self.threshold_absolute)
        
        self.gyro_energy_min = QDoubleSpinBox()
        self.gyro_energy_min.setRange(0.0, 1.0)
        self.gyro_energy_min.setValue(0.05)
        params_form.addRow("gyro_energy_min:", self.gyro_energy_min)
        
        self.mag_var_min = QDoubleSpinBox()
        self.mag_var_min.setRange(0.0, 1.0)
        self.mag_var_min.setValue(0.05)
        params_form.addRow("mag_var_min:", self.mag_var_min)
        
        self.penalty = QDoubleSpinBox()
        self.penalty.setRange(0.0, 5.0)
        self.penalty.setValue(0.5)
        params_form.addRow("penalty:", self.penalty)
        
        self.acf_dominance = QDoubleSpinBox()
        self.acf_dominance.setRange(0.0, 1.0)
        self.acf_dominance.setValue(0.7)
        params_form.addRow("acf_dominance:", self.acf_dominance)
        
        self.g = QDoubleSpinBox()
        self.g.setRange(9.0, 10.0)
        self.g.setValue(9.8)
        params_form.addRow("g:", self.g)
        
        # FSM
        self.peak_prominence = QDoubleSpinBox()
        self.peak_prominence.setRange(0.0, 1.0)
        self.peak_prominence.setValue(0.1)
        params_form.addRow("peak_prominence:", self.peak_prominence)
        
        self.valley_prominence = QDoubleSpinBox()
        self.valley_prominence.setRange(0.0, 1.0)
        self.valley_prominence.setValue(0.05)
        params_form.addRow("valley_prominence:", self.valley_prominence)
        
        self.min_step_period = QDoubleSpinBox()
        self.min_step_period.setRange(0.0, 1.0)
        self.min_step_period.setValue(0.3)
        params_form.addRow("min_step_period:", self.min_step_period)
        
        self.max_step_period = QDoubleSpinBox()
        self.max_step_period.setRange(0.0, 5.0)
        self.max_step_period.setValue(3.0)
        params_form.addRow("max_step_period:", self.max_step_period)
        
        self.state_timeout = QDoubleSpinBox()
        self.state_timeout.setRange(0.0, 3.0)
        self.state_timeout.setValue(1.5)
        params_form.addRow("state_timeout:", self.state_timeout)
        
        self.peak_valley_min_dist = QDoubleSpinBox()
        self.peak_valley_min_dist.setRange(0.0, 2.0)
        self.peak_valley_min_dist.setValue(0.2)
        params_form.addRow("peak_valley_min_dist:", self.peak_valley_min_dist)
        
        # Boolean flags group
        flags_group = QGroupBox("Algorithm Flags")
        flags_layout = QGridLayout(flags_group)
        
        self.gravity_checkbox = QCheckBox("gravity")
        self.gravity_checkbox.setChecked(False)
        flags_layout.addWidget(self.gravity_checkbox, 0, 0)
        
        self.cma_checkbox = QCheckBox("CMA")
        self.cma_checkbox.setChecked(True)
        flags_layout.addWidget(self.cma_checkbox, 0, 1)
        
        self.hampel_checkbox = QCheckBox("hampel")
        self.hampel_checkbox.setChecked(True)
        flags_layout.addWidget(self.hampel_checkbox, 1, 0)
        
        self.savgol_checkbox = QCheckBox("savgol")
        self.savgol_checkbox.setChecked(True)
        flags_layout.addWidget(self.savgol_checkbox, 1, 1)
        
        self.lpf_checkbox = QCheckBox("LPF")
        self.lpf_checkbox.setChecked(True)
        flags_layout.addWidget(self.lpf_checkbox, 2, 0)
        
        self.acf_checkbox = QCheckBox("ACF")
        self.acf_checkbox.setChecked(True)
        flags_layout.addWidget(self.acf_checkbox, 2, 1)
        
        self.fsm_checkbox = QCheckBox("FSM")
        self.fsm_checkbox.setChecked(False)
        flags_layout.addWidget(self.fsm_checkbox, 3, 0)
        
        params_form.addRow(flags_group)
        
        scroll_area.setWidget(params_widget)
        params_layout.addWidget(scroll_area)
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def toggle_model_file(self, state):
        """Enable/disable model file selection based on LSTM checkbox state."""
        enabled = (state == Qt.Checked)
        self.model_file_edit.setEnabled(enabled)
        self.model_file_btn.setEnabled(enabled)
        if not enabled:
            self.model_file_edit.clear()

    def browse_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select CSV files", "", "CSV files (*.csv)")
        if files:
            self.file_edit.setText("; ".join(files))

    def browse_model_file(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select LSTM Model File", "", "HDF5 Files (*.h5);;All Files (*)")
        if file:
            self.model_file_edit.setText(file)

    def get_params(self):
        return {
            "files": self.file_edit.text().split("; "),
            "model_path": self.model_file_edit.text() if self.lstm_checkbox.isChecked() else None,
            "window_sec": self.window_sec.value(),
            "threshold_factor": self.threshold_factor.value(),
            "refractory_sec": self.refractory_sec.value(),
            "filter_len": self.filter_len.value(),
            "cutoff_low": self.cutoff_low.value(),
            "cutoff_high": self.cutoff_high.value(),
            "fs": self.fs.value(),
            "order": self.order.value(),
            "n_sigma": self.n_sigma.value(),
            "polyorder": self.polyorder.value(),
            "gyro_weight": self.gyro_weight.value(),
            "noise_threshold": self.noise_threshold.value(),
            "threshold_absolute": self.threshold_absolute.value(),
            "gyro_energy_min": self.gyro_energy_min.value(),
            "mag_var_min": self.mag_var_min.value(),
            "penalty": self.penalty.value(),
            "acf_dominance": self.acf_dominance.value(),
            "g": self.g.value(),
            "gravity": self.gravity_checkbox.isChecked(),
            "CMA": self.cma_checkbox.isChecked(),
            "hampel": self.hampel_checkbox.isChecked(),
            "savgol": self.savgol_checkbox.isChecked(),
            "LPF": self.lpf_checkbox.isChecked(),
            "ACF": self.acf_checkbox.isChecked(),
            "FSM": self.fsm_checkbox.isChecked(),
            "peak_prominence": self.peak_prominence.value(),
            "valley_prominence": self.valley_prominence.value(),
            "min_step_period": self.min_step_period.value(),
            "max_step_period": self.max_step_period.value(),
            "state_timeout": self.state_timeout.value(),
            "peak_valley_min_dist": self.peak_valley_min_dist.value(),
        }



class OnlineParamDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Online Step Counter")
        self.setMinimumWidth(400)

        layout = QVBoxLayout()

        # Connection settings
        conn_group = QGroupBox("Connection")
        conn_layout = QFormLayout()
        self.ip_edit = QLineEdit("192.168.0.247")
        self.port_edit = QSpinBox()
        self.port_edit.setRange(1024, 65535)
        self.port_edit.setValue(8080)
        conn_layout.addRow("IP Address:", self.ip_edit)
        conn_layout.addRow("Port:", self.port_edit)
        conn_group.setLayout(conn_layout)
        layout.addWidget(conn_group)

        # LSTM Model Group
        lstm_group = QGroupBox("LSTM Model")
        lstm_layout = QVBoxLayout()
        
        self.lstm_checkbox = QCheckBox("Enable LSTM Model (not recommended)")
        self.lstm_checkbox.setChecked(False)
        self.lstm_checkbox.stateChanged.connect(self.toggle_model_file)
        lstm_layout.addWidget(self.lstm_checkbox)
        
        model_file_layout = QHBoxLayout()
        self.model_file_edit = QLineEdit()
        self.model_file_edit.setReadOnly(True)
        self.model_file_edit.setEnabled(False)
        self.model_file_btn = QPushButton("Browse Model...")
        self.model_file_btn.setEnabled(False)
        self.model_file_btn.clicked.connect(self.browse_model_file)
        model_file_layout.addWidget(self.model_file_edit)
        model_file_layout.addWidget(self.model_file_btn)
        lstm_layout.addLayout(model_file_layout)
        
        lstm_group.setLayout(lstm_layout)
        layout.addWidget(lstm_group)

        # Parameters
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout()
        
        # Create scroll area for parameters
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(280)
        scroll_area.setMaximumWidth(350)
        
        params_widget = QWidget()
        params_form = QFormLayout(params_widget)
        
        # Window and basic parameters
        self.window_sec = QDoubleSpinBox()
        self.window_sec.setRange(0.5, 20.0)
        self.window_sec.setValue(2.0)
        params_form.addRow("window_sec:", self.window_sec)
        
        self.threshold_factor = QDoubleSpinBox()
        self.threshold_factor.setRange(0.5, 5.0)
        self.threshold_factor.setValue(0.8)
        params_form.addRow("threshold_factor:", self.threshold_factor)
        
        self.refractory_sec = QDoubleSpinBox()
        self.refractory_sec.setRange(0.1, 1.0)
        self.refractory_sec.setValue(0.75)
        params_form.addRow("refractory_sec:", self.refractory_sec)
        
        self.filter_len = QSpinBox()
        self.filter_len.setRange(1, 50)
        self.filter_len.setValue(5)
        params_form.addRow("filter_len:", self.filter_len)
        
        self.cutoff_low = QDoubleSpinBox()
        self.cutoff_low.setRange(0.1, 2.0)
        self.cutoff_low.setValue(0.5)
        params_form.addRow("cutoff_low:", self.cutoff_low)
        
        self.cutoff_high = QDoubleSpinBox()
        self.cutoff_high.setRange(1.0, 10.0)
        self.cutoff_high.setValue(2.0)
        params_form.addRow("cutoff_high:", self.cutoff_high)
        
        self.fs = QDoubleSpinBox()
        self.fs.setRange(20.0, 200.0)
        self.fs.setValue(50.0)
        params_form.addRow("fs:", self.fs)
        
        self.order = QSpinBox()
        self.order.setRange(2, 8)
        self.order.setValue(3)
        params_form.addRow("order:", self.order)
        
        self.n_sigma = QDoubleSpinBox()
        self.n_sigma.setRange(1.0, 5.0)
        self.n_sigma.setValue(3.0)
        params_form.addRow("n_sigma:", self.n_sigma)
        
        self.polyorder = QSpinBox()
        self.polyorder.setRange(1, 5)
        self.polyorder.setValue(3)
        params_form.addRow("polyorder:", self.polyorder)
        
        self.gyro_weight = QDoubleSpinBox()
        self.gyro_weight.setRange(0.0, 1.0)
        self.gyro_weight.setValue(0.3)
        params_form.addRow("gyro_weight:", self.gyro_weight)
        
        self.noise_threshold = QDoubleSpinBox()
        self.noise_threshold.setRange(0.0, 5.0)
        self.noise_threshold.setValue(0.2)
        params_form.addRow("noise_threshold:", self.noise_threshold)
        
        self.threshold_absolute = QDoubleSpinBox()
        self.threshold_absolute.setRange(0.0, 5.0)
        self.threshold_absolute.setValue(0.6)
        params_form.addRow("threshold_absolute:", self.threshold_absolute)
        
        self.gyro_energy_min = QDoubleSpinBox()
        self.gyro_energy_min.setRange(0.0, 1.0)
        self.gyro_energy_min.setValue(0.05)
        params_form.addRow("gyro_energy_min:", self.gyro_energy_min)
        
        self.mag_var_min = QDoubleSpinBox()
        self.mag_var_min.setRange(0.0, 1.0)
        self.mag_var_min.setValue(0.05)
        params_form.addRow("mag_var_min:", self.mag_var_min)
        
        self.penalty = QDoubleSpinBox()
        self.penalty.setRange(0.0, 5.0)
        self.penalty.setValue(0.5)
        params_form.addRow("penalty:", self.penalty)
        
        self.acf_dominance = QDoubleSpinBox()
        self.acf_dominance.setRange(0.0, 1.0)
        self.acf_dominance.setValue(0.7)
        params_form.addRow("acf_dominance:", self.acf_dominance)
        
        self.g = QDoubleSpinBox()
        self.g.setRange(9.0, 10.0)
        self.g.setValue(9.8)
        params_form.addRow("g:", self.g)
        
        # FSM
        self.peak_prominence = QDoubleSpinBox()
        self.peak_prominence.setRange(0.0, 1.0)
        self.peak_prominence.setValue(0.1)
        params_form.addRow("peak_prominence:", self.peak_prominence)
        
        self.valley_prominence = QDoubleSpinBox()
        self.valley_prominence.setRange(0.0, 1.0)
        self.valley_prominence.setValue(0.05)
        params_form.addRow("valley_prominence:", self.valley_prominence)
        
        self.min_step_period = QDoubleSpinBox()
        self.min_step_period.setRange(0.0, 1.0)
        self.min_step_period.setValue(0.3)
        params_form.addRow("min_step_period:", self.min_step_period)
        
        self.max_step_period = QDoubleSpinBox()
        self.max_step_period.setRange(0.0, 5.0)
        self.max_step_period.setValue(3.0)
        params_form.addRow("max_step_period:", self.max_step_period)
        
        self.state_timeout = QDoubleSpinBox()
        self.state_timeout.setRange(0.0, 3.0)
        self.state_timeout.setValue(1.5)
        params_form.addRow("state_timeout:", self.state_timeout)
        
        self.peak_valley_min_dist = QDoubleSpinBox()
        self.peak_valley_min_dist.setRange(0.0, 2.0)
        self.peak_valley_min_dist.setValue(0.2)
        params_form.addRow("peak_valley_min_dist:", self.peak_valley_min_dist)
        
        # Boolean flags group
        flags_group = QGroupBox("Algorithm Flags")
        flags_layout = QGridLayout(flags_group)
        
        self.gravity_checkbox = QCheckBox("gravity")
        self.gravity_checkbox.setChecked(False)
        flags_layout.addWidget(self.gravity_checkbox, 0, 0)
        
        self.cma_checkbox = QCheckBox("CMA")
        self.cma_checkbox.setChecked(True)
        flags_layout.addWidget(self.cma_checkbox, 0, 1)
        
        self.hampel_checkbox = QCheckBox("hampel")
        self.hampel_checkbox.setChecked(True)
        flags_layout.addWidget(self.hampel_checkbox, 1, 0)
        
        self.savgol_checkbox = QCheckBox("savgol")
        self.savgol_checkbox.setChecked(True)
        flags_layout.addWidget(self.savgol_checkbox, 1, 1)
        
        self.lpf_checkbox = QCheckBox("LPF")
        self.lpf_checkbox.setChecked(True)
        flags_layout.addWidget(self.lpf_checkbox, 2, 0)
        
        self.acf_checkbox = QCheckBox("ACF")
        self.acf_checkbox.setChecked(True)
        flags_layout.addWidget(self.acf_checkbox, 2, 1)
        
        self.fsm_checkbox = QCheckBox("FSM")
        self.fsm_checkbox.setChecked(False)
        flags_layout.addWidget(self.fsm_checkbox, 3, 0)
        
        params_form.addRow(flags_group)
        
        scroll_area.setWidget(params_widget)
        params_layout.addWidget(scroll_area)
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def toggle_model_file(self, state):
        """Enable/disable model file selection based on LSTM checkbox state."""
        enabled = (state == Qt.Checked)
        self.model_file_edit.setEnabled(enabled)
        self.model_file_btn.setEnabled(enabled)
        if not enabled:
            self.model_file_edit.clear()

    def browse_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select CSV files", "", "CSV files (*.csv)")
        if files:
            self.file_edit.setText("; ".join(files))

    def browse_model_file(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select LSTM Model File", "", "HDF5 Files (*.h5);;All Files (*)")
        if file:
            self.model_file_edit.setText(file)

    def get_params(self):
        return {
            "ip": self.ip_edit.text(),
            "port": self.port_edit.value(),
            "model_path": self.model_file_edit.text() if self.lstm_checkbox.isChecked() else None,
            "window_sec": self.window_sec.value(),
            "threshold_factor": self.threshold_factor.value(),
            "refractory_sec": self.refractory_sec.value(),
            "filter_len": self.filter_len.value(),
            "cutoff_low": self.cutoff_low.value(),
            "cutoff_high": self.cutoff_high.value(),
            "fs": self.fs.value(),
            "order": self.order.value(),
            "n_sigma": self.n_sigma.value(),
            "polyorder": self.polyorder.value(),
            "gyro_weight": self.gyro_weight.value(),
            "noise_threshold": self.noise_threshold.value(),
            "threshold_absolute": self.threshold_absolute.value(),
            "gyro_energy_min": self.gyro_energy_min.value(),
            "mag_var_min": self.mag_var_min.value(),
            "penalty": self.penalty.value(),
            "acf_dominance": self.acf_dominance.value(),
            "g": self.g.value(),
            "gravity": self.gravity_checkbox.isChecked(),
            "CMA": self.cma_checkbox.isChecked(),
            "hampel": self.hampel_checkbox.isChecked(),
            "savgol": self.savgol_checkbox.isChecked(),
            "LPF": self.lpf_checkbox.isChecked(),
            "ACF": self.acf_checkbox.isChecked(),
            "FSM": self.fsm_checkbox.isChecked(),
            "peak_prominence": self.peak_prominence.value(),
            "valley_prominence": self.valley_prominence.value(),
            "min_step_period": self.min_step_period.value(),
            "max_step_period": self.max_step_period.value(),
            "state_timeout": self.state_timeout.value(),
            "peak_valley_min_dist": self.peak_valley_min_dist.value(),
        }


class EvaluationParamDialog(QDialog):
    """Evaluation dialog for ablation study with task management."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ablation Study Evaluation")
        self.setMinimumSize(900, 700)
        
        # Task list storage
        self.tasks = []
        
        layout = QVBoxLayout()
        
        # Top section: File selection and parameters
        top_splitter = QSplitter(Qt.Horizontal)
        
        # Left side: Parameters
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Data directory selection
        data_dir_group = QGroupBox("Test Dataset Directory")
        data_dir_layout = QHBoxLayout()
        self.data_dir_edit = QLineEdit()
        self.data_dir_edit.setReadOnly(True)
        self.data_dir_btn = QPushButton("Browse...")
        self.data_dir_btn.clicked.connect(self.browse_data_dir)
        data_dir_layout.addWidget(self.data_dir_edit)
        data_dir_layout.addWidget(self.data_dir_btn)
        data_dir_group.setLayout(data_dir_layout)
        left_layout.addWidget(data_dir_group)
        
        # Output directory selection
        output_dir_group = QGroupBox("Output Directory")
        output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setReadOnly(True)
        self.output_dir_btn = QPushButton("Browse...")
        self.output_dir_btn.clicked.connect(self.browse_output_dir)
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(self.output_dir_btn)
        output_dir_group.setLayout(output_dir_layout)
        left_layout.addWidget(output_dir_group)
        
        # LSTM Model Group
        lstm_group = QGroupBox("LSTM Model")
        lstm_layout = QVBoxLayout()
        
        self.lstm_checkbox = QCheckBox("Enable LSTM Model")
        self.lstm_checkbox.setChecked(False)
        self.lstm_checkbox.stateChanged.connect(self.toggle_model_file)
        lstm_layout.addWidget(self.lstm_checkbox)
        
        model_file_layout = QHBoxLayout()
        self.model_file_edit = QLineEdit()
        self.model_file_edit.setReadOnly(True)
        self.model_file_edit.setEnabled(False)
        self.model_file_btn = QPushButton("Browse Model...")
        self.model_file_btn.setEnabled(False)
        self.model_file_btn.clicked.connect(self.browse_model_file)
        model_file_layout.addWidget(self.model_file_edit)
        model_file_layout.addWidget(self.model_file_btn)
        lstm_layout.addLayout(model_file_layout)
        
        lstm_group.setLayout(lstm_layout)
        left_layout.addWidget(lstm_group)
        
        # Parameters
        params_group = QGroupBox("Parameters")
        params_layout = QFormLayout()
        
        # Create scroll area for parameters
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        # scroll_area.setMinimumWidth(280)
        # scroll_area.setMaximumWidth(350)
        
        params_widget = QWidget()
        params_form = QFormLayout(params_widget)
        
        self.window_sec = QDoubleSpinBox()
        self.window_sec.setRange(0.5, 20.0)
        self.window_sec.setValue(2.0)
        params_form.addRow("window_sec:", self.window_sec)
        
        self.threshold_factor = QDoubleSpinBox()
        self.threshold_factor.setRange(0.5, 5.0)
        self.threshold_factor.setValue(0.8)
        params_form.addRow("threshold_factor:", self.threshold_factor)
        
        self.refractory_sec = QDoubleSpinBox()
        self.refractory_sec.setRange(0.1, 1.0)
        self.refractory_sec.setValue(0.75)
        params_form.addRow("refractory_sec:", self.refractory_sec)
        
        self.filter_len = QSpinBox()
        self.filter_len.setRange(1, 50)
        self.filter_len.setValue(5)
        params_form.addRow("filter_len:", self.filter_len)
        
        self.cutoff_low = QDoubleSpinBox()
        self.cutoff_low.setRange(0.1, 2.0)
        self.cutoff_low.setValue(0.5)
        params_form.addRow("cutoff_low:", self.cutoff_low)
        
        self.cutoff_high = QDoubleSpinBox()
        self.cutoff_high.setRange(1.0, 10.0)
        self.cutoff_high.setValue(2.0)
        params_form.addRow("cutoff_high:", self.cutoff_high)
        
        self.fs = QDoubleSpinBox()
        self.fs.setRange(20.0, 200.0)
        self.fs.setValue(50.0)
        params_form.addRow("fs:", self.fs)
        
        self.order = QSpinBox()
        self.order.setRange(2, 8)
        self.order.setValue(3)
        params_form.addRow("order:", self.order)
        
        self.n_sigma = QDoubleSpinBox()
        self.n_sigma.setRange(1.0, 5.0)
        self.n_sigma.setValue(3.0)
        params_form.addRow("n_sigma:", self.n_sigma)
        
        self.polyorder = QSpinBox()
        self.polyorder.setRange(1, 5)
        self.polyorder.setValue(3)
        params_form.addRow("polyorder:", self.polyorder)
        
        self.gyro_weight = QDoubleSpinBox()
        self.gyro_weight.setRange(0.0, 1.0)
        self.gyro_weight.setValue(0.3)
        params_form.addRow("gyro_weight:", self.gyro_weight)
        
        self.noise_threshold = QDoubleSpinBox()
        self.noise_threshold.setRange(0.0, 5.0)
        self.noise_threshold.setValue(0.2)
        params_form.addRow("noise_threshold:", self.noise_threshold)
        
        self.threshold_absolute = QDoubleSpinBox()
        self.threshold_absolute.setRange(0.0, 5.0)
        self.threshold_absolute.setValue(0.6)
        params_form.addRow("threshold_absolute:", self.threshold_absolute)
        
        self.gyro_energy_min = QDoubleSpinBox()
        self.gyro_energy_min.setRange(0.0, 1.0)
        self.gyro_energy_min.setValue(0.05)
        params_form.addRow("gyro_energy_min:", self.gyro_energy_min)
        
        self.mag_var_min = QDoubleSpinBox()
        self.mag_var_min.setRange(0.0, 1.0)
        self.mag_var_min.setValue(0.05)
        params_form.addRow("mag_var_min:", self.mag_var_min)
        
        self.penalty = QDoubleSpinBox()
        self.penalty.setRange(0.0, 5.0)
        self.penalty.setValue(0.5)
        params_form.addRow("penalty:", self.penalty)
        
        self.acf_dominance = QDoubleSpinBox()
        self.acf_dominance.setRange(0.0, 1.0)
        self.acf_dominance.setValue(0.7)
        params_form.addRow("acf_dominance:", self.acf_dominance)
        
        self.g = QDoubleSpinBox()
        self.g.setRange(9.0, 10.0)
        self.g.setValue(9.8)
        params_form.addRow("g:", self.g)
        
        # FSM
        self.peak_prominence = QDoubleSpinBox()
        self.peak_prominence.setRange(0.0, 1.0)
        self.peak_prominence.setValue(0.1)
        params_form.addRow("peak_prominence:", self.peak_prominence)
        
        self.valley_prominence = QDoubleSpinBox()
        self.valley_prominence.setRange(0.0, 1.0)
        self.valley_prominence.setValue(0.05)
        params_form.addRow("valley_prominence:", self.valley_prominence)
        
        self.min_step_period = QDoubleSpinBox()
        self.min_step_period.setRange(0.0, 1.0)
        self.min_step_period.setValue(0.3)
        params_form.addRow("min_step_period:", self.min_step_period)
        
        self.max_step_period = QDoubleSpinBox()
        self.max_step_period.setRange(0.0, 5.0)
        self.max_step_period.setValue(3.0)
        params_form.addRow("max_step_period:", self.max_step_period)
        
        self.state_timeout = QDoubleSpinBox()
        self.state_timeout.setRange(0.0, 3.0)
        self.state_timeout.setValue(1.5)
        params_form.addRow("state_timeout:", self.state_timeout)
        
        self.peak_valley_min_dist = QDoubleSpinBox()
        self.peak_valley_min_dist.setRange(0.0, 2.0)
        self.peak_valley_min_dist.setValue(0.2)
        params_form.addRow("peak_valley_min_dist:", self.peak_valley_min_dist)
        
        # Boolean flags
        flags_group = QGroupBox("Algorithm Flags")
        flags_layout = QGridLayout(flags_group)
        
        self.gravity_checkbox = QCheckBox("gravity")
        self.gravity_checkbox.setChecked(False)
        flags_layout.addWidget(self.gravity_checkbox, 0, 0)
        
        self.cma_checkbox = QCheckBox("CMA")
        self.cma_checkbox.setChecked(True)
        flags_layout.addWidget(self.cma_checkbox, 0, 1)
        
        self.hampel_checkbox = QCheckBox("hampel")
        self.hampel_checkbox.setChecked(True)
        flags_layout.addWidget(self.hampel_checkbox, 1, 0)
        
        self.savgol_checkbox = QCheckBox("savgol")
        self.savgol_checkbox.setChecked(True)
        flags_layout.addWidget(self.savgol_checkbox, 1, 1)
        
        self.lpf_checkbox = QCheckBox("LPF")
        self.lpf_checkbox.setChecked(True)
        flags_layout.addWidget(self.lpf_checkbox, 2, 0)
        
        self.acf_checkbox = QCheckBox("ACF")
        self.acf_checkbox.setChecked(True)
        flags_layout.addWidget(self.acf_checkbox, 2, 1)
        
        self.fsm_checkbox = QCheckBox("FSM")
        self.fsm_checkbox.setChecked(False)
        flags_layout.addWidget(self.fsm_checkbox, 3, 0)
        
        params_form.addRow(flags_group)
        # params_layout.addRow("Flags:", flags_layout)
        
        scroll_area.setWidget(params_widget)
        params_layout.addWidget(scroll_area)
        params_group.setLayout(params_layout)
        left_layout.addWidget(params_group)
        
        # Add task button
        self.add_task_btn = QPushButton("Add Task")
        self.add_task_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        self.add_task_btn.clicked.connect(self.add_task)
        left_layout.addWidget(self.add_task_btn)
        
        left_widget.setLayout(left_layout)
        top_splitter.addWidget(left_widget)
        
        # Right side: Task list
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        task_list_group = QGroupBox("Task List")
        task_list_layout = QVBoxLayout()
        
        # Add scroll area for task list
        task_scroll_area = QScrollArea()
        task_scroll_area.setMinimumWidth(200)
        task_scroll_area.setMinimumHeight(400)
        task_scroll_area.setWidgetResizable(True)
        task_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        task_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        task_scroll_widget = QWidget()
        self.task_table = QGridLayout(task_scroll_widget)
        self.task_table.setColumnStretch(0, 1)
        self.task_table.setColumnStretch(0, 0)
        
        # Header
        header_label = QLabel("<b>Configuration</b>")
        header_label.setStyleSheet("font-weight: bold; padding: 5px;")
        self.task_table.addWidget(header_label, 0, 0)
        
        delete_header = QLabel("<b>Action</b>")
        delete_header.setStyleSheet("font-weight: bold; padding: 5px;")
        self.task_table.addWidget(delete_header, 0, 1)
        
        task_scroll_area.setWidget(task_scroll_widget)
        task_list_layout.addWidget(task_scroll_area)
        task_list_layout.addStretch()
        
        task_list_group.setLayout(task_list_layout)
        right_layout.addWidget(task_list_group)
        
        # Start button
        self.start_btn = QPushButton("Start Experiment")
        self.start_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 10px; font-size: 14px;")
        self.start_btn.clicked.connect(self.accept)
        right_layout.addWidget(self.start_btn)
        
        right_widget.setLayout(right_layout)
        top_splitter.addWidget(right_widget)
        
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 2)
        
        layout.addWidget(top_splitter)
        
        # Log area
        log_group = QGroupBox("Experiment Log")
        log_layout = QVBoxLayout()
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(150)
        log_layout.addWidget(self.log_area)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        self.setLayout(layout)
    
    def browse_data_dir(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Test Dataset Directory", "")
        if directory:
            self.data_dir_edit.setText(directory)
    
    def browse_output_dir(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", "")
        if directory:
            self.output_dir_edit.setText(directory)
    
    def toggle_model_file(self, state):
        """Enable/disable model file selection based on LSTM checkbox state."""
        enabled = (state == Qt.Checked)
        self.model_file_edit.setEnabled(enabled)
        self.model_file_btn.setEnabled(enabled)
        if not enabled:
            self.model_file_edit.clear()
    
    def browse_model_file(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select LSTM Model File", "", "HDF5 Files (*.h5);;All Files (*)")
        if file:
            self.model_file_edit.setText(file)
    
    def get_current_params(self):
        """Get current parameter settings as a dictionary."""
        return {
            "data_dir": self.data_dir_edit.text(),
            "output_dir": self.output_dir_edit.text(),
            "model_path": self.model_file_edit.text() if self.lstm_checkbox.isChecked() else None,
            "window_sec": self.window_sec.value(),
            "threshold_factor": self.threshold_factor.value(),
            "refractory_sec": self.refractory_sec.value(),
            "filter_len": self.filter_len.value(),
            "cutoff_low": self.cutoff_low.value(),
            "cutoff_high": self.cutoff_high.value(),
            "fs": self.fs.value(),
            "order": self.order.value(),
            "n_sigma": self.n_sigma.value(),
            "polyorder": self.polyorder.value(),
            "gyro_weight": self.gyro_weight.value(),
            "noise_threshold": self.noise_threshold.value(),
            "threshold_absolute": self.threshold_absolute.value(),
            "gyro_energy_min": self.gyro_energy_min.value(),
            "mag_var_min": self.mag_var_min.value(),
            "penalty": self.penalty.value(),
            "acf_dominance": self.acf_dominance.value(),
            "g": self.g.value(),
            "gravity": self.gravity_checkbox.isChecked(),
            "CMA": self.cma_checkbox.isChecked(),
            "hampel": self.hampel_checkbox.isChecked(),
            "savgol": self.savgol_checkbox.isChecked(),
            "LPF": self.lpf_checkbox.isChecked(),
            "ACF": self.acf_checkbox.isChecked(),
            "FSM": self.fsm_checkbox.isChecked(),
            "peak_prominence": self.peak_prominence.value(),
            "valley_prominence": self.valley_prominence.value(),
            "min_step_period": self.min_step_period.value(),
            "max_step_period": self.max_step_period.value(),
            "state_timeout": self.state_timeout.value(),
            "peak_valley_min_dist": self.peak_valley_min_dist.value(),
        }
    
    def add_task(self):
        """Add current configuration to task list."""
        params = self.get_current_params()
        
        # Validate
        if not params["data_dir"]:
            QMessageBox.warning(self, "Warning", "Please select test dataset directory.")
            return
        if not params["output_dir"]:
            QMessageBox.warning(self, "Warning", "Please select output directory.")
            return
        
        # Add to task list
        self.tasks.append(params)
        
        # Update UI
        row = len(self.tasks)
        task_id = row
        
        # Create detailed task description with all parameters
        config_lines = [f"<b>Task {task_id}</b>:"]
        
        # Basic info
        config_lines.append(f"window={params['window_sec']}s")
        config_lines.append(f"thresh={params['threshold_factor']}")
        config_lines.append(f"refractory={params['refractory_sec']}s")
        
        # Filter params
        config_lines.append(f"cutoff_low={params['cutoff_low']}")
        config_lines.append(f"cutoff_high={params['cutoff_high']}")
        config_lines.append(f"fs={params['fs']}")
        config_lines.append(f"order={params['order']}")
        
        # Advanced params
        config_lines.append(f"n_sigma={params['n_sigma']}")
        config_lines.append(f"polyorder={params['polyorder']}")
        config_lines.append(f"gyro_weight={params['gyro_weight']}")
        config_lines.append(f"noise_thresh={params['noise_threshold']}")
        config_lines.append(f"thresh_abs={params['threshold_absolute']}")
        
        # Detection params
        config_lines.append(f"gyro_energy_min={params['gyro_energy_min']}")
        config_lines.append(f"mag_var_min={params['mag_var_min']}")
        config_lines.append(f"penalty={params['penalty']}")
        config_lines.append(f"acf_dom={params['acf_dominance']}")
        config_lines.append(f"g={params['g']}")
        
        # FSM
        config_lines.append(f"peak_prominence={params['peak_prominence']}")
        config_lines.append(f"valley_prominence={params['valley_prominence']}")
        config_lines.append(f"min_step_period={params['min_step_period']}")
        config_lines.append(f"max_step_period={params['max_step_period']}")
        config_lines.append(f"state_timeout={params['state_timeout']}")
        config_lines.append(f"peak_valley_min_dist={params['peak_valley_min_dist']}")
        
        # Flags
        flags = []
        if params['gravity']: flags.append('G')
        if params['CMA']: flags.append('CMA')
        if params['hampel']: flags.append('H')
        if params['savgol']: flags.append('S')
        if params['LPF']: flags.append('L')
        if params['ACF']: flags.append('A')
        if params['FSM']: flags.append('F')
        config_lines.append(f"flags=[{','.join(flags)}]")
        
        # LSTM
        if params['model_path']:
            config_lines.append("LSTM=enabled")
        
        # Join all lines with line breaks
        config_text = '<br>'.join(config_lines)
        
        # Add to grid
        task_label = QLabel(config_text)
        task_label.setStyleSheet("padding: 5px; background-color: #f5f5f5; border: 1px solid #ddd; border-radius: 3px;")
        task_label.setWordWrap(True)
        self.task_table.addWidget(task_label, row, 0)
        
        delete_btn = QPushButton("Delete")
        delete_btn.setStyleSheet("background-color: #f44336; color: white; padding: 3px 10px;")
        delete_btn.clicked.connect(lambda checked, r=row: self.delete_task(r))
        self.task_table.addWidget(delete_btn, row, 1)
        
        self.log(f"Task {task_id} added successfully.")
    
    def delete_task(self, row):
        """Delete a task from the list."""
        if row < 1 or row > len(self.tasks):
            return
        
        # Remove from data
        del self.tasks[row - 1]
        
        # Rebuild UI
        self.rebuild_task_table()
        
        self.log(f"Task {row} deleted.")
    
    def rebuild_task_table(self):
        """Rebuild the task table after deletion."""
        # Clear existing widgets
        while self.task_table.count():
            item = self.task_table.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        # Re-add header
        header_label = QLabel("<b>Configuration</b>")
        header_label.setStyleSheet("font-weight: bold; padding: 5px;")
        self.task_table.addWidget(header_label, 0, 0)
        
        delete_header = QLabel("<b>Action</b>")
        delete_header.setStyleSheet("font-weight: bold; padding: 5px;")
        self.task_table.addWidget(delete_header, 0, 1)
        
        # Re-add tasks
        for i, params in enumerate(self.tasks):
            row = i + 1
            task_id = row
            
            # Create detailed task description with all parameters
            config_lines = [f"<b>Task {task_id}</b>:"]
            
            # Basic info
            config_lines.append(f"window={params['window_sec']}s")
            config_lines.append(f"thresh={params['threshold_factor']}")
            config_lines.append(f"refractory={params['refractory_sec']}s")
            
            # Filter params
            config_lines.append(f"cutoff_low={params['cutoff_low']}")
            config_lines.append(f"cutoff_high={params['cutoff_high']}")
            config_lines.append(f"fs={params['fs']}")
            config_lines.append(f"order={params['order']}")
            
            # Advanced params
            config_lines.append(f"n_sigma={params['n_sigma']}")
            config_lines.append(f"polyorder={params['polyorder']}")
            config_lines.append(f"gyro_weight={params['gyro_weight']}")
            config_lines.append(f"noise_thresh={params['noise_threshold']}")
            config_lines.append(f"thresh_abs={params['threshold_absolute']}")
            
            # Detection params
            config_lines.append(f"gyro_energy_min={params['gyro_energy_min']}")
            config_lines.append(f"mag_var_min={params['mag_var_min']}")
            config_lines.append(f"penalty={params['penalty']}")
            config_lines.append(f"acf_dom={params['acf_dominance']}")
            config_lines.append(f"g={params['g']}")
            
            # FSM
            config_lines.append(f"peak_prominence={params['peak_prominence']}")
            config_lines.append(f"valley_prominence={params['valley_prominence']}")
            config_lines.append(f"min_step_period={params['min_step_period']}")
            config_lines.append(f"max_step_period={params['max_step_period']}")
            config_lines.append(f"state_timeout={params['state_timeout']}")
            config_lines.append(f"peak_valley_min_dist={params['peak_valley_min_dist']}")
            
            # Flags
            flags = []
            if params['gravity']: flags.append('G')
            if params['CMA']: flags.append('CMA')
            if params['hampel']: flags.append('H')
            if params['savgol']: flags.append('S')
            if params['LPF']: flags.append('L')
            if params['ACF']: flags.append('A')
            if params['FSM']: flags.append('F')
            config_lines.append(f"flags=[{','.join(flags)}]")
            
            # LSTM
            if params['model_path']:
                config_lines.append("LSTM=enabled")
            
            # Join all lines with line breaks
            config_text = '<br>'.join(config_lines)
            
            task_label = QLabel(config_text)
            task_label.setStyleSheet("padding: 5px; background-color: #f5f5f5; border: 1px solid #ddd; border-radius: 3px;")
            task_label.setWordWrap(True)
            self.task_table.addWidget(task_label, row, 0)
            
            delete_btn = QPushButton("Delete")
            delete_btn.setStyleSheet("background-color: #f44336; color: white; padding: 3px 10px;")
            delete_btn.clicked.connect(lambda checked, r=row: self.delete_task(r))
            self.task_table.addWidget(delete_btn, row, 1)
    
    def log(self, msg):
        """Append message to log area."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.log_area.append(f"[{timestamp}] {msg}")
        scrollbar = self.log_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def get_tasks(self):
        """Return all tasks."""
        return self.tasks


class OnlinePlotWindow(QMainWindow):
    def __init__(self, update_interval_sec, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Online Step Counter - Live Plot")
        self.setMinimumSize(800, 500)

        # Data storage
        self.times = []      # list of float
        self.mags = []       # list of float
        self.step_times = [] # list of float
        
        # Step counters
        self.total_steps = 0
        self.new_steps = 0
        self.last_step_count = 0

        self.update_interval = int(update_interval_sec * 1000)  # milliseconds, convert to int

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Step counter display
        counter_layout = QHBoxLayout()
        
        # New steps display
        new_steps_label = QLabel("New Steps:")
        new_steps_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.new_steps_display = QLabel("0")
        self.new_steps_display.setStyleSheet("color: #ff4444; font-weight: bold; font-size: 16px; min-width: 50px;")
        
        # Total steps display
        total_steps_label = QLabel("Total Steps:")
        total_steps_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.total_steps_display = QLabel("0")
        self.total_steps_display.setStyleSheet("color: #44aa44; font-weight: bold; font-size: 16px; min-width: 50px;")
        
        # Step frequency display
        step_freq_label = QLabel("Step Frequency:")
        step_freq_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.step_freq_display = QLabel("0")
        self.step_freq_display.setStyleSheet("color: #44aa44; font-weight: bold; font-size: 16px; min-width: 50px;")
        
        counter_layout.addWidget(new_steps_label)
        counter_layout.addWidget(self.new_steps_display)
        counter_layout.addSpacing(30)
        counter_layout.addWidget(total_steps_label)
        counter_layout.addWidget(self.total_steps_display)
        counter_layout.addSpacing(30)
        counter_layout.addWidget(step_freq_label)
        counter_layout.addWidget(self.step_freq_display)
        counter_layout.addStretch()
        layout.addLayout(counter_layout)

        # Control buttons
        button_layout = QHBoxLayout()
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet("background-color: #ff4444; color: white; font-weight: bold; padding: 5px 15px;")
        self.stop_btn.clicked.connect(self.on_stop_clicked)
        button_layout.addStretch()
        button_layout.addWidget(self.stop_btn)
        layout.addLayout(button_layout)

        # Matplotlib figure
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Setup plot
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Acceleration magnitude (m/s²)")
        self.ax.set_title("Real-time Step Detection")
        self.ax.grid(True)

        # Timer for periodic refresh
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_plot)
        self.timer.start(self.update_interval)
        
        # Reference to worker (will be set by main window)
        self.worker = None

    def closeEvent(self, event):
        """Stop the timer when the window is closed."""
        self.timer.stop()
        if self.worker:
            self.worker.stop()
            # Give worker a brief moment to stop
            time.sleep(0.1)
        event.accept()

    def on_stop_clicked(self):
        """Handle stop button click."""
        if self.worker:
            self.worker.stop()
        self.timer.stop()
        # Reset counters when stopping
        self.new_steps = 0
        self.new_steps_display.setText("0")
        
        # Give worker time to stop before closing
        QTimer.singleShot(100, self.close)

    def append_data(self, times, mags):
        """Append new data points (called from main thread via signal)."""
        # Ensure times and mags are numpy arrays or lists
        self.times.extend(times)
        self.mags.extend(mags)

    def add_steps(self, total_steps, step_timestamps, step_freq=0.0):
        """Add new step timestamps (called from main thread via signal)."""
        self.step_times.extend(step_timestamps)
        # Update step counters
        self.total_steps = total_steps
        self.new_steps = total_steps - self.last_step_count
        self.last_step_count = total_steps
        # Update displays
        self.new_steps_display.setText(str(self.new_steps))
        self.total_steps_display.setText(str(self.total_steps))
        if step_freq > 0:
            self.step_freq_display.setText(f"{step_freq:.2f}")

    def refresh_plot(self):
        """Redraw the plot with all accumulated data."""
        if not self.times:
            return

        # Clear axis
        self.ax.clear()

        # Plot acceleration magnitude
        self.ax.plot(self.times, self.mags, label="Acceleration magnitude", alpha=0.7)

        # Mark detected steps
        for ts in self.step_times:
            self.ax.axvline(x=ts, color='red', linestyle='--', alpha=0.7)

        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Acceleration magnitude (m/s²)")
        self.ax.set_title("Real-time Step Detection")
        self.ax.legend()
        self.ax.grid(True)

        # Auto-scale y-axis to current data
        if self.mags:
            ymin = min(self.mags)
            ymax = max(self.mags)
            margin = (ymax - ymin) * 0.1 if ymax > ymin else 1.0
            self.ax.set_ylim(ymin - margin, ymax + margin)

        self.canvas.draw()


class EvaluationWorker(QObject):
    """Worker class for running evaluation experiments in background."""
    
    log_signal = pyqtSignal(str)
    task_started = pyqtSignal(int)
    task_finished = pyqtSignal(int, dict)
    experiment_finished = pyqtSignal()
    
    def __init__(self, tasks):
        super().__init__()
        self.tasks = tasks
        self._stopped = False
    
    def stop(self):
        self._stopped = True
    
    def run(self):
        """Run all evaluation tasks."""
        try:
            preprocessor = DataPreprocessor()
            
            for task_idx, params in enumerate(self.tasks):
                if self._stopped:
                    break
                
                task_num = task_idx + 1
                self.log_signal.emit(f"\n{'='*60}")
                self.log_signal.emit(f"Starting Task {task_num}/{len(self.tasks)}")
                self.log_signal.emit(f"{'='*60}")
                
                # Validate directories
                data_dir = params['data_dir']
                output_dir = params['output_dir']
                
                if not os.path.isdir(data_dir):
                    self.log_signal.emit(f"ERROR: Data directory '{data_dir}' does not exist.")
                    continue
                
                # Create output directory if not exists
                os.makedirs(output_dir, exist_ok=True)
                
                # Find all CSV files
                file_pattern = os.path.join(data_dir, "*.csv")
                all_files = glob.glob(file_pattern)
                
                if not all_files:
                    self.log_signal.emit(f"No CSV files found in '{data_dir}'")
                    continue
                
                # Parse ground truth from filenames
                valid_files = []
                true_steps_list = []
                
                for f in all_files:
                    basename = os.path.basename(f)
                    parts = basename.split('_')
                    if len(parts) != 2 or not parts[1].endswith('.csv'):
                        self.log_signal.emit(f"Skipping file with unexpected name: {basename}")
                        continue
                    try:
                        true_steps = int(parts[1].replace('.csv', ''))
                        valid_files.append(f)
                        true_steps_list.append(true_steps)
                    except ValueError:
                        self.log_signal.emit(f"Skipping file with invalid true steps: {basename}")
                        continue
                
                if not valid_files:
                    self.log_signal.emit("No valid files to process.")
                    continue
                
                self.log_signal.emit(f"Found {len(valid_files)} valid files.")
                
                # Initialize counter with task parameters
                counter = StepCounter(
                    model_path=params.get('model_path'),
                    window_sec=params['window_sec'],
                    threshold_factor=params['threshold_factor'],
                    refractory_sec=params['refractory_sec'],
                    filter_len=params['filter_len'],
                    cutoff_low=params['cutoff_low'],
                    cutoff_high=params['cutoff_high'],
                    fs=params['fs'],
                    order=params['order'],
                    n_sigma=params['n_sigma'],
                    polyorder=params['polyorder'],
                    gyro_weight=params['gyro_weight'],
                    noise_threshold=params['noise_threshold'],
                    threshold_absolute=params['threshold_absolute'],
                    gyro_energy_min=params['gyro_energy_min'],
                    mag_var_min=params['mag_var_min'],
                    penalty=params['penalty'],
                    acf_dominance=params['acf_dominance'],
                    g=params['g'],
                    gravity=params.get('gravity', False),
                    CMA=params.get('CMA', True),
                    hampel=params.get('hampel', True),
                    savgol=params.get('savgol', True),
                    LPF=params.get('LPF', True),
                    ACF=params.get('ACF', True),
                    FSM=params.get('FSM', False),
                    peak_prominence=params['peak_prominence'],
                    valley_prominence=params['valley_prominence'],
                    min_step_period=params['min_step_period'],
                    max_step_period=params['max_step_period'],
                    state_timeout=params['state_timeout'],
                    peak_valley_min_dist=params['peak_valley_min_dist']
                )
                
                # Process each file
                predictions = []
                results_log = []
                
                for file_idx, (f, true_steps) in enumerate(zip(valid_files, true_steps_list)):
                    if self._stopped:
                        break
                    
                    basename = os.path.basename(f)
                    self.log_signal.emit(f"[{file_idx+1}/{len(valid_files)}] Processing {basename}...")
                    
                    try:
                        # Preprocess data
                        data_dict = preprocessor.offline([f])
                        if not data_dict:
                            self.log_signal.emit(f"  Warning: Could not read file, skipping.")
                            continue
                        
                        # Run step counter
                        result = counter.run_offline(data_dict)
                        pred_steps = result['step_count']
                        predictions.append(pred_steps)
                        
                        # Log result
                        result_str = f"  True: {true_steps}, Pred: {pred_steps}"
                        self.log_signal.emit(result_str)
                        results_log.append({
                            'file': basename,
                            'true_steps': true_steps,
                            'predicted_steps': pred_steps,
                            'error': abs(pred_steps - true_steps)
                        })
                        
                        # Save intermediate results
                        self.save_task_results(
                            output_dir, task_num, params, results_log
                        )
                        
                    except Exception as e:
                        error_msg = f"  ERROR processing {basename}: {str(e)}"
                        self.log_signal.emit(error_msg)
                        results_log.append({
                            'file': basename,
                            'error': str(e)
                        })
                        continue
                
                # Calculate final metrics
                if predictions:
                    from sklearn.metrics import mean_absolute_error, mean_squared_error
                    preds = np.array(predictions, dtype=float)
                    trues = np.array(true_steps_list[:len(predictions)], dtype=float)
                    
                    total_true = np.sum(trues)
                    total_pred = np.sum(preds)
                    abs_errors = np.abs(preds - trues)
                    total_abs_error = np.sum(abs_errors)
                    mae = mean_absolute_error(trues, preds)
                    rmse = np.sqrt(mean_squared_error(trues, preds))
                    
                    with np.errstate(divide='ignore', invalid='ignore'):
                        rel_errors = np.where(trues != 0, abs_errors / trues, 0)
                    mean_rel_error = np.mean(rel_errors) * 100
                    
                    exact_matches = np.sum(preds == trues)
                    accuracy = exact_matches / len(preds) * 100
                    
                    # Log metrics
                    self.log_signal.emit(f"\n--- Task {task_num} Results ---")
                    self.log_signal.emit(f"Files processed: {len(predictions)}")
                    self.log_signal.emit(f"Total true steps: {total_true:.0f}")
                    self.log_signal.emit(f"Total predicted steps: {total_pred:.0f}")
                    self.log_signal.emit(f"Total absolute error: {total_abs_error:.0f}")
                    self.log_signal.emit(f"MAE: {mae:.2f}")
                    self.log_signal.emit(f"RMSE: {rmse:.2f}")
                    self.log_signal.emit(f"Mean Relative Error: {mean_rel_error:.2f}%")
                    self.log_signal.emit(f"Exact Match Accuracy: {accuracy:.2f}%")
                    
                    # Save final results
                    metrics = {
                        'total_true': float(total_true),
                        'total_pred': float(total_pred),
                        'total_abs_error': float(total_abs_error),
                        'mae': float(mae),
                        'rmse': float(rmse),
                        'mean_rel_error': float(mean_rel_error),
                        'accuracy': float(accuracy),
                        'num_files': len(predictions)
                    }
                    
                    self.save_final_results(
                        output_dir, task_num, params, results_log, metrics
                    )
                    
                    # Emit task finished signal
                    self.task_finished.emit(task_num, metrics)
                else:
                    self.log_signal.emit(f"Task {task_num}: No predictions were made.")
            
            self.log_signal.emit(f"\n{'='*60}")
            self.log_signal.emit("All tasks completed!")
            self.log_signal.emit(f"{'='*60}")
            
        except Exception as e:
            self.log_signal.emit(f"EXPERIMENT ERROR: {str(e)}")
            import traceback
            self.log_signal.emit(traceback.format_exc())
        
        self.experiment_finished.emit()
    
    def save_task_results(self, output_dir, task_num, params, results_log):
        """Save intermediate results to text file."""
        filename = os.path.join(output_dir, f"task_{task_num}_results.txt")
        
        with open(filename, 'w') as f:
            f.write(f"Task {task_num} - Intermediate Results\n")
            f.write("="*60 + "\n\n")
            
            f.write("Parameters:\n")
            for key, value in params.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            f.write("Results:\n")
            for result in results_log:
                if 'true_steps' in result:
                    f.write(f"  File: {result['file']}\n")
                    f.write(f"    True Steps: {result['true_steps']}\n")
                    f.write(f"    Predicted Steps: {result['predicted_steps']}\n")
                    f.write(f"    Error: {result['error']}\n\n")
                else:
                    f.write(f"  File: {result['file']}\n")
                    f.write(f"    ERROR: {result['error']}\n\n")
    
    def save_final_results(self, output_dir, task_num, params, results_log, metrics):
        """Save final results to text file."""
        filename = os.path.join(output_dir, f"task_{task_num}_final.txt")
        
        with open(filename, 'w') as f:
            f.write(f"Task {task_num} - Final Results\n")
            f.write("="*60 + "\n\n")
            
            f.write("Parameters:\n")
            for key, value in params.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            f.write("Detailed Results:\n")
            for result in results_log:
                if 'true_steps' in result:
                    f.write(f"  File: {result['file']}\n")
                    f.write(f"    True Steps: {result['true_steps']}\n")
                    f.write(f"    Predicted Steps: {result['predicted_steps']}\n")
                    f.write(f"    Absolute Error: {result['error']}\n\n")
                else:
                    f.write(f"  File: {result['file']}\n")
                    f.write(f"    ERROR: {result['error']}\n\n")
            
            f.write("\nSummary Metrics:\n")
            f.write("-"*60 + "\n")
            f.write(f"  Number of Files: {metrics['num_files']}\n")
            f.write(f"  Total True Steps: {metrics['total_true']:.0f}\n")
            f.write(f"  Total Predicted Steps: {metrics['total_pred']:.0f}\n")
            f.write(f"  Total Absolute Error: {metrics['total_abs_error']:.0f}\n")
            f.write(f"  Mean Absolute Error (MAE): {metrics['mae']:.2f}\n")
            f.write(f"  Root Mean Square Error (RMSE): {metrics['rmse']:.2f}\n")
            f.write(f"  Mean Relative Error: {metrics['mean_rel_error']:.2f}%\n")
            f.write(f"  Exact Match Accuracy: {metrics['accuracy']:.2f}%\n")


class OnlineWorker(QObject):
    # Signals
    log = pyqtSignal(str)
    steps_updated = pyqtSignal(int, np.ndarray, float)   # total_steps, new_step_timestamps, step_frequency
    data_chunk = pyqtSignal(list, list)           # times (list), magnitudes (list)
    finished = pyqtSignal()

    def __init__(self, ip, port, params, model_path=None):
        super().__init__()
        self.ip = ip
        self.port = port
        self.params = params
        self.model_path = model_path
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        try:
            pre = DataPreprocessor()
            gen = pre.online(self.ip, self.port)
            counter = StepCounter(
                model_path=self.model_path,
                window_sec=self.params['window_sec'],
                threshold_factor=self.params['threshold_factor'],
                refractory_sec=self.params['refractory_sec'],
                filter_len=self.params['filter_len'],
                cutoff_low=self.params['cutoff_low'],
                cutoff_high=self.params['cutoff_high'],
                fs=self.params['fs'],
                order=self.params['order'],
                n_sigma=self.params['n_sigma'],
                polyorder=self.params['polyorder'],
                gyro_weight=self.params['gyro_weight'],
                noise_threshold=self.params['noise_threshold'],
                threshold_absolute=self.params['threshold_absolute'],
                gyro_energy_min=self.params['gyro_energy_min'],
                mag_var_min=self.params['mag_var_min'],
                penalty=self.params['penalty'],
                acf_dominance=self.params['acf_dominance'],
                g=self.params['g'],
                gravity=self.params.get('gravity', False),
                CMA=self.params.get('CMA', True),
                hampel=self.params.get('hampel', True),
                savgol=self.params.get('savgol', True),
                LPF=self.params.get('LPF', True),
                ACF=self.params.get('ACF', True),
                FSM=self.params.get('FSM', False),
                peak_prominence=self.params.get('peak_prominence', 0.1),
                valley_prominence=self.params.get('valley_prominence', 0.05),
                min_step_period=self.params.get('min_step_period', 0.3),
                max_step_period=self.params.get('max_step_period', 3.0),
                state_timeout=self.params.get('state_timeout', 1.5),
                peak_valley_min_dist=self.params.get('peak_valley_min_dist', 0.2),
            )
            for chunk in gen:
                if not self._running:
                    break
                # Process the chunk
                out = counter.update(chunk)
                # Emit new steps
                if out['new_steps'] > 0:
                    # Calculate step frequency from period (in seconds)
                    period = out['diagnostics'].get('period', 0)
                    step_freq = 1.0 / period if period > 0 else 0.0
                    self.steps_updated.emit(out['total_steps'], out['new_step_timestamps'], step_freq)
                # Compute magnitude for plotting (use raw acceleration from chunk)
                acc = chunk['acc']
                mag = np.sqrt(np.sum(acc ** 2, axis=1))
                # Emit data chunk for plotting
                self.data_chunk.emit(chunk['time'].tolist(), mag.tolist())
        except Exception as e:
            self.log.emit(f"Online error: {str(e)}")
        self.finished.emit()


class EvaluationLogWindow(QMainWindow):
    """Separate window for displaying evaluation experiment logs."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Evaluation Experiment Log")
        self.setMinimumSize(800, 600)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Title
        title_label = QLabel("Experiment Log Output")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title_label)
        
        # Log area
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setFontFamily("Consolas")
        self.log_area.setFontPointSize(10)
        layout.addWidget(self.log_area)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.clear_btn = QPushButton("Clear Log")
        self.clear_btn.clicked.connect(self.clear_log)
        button_layout.addWidget(self.clear_btn)
        
        self.save_btn = QPushButton("Save Log...")
        self.save_btn.clicked.connect(self.save_log)
        button_layout.addWidget(self.save_btn)
        
        button_layout.addStretch()
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
    
    def append_log(self, message):
        """Append a message to the log area."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.log_area.append(f"[{timestamp}] {message}")
        scrollbar = self.log_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def clear_log(self):
        """Clear all log content."""
        self.log_area.clear()
    
    def save_log(self):
        """Save log to text file."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Log File", "", "Text Files (*.txt);;All Files (*)")
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.log_area.toPlainText())
                QMessageBox.information(self, "Success", f"Log saved to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save log:\n{str(e)}")
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Just hide instead of close, so we can reuse it
        self.hide()
        event.ignore()


class ResultWindow(QMainWindow):
    def __init__(self, data, result, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Step Counter Result")
        self.setMinimumSize(800, 600)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Text summary
        step_count = result.get('step_count', len(result.get('step_timestamps', [])))
        self.summary_label = QLabel(f"Step count: {step_count}")
        layout.addWidget(self.summary_label)

        # Matplotlib figure
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.plot_data(data, result)

    def plot_data(self, data, result):
        t = data['time']
        acc = data['acc']
        mag = np.sqrt(np.sum(acc ** 2, axis=1))
        step_timestamps = result.get('step_timestamps', np.array([]))
        filtered = result.get('diagnostics', {}).get('filtered', None)
        fsm = result.get('diagnostics', {}).get('fsm_signal', None)
        # print(mag==filtered)

        ax = self.figure.add_subplot(111)
        ax.plot(t, mag, label='Raw magnitude', alpha=0.5)
        
        # Plot filtered signal if available
        if filtered is not None:
            ax.plot(t, filtered, label='Smoothed', linewidth=1)
        
        if fsm is not None:
            ax.plot(t, fsm, label='FSM', linewidth=1)
        
        for ts in step_timestamps:
            ax.axvline(x=ts, color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration magnitude (m/s²)')
        ax.set_title('Step Detection')
        ax.legend()
        ax.grid(True)
        self.canvas.draw()



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Step Counter GUI")
        self.setMinimumSize(600, 400)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_offline = QPushButton("Offline")
        self.btn_online = QPushButton("Online")
        self.btn_evaluation = QPushButton("Evaluation")
        self.btn_offline.clicked.connect(self.offline_mode)
        self.btn_online.clicked.connect(self.online_mode)
        self.btn_evaluation.clicked.connect(self.evaluation_mode)
        btn_layout.addWidget(self.btn_offline)
        btn_layout.addWidget(self.btn_online)
        btn_layout.addWidget(self.btn_evaluation)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Log area
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        layout.addWidget(QLabel("Log:"))
        layout.addWidget(self.log_area)

        # Online thread management
        self.online_thread = None
        self.online_worker = None
        self.plot_window = None   # reference to live plot window
        
        # Evaluation thread management
        self.evaluation_thread = None
        self.evaluation_worker = None
        self.evaluation_log_window = None

    def log(self, msg):
        self.log_area.append(msg)
        scrollbar = self.log_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def offline_mode(self):
        dlg = OfflineParamDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        params = dlg.get_params()
        files = params.pop("files")
        model_path = params.pop("model_path", None)
        if not files or files == [""]:
            QMessageBox.warning(self, "No files", "Please select at least one CSV file.")
            return

        self.log(f"Starting offline processing with files: {files}")
        try:
            pre = DataPreprocessor()
            data = pre.offline(files)
            if 'acc' not in data:
                self.log("Error: No acceleration data found in the provided files.")
                return

            counter = StepCounter(model_path=model_path, **params)
            result = counter.run_offline(data)

            step_count = result['step_count']
            timestamps = result['step_timestamps']
            self.log(f"Offline processing finished. Step count: {step_count}")

            # self.result_win = ResultWindow(data, timestamps, step_count)
            self.result_win = ResultWindow(data, result)
            self.result_win.show()
        except Exception as e:
            self.log(f"Offline error: {str(e)}")
            QMessageBox.critical(self, "Error", f"Processing failed:\n{str(e)}")

    def online_mode(self):
        dlg = OnlineParamDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        params = dlg.get_params()
        ip = params.pop("ip")
        port = params.pop("port")
        model_path = params.pop("model_path", None)
        plot_interval = params['window_sec']   # use same interval for plot update

        self.log(f"Starting online connection to {ip}:{port}")

        # Stop any existing online thread
        if self.online_worker is not None:
            self.online_worker.stop()
            if self.online_thread is not None:
                self.online_thread.quit()
                self.online_thread.wait()

        # Create and show live plot window
        self.plot_window = OnlinePlotWindow(plot_interval)
        self.plot_window.worker = self.online_worker  # Set worker reference for stop functionality
        self.plot_window.show()

        # Create thread and worker
        self.online_thread = QThread()
        self.online_worker = OnlineWorker(ip, port, params, model_path=model_path)
        self.online_worker.moveToThread(self.online_thread)

        # Connect signals
        self.online_worker.log.connect(self.log)
        self.online_worker.steps_updated.connect(self.on_steps_updated)
        self.online_worker.data_chunk.connect(self.plot_window.append_data)
        self.online_worker.finished.connect(self.on_online_finished)
        self.online_thread.started.connect(self.online_worker.run)
        self.online_thread.finished.connect(self.on_online_thread_finished)

        # Start
        self.online_thread.start()

    def on_steps_updated(self, total_steps, timestamps, step_freq):
        # Also forward steps to the plot window
        if self.plot_window:
            self.plot_window.add_steps(total_steps, timestamps, step_freq)
        self.log(f"Step(s) detected! Total: {total_steps}, Frequency: {step_freq:.2f} Hz")

    def on_online_finished(self):
        self.log("Online worker finished.")
        
        # Clean up thread properly
        if self.online_thread is not None:
            if self.online_thread.isRunning():
                self.online_thread.quit()
                self.online_thread.wait(1000)  # Wait up to 1 second for thread to finish
            self.online_thread = None
        
        self.online_worker = None
        
        # Optionally close plot window when worker stops
        if self.plot_window:
            self.plot_window.close()
            self.plot_window = None
    
    def on_online_thread_finished(self):
        """Called when the online thread is completely finished."""
        self.log("Online thread cleaned up.")
        self.online_thread = None
    
    def evaluation_mode(self):
        """Open evaluation dialog and run ablation study."""
        dlg = EvaluationParamDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        
        tasks = dlg.get_tasks()
        
        if not tasks:
            QMessageBox.warning(self, "No Tasks", "Please add at least one task to the task list.")
            return
        
        self.log(f"Starting evaluation with {len(tasks)} task(s)...")
        
        # Create and show evaluation log window
        self.evaluation_log_window = EvaluationLogWindow()
        self.evaluation_log_window.show()
        
        # Create thread and worker
        self.evaluation_thread = QThread()
        self.evaluation_worker = EvaluationWorker(tasks)
        self.evaluation_worker.moveToThread(self.evaluation_thread)
        
        # Connect signals
        self.evaluation_worker.log_signal.connect(self.evaluation_log_window.append_log)
        self.evaluation_worker.task_started.connect(self.on_evaluation_task_started)
        self.evaluation_worker.task_finished.connect(self.on_evaluation_task_finished)
        self.evaluation_worker.experiment_finished.connect(self.on_evaluation_finished)
        self.evaluation_thread.started.connect(self.evaluation_worker.run)
        self.evaluation_thread.finished.connect(self.on_evaluation_thread_finished)
        
        # Start
        self.evaluation_thread.start()
    
    def on_evaluation_task_started(self, task_num):
        """Called when an evaluation task starts."""
        self.log(f"Evaluation Task {task_num} started.")
    
    def on_evaluation_task_finished(self, task_num, metrics):
        """Called when an evaluation task finishes."""
        self.log(f"Evaluation Task {task_num} completed. MAE: {metrics['mae']:.2f}, Accuracy: {metrics['accuracy']:.2f}%")
    
    def on_evaluation_finished(self):
        """Called when all evaluation tasks are finished."""
        self.log("Evaluation experiment finished.")
        
        # Clean up thread properly
        if self.evaluation_thread is not None:
            if self.evaluation_thread.isRunning():
                self.evaluation_thread.quit()
                self.evaluation_thread.wait(1000)
            self.evaluation_thread = None
        
        self.evaluation_worker = None
    
    def on_evaluation_thread_finished(self):
        """Called when the evaluation thread is completely finished."""
        self.log("Evaluation thread cleaned up.")
        self.evaluation_thread = None
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())