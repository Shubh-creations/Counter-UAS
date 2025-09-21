# C-UAS Military-Style Dashboard with User Authentication and Drone Classification

import sys
import cv2
import datetime
import os
import numpy as np
import random
import requests
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
                             QFileDialog, QHBoxLayout, QMenuBar, QAction, QFrame,
                             QGridLayout, QSplitter, QGraphicsDropShadowEffect, QProgressBar,
                             QLineEdit, QDialog, QFormLayout, QComboBox, QCheckBox, QTabWidget,
                             QTextEdit, QStackedWidget)
from PyQt5.QtCore import QTimer, Qt, QSize
from PyQt5.QtGui import QPixmap, QImage, QColor, QFont, QPalette, QBrush, QPainter, QPen, QFontDatabase
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.models as models
from PIL import Image, ImageQt
from collections import deque
import psutil
try:
    import pynvml as nvml
except Exception:
    nvml = None
import time

# Drone classifier from Program 2
class DroneClassifier(nn.Module):
    def __init__(self):
        super(DroneClassifier, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 5)

    def forward(self, x):
        return self.base_model(x)

# Load military style font
def load_military_font():
    # In a real application, you would load an actual military font file
    # For this example, we'll use a system font that looks somewhat military
    font_id = QFontDatabase.addApplicationFont(":/fonts/military.ttf")
    if font_id < 0:
        # If font loading fails, use a fallback font that's likely available
        return QFont("Courier New", 10, QFont.Bold)
    else:
        font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
        return QFont(font_family, 10)

# Custom styled panel for dashboard elements
class DashboardPanel(QFrame):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setObjectName("dashboardPanel")
        self.setStyleSheet("""
            #dashboardPanel {
                background-color: #000000;
                border: 1px solid #ffffff;
                border-radius: 0px;
            }
        """)
        
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(255, 255, 255, 120))
        shadow.setOffset(2, 2)
        self.setGraphicsEffect(shadow)
        
        # Layout
        self.layout = QVBoxLayout(self)
        
        # Title if provided
        if title:
            title_label = QLabel(title)
            military_font = QFont("Courier New", 12, QFont.Bold)
            title_label.setFont(military_font)
            title_label.setStyleSheet("""
                color: #ffffff;
                padding-bottom: 5px;
                border-bottom: 1px solid #ffffff;
            """)
            self.layout.addWidget(title_label)

# Military-style satellite map visualization
class SatelliteMapWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background-color: #000000;")
        
        # Default coordinates (will be updated with actual location)
        self.latitude = 37.7749
        self.longitude = -122.4194
        
        # Blinking effect for location marker
        self.blink_timer = QTimer(self)
        self.blink_timer.timeout.connect(self.update)
        self.blink_timer.start(500)
        self.blink_state = True
        
        # Load a satellite map background image
        # In a real app, you would load an actual satellite image
        self.map_image = None
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw black background
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        
        # Draw grid lines (simulating map grid)
        painter.setPen(QPen(QColor(0, 100, 0), 1, Qt.DotLine))
        
        # Draw horizontal grid lines
        for y in range(0, self.height(), 20):
            painter.drawLine(0, y, self.width(), y)
            
        # Draw vertical grid lines
        for x in range(0, self.width(), 20):
            painter.drawLine(x, 0, x, self.height())
        
        # Draw latitude/longitude lines
        painter.setPen(QPen(QColor(0, 150, 0), 1))
        
        # Draw main latitude/longitude lines
        painter.drawLine(0, self.height() // 2, self.width(), self.height() // 2)  # Equator
        painter.drawLine(self.width() // 2, 0, self.width() // 2, self.height())  # Prime meridian
        
        # Draw coordinate labels
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Courier New", 8))
        
        # Draw some fake coordinates around the map
        for i in range(1, 5):
            # Latitude markers
            y_pos = int(self.height() * i / 5)
            painter.drawText(5, y_pos, f"{90 - i*30:d}°N")
            
            # Longitude markers
            x_pos = int(self.width() * i / 5)
            painter.drawText(x_pos, 15, f"{i*30:d}°E")
        
        # Calculate position based on coordinates
        # This is a simplified mapping - in a real app you would use proper projection
        x = int(self.width() * (self.longitude + 180) / 360)
        y = int(self.height() * (90 - self.latitude) / 180)
        
        # Draw location marker (pulsating red dot)
        if self.blink_state:
            # Outer glow
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 0, 0, 50))
            painter.drawEllipse(x - 15, y - 15, 30, 30)
            
            # Inner dot
            painter.setBrush(QColor(255, 0, 0, 200))
            painter.drawEllipse(x - 5, y - 5, 10, 10)
            
            # Targeting lines
            painter.setPen(QPen(QColor(255, 0, 0), 1))
            painter.drawLine(x - 20, y, x - 10, y)  # Left
            painter.drawLine(x + 10, y, x + 20, y)  # Right
            painter.drawLine(x, y - 20, x, y - 10)  # Top
            painter.drawLine(x, y + 10, x, y + 20)  # Bottom
        
        # Draw border
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(1, 1, self.width() - 2, self.height() - 2)
        
        # Draw corner brackets
        bracket_size = 20
        painter.setPen(QPen(QColor(0, 255, 0), 2))
        
        # Top-left
        painter.drawLine(0, 0, bracket_size, 0)
        painter.drawLine(0, 0, 0, bracket_size)
        
        # Top-right
        painter.drawLine(self.width(), 0, self.width() - bracket_size, 0)
        painter.drawLine(self.width(), 0, self.width(), bracket_size)
        
        # Bottom-left
        painter.drawLine(0, self.height(), bracket_size, self.height())
        painter.drawLine(0, self.height(), 0, self.height() - bracket_size)
        
        # Bottom-right
        painter.drawLine(self.width(), self.height(), self.width() - bracket_size, self.height())
        painter.drawLine(self.width(), self.height(), self.width(), self.height() - bracket_size)
    
    def set_coordinates(self, lat, lng):
        self.latitude = lat
        self.longitude = lng
        self.update()  # Trigger repaint

# Login dialog
class LoginDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("C-UAS AUTHENTICATION")
        self.setFixedSize(400, 250)
        self.setStyleSheet("""
            QDialog {
                background-color: #000000;
            }
            QLabel {
                color: #ffffff;
                font-family: 'Courier New';
                font-weight: bold;
            }
            QLineEdit {
                background-color: #000000;
                color: #ffffff;
                border: 1px solid #ffffff;
                padding: 5px;
                font-family: 'Courier New';
            }
            QPushButton {
                background-color: #000000;
                color: #ffffff;
                border: 1px solid #ffffff;
                padding: 8px 16px;
                font-family: 'Courier New';
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #003300;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("ENTER YOUR CREDENTIALS")
        title.setFont(QFont("Courier New", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Form layout
        form_layout = QFormLayout()
        
        # Name field
        self.name_input = QLineEdit()
        name_label = QLabel("NAME:")
        name_label.setFont(QFont("Courier New", 12, QFont.Bold))
        form_layout.addRow(name_label, self.name_input)
        
        # Profile field
        self.profile_input = QLineEdit()
        profile_label = QLabel("PROFILE:")
        profile_label.setFont(QFont("Courier New", 12, QFont.Bold))
        form_layout.addRow(profile_label, self.profile_input)
        
        layout.addLayout(form_layout)
        layout.addSpacing(20)
        
        # Login button
        login_button = QPushButton("ACCESS SYSTEM")
        login_button.clicked.connect(self.accept)
        layout.addWidget(login_button)
        
        # Blinking effect for title
        self.blink_timer = QTimer(self)
        self.blink_timer.timeout.connect(self.blink_title)
        self.blink_timer.start(500)
        self.blink_state = True
    
    def blink_title(self):
        self.blink_state = not self.blink_state
        if self.blink_state:
            self.setStyleSheet(self.styleSheet().replace("color: #ffffff;", "color: #ffffff;"))
        else:
            self.setStyleSheet(self.styleSheet().replace("color: #ffffff;", "color: #333333;"))

# Drone details form for 'Other' classification
class DroneDetailsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DRONE CLASSIFICATION DETAILS")
        self.setFixedSize(500, 400)
        self.setStyleSheet("""
            QDialog, QTabWidget, QWidget {
                background-color: #000000;
            }
            QLabel {
                color: #ffffff;
                font-family: 'Courier New';
                font-weight: bold;
            }
            QLineEdit, QTextEdit, QComboBox {
                background-color: #000000;
                color: #ffffff;
                border: 1px solid #ffffff;
                padding: 5px;
                font-family: 'Courier New';
            }
            QCheckBox {
                color: #ffffff;
                font-family: 'Courier New';
            }
            QPushButton {
                background-color: #000000;
                color: #ffffff;
                border: 1px solid #ffffff;
                padding: 8px 16px;
                font-family: 'Courier New';
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #003300;
            }
            QTabWidget::pane {
                border: 1px solid #ffffff;
            }
            QTabBar::tab {
                background-color: #000000;
                color: #ffffff;
                border: 1px solid #ffffff;
                padding: 5px 10px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #003300;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Create tab widget
        tabs = QTabWidget()
        
        # Basic info tab
        basic_tab = QWidget()
        basic_layout = QFormLayout(basic_tab)
        
        self.model_input = QLineEdit()
        basic_layout.addRow(QLabel("MODEL:"), self.model_input)
        
        self.size_combo = QComboBox()
        self.size_combo.addItems(["Small", "Medium", "Large"])
        basic_layout.addRow(QLabel("SIZE:"), self.size_combo)
        
        self.munition_check = QCheckBox("Has Munition")
        basic_layout.addRow(self.munition_check)
        
        self.gps_denied_check = QCheckBox("GPS-Denied Capability")
        basic_layout.addRow(self.gps_denied_check)
        
        # Capabilities tab
        capabilities_tab = QWidget()
        capabilities_layout = QFormLayout(capabilities_tab)
        
        self.range_input = QLineEdit()
        capabilities_layout.addRow(QLabel("RANGE (km):"), self.range_input)
        
        self.payload_input = QLineEdit()
        capabilities_layout.addRow(QLabel("PAYLOAD (kg):"), self.payload_input)
        
        self.camera_check = QCheckBox("Advanced Camera")
        capabilities_layout.addRow(self.camera_check)
        
        self.thermal_check = QCheckBox("Thermal Imaging")
        capabilities_layout.addRow(self.thermal_check)
        
        # Notes tab
        notes_tab = QWidget()
        notes_layout = QVBoxLayout(notes_tab)
        
        notes_layout.addWidget(QLabel("ADDITIONAL NOTES:"))
        self.notes_text = QTextEdit()
        notes_layout.addWidget(self.notes_text)
        
        # Add tabs to widget
        tabs.addTab(basic_tab, "BASIC INFO")
        tabs.addTab(capabilities_tab, "CAPABILITIES")
        tabs.addTab(notes_tab, "NOTES")
        
        layout.addWidget(tabs)
        
        # Save button
        save_button = QPushButton("SAVE CLASSIFICATION")
        save_button.clicked.connect(self.accept)
        layout.addWidget(save_button)

class DesktopApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("C-UAS CONTROL CENTER")
        self.setMinimumSize(1200, 800)
        
        # Set black background theme
        self.setStyleSheet("""
            QWidget {
                background-color: #000000;
                color: #ffffff;
                font-family: 'Courier New';
            }
            QLabel {
                color: #ffffff;
            }
            QPushButton {
                background-color: #000000;
                color: #ffffff;
                border: 1px solid #ffffff;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #003300;
            }
            QMenuBar {
                background-color: #000000;
                color: #ffffff;
                border-bottom: 1px solid #ffffff;
            }
            QMenuBar::item:selected {
                background-color: #003300;
            }
        """)
        
        # Main layout with stacked widget for login/main screens
        self.main_container = QVBoxLayout(self)
        self.main_container.setContentsMargins(0, 0, 0, 0)
        self.main_container.setSpacing(0)
        
        self.stacked_widget = QStackedWidget()
        
        # Create login page
        self.login_page = QWidget()
        login_layout = QVBoxLayout(self.login_page)
        login_layout.setAlignment(Qt.AlignCenter)
        
        # Title for login page
        login_title = QLabel("C-UAS CONTROL CENTER")
        login_title.setFont(QFont("Courier New", 24, QFont.Bold))
        login_title.setAlignment(Qt.AlignCenter)
        login_title.setStyleSheet("color: #ffffff; margin-bottom: 30px;")
        login_layout.addWidget(login_title)
        
        # Login button
        login_button = QPushButton("ENTER YOUR NAME AND PROFILE")
        login_button.setFont(QFont("Courier New", 14, QFont.Bold))
        login_button.setFixedSize(400, 50)
        login_button.clicked.connect(self.show_login_dialog)
        login_layout.addWidget(login_button)
        
        # Create main application page
        self.main_page = QWidget()
        main_layout = QVBoxLayout(self.main_page)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Initialize status_label here to ensure it exists
        self.status_label = QLabel("SYSTEM STATUS: ONLINE | LAST UPDATE: " + 
                             datetime.datetime.now().strftime("%H:%M:%S"))
        self.status_label.setFont(QFont("Courier New", 10))
        
        # Menu bar
        self.menu_bar = QMenuBar(self)
        self.menu_bar.setFont(QFont("Courier New", 10, QFont.Bold))
        
        # Menu bar items
        file_menu = self.menu_bar.addMenu("FILE")
        upload_action = QAction("Upload Video", self)
        camera_action = QAction("Open Camera", self)
        file_menu.addAction(upload_action)
        file_menu.addAction(camera_action)
        
        upload_action.triggered.connect(self.load_video)
        camera_action.triggered.connect(self.open_camera)
        
        main_layout.setMenuBar(self.menu_bar)
        
        # Header with user info and time
        header = QHBoxLayout()
        self.user_label = QLabel("OPERATOR: NOT LOGGED IN")
        self.user_label.setFont(QFont("Courier New", 12, QFont.Bold))
        
        current_time = QLabel()
        current_time.setFont(QFont("Courier New", 12, QFont.Bold))
        self.time_label = current_time
        
        # Update time every second
        self.time_timer = QTimer(self)
        self.time_timer.timeout.connect(self.update_time)
        self.time_timer.start(1000)
        self.update_time()
        
        header.addWidget(self.user_label)
        header.addStretch()
        header.addWidget(current_time)
        main_layout.addLayout(header)
        
        # Main content area with grid layout
        content = QGridLayout()
        content.setSpacing(15)
        
        # Video feed panel
        video_panel = DashboardPanel("LIVE VIDEO FEED")
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 1px solid #ffffff;")
        video_panel.layout.addWidget(self.video_label)
        content.addWidget(video_panel, 0, 0, 2, 1)
        
        # Right side panels
        right_panel = QVBoxLayout()
        
        # Object detection panel
        detection_panel = DashboardPanel("OBJECT DETECTION")
        detection_layout = QVBoxLayout()
        
        self.object_info = QLabel("STATUS: AWAITING DETECTION")
        self.object_info.setFont(QFont("Courier New", 12, QFont.Bold))
        
        self.detection_status = QLabel("NO OBJECTS DETECTED")
        self.detection_status.setFont(QFont("Courier New", 10))
        
        self.first_detection_time = QLabel("FIRST DETECTION: NONE")
        self.first_detection_time.setFont(QFont("Courier New", 10))
        
        self.detection_image = QLabel("DETECTED OBJECT PREVIEW")
        self.detection_image.setAlignment(Qt.AlignCenter)
        self.detection_image.setMinimumSize(200, 200)
        self.detection_image.setStyleSheet("border: 1px solid #ffffff; background-color: #000000;")
        
        self.classification_result = QLabel("CLASSIFICATION: NONE")
        self.classification_result.setFont(QFont("Courier New", 12, QFont.Bold))
        
        self.classification_progress = QProgressBar()
        self.classification_progress.setRange(0, 100)
        self.classification_progress.setValue(0)
        self.classification_progress.setTextVisible(True)
        self.classification_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ffffff;
                text-align: center;
                background-color: #000000;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background-color: #ffffff;
            }
        """)
        
        detection_layout.addWidget(self.object_info)
        detection_layout.addWidget(self.detection_status)
        detection_layout.addWidget(self.first_detection_time)
        detection_layout.addWidget(self.detection_image)
        detection_layout.addWidget(self.classification_result)
        detection_layout.addWidget(self.classification_progress)
        detection_panel.layout.addLayout(detection_layout)
        right_panel.addWidget(detection_panel)
        
        # Location panel with Google Maps
        location_panel = DashboardPanel("CURRENT LOCATION")
        location_layout = QVBoxLayout()
        
        # Coordinates display
        self.coord_label = QLabel("COORDINATES: 37.7749° N, 122.4194° W")
        self.coord_label.setFont(QFont("Courier New", 10, QFont.Bold))
        location_layout.addWidget(self.coord_label)
        
        # Satellite map view
        self.map_view = SatelliteMapWidget()
        location_layout.addWidget(self.map_view)
        
        location_panel.layout.addLayout(location_layout)
        right_panel.addWidget(location_panel)
        
        content.addLayout(right_panel, 0, 1, 2, 1)
        main_layout.addLayout(content)
        
        # Metrics panel (real-time)
        metrics_panel = DashboardPanel("SYSTEM METRICS")
        metrics_layout = QGridLayout()
        metrics_panel.layout.addLayout(metrics_layout)

        self.cpu_label = QLabel("CPU: 0%")
        self.mem_label = QLabel("MEM: 0%")
        self.gpu_label = QLabel("GPU: 0%")
        self.fps_label = QLabel("FPS: 0.0")
        self.latency_label = QLabel("LATENCY: 0 ms")
        self.det_rate_label = QLabel("DETECTION RATE: 0.0 fps")
        self.threats_label = QLabel("THREATS: 0")

        for lab in [self.cpu_label, self.mem_label, self.gpu_label, self.fps_label, self.latency_label, self.det_rate_label, self.threats_label]:
            lab.setFont(QFont("Courier New", 10, QFont.Bold))

        metrics_layout.addWidget(self.cpu_label, 0, 0)
        metrics_layout.addWidget(self.mem_label, 0, 1)
        metrics_layout.addWidget(self.gpu_label, 0, 2)
        metrics_layout.addWidget(self.fps_label, 1, 0)
        metrics_layout.addWidget(self.latency_label, 1, 1)
        metrics_layout.addWidget(self.det_rate_label, 1, 2)
        metrics_layout.addWidget(self.threats_label, 2, 0)

        right_panel.addWidget(metrics_panel)

        # Status bar
        status_bar = QHBoxLayout()
        self.status_label = QLabel("SYSTEM STATUS: ONLINE | LAST UPDATE: " + 
                             datetime.datetime.now().strftime("%H:%M:%S"))
        self.status_label.setFont(QFont("Courier New", 10))
        status_bar.addWidget(self.status_label)
        main_layout.addLayout(status_bar)
        
        # Add pages to stacked widget
        self.stacked_widget.addWidget(self.login_page)
        self.stacked_widget.addWidget(self.main_page)
        
        # Start with login page
        self.stacked_widget.setCurrentIndex(0)
        
        self.main_container.addWidget(self.stacked_widget)
        
        # Initialize video capture and timer
        self.cap = None
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self.update_frame)
        
        # Load object detection model (prefer CUDA)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for UI inference")
        self.torch_device = torch.device('cuda')
        try:
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.torch_device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # First detection flag and image
        self.first_detection = False
        self.first_detection_image = None
        
        # Classification progress simulation
        self.classification_timer = QTimer()
        self.classification_timer.timeout.connect(self.update_classification_progress)
        self.classification_progress_value = 0
        self.classifying = False
        
        # Drone classifier from Program 2
        self.drone_classifier = DroneClassifier().to(self.torch_device)
        self.drone_classifier.eval()

        # Performance metrics state
        self._frame_times = deque(maxlen=60)
        self._det_times = deque(maxlen=300)
        self._threat_count = 0
        self._last_fps = 0.0
        self._last_latency_ms = 0
        self._last_det_rate = 0.0
        self._alert_blink_state = False
        self._metrics_timer = QTimer(self)
        self._metrics_timer.timeout.connect(self.update_metrics)
        self._metrics_timer.start(1000)
        # NVML init (best-effort)
        self._nvml_handle = None
        try:
            if nvml is not None:
                nvml.nvmlInit()
                self._nvml_handle = nvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._nvml_handle = None

    def show_login_dialog(self):
        dialog = LoginDialog(self)
        if dialog.exec_():
            self.user_name = dialog.name_input.text()
            self.user_profile = dialog.profile_input.text()
            if not self.user_name:
                self.user_name = "ANONYMOUS"
            if not self.user_profile:
                self.user_profile = "STANDARD"
                
            self.user_label.setText(f"OPERATOR: {self.user_name} | PROFILE: {self.user_profile}")
            self.stacked_widget.setCurrentIndex(1)  # Switch to main page
            
            # Simulate getting current location
            # In a real app, you would use geolocation services
            self.update_location()
    
    def update_location(self):
        # Simulate random location near default coordinates
        lat = 37.7749 + (random.random() - 0.5) * 0.01
        lng = -122.4194 + (random.random() - 0.5) * 0.01
        
        self.coord_label.setText(f"COORDINATES: {lat:.4f}° N, {lng:.4f}° W")
        self.map_view.set_coordinates(lat, lng)
    
    def update_time(self):
        current_time = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        self.time_label.setText(current_time)
        self.status_label.setText(f"SYSTEM STATUS: ONLINE | LAST UPDATE: {datetime.datetime.now().strftime('%H:%M:%S')}")
        
        # Update location occasionally (every 10 seconds)
        if datetime.datetime.now().second % 10 == 0:
            self.update_location()

    def compute_fps(self):
        if len(self._frame_times) < 2:
            return 0.0
        dt = self._frame_times[-1] - self._frame_times[0]
        if dt <= 0:
            return 0.0
        return (len(self._frame_times) - 1) / dt

    def compute_detection_rate(self, detections_in_frame):
        # prune old timestamps (older than 5s)
        now = time.time()
        while self._det_times and now - self._det_times[0] > 5.0:
            self._det_times.popleft()
        # instantaneous update happens in update_frame; here we just return avg
        if not self._det_times:
            return 0.0
        interval = max(now - self._det_times[0], 1e-3)
        return len(self._det_times) / interval

    def update_metrics(self):
        try:
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            gpu = 0
            try:
                if self._nvml_handle is not None:
                    util = nvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                    gpu = int(util.gpu)
            except Exception:
                gpu = 0
            self.cpu_label.setText(f"CPU: {int(cpu)}%")
            self.mem_label.setText(f"MEM: {int(mem)}%")
            self.gpu_label.setText(f"GPU: {int(gpu)}%")
            self.fps_label.setText(f"FPS: {self._last_fps:.1f}")
            self.latency_label.setText(f"LATENCY: {self._last_latency_ms} ms")
            self.det_rate_label.setText(f"DETECTION RATE: {self._last_det_rate:.1f} fps")
            # Blink threats label if nonzero
            if self._threat_count > 0:
                self._alert_blink_state = not self._alert_blink_state
                color = '#ff0000' if self._alert_blink_state else '#ffffff'
                self.threats_label.setStyleSheet(f"color: {color}; font-weight: bold;")
            self.threats_label.setText(f"THREATS: {self._threat_count}")
        except Exception:
            pass
    
    def load_video(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "Open Video")
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            self.frame_timer.start(30)
            self.reset_detection()

    def open_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.frame_timer.start(30)
        self.reset_detection()
        
    def reset_detection(self):
        self.first_detection = False
        self.first_detection_image = None
        self.object_info.setText("STATUS: AWAITING DETECTION")
        self.detection_status.setText("NO OBJECTS DETECTED")
        self.first_detection_time.setText("FIRST DETECTION: NONE")
        self.classification_result.setText("CLASSIFICATION: NONE")
        self.classification_progress.setValue(0)
        self.classification_progress_value = 0
        self.classifying = False
        self.drone_details_shown = False  # Track if we've shown the details dialog
        
    def update_classification_progress(self):
        if self.classification_progress_value < 100:
            self.classification_progress_value += 5
            self.classification_progress.setValue(self.classification_progress_value)
            if self.classification_progress_value < 50:
                self.classification_result.setText("CLASSIFICATION: THE SYSTEM IS DETECTING...")
            elif self.classification_progress_value < 80:
                self.classification_result.setText("CLASSIFICATION: ANALYZING FEATURES...")
        else:
            self.classification_timer.stop()
            self.classifying = False
            
            # Use the drone classifier to classify the detected object
            if self.first_detection_image is not None:
                # Convert the OpenCV image to PIL format for classification
                pil_image = Image.fromarray(cv2.cvtColor(self.first_detection_image, cv2.COLOR_BGR2RGB))
                
                # Classify using a random selection for demonstration
                # In a real application, you would use the actual classifier here
                classes = ['SURVEILLANCE DRONE', 'SUICIDE DRONE', 'CIVILIAN DRONE', 'CUSTOMIZED DRONE', 'OTHER']
                result = random.choice(classes)
                
                self.classification_result.setText(f"CLASSIFICATION: {result}")
                
                # Set color based on classification
                if result == "SURVEILLANCE DRONE" or result == "SUICIDE DRONE":
                    self.classification_result.setStyleSheet("color: #ff0000; font-weight: bold;")
                    self._threat_count = min(self._threat_count + 1, 99)
                else:
                    self.classification_result.setStyleSheet("color: #ffffff; font-weight: bold;")
                    
                # If classified as OTHER, show the details dialog
                if result == "OTHER" and not hasattr(self, 'drone_details_shown') or not self.drone_details_shown:
                    self.drone_details_shown = True
                    # Use QTimer to show dialog after a short delay
                    QTimer.singleShot(500, self.show_drone_details_dialog)
    
    def show_drone_details_dialog(self):
        dialog = DroneDetailsDialog(self)
        dialog.exec_()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.frame_timer.stop()
            return

        # Convert frame to RGB for PyTorch model
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(rgb_frame).to(self.torch_device)
        
        # Update time on each frame
        current_time = datetime.datetime.now()
        self.time_label.setText(current_time.strftime("%d-%m-%Y %H:%M:%S"))
        
        start_t = time.time()
        # Object detection
        with torch.no_grad():
            predictions = self.model([input_tensor])[0]

        # Check for detections with high confidence
        detection_found = False
        det_this_frame = 0
        for idx, score in enumerate(predictions['scores']):
            if score > 0.7:  # Lower threshold for better detection
                detection_found = True
                box = predictions['boxes'][idx].numpy().astype(int)
                label = predictions['labels'][idx].item()
                
                # Draw bounding box with military-style targeting
                cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 2)
                
                # Add targeting elements
                center_x = (box[0] + box[2]) // 2
                center_y = (box[1] + box[3]) // 2
                
                # Crosshair
                cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 1)
                cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 1)
                
                # Corner brackets
                bracket_len = 15
                # Top-left
                cv2.line(frame, (box[0], box[1]), (box[0] + bracket_len, box[1]), (0, 255, 0), 2)
                cv2.line(frame, (box[0], box[1]), (box[0], box[1] + bracket_len), (0, 255, 0), 2)
                # Top-right
                cv2.line(frame, (box[2], box[1]), (box[2] - bracket_len, box[1]), (0, 255, 0), 2)
                cv2.line(frame, (box[2], box[1]), (box[2], box[1] + bracket_len), (0, 255, 0), 2)
                # Bottom-left
                cv2.line(frame, (box[0], box[3]), (box[0] + bracket_len, box[3]), (0, 255, 0), 2)
                cv2.line(frame, (box[0], box[3]), (box[0], box[3] - bracket_len), (0, 255, 0), 2)
                # Bottom-right
                cv2.line(frame, (box[2], box[3]), (box[2] - bracket_len, box[3]), (0, 255, 0), 2)
                cv2.line(frame, (box[2], box[3]), (box[2], box[3] - bracket_len), (0, 255, 0), 2)
                
                # Extract the cropped object
                if box[1] < box[3] and box[0] < box[2]:  # Ensure valid box dimensions
                    cropped = frame[box[1]:box[3], box[0]:box[2]]
                    
                    if cropped.size > 0:  # Ensure cropped image is not empty
                        # Handle first detection
                        if not self.first_detection:
                            self.first_detection = True
                            self.first_detection_time.setText(f"FIRST DETECTION: {current_time.strftime('%H:%M:%S')}")
                            self.object_info.setText("STATUS: OBJECT DETECTED FOR THE FIRST TIME!")
                            self.object_info.setStyleSheet("color: #ff0000; font-weight: bold;")
                            
                            # Save and display the first detection image
                            self.first_detection_image = cropped.copy()
                            cropped_resized = cv2.resize(cropped, (200, 200))
                            cropped_qt = QImage(cropped_resized.data, cropped_resized.shape[1], 
                                               cropped_resized.shape[0], cropped_resized.strides[0], 
                                               QImage.Format_BGR888)
                            self.detection_image.setPixmap(QPixmap.fromImage(cropped_qt))
                            
                            # Start classification process
                            self.classifying = True
                            self.classification_progress_value = 0
                            self.classification_timer.start(100)
                        else:
                            self.detection_status.setText(f"OBJECT TRACKED AT {current_time.strftime('%H:%M:%S')}")
                        det_this_frame += 1
                break
        
        if not detection_found and not self.first_detection:
            self.object_info.setText("STATUS: SCANNING FOR OBJECTS...")
            self.object_info.setStyleSheet("color: #ffffff;")

        # Add military-style HUD elements to frame
        h, w = frame.shape[:2]
        
        # Add frame border
        cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 255, 0), 1)
        
        # Add corner brackets to frame
        bracket_size = 20
        # Top-left
        cv2.line(frame, (0, 0), (bracket_size, 0), (0, 255, 0), 2)
        cv2.line(frame, (0, 0), (0, bracket_size), (0, 255, 0), 2)
        # Top-right
        cv2.line(frame, (w-1, 0), (w-1-bracket_size, 0), (0, 255, 0), 2)
        cv2.line(frame, (w-1, 0), (w-1, bracket_size), (0, 255, 0), 2)
        # Bottom-left
        cv2.line(frame, (0, h-1), (bracket_size, h-1), (0, 255, 0), 2)
        cv2.line(frame, (0, h-1), (0, h-1-bracket_size), (0, 255, 0), 2)
        # Bottom-right
        cv2.line(frame, (w-1, h-1), (w-1-bracket_size, h-1), (0, 255, 0), 2)
        cv2.line(frame, (w-1, h-1), (w-1, h-1-bracket_size), (0, 255, 0), 2)
        
        # Add timestamp to frame
        timestamp = current_time.strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (w-120, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Overlay HUD metrics
        det_rate = self.compute_detection_rate(det_this_frame)
        fps = self.compute_fps()
        latency_ms = int((time.time() - start_t) * 1000)
        self._last_fps = fps
        self._last_latency_ms = latency_ms
        self._last_det_rate = det_rate
        hud_text = f"FPS:{fps:.1f}  LAT:{latency_ms}ms  DET:{det_rate:.1f}/s"
        cv2.putText(frame, hud_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Resize and display the main frame
        frame = cv2.resize(frame, (640, 480))
        img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_BGR888)
        self.video_label.setPixmap(QPixmap.fromImage(img))

        # Record timing for FPS
        self._frame_times.append(time.time())

        # Mark detections for rate
        if det_this_frame > 0:
            now = time.time()
            for _ in range(det_this_frame):
                self._det_times.append(now)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for better dark theme support
    window = DesktopApp()
    window.show()
    sys.exit(app.exec_())


# Program 2: Drone Image Classifier

import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models

# Dummy classifier for simplicity
class DroneClassifier(nn.Module):
    def __init__(self):
        super(DroneClassifier, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 5)

    def forward(self, x):
        return self.base_model(x)


def classify_image(img_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DroneClassifier().to(device)
    model.eval()  # Assume model is pretrained for actual use

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(img_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    classes = ['Surveillance Drone', 'Suicide Drone', 'Civilian Drone', 'Customized Drone', 'Other']
    return classes[predicted.item()]

# Example usage:
# result = classify_image('drone_image.jpg')
# print("Predicted class:", result)
