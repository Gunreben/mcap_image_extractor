"""
PySide6 GUI for the MCAP Image Extractor.

Provides a modern interface for:
- Adding / removing multiple MCAP bag files
- Discovering and selecting topics (images, GPS, IMU, pointclouds, odom)
- Configuring output format, frame interval, and metadata options
- Optional image rectification with auto-mapped camera intrinsics
- Persistent user preferences (default output dir, calibration path)
- Running extraction with live progress and log output
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem, QLineEdit,
    QFileDialog, QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox,
    QGroupBox, QProgressBar, QTextEdit, QMessageBox, QSplitter,
    QAbstractItemView, QDialog, QDialogButtonBox, QTableWidget,
    QTableWidgetItem, QHeaderView,
)
from PySide6.QtCore import Qt, QThread, Signal, QSettings
from PySide6.QtGui import QFont, QColor

import qt_material

from .extractor import (
    BagSummary, CameraIntrinsics, ExtractionConfig, ExtractionStats,
    McapImageExtractor, TopicInfo, inspect_bag,
    scan_calibration_dir, auto_map_topics_to_calibrations,
)

logger = logging.getLogger(__name__)

# Settings keys
SETTINGS_ORG = 'McapExtractor'
SETTINGS_APP = 'McapImageExtractor'
PREF_CALIBRATION_DIR = 'preferences/calibration_dir'
PREF_OUTPUT_DIR = 'preferences/output_dir'

DEFAULT_CALIBRATION_DIR = (
    '/home/gunreben/ros2_ws/src/tractor_multi_cam_publisher/calibration'
)

# ---------------------------------------------------------------------------
# Category colours and labels for the topic list
# ---------------------------------------------------------------------------

CATEGORY_COLORS = {
    'image':      '#4fc3f7',   # light blue
    'gps':        '#81c784',   # green
    'imu':        '#ffb74d',   # orange
    'pointcloud': '#ce93d8',   # purple
    'odom':       '#fff176',   # yellow
    'other':      '#90a4ae',   # grey
}

CATEGORY_LABELS = {
    'image':      'IMAGE',
    'gps':        'GPS',
    'imu':        'IMU',
    'pointcloud': 'POINTCLOUD',
    'odom':       'ODOM',
    'other':      'OTHER',
}


# ---------------------------------------------------------------------------
# Preferences dialog
# ---------------------------------------------------------------------------

class PreferencesDialog(QDialog):
    """Modal dialog for persistent application preferences."""

    def __init__(self, settings: QSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Preferences')
        self.setMinimumWidth(550)
        self.settings = settings

        lay = QVBoxLayout(self)

        # --- Default calibration directory ---
        calib_group = QGroupBox("Default Calibration Directory")
        calib_lay = QHBoxLayout(calib_group)
        self.calib_edit = QLineEdit()
        self.calib_edit.setText(
            settings.value(PREF_CALIBRATION_DIR, DEFAULT_CALIBRATION_DIR))
        self.calib_edit.setPlaceholderText("Path to *.intrinsics.yaml files")
        calib_lay.addWidget(self.calib_edit, stretch=1)
        calib_browse = QPushButton("Browse")
        calib_browse.clicked.connect(self._browse_calib)
        calib_lay.addWidget(calib_browse)
        lay.addWidget(calib_group)

        # --- Default output directory ---
        out_group = QGroupBox("Default Output Directory")
        out_lay = QHBoxLayout(out_group)
        self.output_edit = QLineEdit()
        self.output_edit.setText(
            settings.value(PREF_OUTPUT_DIR, ''))
        self.output_edit.setPlaceholderText(
            "Leave empty to auto-set from bag location")
        out_lay.addWidget(self.output_edit, stretch=1)
        out_browse = QPushButton("Browse")
        out_browse.clicked.connect(self._browse_output)
        out_lay.addWidget(out_browse)
        lay.addWidget(out_group)

        # --- Buttons ---
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._save_and_accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def _browse_calib(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select Calibration Directory", self.calib_edit.text())
        if d:
            self.calib_edit.setText(d)

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select Default Output Directory", self.output_edit.text())
        if d:
            self.output_edit.setText(d)

    def _save_and_accept(self):
        self.settings.setValue(PREF_CALIBRATION_DIR, self.calib_edit.text())
        self.settings.setValue(PREF_OUTPUT_DIR, self.output_edit.text())
        self.accept()


# ---------------------------------------------------------------------------
# Background threads
# ---------------------------------------------------------------------------

class BagInspectThread(QThread):
    """Inspect MCAP bags in the background to discover topics."""
    result_ready = Signal(dict, dict)  # {bag_path: [TopicInfo, ...]}, {bag_path: duration}
    error_occurred = Signal(str)
    progress = Signal(str)

    def __init__(self, bag_paths: List[str]):
        super().__init__()
        self.bag_paths = bag_paths

    def run(self):
        results: Dict[str, List[TopicInfo]] = {}
        durations: Dict[str, float] = {}
        for i, path in enumerate(self.bag_paths):
            name = Path(path).name
            self.progress.emit(f"Inspecting {name} ({i + 1}/{len(self.bag_paths)})...")
            try:
                summary = inspect_bag(path)
                results[path] = summary.topics
                durations[path] = summary.duration
            except Exception as e:
                self.error_occurred.emit(f"Error inspecting {name}: {e}")
                results[path] = []
                durations[path] = 0.0
        self.result_ready.emit(results, durations)


class ExtractionThread(QThread):
    """Run the extraction pipeline in the background."""
    progress_update = Signal(float, str)   # fraction, message
    log_message = Signal(str)
    finished_signal = Signal(object)       # ExtractionStats
    error_occurred = Signal(str)

    def __init__(self, config: ExtractionConfig):
        super().__init__()
        self.config = config
        self.extractor: Optional[McapImageExtractor] = None

    def run(self):
        try:
            self.extractor = McapImageExtractor(self.config)
            self.extractor.set_progress_callback(self._on_progress)
            stats = self.extractor.extract()
            self.finished_signal.emit(stats)
        except Exception as e:
            self.error_occurred.emit(str(e))

    def _on_progress(self, fraction: float, message: str):
        self.progress_update.emit(fraction, message)
        self.log_message.emit(message)

    def cancel(self):
        if self.extractor:
            self.extractor.cancel()


# ---------------------------------------------------------------------------
# Main GUI
# ---------------------------------------------------------------------------

class McapImageExtractorGUI(QWidget):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle('MCAP Image Extractor — CVAT Data Preparation')
        self.setGeometry(100, 80, 900, 820)
        self.setMinimumSize(750, 650)

        # Persistent settings
        self.settings = QSettings(SETTINGS_ORG, SETTINGS_APP)

        # State
        self.bag_paths: List[str] = []
        self.all_topics: Dict[str, List[TopicInfo]] = {}   # per-bag topics
        self.bag_durations: Dict[str, float] = {}           # per-bag duration (sec)
        self.merged_topics: List[TopicInfo] = []
        self.inspect_thread: Optional[BagInspectThread] = None
        self.extract_thread: Optional[ExtractionThread] = None

        # Calibration state
        self.calibrations: Dict[str, CameraIntrinsics] = {}  # camera_name → intrinsics
        self.calib_mapping: Dict[str, str] = {}               # topic → camera_name

        self._build_ui()
        self._load_preferences()

    # ----- preferences -----

    def _load_preferences(self):
        """Populate fields from saved QSettings."""
        default_output = self.settings.value(PREF_OUTPUT_DIR, '')
        if default_output:
            self.output_dir_edit.setText(default_output)

        calib_dir = self.settings.value(
            PREF_CALIBRATION_DIR, DEFAULT_CALIBRATION_DIR)
        if calib_dir and Path(calib_dir).is_dir():
            self.calibrations = scan_calibration_dir(calib_dir)
            self.calib_dir_label.setText(calib_dir)

    def _open_preferences(self):
        dlg = PreferencesDialog(self.settings, parent=self)
        if dlg.exec() == QDialog.Accepted:
            # Refresh calibration data
            calib_dir = self.settings.value(
                PREF_CALIBRATION_DIR, DEFAULT_CALIBRATION_DIR)
            if calib_dir and Path(calib_dir).is_dir():
                self.calibrations = scan_calibration_dir(calib_dir)
                self.calib_dir_label.setText(calib_dir)
                self._log(f"Loaded {len(self.calibrations)} calibration(s) "
                          f"from {calib_dir}")
            else:
                self.calibrations = {}
                self.calib_dir_label.setText("(not set)")

            # Refresh default output dir
            default_output = self.settings.value(PREF_OUTPUT_DIR, '')
            if default_output and not self.output_dir_edit.text().strip():
                self.output_dir_edit.setText(default_output)

            self._refresh_calibration_mapping()

    # ----- UI construction -----

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # Title row with Preferences button
        title_row = QHBoxLayout()
        title = QLabel("MCAP Image Extractor")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title_row.addStretch()
        title_row.addWidget(title)
        title_row.addStretch()
        self.prefs_btn = QPushButton("  Preferences  ")
        self.prefs_btn.setMinimumHeight(30)
        self.prefs_btn.clicked.connect(self._open_preferences)
        title_row.addWidget(self.prefs_btn)
        root.addLayout(title_row)

        subtitle = QLabel("Extract image frames from ROS 2 bags for CVAT labeling")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #aaa; margin-bottom: 6px;")
        root.addWidget(subtitle)

        # Splitter: top (config) / bottom (log)
        splitter = QSplitter(Qt.Vertical)
        root.addWidget(splitter, stretch=1)

        # --- Top section ---
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(8)

        # Bag files
        top_layout.addWidget(self._build_bag_section())

        # Topics
        top_layout.addWidget(self._build_topic_section(), stretch=1)

        # Output & Settings
        top_layout.addWidget(self._build_output_section())
        top_layout.addWidget(self._build_settings_section())

        # Calibration / rectification
        top_layout.addWidget(self._build_calibration_section())

        # Extraction preview
        top_layout.addWidget(self._build_preview_section())

        splitter.addWidget(top_widget)

        # --- Bottom section (progress + log) ---
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(6)

        # Progress
        prog_group = QGroupBox("Progress")
        prog_lay = QVBoxLayout(prog_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        prog_lay.addWidget(self.progress_bar)
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #aaa;")
        prog_lay.addWidget(self.status_label)
        bottom_layout.addWidget(prog_group)

        # Log
        log_group = QGroupBox("Log")
        log_lay = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Monospace", 9))
        self.log_text.setMaximumHeight(180)
        log_lay.addWidget(self.log_text)
        bottom_layout.addWidget(log_group, stretch=1)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.extract_btn = QPushButton("  Extract  ")
        self.extract_btn.setEnabled(False)
        self.extract_btn.setMinimumHeight(36)
        self.extract_btn.clicked.connect(self._start_extraction)
        self.cancel_btn = QPushButton("  Cancel  ")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setMinimumHeight(36)
        self.cancel_btn.clicked.connect(self._cancel_extraction)
        btn_layout.addWidget(self.extract_btn)
        btn_layout.addWidget(self.cancel_btn)
        bottom_layout.addLayout(btn_layout)

        splitter.addWidget(bottom_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

    # ----- section builders -----

    def _build_bag_section(self) -> QGroupBox:
        group = QGroupBox("Bag Files")
        lay = QHBoxLayout(group)

        self.bag_list = QListWidget()
        self.bag_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.bag_list.setMaximumHeight(90)
        lay.addWidget(self.bag_list, stretch=1)

        btn_col = QVBoxLayout()
        add_btn = QPushButton("Add Bags")
        add_btn.clicked.connect(self._add_bags)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._remove_bags)
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._clear_bags)
        btn_col.addWidget(add_btn)
        btn_col.addWidget(remove_btn)
        btn_col.addWidget(clear_btn)
        btn_col.addStretch()
        lay.addLayout(btn_col)

        return group

    def _build_topic_section(self) -> QGroupBox:
        group = QGroupBox("Topics")
        lay = QVBoxLayout(group)

        # Quick-select buttons
        sel_row = QHBoxLayout()
        self.select_all_images_btn = QPushButton("Select All Images")
        self.select_all_images_btn.clicked.connect(
            lambda: self._select_category('image'))
        self.select_all_gps_btn = QPushButton("Select All GPS")
        self.select_all_gps_btn.clicked.connect(
            lambda: self._select_category('gps'))
        self.select_all_odom_btn = QPushButton("Select All Odom")
        self.select_all_odom_btn.clicked.connect(
            lambda: self._select_category('odom'))
        self.select_none_btn = QPushButton("Deselect All")
        self.select_none_btn.clicked.connect(self._deselect_all)
        sel_row.addWidget(self.select_all_images_btn)
        sel_row.addWidget(self.select_all_gps_btn)
        sel_row.addWidget(self.select_all_odom_btn)
        sel_row.addStretch()
        sel_row.addWidget(self.select_none_btn)
        lay.addLayout(sel_row)

        self.topic_list = QListWidget()
        self.topic_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.topic_list.itemChanged.connect(self._on_topic_changed)
        lay.addWidget(self.topic_list, stretch=1)

        return group

    def _build_output_section(self) -> QGroupBox:
        group = QGroupBox("Output")
        lay = QHBoxLayout(group)

        lay.addWidget(QLabel("Directory:"))
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Select output directory...")
        lay.addWidget(self.output_dir_edit, stretch=1)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_output)
        lay.addWidget(browse_btn)

        return group

    def _build_settings_section(self) -> QGroupBox:
        group = QGroupBox("Extraction Settings")
        grid = QGridLayout(group)

        # Row 0: Frame interval
        grid.addWidget(QLabel("Frame interval (sec):"), 0, 0)
        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.1, 60.0)
        self.interval_spin.setValue(1.0)
        self.interval_spin.setSingleStep(0.5)
        self.interval_spin.setToolTip(
            "Minimum time gap between extracted frames.\n"
            "1.0 = ~1 frame per second, avoids near-duplicate frames.")
        self.interval_spin.valueChanged.connect(lambda _: self._update_preview())
        grid.addWidget(self.interval_spin, 0, 1)

        # Row 0: Image format
        grid.addWidget(QLabel("Image format:"), 0, 2)
        self.format_combo = QComboBox()
        self.format_combo.addItems(['png', 'jpg'])
        self.format_combo.currentTextChanged.connect(self._on_format_changed)
        grid.addWidget(self.format_combo, 0, 3)

        # Row 0: JPEG quality
        self.quality_label = QLabel("JPEG quality:")
        grid.addWidget(self.quality_label, 0, 4)
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setValue(95)
        grid.addWidget(self.quality_spin, 0, 5)
        # Initially hidden (PNG default)
        self.quality_label.setVisible(False)
        self.quality_spin.setVisible(False)

        # Row 1: checkboxes
        self.csv_check = QCheckBox("Generate metadata CSV")
        self.csv_check.setChecked(True)
        self.csv_check.setToolTip(
            "Create metadata.csv with filename, timestamp, GPS, heading, speed")
        grid.addWidget(self.csv_check, 1, 0, 1, 2)

        self.pointcloud_check = QCheckBox("Extract pointclouds (PLY)")
        self.pointcloud_check.setChecked(False)
        self.pointcloud_check.setToolTip(
            "Save selected PointCloud2 topics as binary PLY files")
        grid.addWidget(self.pointcloud_check, 1, 2, 1, 2)

        return group

    def _build_calibration_section(self) -> QGroupBox:
        group = QGroupBox("Image Rectification")
        lay = QVBoxLayout(group)

        # Top row: checkbox + calibration dir info
        top_row = QHBoxLayout()
        self.rectify_check = QCheckBox("Rectify images (undistort)")
        self.rectify_check.setChecked(False)
        self.rectify_check.setToolTip(
            "Apply lens distortion correction using camera intrinsics.\n"
            "Requires calibration YAML files.")
        self.rectify_check.stateChanged.connect(self._on_rectify_changed)
        top_row.addWidget(self.rectify_check)
        top_row.addStretch()
        top_row.addWidget(QLabel("Calibration dir:"))
        self.calib_dir_label = QLabel("(not set)")
        self.calib_dir_label.setStyleSheet("color: #aaa;")
        top_row.addWidget(self.calib_dir_label)
        lay.addLayout(top_row)

        # Mapping table: topic → calibration file
        self.calib_table = QTableWidget(0, 3)
        self.calib_table.setHorizontalHeaderLabels(
            ['Image Topic', 'Calibration File', ''])
        header = self.calib_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.calib_table.verticalHeader().setVisible(False)
        self.calib_table.setMaximumHeight(140)
        self.calib_table.setVisible(False)
        lay.addWidget(self.calib_table)

        return group

    def _build_preview_section(self) -> QGroupBox:
        group = QGroupBox("Extraction Preview")
        lay = QVBoxLayout(group)
        self.preview_label = QLabel("Select image topics and bags to see an estimate.")
        self.preview_label.setWordWrap(True)
        self.preview_label.setStyleSheet("color: #ccc; padding: 4px;")
        lay.addWidget(self.preview_label)
        return group

    # ----- event handlers -----

    def _add_bags(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select MCAP Bag Files",
            "",
            "MCAP files (*.mcap);;All files (*.*)",
        )
        if not paths:
            return

        new_paths = [p for p in paths if p not in self.bag_paths]
        if not new_paths:
            return

        self.bag_paths.extend(new_paths)
        for p in new_paths:
            self.bag_list.addItem(Path(p).name)

        # Auto-set output dir: prefer preferences default, then bag parent
        if not self.output_dir_edit.text():
            default_output = self.settings.value(PREF_OUTPUT_DIR, '')
            if default_output:
                self.output_dir_edit.setText(default_output)
            else:
                parent = str(Path(new_paths[0]).parent / 'extracted')
                self.output_dir_edit.setText(parent)

        # Inspect new bags
        self._inspect_bags(new_paths)

    def _remove_bags(self):
        selected = self.bag_list.selectedItems()
        if not selected:
            return
        for item in selected:
            row = self.bag_list.row(item)
            path = self.bag_paths[row]
            self.bag_paths.pop(row)
            self.bag_list.takeItem(row)
            self.all_topics.pop(path, None)
            self.bag_durations.pop(path, None)
        self._rebuild_topic_list()
        self._update_preview()
        self._refresh_calibration_mapping()

    def _clear_bags(self):
        self.bag_paths.clear()
        self.bag_list.clear()
        self.all_topics.clear()
        self.bag_durations.clear()
        self.merged_topics.clear()
        self.topic_list.clear()
        self.calib_mapping.clear()
        self.extract_btn.setEnabled(False)
        self._update_preview()
        self._refresh_calibration_mapping()

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self.output_dir_edit.setText(d)

    def _on_format_changed(self, fmt: str):
        is_jpeg = fmt in ('jpg', 'jpeg')
        self.quality_label.setVisible(is_jpeg)
        self.quality_spin.setVisible(is_jpeg)

    def _select_category(self, category: str):
        for i in range(self.topic_list.count()):
            item = self.topic_list.item(i)
            cat = item.data(Qt.UserRole)
            if cat == category:
                item.setCheckState(Qt.Checked)

    def _deselect_all(self):
        for i in range(self.topic_list.count()):
            self.topic_list.item(i).setCheckState(Qt.Unchecked)

    def _on_topic_changed(self, item: QListWidgetItem):
        self._update_extract_button()
        self._update_preview()
        self._refresh_calibration_mapping()

    def _update_extract_button(self):
        has_checked = any(
            self.topic_list.item(i).checkState() == Qt.Checked
            for i in range(self.topic_list.count())
        )
        has_output = bool(self.output_dir_edit.text().strip())
        not_running = (self.extract_thread is None
                       or not self.extract_thread.isRunning())
        self.extract_btn.setEnabled(has_checked and has_output and not_running)

    def _update_preview(self):
        """Recompute and display estimated image counts per topic and total."""
        interval = self.interval_spin.value()

        # Gather selected image topics
        selected_image_topics: List[str] = []
        for i in range(self.topic_list.count()):
            item = self.topic_list.item(i)
            if (item.checkState() == Qt.Checked
                    and item.data(Qt.UserRole) == 'image'):
                selected_image_topics.append(item.data(Qt.UserRole + 1))

        if not selected_image_topics:
            self.preview_label.setText(
                "Select image topics and bags to see an estimate.")
            return

        # Estimate per-topic across all bags
        per_topic: Dict[str, int] = {}
        for topic_name in selected_image_topics:
            total_est = 0
            for bag_path, topics in self.all_topics.items():
                duration = self.bag_durations.get(bag_path, 0.0)
                for t in topics:
                    if t.name == topic_name and t.category == 'image':
                        if duration > 0 and t.message_count > 0:
                            max_at_interval = int(duration / interval) + 1
                            total_est += min(t.message_count, max_at_interval)
                        else:
                            # No duration info; fall back to message count
                            total_est += t.message_count
                        break
            per_topic[topic_name] = total_est

        grand_total = sum(per_topic.values())

        # Build display text
        lines = []
        for topic_name, est in per_topic.items():
            lines.append(f"  {topic_name}:  ~{est:,} images")
        lines.append(f"\n  Total:  ~{grand_total:,} images")

        self.preview_label.setText(
            f"Estimated output at {interval:.1f}s interval:\n"
            + "\n".join(lines)
        )

    # ----- calibration / rectification -----

    def _on_rectify_changed(self, state: int):
        enabled = bool(state)
        self.calib_table.setVisible(enabled)
        if enabled:
            self._refresh_calibration_mapping()

    def _refresh_calibration_mapping(self):
        """Re-run auto-mapping and update the calibration table."""
        # Gather currently checked image topics
        image_topics: List[str] = []
        for i in range(self.topic_list.count()):
            item = self.topic_list.item(i)
            if (item.checkState() == Qt.Checked
                    and item.data(Qt.UserRole) == 'image'):
                image_topics.append(item.data(Qt.UserRole + 1))

        # Auto-map
        if self.calibrations:
            auto = auto_map_topics_to_calibrations(
                image_topics, self.calibrations)
            # Merge: keep any manual overrides that are still valid
            new_mapping: Dict[str, str] = {}
            for topic in image_topics:
                if topic in self.calib_mapping and \
                        self.calib_mapping[topic] in self.calibrations:
                    new_mapping[topic] = self.calib_mapping[topic]
                elif topic in auto:
                    new_mapping[topic] = auto[topic]
            self.calib_mapping = new_mapping
        else:
            # Keep only entries whose topics are still selected
            self.calib_mapping = {
                t: c for t, c in self.calib_mapping.items()
                if t in image_topics
            }

        self._rebuild_calib_table(image_topics)

    def _rebuild_calib_table(self, image_topics: List[str]):
        """Populate the calibration mapping table."""
        self.calib_table.setRowCount(len(image_topics))
        for row, topic in enumerate(image_topics):
            # Column 0: topic name (read-only)
            topic_item = QTableWidgetItem(topic)
            topic_item.setFlags(topic_item.flags() & ~Qt.ItemIsEditable)
            self.calib_table.setItem(row, 0, topic_item)

            # Column 1: mapped calibration file
            cam_name = self.calib_mapping.get(topic, '')
            if cam_name and cam_name in self.calibrations:
                calib = self.calibrations[cam_name]
                display = f"{cam_name}  ({calib.distortion_model})"
                calib_item = QTableWidgetItem(display)
                calib_item.setForeground(QColor('#81c784'))
            else:
                calib_item = QTableWidgetItem("— unmapped —")
                calib_item.setForeground(QColor('#ef5350'))
            calib_item.setFlags(calib_item.flags() & ~Qt.ItemIsEditable)
            self.calib_table.setItem(row, 1, calib_item)

            # Column 2: browse button
            browse_btn = QPushButton("...")
            browse_btn.setToolTip("Manually select a calibration YAML file")
            browse_btn.setMaximumWidth(32)
            browse_btn.clicked.connect(
                lambda checked=False, t=topic: self._manual_set_calibration(t))
            self.calib_table.setCellWidget(row, 2, browse_btn)

    def _manual_set_calibration(self, topic: str):
        """Let the user pick a specific intrinsics YAML for a topic."""
        calib_dir = self.settings.value(
            PREF_CALIBRATION_DIR, DEFAULT_CALIBRATION_DIR)
        path, _ = QFileDialog.getOpenFileName(
            self, f"Select Calibration for {topic}",
            calib_dir,
            "YAML files (*.yaml *.yml);;All files (*.*)",
        )
        if not path:
            return

        from .extractor import load_intrinsics_yaml
        intr = load_intrinsics_yaml(path)
        if intr is None or not intr.camera_name:
            QMessageBox.warning(
                self, "Invalid File",
                f"Could not parse camera intrinsics from:\n{path}")
            return

        # Register in calibrations dict and mapping
        self.calibrations[intr.camera_name] = intr
        self.calib_mapping[topic] = intr.camera_name
        self._log(f"Mapped {topic} → {intr.camera_name} "
                  f"({intr.distortion_model}) from {Path(path).name}")
        self._refresh_calibration_mapping()

    # ----- bag inspection -----

    def _inspect_bags(self, paths: List[str]):
        self._set_controls_enabled(False)
        self.status_label.setText("Inspecting bags...")
        self.inspect_thread = BagInspectThread(paths)
        self.inspect_thread.result_ready.connect(self._on_inspect_done)
        self.inspect_thread.error_occurred.connect(self._on_inspect_error)
        self.inspect_thread.progress.connect(
            lambda m: self.status_label.setText(m))
        self.inspect_thread.finished.connect(
            lambda: self._set_controls_enabled(True))
        self.inspect_thread.start()

    def _on_inspect_done(self, results: Dict[str, List[TopicInfo]],
                         durations: Dict[str, float]):
        self.all_topics.update(results)
        self.bag_durations.update(durations)
        for path, topics in results.items():
            name = Path(path).name
            dur = durations.get(path, 0.0)
            count = sum(t.message_count for t in topics)
            self._log(f"Loaded {name}: {len(topics)} topics, "
                      f"{count:,} messages, {dur:.1f}s duration")
        self._rebuild_topic_list()
        self._update_preview()
        self._refresh_calibration_mapping()
        self.status_label.setText("Ready")

    def _on_inspect_error(self, msg: str):
        self._log(f"ERROR: {msg}")
        QMessageBox.warning(self, "Inspection Error", msg)

    def _rebuild_topic_list(self):
        """Merge topics from all loaded bags and rebuild the list widget."""
        # Block signals during rebuild
        self.topic_list.blockSignals(True)

        # Remember which topics were checked
        previously_checked: Set[str] = set()
        for i in range(self.topic_list.count()):
            item = self.topic_list.item(i)
            if item.checkState() == Qt.Checked:
                previously_checked.add(item.data(Qt.UserRole + 1))

        self.topic_list.clear()

        # Merge: keep the first occurrence, sum message counts
        seen: Dict[str, TopicInfo] = {}
        for topics in self.all_topics.values():
            for t in topics:
                if t.name in seen:
                    seen[t.name] = TopicInfo(
                        name=t.name,
                        type=t.type,
                        message_count=seen[t.name].message_count + t.message_count,
                        category=t.category,
                    )
                else:
                    seen[t.name] = TopicInfo(
                        name=t.name,
                        type=t.type,
                        message_count=t.message_count,
                        category=t.category,
                    )

        self.merged_topics = sorted(seen.values(), key=lambda t: (
            {'image': 0, 'gps': 1, 'odom': 2, 'imu': 3,
             'pointcloud': 4, 'other': 5}.get(t.category, 9),
            t.name,
        ))

        for t in self.merged_topics:
            cat_label = CATEGORY_LABELS.get(t.category, 'OTHER')
            cat_color = CATEGORY_COLORS.get(t.category, '#90a4ae')
            display = (f"[{cat_label}]  {t.name}    "
                       f"({t.type}, {t.message_count:,} msgs)")

            item = QListWidgetItem(display)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            # Restore check state or default-check image topics
            if t.name in previously_checked:
                item.setCheckState(Qt.Checked)
            elif not previously_checked and t.category == 'image':
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            item.setForeground(QColor(cat_color))
            item.setData(Qt.UserRole, t.category)
            item.setData(Qt.UserRole + 1, t.name)    # topic name
            item.setData(Qt.UserRole + 2, t.type)    # msg type
            self.topic_list.addItem(item)

        self.topic_list.blockSignals(False)
        self._update_extract_button()

    # ----- extraction -----

    def _start_extraction(self):
        output_dir = self.output_dir_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "Error", "Please select an output directory.")
            return

        # Collect selected topics by category
        image_topics: List[str] = []
        gps_topics: List[str] = []
        imu_topics: List[str] = []
        pc_topics: List[str] = []
        odom_topics: List[str] = []

        for i in range(self.topic_list.count()):
            item = self.topic_list.item(i)
            if item.checkState() != Qt.Checked:
                continue
            cat = item.data(Qt.UserRole)
            name = item.data(Qt.UserRole + 1)
            if cat == 'image':
                image_topics.append(name)
            elif cat == 'gps':
                gps_topics.append(name)
            elif cat == 'imu':
                imu_topics.append(name)
            elif cat == 'pointcloud':
                pc_topics.append(name)
            elif cat == 'odom':
                odom_topics.append(name)

        if not image_topics and not pc_topics:
            QMessageBox.warning(
                self, "Error",
                "Please select at least one image or pointcloud topic.")
            return

        # Build calibration map for extraction
        do_rectify = self.rectify_check.isChecked()
        calibration_map: Dict[str, CameraIntrinsics] = {}
        if do_rectify:
            unmapped = []
            for topic in image_topics:
                cam_name = self.calib_mapping.get(topic)
                if cam_name and cam_name in self.calibrations:
                    calibration_map[topic] = self.calibrations[cam_name]
                else:
                    unmapped.append(topic)
            if unmapped:
                reply = QMessageBox.question(
                    self, "Unmapped Topics",
                    f"The following image topics have no calibration mapping "
                    f"and will NOT be rectified:\n\n"
                    + "\n".join(f"  • {t}" for t in unmapped)
                    + "\n\nContinue anyway?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
                )
                if reply == QMessageBox.No:
                    return

        config = ExtractionConfig(
            bag_paths=list(self.bag_paths),
            output_dir=output_dir,
            image_topics=image_topics,
            gps_topics=gps_topics,
            imu_topics=imu_topics,
            pointcloud_topics=pc_topics,
            odom_topics=odom_topics,
            frame_interval=self.interval_spin.value(),
            image_format=self.format_combo.currentText(),
            jpeg_quality=self.quality_spin.value(),
            generate_metadata_csv=self.csv_check.isChecked(),
            extract_pointclouds=self.pointcloud_check.isChecked(),
            rectify=do_rectify,
            calibration_map=calibration_map,
        )

        self._log("\n" + "=" * 60)
        self._log("Starting extraction")
        self._log(f"  Bags: {len(config.bag_paths)}")
        self._log(f"  Image topics: {len(image_topics)}")
        self._log(f"  GPS topics: {len(gps_topics)}")
        self._log(f"  Odom topics: {len(odom_topics)}")
        self._log(f"  Pointcloud topics: {len(pc_topics)}")
        self._log(f"  Frame interval: {config.frame_interval:.1f}s")
        self._log(f"  Format: {config.image_format.upper()}")
        self._log(f"  Rectify: {do_rectify}"
                  + (f" ({len(calibration_map)} mapped)" if do_rectify else ""))
        self._log(f"  Output: {output_dir}")
        self._log("=" * 60)

        self._set_controls_enabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)

        self.extract_thread = ExtractionThread(config)
        self.extract_thread.progress_update.connect(self._on_extract_progress)
        self.extract_thread.log_message.connect(self._log)
        self.extract_thread.finished_signal.connect(self._on_extract_done)
        self.extract_thread.error_occurred.connect(self._on_extract_error)
        self.extract_thread.finished.connect(self._on_extract_thread_finished)
        self.extract_thread.start()

    def _cancel_extraction(self):
        if self.extract_thread and self.extract_thread.isRunning():
            self._log("Cancellation requested...")
            self.extract_thread.cancel()
            self.cancel_btn.setEnabled(False)

    def _on_extract_progress(self, fraction: float, message: str):
        self.progress_bar.setValue(int(fraction * 1000))
        self.status_label.setText(message)

    def _on_extract_done(self, stats: ExtractionStats):
        self._log("\n" + "=" * 60)
        self._log("EXTRACTION COMPLETE")
        self._log(f"  Bags processed:      {stats.bags_processed}")
        self._log(f"  Images extracted:     {stats.images_extracted:,}")
        self._log(f"  Images skipped:       {stats.images_skipped:,}")
        self._log(f"  Pointclouds saved:    {stats.pointclouds_extracted:,}")
        self._log(f"  GPS messages read:    {stats.gps_messages:,}")
        self._log(f"  Total messages:       {stats.total_messages:,}")
        if stats.errors:
            self._log(f"  Errors: {len(stats.errors)}")
            for err in stats.errors[:10]:
                self._log(f"    - {err}")
        self._log("=" * 60)

        self.progress_bar.setValue(1000)
        self.status_label.setText("Extraction complete!")

        summary = (
            f"Extraction complete!\n\n"
            f"Images extracted: {stats.images_extracted:,}\n"
            f"Images skipped: {stats.images_skipped:,}\n"
            f"Pointclouds: {stats.pointclouds_extracted:,}\n"
            f"GPS readings: {stats.gps_messages:,}"
        )
        if stats.errors:
            summary += f"\n\nErrors: {len(stats.errors)}"
            QMessageBox.warning(self, "Extraction Complete (with errors)",
                                summary)
        else:
            QMessageBox.information(self, "Extraction Complete", summary)

    def _on_extract_error(self, error_msg: str):
        self._log(f"EXTRACTION ERROR: {error_msg}")
        QMessageBox.critical(self, "Extraction Error", error_msg)

    def _on_extract_thread_finished(self):
        self.extract_thread = None
        self._set_controls_enabled(True)
        self.cancel_btn.setEnabled(False)
        self._update_extract_button()

    # ----- utilities -----

    def _set_controls_enabled(self, enabled: bool):
        """Enable or disable interactive controls during operations."""
        self.bag_list.setEnabled(enabled)
        self.topic_list.setEnabled(enabled)
        self.output_dir_edit.setEnabled(enabled)
        self.interval_spin.setEnabled(enabled)
        self.format_combo.setEnabled(enabled)
        self.quality_spin.setEnabled(enabled)
        self.csv_check.setEnabled(enabled)
        self.pointcloud_check.setEnabled(enabled)
        self.rectify_check.setEnabled(enabled)
        self.calib_table.setEnabled(enabled)
        # Buttons in bag section
        for btn in self.findChildren(QPushButton):
            if btn not in (self.cancel_btn,):
                btn.setEnabled(enabled)
        if enabled:
            self._update_extract_button()

    def _log(self, message: str):
        self.log_text.append(message)
        # Auto-scroll
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def closeEvent(self, event):
        if self.extract_thread and self.extract_thread.isRunning():
            reply = QMessageBox.question(
                self, 'Confirm Exit',
                "Extraction is in progress. Exit anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.extract_thread.cancel()
                self.extract_thread.wait(5000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)

    extra = {
        'primaryTextColor': '#ffffff',
        'secondaryTextColor': '#ffffff',
    }
    qt_material.apply_stylesheet(app, theme='dark_blue.xml', extra=extra)

    window = McapImageExtractorGUI()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
