from __future__ import annotations

import os
from datetime import datetime
import json
from typing import Dict, List, Optional

import cv2
from PySide6 import QtCore, QtGui, QtWidgets

from .extractor import ExtractParams, extract_highres_frame, extract_pages
from .models import PageItem, RawFrame


class ExtractWorker(QtCore.QObject):
    progress = QtCore.Signal(int, str)
    item_ready = QtCore.Signal(object)
    finished = QtCore.Signal(bool, str, str, object)

    def __init__(self, video_path: str, params: ExtractParams, output_dir: str):
        super().__init__()
        self._video_path = video_path
        self._params = params
        self._output_dir = output_dir
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        def _progress(percent: int, message: str) -> None:
            self.progress.emit(int(percent), message)

        def _should_stop() -> bool:
            return self._stop

        try:
            output_dir, _results, raw_frames = extract_pages(
                self._video_path,
                self._params,
                output_dir=self._output_dir,
                progress_cb=_progress,
                should_stop=_should_stop,
                on_item=lambda item: self.item_ready.emit(item),
            )
            if self._stop:
                self.finished.emit(False, "Canceled", output_dir, raw_frames)
            else:
                self.finished.emit(True, "Done", output_dir, raw_frames)
        except Exception as exc:
            self.finished.emit(False, f"Error: {exc}", self._output_dir, [])


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Speedread Extractor")
        self.resize(1200, 720)
        self.setAcceptDrops(True)

        self._settings = QtCore.QSettings("speedread", "speedread")
        self._video_path: Optional[str] = None
        self._output_dir: Optional[str] = None
        self._output_root: Optional[str] = self._load_output_root()
        self._extract_thread: Optional[QtCore.QThread] = None
        self._extract_worker: Optional[ExtractWorker] = None
        self._raw_frames: List[RawFrame] = []
        self._preview_time_s = 1.0
        self._page_item_map: Dict[int, QtWidgets.QListWidgetItem] = {}

        self._preset_map = self._build_presets()

        self._build_ui()

    def _build_presets(self) -> Dict[str, ExtractParams]:
        return {
            "UltraFast": ExtractParams(
                analysis_interval_s=3.0,
                analysis_long_side=0,
                max_interval_s=5.0,
                llm_max_tokens=256,
            ),
            "Fast": ExtractParams(
                analysis_interval_s=2.0,
                analysis_long_side=0,
                max_interval_s=5.0,
                llm_max_tokens=384,
            ),
            "Balanced": ExtractParams(
                analysis_interval_s=1.0,
                analysis_long_side=0,
                max_interval_s=5.0,
                llm_max_tokens=512,
            ),
            "Robust": ExtractParams(
                analysis_interval_s=0.5,
                analysis_long_side=0,
                max_interval_s=5.0,
                llm_max_tokens=1024,
            ),
        }

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        main_layout = QtWidgets.QHBoxLayout(root)

        left_panel = self._build_left_panel()
        center_panel = self._build_center_panel()
        right_panel = self._build_right_panel()

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(left_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 3)
        base_width = max(1, self.width())
        splitter.setSizes(
            [
                int(base_width * 0.2),
                int(base_width * 0.3),
                int(base_width * 0.5),
            ]
        )
        main_layout.addWidget(splitter)

    def _build_left_panel(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        input_group = QtWidgets.QGroupBox("Input")
        input_layout = QtWidgets.QFormLayout(input_group)

        path_layout = QtWidgets.QHBoxLayout()
        self.video_path_edit = QtWidgets.QLineEdit()
        self.video_path_edit.setReadOnly(True)
        browse_btn = QtWidgets.QPushButton("Select video")
        browse_btn.clicked.connect(self._browse_video)
        path_layout.addWidget(self.video_path_edit, 1)
        path_layout.addWidget(browse_btn)
        input_layout.addRow("Input video", path_layout)

        layout.addWidget(input_group)

        output_group = QtWidgets.QGroupBox("Output")
        output_layout = QtWidgets.QVBoxLayout(output_group)

        output_root_layout = QtWidgets.QHBoxLayout()
        self.output_root_label = QtWidgets.QLabel()
        self._refresh_output_root_label()
        self.output_root_btn = QtWidgets.QPushButton("Set output root")
        self.output_root_btn.clicked.connect(self._select_output_root)
        output_root_layout.addWidget(self.output_root_label, 1)
        output_root_layout.addWidget(self.output_root_btn)
        output_layout.addLayout(output_root_layout)

        resume_layout = QtWidgets.QHBoxLayout()
        self.resume_btn = QtWidgets.QPushButton("Resume (select folder)")
        self.resume_btn.clicked.connect(self._resume_extract)
        resume_layout.addStretch()
        resume_layout.addWidget(self.resume_btn)
        output_layout.addLayout(resume_layout)

        layout.addWidget(output_group)

        options_group = QtWidgets.QGroupBox("Options")
        options_layout = QtWidgets.QFormLayout(options_group)
        self.preset_combo = QtWidgets.QComboBox()
        for name in self._preset_map.keys():
            self.preset_combo.addItem(name)
        self.preset_combo.setCurrentText("Balanced")
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        options_layout.addRow("Preset", self.preset_combo)

        self.analysis_interval_spin = QtWidgets.QDoubleSpinBox()
        self.analysis_interval_spin.setRange(0.1, 10.0)
        self.analysis_interval_spin.setSingleStep(0.1)
        self.analysis_long_side_combo = QtWidgets.QComboBox()
        self.analysis_long_side_combo.addItem("No resize", 0)
        self.analysis_long_side_combo.addItem("1920", 1920)
        self.analysis_long_side_combo.addItem("1280", 1280)
        self.analysis_long_side_combo.addItem("720", 720)
        self.max_interval_spin = QtWidgets.QDoubleSpinBox()
        self.max_interval_spin.setRange(0.1, 10.0)
        self.max_interval_spin.setSingleStep(0.1)

        self.rotation_combo = QtWidgets.QComboBox()
        self.rotation_combo.addItem("0 deg", 0)
        self.rotation_combo.addItem("90 deg", 90)
        self.rotation_combo.addItem("180 deg", 180)
        self.rotation_combo.addItem("270 deg", 270)
        self.rotation_combo.currentIndexChanged.connect(self._refresh_preview)

        self.llm_url_edit = QtWidgets.QLineEdit("http://127.0.0.1:1234")
        self.llm_model_edit = QtWidgets.QLineEdit("qwen/qwen3-vl-8b")
        self.llm_prompt_combo = QtWidgets.QComboBox()
        self.llm_prompt_combo.addItem("General", "general")
        self.llm_prompt_combo.addItem("Japanese", "ja")
        self.llm_prompt_combo.addItem("Japanese (vertical)", "ja_vert")
        self.llm_max_tokens_spin = QtWidgets.QSpinBox()
        self.llm_max_tokens_spin.setRange(64, 4096)
        self.llm_max_tokens_spin.setSingleStep(64)

        options_layout.addRow("Base interval (s)", self.analysis_interval_spin)
        options_layout.addRow("Max interval (s)", self.max_interval_spin)
        options_layout.addRow("Rotation", self.rotation_combo)
        options_layout.addRow("Analysis long side", self.analysis_long_side_combo)
        self.llm_split_check = QtWidgets.QCheckBox("(vertical only; 4 parts for LLM)")
        self.llm_split_check.setChecked(True)
        options_layout.addRow("Split pages", self.llm_split_check)
        options_layout.addRow("LLM base URL", self.llm_url_edit)
        options_layout.addRow("LLM model", self.llm_model_edit)
        options_layout.addRow("Prompt", self.llm_prompt_combo)
        options_layout.addRow("Max tokens", self.llm_max_tokens_spin)

        layout.addWidget(options_group)

        preview_group = QtWidgets.QGroupBox("Preview")
        preview_layout = QtWidgets.QVBoxLayout(preview_group)
        self.preview_label = QtWidgets.QLabel("No preview")
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setMinimumSize(200, 260)
        self.preview_label.setFrameShape(QtWidgets.QFrame.Box)
        self.preview_label.setFrameShadow(QtWidgets.QFrame.Sunken)
        preview_layout.addWidget(self.preview_label)
        layout.addWidget(preview_group)

        actions_group = QtWidgets.QGroupBox("Actions")
        actions_layout = QtWidgets.QVBoxLayout(actions_group)
        self.extract_btn = QtWidgets.QPushButton("Extract pages")
        self.extract_btn.clicked.connect(self._start_extract)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_extract)
        self.stop_btn.setEnabled(False)
        self.open_folder_btn = QtWidgets.QPushButton("Open output folder")
        self.open_folder_btn.clicked.connect(self._open_output_folder)
        self.open_folder_btn.setEnabled(False)

        actions_layout.addWidget(self.extract_btn)
        actions_layout.addWidget(self.stop_btn)
        actions_layout.addWidget(self.open_folder_btn)
        actions_layout.addStretch()
        layout.addWidget(actions_group)

        layout.addStretch()

        self._apply_preset(self.preset_combo.currentText())
        return widget

    def _build_center_panel(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        status_group = QtWidgets.QGroupBox("Progress")
        status_layout = QtWidgets.QVBoxLayout(status_group)
        self.status_label = QtWidgets.QLabel("Idle")
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar)

        self.log_view = QtWidgets.QTextEdit()
        self.log_view.setReadOnly(True)
        status_layout.addWidget(self.log_view, 1)

        layout.addWidget(status_group)
        return widget

    def _build_right_panel(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        header_layout = QtWidgets.QHBoxLayout()
        self.page_count_label = QtWidgets.QLabel("Pages: 0")
        header_layout.addWidget(self.page_count_label)
        header_layout.addStretch()

        self.thumb_container = QtWidgets.QWidget()
        thumb_layout = QtWidgets.QVBoxLayout(self.thumb_container)
        thumb_layout.setContentsMargins(0, 0, 0, 0)
        thumb_layout.setSpacing(6)

        self.thumb_list = QtWidgets.QListWidget()
        self.thumb_list.setViewMode(QtWidgets.QListView.ListMode)
        self.thumb_list.setIconSize(QtCore.QSize(72, 100))
        self.thumb_list.setResizeMode(QtWidgets.QListView.Adjust)
        self.thumb_list.setDragDropMode(QtWidgets.QAbstractItemView.NoDragDrop)
        self.thumb_list.setDragEnabled(False)
        self.thumb_list.setAcceptDrops(False)
        self.thumb_list.setDefaultDropAction(QtCore.Qt.IgnoreAction)
        self.thumb_list.setMovement(QtWidgets.QListView.Static)
        self.thumb_list.setWrapping(False)
        self.thumb_list.setUniformItemSizes(True)
        self.thumb_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.thumb_list.setSpacing(6)
        self.thumb_list.itemSelectionChanged.connect(self._update_transcript_view)

        thumb_layout.addLayout(header_layout)
        thumb_layout.addWidget(self.thumb_list, 1)

        transcript_group = QtWidgets.QGroupBox("Transcription")
        transcript_layout = QtWidgets.QVBoxLayout(transcript_group)
        self.transcript_view = QtWidgets.QTextEdit()
        self.transcript_view.setReadOnly(True)
        transcript_layout.addWidget(self.transcript_view)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter.addWidget(self.thumb_container)
        splitter.addWidget(transcript_group)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter, 1)
        return widget

    def _apply_preset(self, name: str) -> None:
        params = self._preset_map.get(name)
        if not params:
            return
        self.analysis_interval_spin.setValue(params.analysis_interval_s)
        self._set_analysis_long_side(params.analysis_long_side)
        self.max_interval_spin.setValue(params.max_interval_s)
        self.llm_max_tokens_spin.setValue(params.llm_max_tokens)

    def _collect_params(self) -> ExtractParams:
        rotation_value = self.rotation_combo.currentData()
        if rotation_value is None:
            rotation_value = 0
        llm_url = self.llm_url_edit.text().strip() or "http://127.0.0.1:1234"
        llm_model = self.llm_model_edit.text().strip() or "qwen/qwen3-vl-8b"
        prompt_key = self.llm_prompt_combo.currentData()
        if not prompt_key:
            prompt_key = "general"
        return ExtractParams(
            analysis_interval_s=self.analysis_interval_spin.value(),
            analysis_long_side=int(self.analysis_long_side_combo.currentData() or 0),
            rotation_degrees=int(rotation_value),
            max_interval_s=self.max_interval_spin.value(),
            llm_base_url=llm_url,
            llm_model=llm_model,
            llm_prompt_key=str(prompt_key),
            llm_split_4=self.llm_split_check.isChecked(),
            llm_max_tokens=self.llm_max_tokens_spin.value(),
        )

    def _set_analysis_long_side(self, value: int) -> None:
        for i in range(self.analysis_long_side_combo.count()):
            if int(self.analysis_long_side_combo.itemData(i)) == int(value):
                self.analysis_long_side_combo.setCurrentIndex(i)
                return
        self.analysis_long_side_combo.setCurrentIndex(0)

    def _browse_video(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select video", "", "Video Files (*.mp4 *.mov *.mkv *.avi)"
        )
        if path:
            self._set_video(path)

    def _set_video(self, path: str) -> None:
        self._video_path = path
        self.video_path_edit.setText(path)
        self._log(f"Loaded video: {path}")
        self._refresh_preview()

    def _load_output_root(self) -> Optional[str]:
        value = self._settings.value("output_root", "", str)
        return value or None

    def _save_output_root(self, path: str) -> None:
        self._settings.setValue("output_root", path)
        self._output_root = path
        self._refresh_output_root_label()

    def _refresh_output_root_label(self) -> None:
        if self._output_root:
            self.output_root_label.setText(f"Output root: {self._output_root}")
        else:
            self.output_root_label.setText("Output root: (not set)")

    def _select_output_root(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output root", self._output_root or ""
        )
        if not path:
            return
        self._save_output_root(path)

    def _ensure_output_root(self) -> bool:
        if self._output_root and os.path.isdir(self._output_root):
            return True
        self._select_output_root()
        return bool(self._output_root and os.path.isdir(self._output_root))

    def _create_output_dir(self) -> Optional[str]:
        if not self._output_root:
            return None
        base = os.path.splitext(os.path.basename(self._video_path or "video"))[0]
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"{stamp}-{base}"
        candidate = os.path.join(self._output_root, name)
        if not os.path.exists(candidate):
            os.makedirs(candidate, exist_ok=False)
            return candidate
        for i in range(1, 100):
            alt = f"{candidate}-{i}"
            if not os.path.exists(alt):
                os.makedirs(alt, exist_ok=False)
                return alt
        return None

    def _write_session(self, output_dir: str, params: ExtractParams) -> None:
        data = {
            "video_path": self._video_path,
            "base_interval_s": params.analysis_interval_s,
        }
        path = os.path.join(output_dir, "session.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_session(self, output_dir: str) -> Optional[Dict[str, object]]:
        path = os.path.join(output_dir, "session.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _is_output_dir_incomplete(self, path: str) -> bool:
        if not os.path.isdir(path):
            return False
        if os.path.exists(os.path.join(path, "final_summary.txt")):
            return False
        pages_dir = os.path.join(path, "pages")
        if os.path.exists(os.path.join(path, "pages_raw.json")):
            return True
        return os.path.isdir(pages_dir) and bool(os.listdir(pages_dir))

    def _start_extract(self) -> None:
        if not self._video_path:
            QtWidgets.QMessageBox.warning(self, "No video", "Please select a video file")
            return
        if self._extract_worker:
            QtWidgets.QMessageBox.warning(self, "Busy", "Extraction is already running")
            return

        if not self._ensure_output_root():
            return

        output_dir: Optional[str] = None
        if self._output_dir and self._is_output_dir_incomplete(self._output_dir):
            session = self._load_session(self._output_dir)
            session_video = session.get("video_path") if isinstance(session, dict) else None
            if session_video and os.path.normpath(str(session_video)) == os.path.normpath(
                self._video_path
            ):
                output_dir = self._output_dir
                base_interval = (
                    session.get("base_interval_s") if isinstance(session, dict) else None
                )
                if base_interval is None and isinstance(session, dict):
                    analysis_fps = session.get("analysis_fps")
                    if analysis_fps:
                        base_interval = 1.0 / float(analysis_fps)
                if base_interval is not None:
                    try:
                        self.analysis_interval_spin.setValue(float(base_interval))
                    except (TypeError, ValueError, ZeroDivisionError):
                        pass
        if output_dir is None:
            output_dir = self._create_output_dir()
            if not output_dir:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Output folder unavailable",
                    "Failed to create a new output folder.",
                )
                return

        self._reset_results()
        self._output_dir = output_dir
        if output_dir == self._output_dir and os.path.exists(
            os.path.join(output_dir, "session.json")
        ):
            self._log(f"Resuming output folder: {output_dir}")
        else:
            self._log(f"Output folder: {output_dir}")

        params = self._collect_params()
        if not os.path.exists(os.path.join(output_dir, "session.json")):
            self._write_session(output_dir, params)
        self._extract_thread = QtCore.QThread()
        self._extract_worker = ExtractWorker(self._video_path, params, self._output_dir)
        self._extract_worker.moveToThread(self._extract_thread)

        self._extract_thread.started.connect(self._extract_worker.run)
        self._extract_worker.progress.connect(self._on_progress)
        self._extract_worker.item_ready.connect(self._on_item_ready)
        self._extract_worker.finished.connect(self._on_extract_finished)
        self._extract_thread.start()

        self.extract_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.open_folder_btn.setEnabled(True)

    def _stop_extract(self) -> None:
        if self._extract_worker:
            self._extract_worker.stop()
            self._log("Stopping extraction...")

    def _resume_extract(self) -> None:
        if self._extract_worker:
            QtWidgets.QMessageBox.warning(self, "Busy", "Extraction is already running")
            return
        start_dir = self._output_root or ""
        output_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output folder", start_dir
        )
        if not output_dir:
            return
        session = self._load_session(output_dir)
        if not session:
            QtWidgets.QMessageBox.warning(
                self, "Resume", "session.json not found in the selected folder."
            )
            return
        video_path = session.get("video_path") if isinstance(session, dict) else None
        if not video_path or not os.path.exists(str(video_path)):
            QtWidgets.QMessageBox.warning(
                self, "Resume", "Original video path not found."
            )
            return
        self._set_video(str(video_path))
        base_interval = session.get("base_interval_s") if isinstance(session, dict) else None
        if base_interval is None and isinstance(session, dict):
            analysis_fps = session.get("analysis_fps")
            if analysis_fps:
                base_interval = 1.0 / float(analysis_fps)
        if base_interval is not None:
            try:
                self.analysis_interval_spin.setValue(float(base_interval))
            except (TypeError, ValueError, ZeroDivisionError):
                pass

        self._reset_results()
        self._output_dir = output_dir
        self._log(f"Resuming output folder: {output_dir}")

        params = self._collect_params()
        self._extract_thread = QtCore.QThread()
        self._extract_worker = ExtractWorker(self._video_path, params, self._output_dir)
        self._extract_worker.moveToThread(self._extract_thread)

        self._extract_thread.started.connect(self._extract_worker.run)
        self._extract_worker.progress.connect(self._on_progress)
        self._extract_worker.item_ready.connect(self._on_item_ready)
        self._extract_worker.finished.connect(self._on_extract_finished)
        self._extract_thread.start()

        self.extract_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.open_folder_btn.setEnabled(True)

    def _on_progress(self, percent: int, message: str) -> None:
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)
        self._log(message)

    def _on_item_ready(self, item: PageItem) -> None:
        self._add_page_item(item)

    def _on_extract_finished(
        self, success: bool, message: str, output_dir: str, raw_frames: object
    ) -> None:
        self.progress_bar.setValue(100 if success else self.progress_bar.value())
        self.status_label.setText(message)
        self._log(message)

        if self._extract_thread:
            self._extract_thread.quit()
            self._extract_thread.wait()
        self._extract_thread = None
        self._extract_worker = None

        self._raw_frames = raw_frames if isinstance(raw_frames, list) else []
        self.extract_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._output_dir = output_dir
        self.open_folder_btn.setEnabled(self._output_dir is not None)

    def _add_page_item(self, item: PageItem) -> None:
        pixmap = QtGui.QPixmap(item.image_path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(
                self.thumb_list.iconSize(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )

        existing = self._page_item_map.get(item.analysis_index)
        if existing is not None:
            if not pixmap.isNull():
                existing.setIcon(QtGui.QIcon(pixmap))
            existing.setData(QtCore.Qt.UserRole, item)
            self._apply_page_item_style(existing, item)
            self._refresh_item_labels()
            if existing.isSelected():
                self._update_transcript_view()
            return

        list_item = QtWidgets.QListWidgetItem()
        list_item.setIcon(QtGui.QIcon(pixmap))
        list_item.setData(QtCore.Qt.UserRole, item)
        self._apply_page_item_style(list_item, item)
        self.thumb_list.addItem(list_item)
        self._page_item_map[item.analysis_index] = list_item
        self._refresh_item_labels()

    def _update_page_count(self) -> None:
        self.page_count_label.setText(f"Pages: {self.thumb_list.count()}")

    def _apply_page_item_style(self, list_item: QtWidgets.QListWidgetItem, item: PageItem) -> None:
        if item.is_selected is False:
            list_item.setForeground(QtGui.QColor(140, 140, 140))
        else:
            list_item.setForeground(self.thumb_list.palette().brush(QtGui.QPalette.Text))

    def _reset_results(self) -> None:
        self.thumb_list.clear()
        self._update_page_count()
        self.progress_bar.setValue(0)
        self.status_label.setText("Idle")
        self.log_view.clear()
        self._raw_frames = []
        self.transcript_view.clear()
        self._refresh_preview()
        self._output_dir = None
        self._page_item_map = {}
        if hasattr(self, "open_folder_btn"):
            self.open_folder_btn.setEnabled(False)

    def _refresh_preview(self, *args: object) -> None:
        if not self._video_path:
            self.preview_label.setText("No preview")
            self.preview_label.setPixmap(QtGui.QPixmap())
            return

        rotation_value = self.rotation_combo.currentData()
        if rotation_value is None:
            rotation_value = 0

        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            self.preview_label.setText("Preview unavailable")
            self.preview_label.setPixmap(QtGui.QPixmap())
            return

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_s = frame_count / fps if fps > 0 else 0.0
            preview_time = self._preview_time_s
            if duration_s > 0 and preview_time > duration_s:
                preview_time = max(0.0, duration_s / 2.0)

            if frame_count > 0:
                if duration_s > 0:
                    ratio = min(1.0, preview_time / duration_s)
                    frame_index = int(round((frame_count - 1) * ratio))
                else:
                    frame_index = 0
                frame = extract_highres_frame(
                    cap,
                    frame_index=frame_index,
                    rotation_degrees=int(rotation_value),
                )
            else:
                frame = extract_highres_frame(
                    cap,
                    time_ms=int(preview_time * 1000),
                    rotation_degrees=int(rotation_value),
                )
        except Exception:
            self.preview_label.setText("Preview unavailable")
            self.preview_label.setPixmap(QtGui.QPixmap())
            return
        finally:
            cap.release()

        if frame is None:
            self.preview_label.setText("Preview unavailable")
            self.preview_label.setPixmap(QtGui.QPixmap())
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        image = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()
        pixmap = QtGui.QPixmap.fromImage(image)
        pixmap = pixmap.scaled(
            self.preview_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.preview_label.setText("")
        self.preview_label.setPixmap(pixmap)

    def _refresh_item_labels(self, *args: object) -> None:
        for i in range(self.thumb_list.count()):
            item = self.thumb_list.item(i)
            data = item.data(QtCore.Qt.UserRole)
            if isinstance(data, PageItem):
                timestamp_sec = data.timestamp_ms / 1000.0
                suffix = "" if data.is_selected is not False else "  dup"
            else:
                timestamp_sec = 0.0
                suffix = ""
            item.setText(f"{i + 1:04d}  {timestamp_sec:.2f}s{suffix}")
        self._update_page_count()

    def _open_output_folder(self) -> None:
        if not self._output_dir or not os.path.isdir(self._output_dir):
            QtWidgets.QMessageBox.warning(
                self, "Output folder missing", "Please run extraction first."
            )
            return
        ok = QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(self._output_dir)
        )
        if not ok:
            QtWidgets.QMessageBox.warning(
                self, "Open folder", "Failed to open the output folder."
            )

    def _update_transcript_view(self) -> None:
        items = self.thumb_list.selectedItems()
        if not items:
            self.transcript_view.clear()
            return
        if len(items) > 1:
            self.transcript_view.setPlainText("Multiple pages selected.")
            return
        data = items[0].data(QtCore.Qt.UserRole)
        if isinstance(data, PageItem):
            if data.ocr_text is None:
                self.transcript_view.setPlainText("Transcription pending.")
            elif data.ocr_text:
                self.transcript_view.setPlainText(data.ocr_text)
            else:
                self.transcript_view.setPlainText("(No transcription)")
        else:
            self.transcript_view.setPlainText("(No transcription)")

    def _log(self, message: str) -> None:
        self.log_view.append(message)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if path:
            self._set_video(path)


def run() -> None:
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
