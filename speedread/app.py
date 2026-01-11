from __future__ import annotations

import os
import tempfile
from typing import Dict, List, Optional

import cv2
from PySide6 import QtCore, QtGui, QtWidgets

from .extractor import (
    ExtractParams,
    extract_highres_frame,
    extract_pages,
    extract_single_frame,
    export_results,
)
from .models import PageResult, RawFrame


class ExtractWorker(QtCore.QObject):
    progress = QtCore.Signal(int, str)
    item_ready = QtCore.Signal(object)
    finished = QtCore.Signal(bool, str, str, object)

    def __init__(self, video_path: str, params: ExtractParams, work_dir: str):
        super().__init__()
        self._video_path = video_path
        self._params = params
        self._work_dir = work_dir
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        def _progress(percent: int, message: str) -> None:
            self.progress.emit(int(percent), message)

        def _should_stop() -> bool:
            return self._stop

        try:
            work_dir, _results, raw_frames = extract_pages(
                self._video_path,
                self._params,
                work_dir=self._work_dir,
                progress_cb=_progress,
                should_stop=_should_stop,
                on_item=lambda item: self.item_ready.emit(item),
            )
            if self._stop:
                self.finished.emit(False, "Canceled", work_dir, raw_frames)
            else:
                self.finished.emit(True, "Done", work_dir, raw_frames)
        except Exception as exc:
            self.finished.emit(False, f"Error: {exc}", self._work_dir, [])


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Speedread Extractor")
        self.resize(1200, 720)
        self.setAcceptDrops(True)

        self._video_path: Optional[str] = None
        self._work_dir: Optional[str] = None
        self._next_temp_index = 1
        self._extract_thread: Optional[QtCore.QThread] = None
        self._extract_worker: Optional[ExtractWorker] = None
        self._raw_frames: List[RawFrame] = []
        self._preview_time_s = 1.0

        self._preset_map = self._build_presets()

        self._build_ui()

    def _build_presets(self) -> Dict[str, ExtractParams]:
        return {
            "Fast": ExtractParams(
                analysis_fps=1.0,
                analysis_long_side=0,
                max_interval_s=5.0,
            ),
            "Balanced": ExtractParams(
                analysis_fps=1.0,
                analysis_long_side=0,
                max_interval_s=5.0,
            ),
            "Robust": ExtractParams(
                analysis_fps=1.0,
                analysis_long_side=0,
                max_interval_s=5.0,
            ),
        }

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        main_layout = QtWidgets.QHBoxLayout(root)

        left_panel = self._build_left_panel()
        center_panel = self._build_center_panel()
        right_panel = self._build_right_panel()

        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(center_panel, 2)
        main_layout.addWidget(right_panel, 3)

    def _build_left_panel(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        input_group = QtWidgets.QGroupBox("Input")
        input_layout = QtWidgets.QVBoxLayout(input_group)

        path_layout = QtWidgets.QHBoxLayout()
        self.video_path_edit = QtWidgets.QLineEdit()
        self.video_path_edit.setReadOnly(True)
        browse_btn = QtWidgets.QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_video)
        path_layout.addWidget(self.video_path_edit, 1)
        path_layout.addWidget(browse_btn)
        input_layout.addLayout(path_layout)

        preset_layout = QtWidgets.QHBoxLayout()
        preset_label = QtWidgets.QLabel("Preset")
        self.preset_combo = QtWidgets.QComboBox()
        for name in self._preset_map.keys():
            self.preset_combo.addItem(name)
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        preset_layout.addWidget(preset_label)
        preset_layout.addWidget(self.preset_combo, 1)
        input_layout.addLayout(preset_layout)

        layout.addWidget(input_group)

        options_group = QtWidgets.QGroupBox("Options")
        options_layout = QtWidgets.QVBoxLayout(options_group)
        self.llm_split_check = QtWidgets.QCheckBox("LLM split into 4 (vertical only)")
        self.llm_split_check.setChecked(True)
        options_layout.addWidget(self.llm_split_check)
        layout.addWidget(options_group)

        advanced_group = QtWidgets.QGroupBox("Extraction")
        advanced_layout = QtWidgets.QFormLayout(advanced_group)

        self.analysis_fps_spin = QtWidgets.QDoubleSpinBox()
        self.analysis_fps_spin.setRange(0.1, 30.0)
        self.analysis_fps_spin.setSingleStep(0.1)
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

        advanced_layout.addRow("Base FPS", self.analysis_fps_spin)
        advanced_layout.addRow("Analysis long side", self.analysis_long_side_combo)
        advanced_layout.addRow("Max interval (s)", self.max_interval_spin)
        advanced_layout.addRow("Rotation", self.rotation_combo)
        advanced_layout.addRow("LLM base URL", self.llm_url_edit)
        advanced_layout.addRow("LLM model", self.llm_model_edit)
        advanced_layout.addRow("Prompt", self.llm_prompt_combo)
        advanced_layout.addRow("Max tokens", self.llm_max_tokens_spin)

        layout.addWidget(advanced_group)

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
        self.add_btn = QtWidgets.QPushButton("Add frame")
        self.add_btn.clicked.connect(self._add_frame)
        self.add_btn.setEnabled(False)
        self.export_btn = QtWidgets.QPushButton("Export")
        self.export_btn.clicked.connect(self._export_results)
        self.export_btn.setEnabled(False)
        self.export_pdf_check = QtWidgets.QCheckBox("Export PDF")

        actions_layout.addWidget(self.extract_btn)
        actions_layout.addWidget(self.stop_btn)
        actions_layout.addWidget(self.add_btn)
        actions_layout.addWidget(self.export_btn)
        actions_layout.addWidget(self.export_pdf_check)
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
        delete_btn = QtWidgets.QPushButton("Delete")
        delete_btn.clicked.connect(self._delete_selected)
        duplicate_btn = QtWidgets.QPushButton("Duplicate")
        duplicate_btn.clicked.connect(self._duplicate_selected)
        header_layout.addWidget(self.page_count_label)
        header_layout.addStretch()
        header_layout.addWidget(delete_btn)
        header_layout.addWidget(duplicate_btn)

        self.thumb_container = QtWidgets.QWidget()
        thumb_layout = QtWidgets.QVBoxLayout(self.thumb_container)
        thumb_layout.setContentsMargins(0, 0, 0, 0)
        thumb_layout.setSpacing(6)

        self.thumb_list = QtWidgets.QListWidget()
        self.thumb_list.setViewMode(QtWidgets.QListView.IconMode)
        self.thumb_list.setIconSize(QtCore.QSize(160, 220))
        self.thumb_list.setResizeMode(QtWidgets.QListView.Adjust)
        self.thumb_list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.thumb_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.thumb_list.setSpacing(8)
        self.thumb_list.model().rowsMoved.connect(self._refresh_item_labels)
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
        self.analysis_fps_spin.setValue(params.analysis_fps)
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
            analysis_fps=self.analysis_fps_spin.value(),
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
        self.add_btn.setEnabled(True)
        self._log(f"Loaded video: {path}")
        self._refresh_preview()

    def _start_extract(self) -> None:
        if not self._video_path:
            QtWidgets.QMessageBox.warning(self, "No video", "Please select a video file")
            return
        if self._extract_worker:
            QtWidgets.QMessageBox.warning(self, "Busy", "Extraction is already running")
            return

        self._reset_results()
        self._work_dir = tempfile.mkdtemp(prefix="speedread_")
        self._next_temp_index = 1

        params = self._collect_params()
        self._extract_thread = QtCore.QThread()
        self._extract_worker = ExtractWorker(self._video_path, params, self._work_dir)
        self._extract_worker.moveToThread(self._extract_thread)

        self._extract_thread.started.connect(self._extract_worker.run)
        self._extract_worker.progress.connect(self._on_progress)
        self._extract_worker.item_ready.connect(self._on_item_ready)
        self._extract_worker.finished.connect(self._on_extract_finished)
        self._extract_thread.start()

        self.extract_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.export_btn.setEnabled(False)
        self.add_btn.setEnabled(False)

    def _stop_extract(self) -> None:
        if self._extract_worker:
            self._extract_worker.stop()
            self._log("Stopping extraction...")

    def _on_progress(self, percent: int, message: str) -> None:
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)
        self._log(message)

    def _on_item_ready(self, item: PageResult) -> None:
        self._next_temp_index += 1
        self._add_page_item(item)

    def _on_extract_finished(
        self, success: bool, message: str, work_dir: str, raw_frames: object
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
        self.export_btn.setEnabled(self.thumb_list.count() > 0 or len(self._raw_frames) > 0)
        self.add_btn.setEnabled(True)
        self._work_dir = work_dir

    def _add_page_item(self, item: PageResult) -> None:
        pixmap = QtGui.QPixmap(item.image_path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(
                self.thumb_list.iconSize(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )

        list_item = QtWidgets.QListWidgetItem()
        list_item.setIcon(QtGui.QIcon(pixmap))
        timestamp_sec = item.timestamp_ms / 1000.0
        list_item.setText(f"{self.thumb_list.count() + 1:04d}\n{timestamp_sec:.2f}s")
        list_item.setData(QtCore.Qt.UserRole, item)
        self.thumb_list.addItem(list_item)
        self._refresh_item_labels()

    def _update_page_count(self) -> None:
        self.page_count_label.setText(f"Pages: {self.thumb_list.count()}")

    def _reset_results(self) -> None:
        self.thumb_list.clear()
        self._update_page_count()
        self.progress_bar.setValue(0)
        self.status_label.setText("Idle")
        self.log_view.clear()
        self._raw_frames = []
        self.transcript_view.clear()
        self._refresh_preview()

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
            if duration_s and duration_s < preview_time:
                preview_time = max(0.0, duration_s / 2.0)
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
            if isinstance(data, PageResult):
                timestamp_sec = data.timestamp_ms / 1000.0
            else:
                timestamp_sec = 0.0
            item.setText(f"{i + 1:04d}\n{timestamp_sec:.2f}s")
        self._update_page_count()

    def _delete_selected(self) -> None:
        for item in self.thumb_list.selectedItems():
            self.thumb_list.takeItem(self.thumb_list.row(item))
        self._refresh_item_labels()
        self._update_transcript_view()

    def _duplicate_selected(self) -> None:
        items = self.thumb_list.selectedItems()
        for item in items:
            data = item.data(QtCore.Qt.UserRole)
            if not isinstance(data, PageResult):
                continue
            dup_item = QtWidgets.QListWidgetItem()
            dup_item.setIcon(item.icon())
            dup_item.setText(item.text())
            dup_item.setData(QtCore.Qt.UserRole, data)
            self.thumb_list.addItem(dup_item)
        self._refresh_item_labels()
        self._update_transcript_view()

    def _add_frame(self) -> None:
        if not self._video_path:
            QtWidgets.QMessageBox.warning(self, "No video", "Please select a video file")
            return
        if not self._work_dir:
            self._work_dir = tempfile.mkdtemp(prefix="speedread_")

        seconds, ok = QtWidgets.QInputDialog.getDouble(
            self, "Add frame", "Timestamp (seconds)", 0.0, 0.0, 10_000.0, 2
        )
        if not ok:
            return

        params = self._collect_params()
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            result = extract_single_frame(
                self._video_path,
                int(seconds * 1000),
                params,
                self._work_dir,
                self._next_temp_index,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to add frame: {exc}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self._next_temp_index += 1
        self._add_page_item(result)
        self.export_btn.setEnabled(True)
        self._update_transcript_view()

    def _export_results(self) -> None:
        if not self._video_path:
            return
        if self.thumb_list.count() == 0 and not self._raw_frames:
            QtWidgets.QMessageBox.warning(self, "No pages", "No extracted pages to export")
            return

        output_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder")
        if not output_dir:
            return

        results = []
        for i in range(self.thumb_list.count()):
            item = self.thumb_list.item(i)
            data = item.data(QtCore.Qt.UserRole)
            if isinstance(data, PageResult):
                results.append(data)

        ok, message = export_results(
            results,
            output_dir,
            source_video_path=self._video_path,
            export_pdf=self.export_pdf_check.isChecked(),
            raw_frames=self._raw_frames,
        )
        if not ok:
            QtWidgets.QMessageBox.warning(self, "Export", message)
        else:
            QtWidgets.QMessageBox.information(self, "Export", message)

    def _update_transcript_view(self) -> None:
        items = self.thumb_list.selectedItems()
        if not items:
            self.transcript_view.clear()
            return
        if len(items) > 1:
            self.transcript_view.setPlainText("Multiple pages selected.")
            return
        data = items[0].data(QtCore.Qt.UserRole)
        if isinstance(data, PageResult) and data.ocr_text:
            self.transcript_view.setPlainText(data.ocr_text)
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
