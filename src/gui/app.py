from __future__ import annotations

import json
import math
import mimetypes
import sys
import uuid
from pathlib import Path
from urllib import request

from PySide6.QtCore import (
    QEasingCurve, QPropertyAnimation, QTimer, Qt, QThread,
    Signal, QParallelAnimationGroup, QPoint, QRectF, QEvent, QUrl
)
from PySide6.QtGui import (
    QColor, QIcon, QLinearGradient, QPainter, QPixmap,
    QRadialGradient, QBrush, QPen, QFont, QFontDatabase,
    QPalette, QPainterPath, QDesktopServices
)
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QGraphicsOpacityEffect,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QProgressBar, QScrollArea, QStackedLayout, QVBoxLayout,
    QWidget, QSizePolicy
)

try:
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
except ImportError:  # pragma: no cover - depends on the local Qt install
    QAudioOutput = None
    QMediaPlayer = None


MEDIA_STATUS_LOADED = getattr(QMediaPlayer.MediaStatus, "LoadedMedia", None) if QMediaPlayer is not None else None


class SearchWorker(QThread):
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, backend_url: str, image_path: Path) -> None:
        super().__init__()
        self.backend_url = backend_url.rstrip("/")
        self.image_path = image_path

    def run(self) -> None:
        try:
            payload = self._post_image()
            self.finished.emit(payload)
        except Exception as exc:
            self.failed.emit(str(exc))

    def _post_image(self) -> dict:
        boundary = f"----Reflectra{uuid.uuid4().hex}"
        mime_type = mimetypes.guess_type(str(self.image_path))[0] or "application/octet-stream"
        image_bytes = self.image_path.read_bytes()

        body = b"".join(
            [
                f"--{boundary}\r\n".encode(),
                (
                    'Content-Disposition: form-data; name="image"; '
                    f'filename="{self.image_path.name}"\r\n'
                ).encode(),
                f"Content-Type: {mime_type}\r\n\r\n".encode(),
                image_bytes,
                f"\r\n--{boundary}--\r\n".encode(),
            ]
        )

        req = request.Request(
            f"{self.backend_url}/api/search",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )

        with request.urlopen(req, timeout=600) as response:
            return json.loads(response.read().decode("utf-8"))


class AudioDownloadWorker(QThread):
    finished = Signal(int, str, bool)
    failed = Signal(int, str)

    def __init__(self, index: int, item: dict, output_dir: Path, play_after: bool) -> None:
        super().__init__()
        self.index = index
        self.item = item
        self.output_dir = output_dir
        self.play_after = play_after

    def run(self) -> None:
        try:
            from study.audio_parts import download_audio_from_metadata

            audio_path = download_audio_from_metadata(
                metadata=self.item,
                output_dir=self.output_dir,
            )
            self.finished.emit(self.index, str(audio_path), self.play_after)
        except Exception as exc:
            self.failed.emit(self.index, str(exc))


class StatusWorker(QThread):
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, backend_url: str) -> None:
        super().__init__()
        self.backend_url = backend_url.rstrip("/")

    def run(self) -> None:
        try:
            with request.urlopen(f"{self.backend_url}/api/status", timeout=4) as response:
                self.finished.emit(json.loads(response.read().decode("utf-8")))
        except Exception as exc:
            self.failed.emit(str(exc))


class AnimatedBackground(QWidget):
    def __init__(self, logo_path: Path) -> None:
        super().__init__()
        self.image = QPixmap()
        self.blur = False
        self.animation_enabled = False
        self.animation_progress = 0.0
        
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_animation)
    
    def _update_animation(self) -> None:
        if not self.animation_enabled:
            return
        self.animation_progress += 0.0035
        if self.animation_progress > 1.0:
            self.animation_progress = 0.0
        self.update()

    def set_animation_enabled(self, enabled: bool) -> None:
        self.animation_enabled = enabled
        if enabled and self.image.isNull():
            if not self.timer.isActive():
                self.timer.start(90)
        else:
            self.timer.stop()
        self.update()

    def set_image(self, path: Path) -> None:
        self.image = QPixmap(str(path))
        self.set_animation_enabled(False)
        self.update()

    def set_blur(self, enabled: bool) -> None:
        self.blur = enabled
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Dynamic gradient with animation
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0.0, QColor("#0a0e12"))
        gradient.setColorAt(0.3, QColor("#141c22"))
        gradient.setColorAt(0.6, QColor("#1a242b"))
        gradient.setColorAt(1.0, QColor("#080c0f"))
        
        # Add subtle gradient shift
        shift = self.animation_progress * 0.3
        gradient.setColorAt(0.0 + shift, QColor("#0f1418"))
        gradient.setColorAt(0.5 + shift, QColor("#1a242b"))
        
        painter.fillRect(self.rect(), gradient)

        if self.animation_enabled and self.image.isNull():
            self._draw_disco_ball(painter)

        # Draw background image if available
        if not self.image.isNull():
            scaled = self.image.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            if self.blur:
                # Progressive blur effect
                blur_factor = 12 + (self.animation_progress * 4)
                softened_width = max(int(scaled.width() / blur_factor), 1)
                softened_height = max(int(scaled.height() / blur_factor), 1)
                if softened_width > 0 and softened_height > 0:
                    scaled = scaled.scaled(
                        softened_width,
                        softened_height,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    ).scaled(
                        self.size(),
                        Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                        Qt.TransformationMode.SmoothTransformation,
                    )
            
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.setOpacity(0.15 if self.blur else 0.4)
            painter.drawPixmap(x, y, scaled)

        # Overlay with subtle gradient
        painter.setOpacity(1.0)
        overlay = QLinearGradient(0, 0, 0, self.height())
        overlay.setColorAt(0.0, QColor(10, 14, 18, 180))
        overlay.setColorAt(0.5, QColor(10, 14, 18, 160))
        overlay.setColorAt(1.0, QColor(10, 14, 18, 200))
        painter.fillRect(self.rect(), overlay)
        
        super().paintEvent(event)

    def _draw_disco_ball(self, painter: QPainter) -> None:
        if self.width() <= 0 or self.height() <= 0:
            return

        width = self.width()
        height = self.height()
        phase = self.animation_progress * math.tau

        center_x = width * 0.5
        center_y = height * 0.47
        radius = min(width, height) * 0.28
        beam_colors = [
            QColor(255, 212, 94, 34),
            QColor(255, 62, 136, 30),
            QColor(84, 226, 202, 30),
            QColor(128, 96, 255, 30),
            QColor(255, 255, 255, 24),
        ]

        painter.setPen(Qt.PenStyle.NoPen)
        for index, color in enumerate(beam_colors):
            angle = phase + index * math.tau / len(beam_colors)
            spread = 0.34
            far = max(width, height) * 1.1
            path = QPainterPath()
            path.moveTo(center_x, center_y)
            path.lineTo(
                center_x + math.cos(angle - spread) * far,
                center_y + math.sin(angle - spread) * far,
            )
            path.lineTo(
                center_x + math.cos(angle + spread) * far,
                center_y + math.sin(angle + spread) * far,
            )
            path.closeSubpath()
            painter.fillPath(path, color)

        glow = QRadialGradient(center_x, center_y, radius * 1.55)
        glow.setColorAt(0.0, QColor(255, 255, 255, 45))
        glow.setColorAt(0.42, QColor(255, 212, 94, 22))
        glow.setColorAt(1.0, QColor(255, 255, 255, 0))
        painter.setBrush(QBrush(glow))
        painter.drawEllipse(QRectF(center_x - radius * 1.5, center_y - radius * 1.5, radius * 3, radius * 3))

        ball_gradient = QRadialGradient(center_x - radius * 0.25, center_y - radius * 0.35, radius * 1.35)
        ball_gradient.setColorAt(0.0, QColor("#233342"))
        ball_gradient.setColorAt(0.58, QColor("#12181f"))
        ball_gradient.setColorAt(1.0, QColor("#05070a"))
        painter.setBrush(QBrush(ball_gradient))
        painter.drawEllipse(QRectF(center_x - radius, center_y - radius, radius * 2, radius * 2))

        spot_colors = [
            QColor("#ffd45e"),
            QColor("#ff3e88"),
            QColor("#54e2ca"),
            QColor("#8060ff"),
            QColor("#ffffff"),
            QColor("#ff7a45"),
        ]
        spot_count = 18
        for index in range(spot_count):
            ring = 0.38 + (index % 3) * 0.2
            angle = phase * 1.4 + index * math.tau / spot_count
            x = center_x + math.cos(angle) * radius * ring
            y = center_y + math.sin(angle) * radius * ring * 0.72
            depth = 0.62 + math.sin(angle + phase) * 0.28
            spot_radius = radius * (0.08 + 0.035 * ((index % 4) / 3)) * max(depth, 0.35)
            color = QColor(spot_colors[index % len(spot_colors)])
            color.setAlpha(int(150 + 80 * max(depth, 0)))

            lens = QRadialGradient(x - spot_radius * 0.25, y - spot_radius * 0.35, spot_radius * 1.2)
            lens.setColorAt(0.0, QColor(255, 255, 255, 230))
            lens.setColorAt(0.25, color.lighter(140))
            lens.setColorAt(1.0, color.darker(150))
            painter.setBrush(QBrush(lens))
            painter.drawEllipse(QRectF(x - spot_radius, y - spot_radius, spot_radius * 2, spot_radius * 2))

        base_y = center_y + radius * 1.06
        base_gradient = QLinearGradient(center_x - radius * 0.5, base_y, center_x + radius * 0.5, base_y + radius * 0.58)
        base_gradient.setColorAt(0.0, QColor(20, 24, 28, 120))
        base_gradient.setColorAt(0.5, QColor(70, 76, 82, 150))
        base_gradient.setColorAt(1.0, QColor(10, 12, 14, 120))
        painter.setBrush(QBrush(base_gradient))
        painter.drawRoundedRect(
            QRectF(center_x - radius * 0.42, base_y, radius * 0.84, radius * 0.46),
            radius * 0.08,
            radius * 0.08,
        )

        base_top = base_y + radius * 0.34
        painter.setBrush(QColor(12, 14, 16, 135))
        painter.drawEllipse(QRectF(center_x - radius * 0.88, base_top, radius * 1.76, radius * 0.44))


class WaveformSeekBar(QWidget):
    seek_requested = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self.progress = 0.0
        self.active = False
        self.phase = 0.0
        self.setMinimumHeight(40)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._animate)
        self.amplitudes = [
            0.35, 0.72, 0.5, 0.88, 0.42, 0.95, 0.65, 0.38,
            0.8, 0.55, 0.92, 0.48, 0.7, 0.9, 0.45, 0.75,
            0.58, 0.86, 0.52, 0.68, 0.96, 0.44, 0.74, 0.6,
            0.84, 0.5, 0.78, 0.36, 0.66, 0.92, 0.54, 0.72,
        ]

    def set_progress(self, value: float) -> None:
        self.progress = min(max(value, 0.0), 1.0)
        self.update()

    def set_active(self, active: bool) -> None:
        self.active = active
        if active and not self.timer.isActive():
            self.timer.start(90)
        elif not active:
            self.timer.stop()
            self.phase = 0.0
        self.update()

    def _animate(self) -> None:
        self.phase = (self.phase + 0.16) % math.tau
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() != Qt.MouseButton.LeftButton or self.width() <= 0:
            return
        ratio = min(max(event.position().x() / self.width(), 0.0), 1.0)
        self.set_progress(ratio)
        self.seek_requested.emit(ratio)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)

        rect = self.rect().adjusted(0, 4, 0, -4)
        if rect.width() <= 0 or rect.height() <= 0:
            return

        count = len(self.amplitudes)
        gap = max(rect.width() / (count * 1.8), 3.0)
        bar_width = max(gap * 0.45, 2.0)
        step = rect.width() / count
        center_y = rect.center().y()
        played_x = rect.left() + rect.width() * self.progress

        for index, amplitude in enumerate(self.amplitudes):
            x = rect.left() + index * step + (step - bar_width) / 2
            pulse = 0.0
            if self.active:
                pulse = 0.16 * math.sin(self.phase + index * 0.72)
            height = rect.height() * min(max(amplitude + pulse, 0.22), 1.0)
            y = center_y - height / 2

            if x <= played_x:
                color = QColor("#ffd45e")
            elif self.active:
                color = QColor("#54e2ca")
                color.setAlpha(170)
            else:
                color = QColor(247, 244, 236, 150)

            painter.setBrush(color)
            painter.drawRoundedRect(QRectF(x, y, bar_width, height), bar_width / 2, bar_width / 2)


class ModernResultCard(QFrame):
    download_requested = Signal(int, dict, bool)
    stop_requested = Signal(int)
    seek_requested = Signal(int, dict, float)

    def __init__(self, item: dict, index: int) -> None:
        super().__init__()
        self.setObjectName("resultCard")
        self.index = index
        self.item = item
        self.busy_step = 0
        self.busy_play_after = False
        self.busy_timer = QTimer(self)
        self.busy_timer.timeout.connect(self._tick_busy)
        self.setFixedHeight(214)
        self.setStyleSheet("""
            QFrame#resultCard {
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 12px;
                background: rgba(12, 18, 20, 0.72);
            }
        """)
        
        # Layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(16)
        
        # Left side - Number or icon
        number_label = QLabel(f"#{index + 1:02d}")
        number_label.setObjectName("resultNumber")
        number_label.setFixedWidth(40)
        number_label.setStyleSheet("color: #ffd45e; font-weight: 700; font-size: 16px;")
        layout.addWidget(number_label)
        
        # Content area
        content_layout = QVBoxLayout()
        content_layout.setSpacing(6)
        
        payload = item.get("payload") or {}
        captions = payload.get("captions") or []
        
        title = QLabel(self._display_title(payload))
        title.setObjectName("cardTitle")
        title.setStyleSheet("font-weight: 800; font-size: 17px; color: #f7f4ec;")
        title.setWordWrap(True)
        content_layout.addWidget(title)

        source_label = QLabel(self._source_label(payload))
        source_label.setObjectName("cardMeta")
        source_label.setStyleSheet("color: #b8c0b4; font-size: 12px; font-weight: 650;")
        source_label.setWordWrap(True)
        content_layout.addWidget(source_label)
        
        # Metadata with visual indicators
        meta_layout = QHBoxLayout()
        meta_layout.setSpacing(12)
        
        # Score indicator
        bi_score = float(item.get('bi_encoder_score') or 0)
        score_bar = self._create_score_bar(bi_score)
        meta_layout.addWidget(score_bar)
        
        # Text metadata
        meta_text = QLabel(f"dataset_id: {str(payload.get('dataset_id', 'unknown'))}")
        meta_text.setObjectName("cardMeta")
        meta_text.setStyleSheet("color: #b8c0b4; font-size: 12px;")
        meta_text.setWordWrap(True)
        meta_layout.addWidget(meta_text)
        
        rerank = item.get('rerank_score')
        if rerank is not None:
            rerank_label = QLabel(f"Rerank: {rerank:.3f}")
            rerank_label.setObjectName("cardMeta")
            rerank_label.setStyleSheet("color: #b8c0b4; font-size: 12px;")
            meta_layout.addWidget(rerank_label)
        
        meta_layout.addStretch()
        content_layout.addLayout(meta_layout)
        
        # Captions
        caption = QLabel(" ".join(str(text) for text in captions[:2]) or "No captions available")
        caption.setObjectName("caption")
        caption.setStyleSheet("color: #e5e8df; font-size: 13px; line-height: 1.4;")
        caption.setWordWrap(True)
        content_layout.addWidget(caption)

        self.waveform = WaveformSeekBar()
        self.waveform.seek_requested.connect(lambda ratio: self.seek_requested.emit(self.index, self.item, ratio))
        content_layout.addWidget(self.waveform)
        
        layout.addLayout(content_layout, 1)

        actions = QVBoxLayout()
        actions.setSpacing(7)
        self.play_button = QPushButton("Play")
        self.play_button.setObjectName("resultActionPrimary")
        self.play_button.setFixedSize(116, 34)
        self.play_button.clicked.connect(lambda: self.download_requested.emit(self.index, self.item, True))
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("resultAction")
        self.stop_button.setFixedSize(116, 34)
        self.stop_button.clicked.connect(lambda: self.stop_requested.emit(self.index))
        self.download_button = QPushButton("Save")
        self.download_button.setObjectName("resultAction")
        self.download_button.setFixedSize(116, 34)
        self.download_button.clicked.connect(lambda: self.download_requested.emit(self.index, self.item, False))
        button_style = """
            QPushButton {
                border-radius: 8px;
                padding: 8px 10px;
                font-size: 12px;
                font-weight: 700;
            }
            QPushButton#resultActionPrimary {
                color: #1a1a1a;
                background: #ffd45e;
                border: none;
            }
            QPushButton#resultActionPrimary:hover {
                background: #ffe06e;
            }
            QPushButton#resultAction {
                color: #f7f4ec;
                background: rgba(255, 255, 255, 0.13);
                border: 1px solid rgba(255, 255, 255, 0.22);
            }
            QPushButton#resultAction:hover {
                background: rgba(255, 255, 255, 0.2);
            }
            QPushButton:disabled {
                color: rgba(247, 244, 236, 0.45);
                background: rgba(255, 255, 255, 0.05);
            }
        """
        self.play_button.setStyleSheet(button_style)
        self.stop_button.setStyleSheet(button_style)
        self.download_button.setStyleSheet(button_style)
        self.download_progress = QProgressBar()
        self.download_progress.setRange(0, 0)
        self.download_progress.setTextVisible(False)
        self.download_progress.setFixedSize(116, 6)
        self.download_progress.hide()
        self.download_progress.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 3px;
                background: rgba(255, 255, 255, 0.08);
            }
            QProgressBar::chunk {
                border-radius: 3px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #54e2ca, stop:0.5 #ffd45e, stop:1 #ff9a56);
            }
        """)
        self.action_status = QLabel("")
        self.action_status.setFixedWidth(116)
        self.action_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.action_status.setStyleSheet("color: #b8c0b4; font-size: 11px;")
        actions.addWidget(self.play_button)
        actions.addWidget(self.stop_button)
        actions.addWidget(self.download_button)
        actions.addWidget(self.download_progress)
        actions.addWidget(self.action_status)
        actions.addStretch()
        layout.addLayout(actions)

    def _display_title(self, payload: dict) -> str:
        for key in ("filename", "stem", "dataset_id", "audio_id"):
            value = str(payload.get(key, "")).strip()
            if value:
                return value
        return "Unknown track"

    def _source_label(self, payload: dict) -> str:
        source = str(payload.get("source_dataset") or payload.get("source") or "unknown source")
        audio_id = str(payload.get("audio_id") or "").strip()
        if audio_id and audio_id != str(payload.get("dataset_id", "")).strip():
            return f"{source} | audio_id: {audio_id}"
        return source
    
    def _create_score_bar(self, score: float) -> QWidget:
        widget = QWidget()
        widget.setFixedSize(60, 6)
        
        # Background
        bg = QWidget(widget)
        bg.setGeometry(0, 0, 60, 6)
        bg.setStyleSheet("background: rgba(255,255,255,0.1); border-radius: 3px;")
        
        # Fill
        fill = QWidget(bg)
        fill_width = max(int(60 * min(score, 1.0)), 2)
        fill.setGeometry(0, 0, fill_width, 6)
        fill.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #ffd45e, stop:1 #ff9a56);
            border-radius: 3px;
        """)
        
        return widget
    
    def set_busy(self, busy: bool, play_after: bool = False) -> None:
        self.play_button.setEnabled(not busy)
        self.download_button.setEnabled(not busy)
        self.busy_play_after = play_after
        if busy:
            self.busy_step = 0
            self.download_progress.show()
            self.action_status.setText("Fetching audio" if play_after else "Saving audio")
            self._tick_busy()
            self.busy_timer.start(260)
        else:
            self.busy_timer.stop()
            self.download_progress.hide()
            self.action_status.setText("")
            self.play_button.setText("Play")
            self.download_button.setText("Save")

    def set_downloaded(self) -> None:
        self.download_button.setText("Saved")
        self.action_status.setText("Ready")

    def set_playing(self, playing: bool) -> None:
        self.waveform.set_active(playing)
        self.play_button.setText("Playing" if playing else "Play")
        self.action_status.setText("Playing" if playing else "")

    def set_playback_progress(self, value: float) -> None:
        self.waveform.set_progress(value)

    def _tick_busy(self) -> None:
        dots = "." * ((self.busy_step % 3) + 1)
        self.busy_step += 1
        if self.busy_play_after:
            self.play_button.setText(f"Loading{dots}")
            self.download_button.setText("Save")
        else:
            self.download_button.setText(f"Saving{dots}")
            self.play_button.setText("Play")


class ReflectraWindow(QMainWindow):
    def __init__(self, backend_url: str, gui_dir: Path) -> None:
        super().__init__()
        self.backend_url = backend_url
        self.gui_dir = gui_dir
        self.logo_path = resolve_logo_path(gui_dir)
        self.selected_image: Path | None = None
        self.worker: SearchWorker | None = None
        self.status_worker: StatusWorker | None = None
        self.audio_workers: dict[int, AudioDownloadWorker] = {}
        self.result_cards: dict[int, ModernResultCard] = {}
        self.downloaded_audio: dict[str, Path] = {}
        self.audio_output_dir = Path("data/study_downloaded_audio")
        self.media_player = None
        self.audio_output = None
        self.current_audio_index: int | None = None
        self.pending_start_ratio = 0.0
        self.pending_start_ratios: dict[str, float] = {}
        self.canceled_audio_downloads: set[int] = set()
        self.backend_status_failures = 0
        self.backend_ready = False
        self._is_maximized = False
        self._drag_position: QPoint | None = None
        self.title_bar: QWidget | None = None
        self.setAcceptDrops(True)
        
        # Window setup
        self.setWindowTitle("Reflectra")
        self.setMinimumSize(1000, 720)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        if self.logo_path.exists():
            self.setWindowIcon(QIcon(str(self.logo_path)))
        self.resize(1200, 800)
        
        # Main container
        self.central_widget = QWidget()
        self.central_widget.setObjectName("centralWidget")
        self.setCentralWidget(self.central_widget)
        
        # Setup background
        self.background = AnimatedBackground(self.logo_path)
        self.background_layout = QVBoxLayout(self.central_widget)
        self.background_layout.setContentsMargins(0, 0, 0, 0)
        self.background_layout.addWidget(self.background)
        
        # Stack for pages
        self.stack = QStackedLayout(self.background)
        self.stack.setContentsMargins(0, 0, 0, 0)
        
        # Build pages
        self.loading_page = self.build_loading_page()
        self.main_page = self.build_main_page()
        self.stack.addWidget(self.loading_page)
        self.stack.addWidget(self.main_page)
        self.stack.setCurrentWidget(self.loading_page)

        # Setup animations
        self.setup_animations()

        # Apply modern styles
        self.setStyleSheet(self.modern_stylesheet())

        # Setup window controls
        self.setup_window_controls()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    def setup_audio_player(self) -> bool:
        if QMediaPlayer is None or QAudioOutput is None:
            return False
        if self.media_player is not None:
            return True
        self.media_player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(0.8)
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.positionChanged.connect(self.update_playback_position)
        self.media_player.durationChanged.connect(self.update_playback_position)
        if MEDIA_STATUS_LOADED is not None:
            self.media_player.mediaStatusChanged.connect(self.apply_pending_start_position)
        return True
    
    def setup_window_controls(self) -> None:
        # Title bar
        self.title_bar = QWidget(self.main_page)
        self.title_bar.setObjectName("titleBar")
        self.title_bar.setFixedHeight(44)
        self.title_bar.setStyleSheet("""
            QWidget#titleBar {
                background: transparent;
                border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            }
        """)
        title_bar_layout = QHBoxLayout(self.title_bar)
        title_bar_layout.setContentsMargins(12, 0, 12, 0)
        
        # Title
        title_label = QLabel("Reflectra")
        title_label.setObjectName("windowTitle")
        title_label.setStyleSheet("font-size: 14px; font-weight: 600; color: #f7f4ec;")
        title_bar_layout.addWidget(title_label)
        title_bar_layout.addStretch()
        
        # Window controls
        for label, action in [
            ("—", self.showMinimized),
            ("□", self.toggle_maximize),
            ("✕", self.close)
        ]:
            btn = QPushButton(label)
            btn.setObjectName("windowControl")
            btn.setFixedSize(32, 32)
            btn.clicked.connect(action)
            btn.setStyleSheet("""
                QPushButton#windowControl {
                    background: transparent;
                    border: none;
                    color: #f7f4ec;
                    font-size: 14px;
                    border-radius: 16px;
                }
                QPushButton#windowControl:hover {
                    background: rgba(255,255,255,0.1);
                }
                QPushButton#windowControl:pressed {
                    background: rgba(255,255,255,0.2);
                }
            """)
            title_bar_layout.addWidget(btn)
        
        # Add title bar to main page
        main_layout = self.main_page.layout()
        if main_layout:
            main_layout.insertWidget(0, self.title_bar)
    
    def toggle_maximize(self) -> None:
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton and self._can_drag_from(event.position().toPoint()):
            self._begin_window_drag(event)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._drag_position is not None and event.buttons() & Qt.MouseButton.LeftButton:
            if self.isMaximized():
                self.showNormal()
            self.move(event.globalPosition().toPoint() - self._drag_position)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._drag_position = None
        super().mouseReleaseEvent(event)

    def _can_drag_from(self, position: QPoint) -> bool:
        child = self.childAt(position)
        while child is not None:
            if isinstance(child, QPushButton):
                return False
            if child is self.title_bar:
                return True
            child = child.parentWidget()
        return 0 <= position.y() <= 44

    def _can_drag_widget(self, widget: QWidget) -> bool:
        current: QWidget | None = widget
        while current is not None:
            if isinstance(current, QPushButton):
                return False
            if current is self.title_bar:
                return True
            current = current.parentWidget()
        return False

    def _begin_window_drag(self, event) -> None:
        if self.isMaximized():
            self.showNormal()
        handle = self.windowHandle()
        if handle is not None and handle.startSystemMove():
            self._drag_position = None
            return
        self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API
        if not isinstance(watched, QWidget) or not self._contains_widget(watched):
            return super().eventFilter(watched, event)

        if event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                position = watched.mapTo(self, event.position().toPoint())
                if self._can_drag_widget(watched) or self._can_drag_from(position):
                    self._begin_window_drag(event)
                    return True

        if event.type() == QEvent.Type.MouseMove:
            if self._drag_position is not None and event.buttons() & Qt.MouseButton.LeftButton:
                if self.isMaximized():
                    self.showNormal()
                self.move(event.globalPosition().toPoint() - self._drag_position)
                return True

        if event.type() == QEvent.Type.MouseButtonRelease:
            self._drag_position = None

        return super().eventFilter(watched, event)

    def _contains_widget(self, widget: QWidget) -> bool:
        while widget is not None:
            if widget is self:
                return True
            widget = widget.parentWidget()
        return False
    
    def build_loading_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("loadingPage")
        page.setStyleSheet("background: transparent;")
        
        layout = QVBoxLayout(page)
        layout.setContentsMargins(34, 34, 34, 34)
        layout.setSpacing(16)
        
        logo = QLabel()
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if self.logo_path.exists():
            logo.setPixmap(
                QPixmap(str(self.logo_path)).scaled(
                    132,
                    132,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

        title = QLabel("Reflectra")
        title.setObjectName("loadingTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("""
            color: #f7f4ec;
            font-size: 52px;
            font-weight: 850;
            background: transparent;
        """)

        self.loading_dots = QLabel(".")
        self.loading_dots.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_dots.setStyleSheet("""
            color: #ffd45e;
            font-size: 28px;
            font-weight: 700;
            background: transparent;
        """)

        layout.addStretch(1)
        layout.addWidget(logo)
        layout.addWidget(title)
        layout.addWidget(self.loading_dots)
        layout.addStretch(1)

        self.start_loading_dots()
        
        return page

    def start_loading_dots(self) -> None:
        self.dot_count = 1
        self.dot_timer = QTimer(self)
        self.dot_timer.timeout.connect(self.update_loading_dots)
        self.dot_timer.start(420)

    def update_loading_dots(self) -> None:
        self.dot_count = (self.dot_count % 3) + 1
        self.loading_dots.setText("." * self.dot_count)
    
    def build_main_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("mainPage")
        root = QVBoxLayout(page)
        root.setContentsMargins(30, 0, 30, 30)
        root.setSpacing(20)

        self.notification = QFrame()
        self.notification.setObjectName("notification")
        self.notification.hide()
        notification_layout = QHBoxLayout(self.notification)
        notification_layout.setContentsMargins(16, 12, 16, 12)
        notification_layout.setSpacing(10)
        self.notification_text = QLabel()
        self.notification_text.setWordWrap(True)
        self.notification_text.setStyleSheet("font-size: 13px; font-weight: 650;")
        notification_layout.addWidget(self.notification_text, 1)
        root.addWidget(self.notification)
        
        # Content area
        content = QHBoxLayout()
        content.setSpacing(20)
        
        # Left panel
        left_panel = QFrame()
        left_panel.setObjectName("glassPanel")
        left_panel.setFixedWidth(380)
        left_panel.setStyleSheet("""
            QFrame#glassPanel {
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 16px;
                background: rgba(12, 18, 20, 0.6);
            }
        """)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(24, 24, 24, 24)
        left_layout.setSpacing(16)
        
        # Logo in main view
        logo_container = QWidget()
        logo_container.setStyleSheet("background: transparent;")
        logo_layout = QHBoxLayout(logo_container)
        logo_layout.setContentsMargins(0, 0, 0, 0)
        
        mini_logo = QLabel()
        if self.logo_path.exists():
            mini_logo.setPixmap(QPixmap(str(self.logo_path)).scaled(
                48, 48,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
        logo_layout.addWidget(mini_logo)
        
        title_container = QVBoxLayout()
        title = QLabel("Reflectra")
        title.setObjectName("mainTitle")
        title.setStyleSheet("font-size: 28px; font-weight: 800; color: #f7f4ec;")
        subtitle = QLabel("Image to Music Search")
        subtitle.setObjectName("muted")
        subtitle.setStyleSheet("color: #b8c0b4; font-size: 13px;")
        title_container.addWidget(title)
        title_container.addWidget(subtitle)
        logo_layout.addLayout(title_container)
        logo_layout.addStretch()
        left_layout.addWidget(logo_container)
        
        # Drop area
        self.drop_label = QLabel("Drop an image here\nor choose one from disk")
        self.drop_label.setObjectName("dropLabel")
        self.drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_label.setMinimumHeight(200)
        self.drop_label.setStyleSheet("""
            QLabel#dropLabel {
                border: 2px dashed rgba(255, 255, 255, 0.15);
                border-radius: 16px;
                background: rgba(255, 255, 255, 0.03);
                font-size: 18px;
                font-weight: 500;
                color: #c0c8bc;
                padding: 20px;
            }
            QLabel#dropLabel:hover {
                border-color: rgba(255, 212, 94, 0.4);
                background: rgba(255, 212, 94, 0.05);
            }
        """)
        left_layout.addWidget(self.drop_label)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        
        self.choose_button = QPushButton("📁 Choose Image")
        self.choose_button.setObjectName("secondaryButton")
        self.choose_button.clicked.connect(self.choose_image)
        self.choose_button.setStyleSheet("""
            QPushButton#secondaryButton {
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 10px;
                padding: 12px 20px;
                color: #f7f4ec;
                font-weight: 600;
                font-size: 14px;
            }
            QPushButton#secondaryButton:hover {
                background: rgba(255, 255, 255, 0.15);
            }
        """)
        
        self.search_button = QPushButton("🔍 Search Songs")
        self.search_button.setObjectName("primaryButton")
        self.search_button.clicked.connect(self.search)
        self.search_button.setStyleSheet("""
            QPushButton#primaryButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ffd45e, stop:1 #ff9a56);
                border: none;
                border-radius: 10px;
                padding: 12px 24px;
                color: #1a1a1a;
                font-weight: 700;
                font-size: 14px;
            }
            QPushButton#primaryButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ffe06e, stop:1 #ffaa66);
            }
            QPushButton#primaryButton:disabled {
                opacity: 0.5;
            }
        """)
        
        btn_layout.addWidget(self.choose_button)
        btn_layout.addWidget(self.search_button)
        left_layout.addLayout(btn_layout)
        
        # Status
        self.status = QLabel("✨ Ready to create music from your images")
        self.status.setObjectName("statusLabel")
        self.status.setStyleSheet("""
            color: #b8c0b4;
            font-size: 13px;
            padding: 8px 12px;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 8px;
        """)
        self.status.setWordWrap(True)
        left_layout.addWidget(self.status)
        
        left_layout.addStretch()
        content.addWidget(left_panel)
        
        # Right panel - Results
        right_panel = QWidget()
        right_panel.setObjectName("resultsPanel")
        right_panel.setStyleSheet("background: transparent;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)
        
        # Results header
        results_header = QWidget()
        results_header.setStyleSheet("background: transparent;")
        results_header_layout = QHBoxLayout(results_header)
        results_header_layout.setContentsMargins(0, 0, 0, 0)
        
        self.results_title = QLabel("Results")
        self.results_title.setObjectName("resultsTitle")
        self.results_title.setStyleSheet("font-size: 20px; font-weight: 700; color: #f7f4ec;")
        results_header_layout.addWidget(self.results_title)
        results_header_layout.addStretch()
        
        self.result_count = QLabel("")
        self.result_count.setObjectName("resultCount")
        self.result_count.setStyleSheet("color: #b8c0b4; font-size: 14px;")
        results_header_layout.addWidget(self.result_count)
        
        right_layout.addWidget(results_header)
        
        # Scroll area for results
        self.scroll = QScrollArea()
        self.scroll.setObjectName("resultsScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.verticalScrollBar().valueChanged.connect(self.on_scroll)
        self.scroll.setStyleSheet("""
            QScrollArea#resultsScroll {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                width: 6px;
                background: transparent;
                margin: 4px;
            }
            QScrollBar::handle:vertical {
                border-radius: 3px;
                background: rgba(255, 255, 255, 0.2);
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(255, 255, 255, 0.3);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
        """)
        
        self.results_widget = QWidget()
        self.results_widget.setObjectName("resultsWidget")
        self.results_widget.setStyleSheet("background: transparent;")
        self.results_layout = QVBoxLayout(self.results_widget)
        self.results_layout.setContentsMargins(0, 0, 8, 0)
        self.results_layout.setSpacing(12)
        self.results_layout.addStretch()
        
        self.scroll.setWidget(self.results_widget)
        self.scroll.hide()
        
        right_layout.addWidget(self.scroll)
        
        # Empty state
        self.empty_state = QLabel("🎵\nSearch results will appear here")
        self.empty_state.setObjectName("emptyState")
        self.empty_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_state.setStyleSheet("""
            color: #b8c0b4;
            font-size: 16px;
            padding: 40px;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        """)
        right_layout.addWidget(self.empty_state)
        
        content.addWidget(right_panel, 1)
        root.addLayout(content, 1)
        
        return page
    
    def modern_stylesheet(self) -> str:
        return """
            QLabel { color: #f7f4ec; }
            QWidget { background: transparent; }
        """
    
    def setup_animations(self) -> None:
        # Loading page opacity
        self.loading_opacity = QGraphicsOpacityEffect(self.loading_page)
        self.loading_page.setGraphicsEffect(self.loading_opacity)
        self.loading_opacity.setOpacity(1.0)
        
        # Main page opacity
        self.main_opacity = QGraphicsOpacityEffect(self.main_page)
        self.main_page.setGraphicsEffect(self.main_opacity)
        self.main_opacity.setOpacity(0.0)
        
        # Fade out loading
        self.fade_out = QPropertyAnimation(self.loading_opacity, b"opacity")
        self.fade_out.setDuration(800)
        self.fade_out.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self.fade_out.finished.connect(self.show_main_page)
        
        # Fade in main
        self.fade_in = QPropertyAnimation(self.main_opacity, b"opacity")
        self.fade_in.setDuration(600)
        self.fade_in.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        # Slide animation for main content
        self.slide_animation = QPropertyAnimation(self.main_page, b"pos")
        self.slide_animation.setDuration(600)
        self.slide_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
    
    def transition_to_main(self) -> None:
        if hasattr(self, 'dot_timer'):
            self.dot_timer.stop()
        
        self.fade_out.stop()
        self.fade_out.setStartValue(self.loading_opacity.opacity())
        self.fade_out.setEndValue(0.0)
        self.fade_out.start()
    
    def show_main_page(self) -> None:
        self.stack.setCurrentWidget(self.main_page)
        self.main_opacity.setOpacity(0.0)
        
        # Slide from bottom
        pos = self.main_page.pos()
        self.slide_animation.setStartValue(QPoint(pos.x(), pos.y() + 50))
        self.slide_animation.setEndValue(pos)
        
        # Start animations
        self.fade_in.stop()
        self.fade_in.setStartValue(0.0)
        self.fade_in.setEndValue(1.0)
        
        # Parallel animation group for smoother transition
        self.transition_group = QParallelAnimationGroup()
        self.transition_group.addAnimation(self.fade_in)
        self.transition_group.addAnimation(self.slide_animation)
        self.transition_group.start()
        self.background.set_animation_enabled(True)
        QTimer.singleShot(1800, self.check_backend_status)

    def check_backend_status(self) -> None:
        if self.status_worker is not None and self.status_worker.isRunning():
            return

        self.status_worker = StatusWorker(self.backend_url)
        self.status_worker.finished.connect(self.render_backend_status)
        self.status_worker.failed.connect(self.show_backend_unreachable)
        self.status_worker.start()

    def render_backend_status(self, payload: dict) -> None:
        self.backend_status_failures = 0
        self.backend_ready = bool(payload.get("ok", False))
        messages: list[str] = []
        qdrant = payload.get("qdrant") or {}
        jaeger = payload.get("jaeger") or {}

        if not qdrant.get("ok", False):
            messages.append(str(qdrant.get("message") or "Qdrant is not reachable."))
        if jaeger.get("enabled") and not jaeger.get("ok", False):
            messages.append(str(jaeger.get("message") or "Jaeger UI is not reachable."))

        if messages:
            self.show_notification("warning", "  ".join(messages))
            QTimer.singleShot(10000, self.check_backend_status)
        else:
            self.show_notification("ok", "Services are ready: Qdrant is online" + (" and Jaeger is visible." if jaeger.get("enabled") else "."))

    def show_backend_unreachable(self, message: str) -> None:
        self.backend_status_failures += 1
        if self.backend_ready:
            QTimer.singleShot(5000, self.check_backend_status)
            return
        if self.backend_status_failures < 4:
            self.status.setText("Backend is warming up...")
            QTimer.singleShot(2000, self.check_backend_status)
            return
        self.show_notification(
            "error",
            (
                f"Reflectra backend is not reachable at {self.backend_url}. "
                "Start the app with reflectra-gui, or run reflectra-gui --server-only "
                f"for this URL. {message}"
            ),
        )
        QTimer.singleShot(3000, self.check_backend_status)

    def show_notification(self, kind: str, message: str) -> None:
        colors = {
            "ok": ("#54e2ca", "rgba(84, 226, 202, 0.12)", "rgba(84, 226, 202, 0.28)"),
            "warning": ("#ffd45e", "rgba(255, 212, 94, 0.13)", "rgba(255, 212, 94, 0.32)"),
            "error": ("#ff6b6b", "rgba(255, 107, 107, 0.13)", "rgba(255, 107, 107, 0.32)"),
        }
        fg, bg, border = colors.get(kind, colors["warning"])
        self.notification_text.setText(message)
        self.notification.setStyleSheet(
            f"""
            QFrame#notification {{
                border: 1px solid {border};
                border-radius: 12px;
                background: {bg};
            }}
            QFrame#notification QLabel {{
                color: {fg};
            }}
            """
        )
        self.notification.show()
    
    def choose_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Image",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if not file_path:
            return
        
        self.selected_image = Path(file_path)
        self.set_selected_image(self.selected_image)
    
    def set_selected_image(self, path: Path) -> None:
        self.selected_image = path
        self.background.set_image(path)
        self.drop_label.setText(f"📸 {path.name}\nClick search to find music")
        self.status.setText("🎵 Ready to search for matching songs")
    
    def search(self) -> None:
        if self.selected_image is None:
            QMessageBox.warning(
                self,
                "Reflectra",
                "Please choose an image first to search for matching songs."
            )
            return
        
        self.status.setText("🔍 Searching for matching songs...")
        self.search_button.setEnabled(False)
        self.choose_button.setEnabled(False)
        self.empty_state.hide()
        
        # Show loading in results area
        self.scroll.show()
        self.clear_results()
        loading_widget = QLabel("⏳ Searching...")
        loading_widget.setStyleSheet("""
            color: #b8c0b4;
            font-size: 16px;
            padding: 40px;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 16px;
        """)
        loading_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.results_layout.insertWidget(0, loading_widget)
        
        self.worker = SearchWorker(self.backend_url, self.selected_image)
        self.worker.finished.connect(self.render_results)
        self.worker.failed.connect(self.show_error)
        self.worker.start()
    
    def render_results(self, payload: dict) -> None:
        self.backend_ready = True
        self.backend_status_failures = 0
        self.clear_results()
        results = payload.get("results", [])
        
        if not results:
            empty = QLabel("🎵\nNo matching songs found.\nTry a different image!")
            empty.setStyleSheet("""
                color: #b8c0b4;
                font-size: 16px;
                padding: 40px;
                background: rgba(255, 255, 255, 0.02);
                border-radius: 16px;
                border: 1px solid rgba(255, 255, 255, 0.05);
            """)
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_layout.addWidget(empty)
            self.result_count.setText("No results")
        else:
            for i, item in enumerate(results):
                card = ModernResultCard(item, i)
                card.download_requested.connect(self.handle_audio_request)
                card.stop_requested.connect(self.stop_audio)
                card.seek_requested.connect(self.handle_audio_seek)
                self.result_cards[i] = card
                self.results_layout.insertWidget(
                    self.results_layout.count() - 1,
                    card
                )
            self.result_count.setText(f"{len(results)} results")
        
        self.status.setText(f"✅ Found {len(results)} matching song{'s' if len(results) != 1 else ''}")
        self.search_button.setEnabled(True)
        self.choose_button.setEnabled(True)
        self.scroll.show()
        self.scroll.verticalScrollBar().setValue(0)
    
    def show_error(self, message: str) -> None:
        self.status.setText("❌ Search failed")
        self.search_button.setEnabled(True)
        self.choose_button.setEnabled(True)
        self.show_notification("error", f"Search failed. {message}")
        
        # Show error in results
        self.clear_results()
        error_widget = QLabel(f"⚠️ Error: {message}")
        error_widget.setStyleSheet("""
            color: #ff6b6b;
            font-size: 14px;
            padding: 20px;
            background: rgba(255, 50, 50, 0.1);
            border-radius: 12px;
            border: 1px solid rgba(255, 50, 50, 0.2);
        """)
        error_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        error_widget.setWordWrap(True)
        self.results_layout.insertWidget(0, error_widget)
        
        QMessageBox.critical(self, "Reflectra", f"Search failed: {message}")
    
    def clear_results(self) -> None:
        self.result_cards.clear()
        while self.results_layout.count() > 0:
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.results_layout.addStretch()

    def audio_cache_key(self, item: dict) -> str:
        payload = item.get("payload") or item
        if not isinstance(payload, dict):
            return str(id(item))
        return "|".join(
            [
                str(payload.get("source_dataset", "")),
                str(payload.get("dataset_id", "")),
                str(payload.get("audio_id", "")),
            ]
        )

    def handle_audio_request(self, index: int, item: dict, play_after: bool) -> None:
        key = self.audio_cache_key(item)
        cached_path = self.downloaded_audio.get(key)
        if cached_path is not None and cached_path.exists():
            if play_after:
                self.play_audio(cached_path, index=index)
            else:
                self.show_notification("ok", f"Audio already saved: {cached_path}")
            return

        if index in self.audio_workers and self.audio_workers[index].isRunning():
            self.show_notification("warning", "Audio is already downloading for this result.")
            return

        card = self.result_cards.get(index)
        if card is not None:
            card.set_busy(True, play_after=play_after)

        self.status.setText("Downloading audio from the source dataset...")
        self.canceled_audio_downloads.discard(index)
        worker = AudioDownloadWorker(index, item, self.audio_output_dir, play_after)
        worker.finished.connect(lambda done_index, path, should_play, key=key: self.audio_download_finished(done_index, path, should_play, key))
        worker.failed.connect(self.audio_download_failed)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        self.audio_workers[index] = worker
        worker.start()

    def audio_download_finished(self, index: int, path: str, play_after: bool, key: str) -> None:
        audio_path = Path(path)
        self.downloaded_audio[key] = audio_path
        start_ratio = self.pending_start_ratios.pop(key, 0.0)
        self.audio_workers.pop(index, None)
        canceled = index in self.canceled_audio_downloads
        self.canceled_audio_downloads.discard(index)
        card = self.result_cards.get(index)
        if card is not None:
            card.set_busy(False)
            card.set_downloaded()
            if canceled:
                card.set_playing(False)
        self.status.setText(f"Audio saved: {audio_path.name}")
        self.show_notification("ok", f"Audio saved to {audio_path}")
        if play_after and not canceled:
            self.play_audio(audio_path, index=index, start_ratio=start_ratio)

    def audio_download_failed(self, index: int, message: str) -> None:
        self.audio_workers.pop(index, None)
        card = self.result_cards.get(index)
        if card is not None:
            card.set_busy(False)
        self.status.setText("Audio download failed")
        self.show_notification("error", f"Audio download failed. {message}")

    def handle_audio_seek(self, index: int, item: dict, ratio: float) -> None:
        key = self.audio_cache_key(item)
        cached_path = self.downloaded_audio.get(key)
        if cached_path is not None and cached_path.exists():
            self.play_audio(cached_path, index=index, start_ratio=ratio)
            return

        self.pending_start_ratios[key] = ratio
        self.handle_audio_request(index, item, True)

    def play_audio(self, path: Path, index: int | None = None, start_ratio: float = 0.0) -> None:
        self.stop_current_card_animation()
        if self.setup_audio_player() and self.media_player is not None:
            self.current_audio_index = index
            self.pending_start_ratio = min(max(start_ratio, 0.0), 1.0)
            self.media_player.setSource(QUrl.fromLocalFile(str(path.resolve())))
            self.apply_pending_start_position()
            self.media_player.play()
            if index is not None and index in self.result_cards:
                self.result_cards[index].set_playing(True)
            self.status.setText(f"Playing: {path.name}")
            return

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))
        self.status.setText(f"Opened audio: {path.name}")

    def stop_audio(self, index: int | None = None) -> None:
        if index is not None and index in self.audio_workers:
            self.canceled_audio_downloads.add(index)
            card = self.result_cards.get(index)
            if card is not None:
                self.pending_start_ratios.pop(self.audio_cache_key(card.item), None)
            if card is not None:
                card.set_busy(False)
                card.set_playing(False)
            self.status.setText("Playback start canceled")
            return

        if self.media_player is not None:
            self.media_player.stop()
        target_index = self.current_audio_index if index is None else index
        if target_index is not None and target_index in self.result_cards:
            self.result_cards[target_index].set_playing(False)
        if index is None or index == self.current_audio_index:
            self.current_audio_index = None
        self.status.setText("Playback stopped")

    def stop_current_card_animation(self) -> None:
        if self.current_audio_index is not None and self.current_audio_index in self.result_cards:
            self.result_cards[self.current_audio_index].set_playing(False)

    def apply_pending_start_position(self, status=None) -> None:
        if self.media_player is None or self.pending_start_ratio <= 0:
            return
        if status is not None and MEDIA_STATUS_LOADED is not None and status != MEDIA_STATUS_LOADED:
            return
        duration = int(self.media_player.duration() or 0)
        if duration <= 0:
            return
        self.media_player.setPosition(int(duration * self.pending_start_ratio))
        self.pending_start_ratio = 0.0

    def update_playback_position(self, _value: int | None = None) -> None:
        if self.media_player is None or self.current_audio_index is None:
            return
        duration = int(self.media_player.duration() or 0)
        if duration <= 0:
            return
        position = int(self.media_player.position() or 0)
        card = self.result_cards.get(self.current_audio_index)
        if card is not None:
            card.set_playback_progress(position / duration)
    
    def on_scroll(self, value: int) -> None:
        if hasattr(self, 'background'):
            self.background.set_blur(self.scroll.isVisible() and value > 24)
    
    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event) -> None:  # noqa: N802 - Qt API
        urls = event.mimeData().urls()
        if not urls:
            return
        path = Path(urls[0].toLocalFile())
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            QMessageBox.warning(self, "Reflectra", "Please drop an image file.")
            return
        self.set_selected_image(path)
    
    def closeEvent(self, event) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.quit()
            self.worker.wait()
        for worker in list(self.audio_workers.values()):
            if worker.isRunning():
                worker.quit()
                worker.wait()
        event.accept()
    
    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.close()
        super().keyPressEvent(event)


def resolve_logo_path(gui_dir: Path) -> Path:
    for candidate_root in [gui_dir, *gui_dir.parents]:
        logo_path = candidate_root / "assets" / "logo.png"
        if logo_path.exists():
            return logo_path
    return gui_dir.parent / "assets" / "logo.png"


def run_app(backend_url: str, gui_dir: Path) -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Reflectra")
    app.setStyle("Fusion")
    
    # Set dark palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(10, 14, 18))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(247, 244, 236))
    app.setPalette(palette)
    
    window = ReflectraWindow(backend_url=backend_url, gui_dir=gui_dir)
    app.main_window = window
    window.show()
    
    # Transition after 2 seconds
    QTimer.singleShot(2000, window.transition_to_main)
    
    return app.exec()
