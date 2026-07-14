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
    Signal, QParallelAnimationGroup, QPoint, QRectF, QEvent
)
from PySide6.QtGui import (
    QColor, QIcon, QLinearGradient, QPainter, QPixmap,
    QRadialGradient, QBrush, QPen, QFont, QFontDatabase,
    QPalette, QPainterPath
)
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QGraphicsOpacityEffect,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QScrollArea, QStackedLayout, QVBoxLayout,
    QWidget, QSizePolicy
)


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


class ModernResultCard(QFrame):
    def __init__(self, item: dict, index: int) -> None:
        super().__init__()
        self.setObjectName("resultCard")
        self.index = index
        self.setFixedHeight(120)
        
        # Setup opacity for fade-in
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        self.opacity_effect.setOpacity(0.0)
        
        # Layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
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
        
        # Title with source
        title = QLabel(str(payload.get("source_dataset", "Unknown")))
        title.setObjectName("cardTitle")
        title.setStyleSheet("font-weight: 700; font-size: 16px; color: #f7f4ec;")
        title.setWordWrap(True)
        content_layout.addWidget(title)
        
        # Metadata with visual indicators
        meta_layout = QHBoxLayout()
        meta_layout.setSpacing(12)
        
        # Score indicator
        bi_score = float(item.get('bi_encoder_score') or 0)
        score_bar = self._create_score_bar(bi_score)
        meta_layout.addWidget(score_bar)
        
        # Text metadata
        meta_text = QLabel(
            f"ID: {str(payload.get('dataset_id', payload.get('audio_id', 'unknown')))}"
        )
        meta_text.setObjectName("cardMeta")
        meta_text.setStyleSheet("color: #b8c0b4; font-size: 12px;")
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
        caption = QLabel(" ".join(str(text) for text in captions[:3]) or "No captions available")
        caption.setObjectName("caption")
        caption.setStyleSheet("color: #e5e8df; font-size: 13px; line-height: 1.4;")
        caption.setWordWrap(True)
        content_layout.addWidget(caption)
        
        layout.addLayout(content_layout, 1)
        
        # Animate in
        QTimer.singleShot(50 + index * 40, self._fade_in)
    
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
    
    def _fade_in(self) -> None:
        self.fade_animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_animation.setDuration(400)
        self.fade_animation.setStartValue(0.0)
        self.fade_animation.setEndValue(1.0)
        self.fade_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.fade_animation.start()


class ReflectraWindow(QMainWindow):
    def __init__(self, backend_url: str, gui_dir: Path) -> None:
        super().__init__()
        self.backend_url = backend_url
        self.gui_dir = gui_dir
        self.logo_path = resolve_logo_path(gui_dir)
        self.selected_image: Path | None = None
        self.worker: SearchWorker | None = None
        self.status_worker: StatusWorker | None = None
        self._is_maximized = False
        self._drag_position: QPoint | None = None
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
    
    def setup_window_controls(self) -> None:
        # Title bar
        title_bar = QWidget(self.main_page)
        title_bar.setObjectName("titleBar")
        title_bar.setFixedHeight(44)
        title_bar.setStyleSheet("""
            QWidget#titleBar {
                background: transparent;
                border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            }
        """)
        title_bar_layout = QHBoxLayout(title_bar)
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
            main_layout.insertWidget(0, title_bar)
    
    def toggle_maximize(self) -> None:
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton and self._can_drag_from(event.position().toPoint()):
            self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
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
            if isinstance(child, (QPushButton, QScrollArea)):
                return False
            child = child.parentWidget()
        return True

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API
        if not isinstance(watched, QWidget) or not self._contains_widget(watched):
            return super().eventFilter(watched, event)

        if event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                position = watched.mapTo(self, event.position().toPoint())
                if self._can_drag_from(position):
                    self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
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
        QTimer.singleShot(500, self.check_backend_status)

    def check_backend_status(self) -> None:
        if self.status_worker is not None and self.status_worker.isRunning():
            return

        self.status_worker = StatusWorker(self.backend_url)
        self.status_worker.finished.connect(self.render_backend_status)
        self.status_worker.failed.connect(self.show_backend_unreachable)
        self.status_worker.start()

    def render_backend_status(self, payload: dict) -> None:
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
        self.show_notification(
            "error",
            f"Reflectra backend is still warming or unavailable. {message}",
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
                self.results_layout.insertWidget(
                    self.results_layout.count() - 1,
                    ModernResultCard(item, i)
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
        while self.results_layout.count() > 0:
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
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
