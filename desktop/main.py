"""
DocFlow — PySide6 Desktop Application
Drag & drop document conversion with real-time progress, OCR preview, and batch export.
"""
from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

from PySide6.QtCore import (
    QMimeData, QObject, QRunnable, QThread, QThreadPool, Qt, Signal, Slot
)
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QFont, QPalette
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QSplitter, QStatusBar, QTabWidget,
    QTextEdit, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)


# ── Dark Palette ──────────────────────────────────────────────────────────────
def apply_dark_palette(app: QApplication) -> None:
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(10, 14, 26))
    palette.setColor(QPalette.WindowText, QColor(226, 232, 240))
    palette.setColor(QPalette.Base, QColor(17, 24, 39))
    palette.setColor(QPalette.AlternateBase, QColor(26, 34, 51))
    palette.setColor(QPalette.ToolTipBase, QColor(0, 212, 255))
    palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
    palette.setColor(QPalette.Text, QColor(226, 232, 240))
    palette.setColor(QPalette.Button, QColor(17, 24, 39))
    palette.setColor(QPalette.ButtonText, QColor(226, 232, 240))
    palette.setColor(QPalette.Highlight, QColor(0, 212, 255))
    palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)


# ── Worker Signal ─────────────────────────────────────────────────────────────
class WorkerSignals(QObject):
    progress = Signal(str, int)   # (filename, percent)
    finished = Signal(str, dict)  # (filename, result_dict)
    error = Signal(str, str)      # (filename, error_msg)


class ProcessWorker(QRunnable):
    """Background worker thread for document processing."""

    def __init__(self, file_path: Path, options: dict):
        super().__init__()
        self.file_path = file_path
        self.options = options
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            self.signals.progress.emit(self.file_path.name, 10)

            # Import pipeline
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from core.pipeline import DocFlowPipeline, ProcessingOptions

            pipeline = DocFlowPipeline()
            opts = ProcessingOptions(**self.options)

            self.signals.progress.emit(self.file_path.name, 30)

            result = asyncio.run(pipeline.process(self.file_path, opts))

            self.signals.progress.emit(self.file_path.name, 90)
            self.signals.finished.emit(self.file_path.name, result.to_dict())
            self.signals.progress.emit(self.file_path.name, 100)
        except Exception as exc:
            self.signals.error.emit(self.file_path.name, str(exc))


# ── Drop Zone ─────────────────────────────────────────────────────────────────
class DropZone(QFrame):
    files_dropped = Signal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setMinimumHeight(120)
        self.setFrameStyle(QFrame.Box | QFrame.Rounded)
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #00d4ff;
                border-radius: 12px;
                background: #111827;
                color: #64748b;
                font-size: 14px;
            }
            QFrame:hover { border-color: #ff6b35; background: #1a2233; }
        """)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        label = QLabel("🗂  Drop files here  —  or click to browse")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: #64748b; font-size: 14px; border: none;")
        layout.addWidget(label)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls()]
        files = []
        for p in paths:
            if p.is_file():
                files.append(p)
            elif p.is_dir():
                files.extend(p.rglob("*.*"))
        self.files_dropped.emit(files)

    def mousePressEvent(self, event):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.ExistingFiles)
        if dialog.exec():
            self.files_dropped.emit([Path(f) for f in dialog.selectedFiles()])


# ── Main Window ───────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 DocFlow — Document Intelligence")
        self.resize(1200, 800)
        self.thread_pool = QThreadPool()
        self.results: dict[str, dict] = {}
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Title
        title = QLabel("🧠 DocFlow")
        title.setStyleSheet("font-size: 24px; font-weight: 700; color: #00d4ff;")
        layout.addWidget(title)

        # Drop zone
        self.drop_zone = DropZone()
        self.drop_zone.files_dropped.connect(self._on_files_dropped)
        layout.addWidget(self.drop_zone)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setStyleSheet("QProgressBar::chunk { background: #00d4ff; }")
        layout.addWidget(self.progress)

        # Splitter: file tree | content tabs
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, stretch=1)

        # File tree
        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabels(["File", "Status", "Chunks"])
        self.file_tree.setMinimumWidth(280)
        self.file_tree.itemClicked.connect(self._on_file_selected)
        self.file_tree.setStyleSheet("background: #111827; color: #e2e8f0; border: 1px solid #1e293b;")
        splitter.addWidget(self.file_tree)

        # Content tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabBar::tab { min-width: 100px; }")

        self.markdown_view = QTextEdit()
        self.markdown_view.setReadOnly(True)
        self.markdown_view.setFont(QFont("Courier New", 11))
        self.markdown_view.setStyleSheet("background: #0a0e1a; color: #e2e8f0;")
        self.tabs.addTab(self.markdown_view, "📝 Markdown")

        self.json_view = QTextEdit()
        self.json_view.setReadOnly(True)
        self.json_view.setFont(QFont("Courier New", 10))
        self.json_view.setStyleSheet("background: #0a0e1a; color: #10b981;")
        self.tabs.addTab(self.json_view, "📋 JSON")

        self.chunks_view = QTextEdit()
        self.chunks_view.setReadOnly(True)
        self.chunks_view.setStyleSheet("background: #0a0e1a; color: #e2e8f0;")
        self.tabs.addTab(self.chunks_view, "🧩 Chunks")

        splitter.addWidget(self.tabs)
        splitter.setSizes([280, 900])

        # Toolbar
        toolbar = QHBoxLayout()
        self.export_btn = QPushButton("⬇️ Export Markdown")
        self.export_btn.clicked.connect(self._export_markdown)
        self.export_btn.setStyleSheet("background: #00d4ff; color: #000; font-weight: bold; padding: 8px 16px; border-radius: 6px;")
        toolbar.addWidget(self.export_btn)

        self.clear_btn = QPushButton("🗑 Clear All")
        self.clear_btn.clicked.connect(self._clear_all)
        self.clear_btn.setStyleSheet("background: #374151; color: #e2e8f0; padding: 8px 16px; border-radius: 6px;")
        toolbar.addWidget(self.clear_btn)

        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Status bar
        self.statusBar().showMessage("Ready — drop files to begin")
        self.statusBar().setStyleSheet("color: #64748b;")

    def _on_files_dropped(self, files: list[Path]):
        valid_exts = {".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".txt", ".html",
                      ".md", ".json", ".png", ".jpg", ".jpeg", ".mp3", ".wav", ".zip"}
        files = [f for f in files if f.suffix.lower() in valid_exts]
        if not files:
            QMessageBox.warning(self, "No valid files", "No supported file types found.")
            return

        self.progress.setVisible(True)
        self.progress.setMaximum(len(files))
        self.progress.setValue(0)

        options = {
            "enable_ocr": True,
            "enable_summarization": False,
            "enable_embeddings": False,
            "chunk_strategy": "recursive",
            "chunk_size": 512,
        }

        for f in files:
            item = QTreeWidgetItem(self.file_tree, [f.name, "⏳ Processing...", ""])
            item.setData(0, Qt.UserRole, str(f))
            self.file_tree.addTopLevelItem(item)

            worker = ProcessWorker(f, options)
            worker.signals.finished.connect(self._on_result)
            worker.signals.error.connect(self._on_error)
            self.thread_pool.start(worker)

    def _on_result(self, filename: str, result: dict):
        self.results[filename] = result
        for i in range(self.file_tree.topLevelItemCount()):
            item = self.file_tree.topLevelItem(i)
            if item.text(0) == filename:
                if result.get("success"):
                    item.setText(1, "✅ Done")
                    item.setForeground(1, QColor("#10b981"))
                    item.setText(2, str(len(result.get("chunks", []))))
                else:
                    item.setText(1, "❌ Error")
                    item.setForeground(1, QColor("#ef4444"))
                break
        done = sum(1 for i in range(self.file_tree.topLevelItemCount())
                   if "Done" in self.file_tree.topLevelItem(i).text(1)
                   or "Error" in self.file_tree.topLevelItem(i).text(1))
        self.progress.setValue(done)
        self.statusBar().showMessage(f"Processed {done}/{self.file_tree.topLevelItemCount()} files")

    def _on_error(self, filename: str, error: str):
        self._on_result(filename, {"success": False, "errors": [error], "chunks": []})

    def _on_file_selected(self, item: QTreeWidgetItem, _):
        filename = item.text(0)
        result = self.results.get(filename, {})
        self.markdown_view.setPlainText(result.get("markdown", ""))

        import json
        self.json_view.setPlainText(json.dumps({
            "metadata": result.get("metadata", {}),
            "file_id": result.get("file_id", ""),
            "processing_time_s": result.get("processing_time_s", 0),
            "tables": len(result.get("tables", [])),
        }, indent=2))

        chunks = result.get("chunks", [])
        chunks_text = "\n\n".join(
            f"--- Chunk {c['chunk_index']+1} ({c.get('token_count', '?')} tokens) ---\n{c['text']}"
            for c in chunks
        )
        self.chunks_view.setPlainText(chunks_text)

    def _export_markdown(self):
        items = self.file_tree.selectedItems()
        if not items:
            return
        filename = items[0].text(0)
        result = self.results.get(filename, {})
        md = result.get("markdown", "")
        if not md:
            return
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Markdown", f"{filename}.md", "Markdown (*.md)")
        if save_path:
            Path(save_path).write_text(md, encoding="utf-8")
            self.statusBar().showMessage(f"Saved: {save_path}")

    def _clear_all(self):
        self.file_tree.clear()
        self.results.clear()
        self.markdown_view.clear()
        self.json_view.clear()
        self.chunks_view.clear()
        self.progress.setVisible(False)
        self.statusBar().showMessage("Cleared")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("DocFlow")
    apply_dark_palette(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
