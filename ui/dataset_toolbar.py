from __future__ import annotations

from dataclasses import dataclass
from typing import List

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QLineEdit, QWidget


@dataclass
class DatasetOption:
    key: str
    title: str


class DatasetToolbarWidget(QWidget):
    selection_changed = Signal(str)
    output_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(6, 0, 6, 0)
        row.setSpacing(6)

        row.addWidget(QLabel("Входной файл:"))
        self.input_combo = QComboBox()
        self.input_combo.setMinimumWidth(220)
        self.input_combo.currentIndexChanged.connect(self._emit_selection_changed)
        row.addWidget(self.input_combo)

        row.addWidget(QLabel("→"))
        row.addWidget(QLabel("Выходной файл:"))
        self.output_edit = QLineEdit()
        self.output_edit.setMinimumWidth(220)
        self.output_edit.textEdited.connect(self.output_changed)
        row.addWidget(self.output_edit)

    def set_options(self, options: List[DatasetOption], selected_key: str | None):
        self.input_combo.blockSignals(True)
        self.input_combo.clear()
        for opt in options:
            self.input_combo.addItem(opt.title, userData=opt.key)

        if options:
            idx = 0
            if selected_key is not None:
                found = self.input_combo.findData(selected_key)
                if found >= 0:
                    idx = found
            self.input_combo.setCurrentIndex(idx)

        self.input_combo.blockSignals(False)
        self._emit_selection_changed()

    def set_output_text(self, text: str):
        self.output_edit.blockSignals(True)
        self.output_edit.setText(text)
        self.output_edit.blockSignals(False)

    def current_input_key(self) -> str | None:
        return self.input_combo.currentData()

    def current_output_name(self) -> str:
        return self.output_edit.text().strip()

    def _emit_selection_changed(self):
        key = self.current_input_key()
        if key:
            self.selection_changed.emit(key)