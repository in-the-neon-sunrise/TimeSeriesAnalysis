from PySide6.QtCore import QObject, QEvent
from PySide6.QtWidgets import QComboBox


class ComboBoxWheelBlocker(QObject):

    def eventFilter(self, obj, event):
        if isinstance(obj, QComboBox) and event.type() == QEvent.Type.Wheel:
            if not obj.view().isVisible():
                event.ignore()
                return True

        return super().eventFilter(obj, event)