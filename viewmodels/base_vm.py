from PySide6.QtCore import QObject, Signal

class BaseViewModel(QObject):
    error_occurred = Signal(str)
    info_changed = Signal(str)

    def __init__(self):
        super().__init__()