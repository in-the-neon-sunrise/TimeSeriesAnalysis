from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtWidgets import QApplication

class ScrollFriendlyCanvas(FigureCanvasQTAgg):
    def wheelEvent(self, event):
        # Передаём событие родителю (ScrollArea)
        QApplication.sendEvent(self.parent(), event)