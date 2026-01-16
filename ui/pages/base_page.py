from PySide6.QtWidgets import QWidget

class BasePage(QWidget):
    def on_enter(self):
        """Вызывается при переходе на страницу"""
        pass

    def on_leave(self):
        """Вызывается при уходе со страницы"""
        pass