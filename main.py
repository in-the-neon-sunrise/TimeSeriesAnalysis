import sys
from PySide6.QtWidgets import QApplication
from app import Application
from ui.combobox_wheel_blocker import ComboBoxWheelBlocker

def main():
    app = QApplication(sys.argv)
    combo_wheel_blocker = ComboBoxWheelBlocker()
    app.installEventFilter(combo_wheel_blocker)
    application = Application()
    application.run()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()