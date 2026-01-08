import sys
from PySide6.QtWidgets import QApplication
from app import Application

def main():
    app = QApplication(sys.argv)
    application = Application()
    application.run()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()