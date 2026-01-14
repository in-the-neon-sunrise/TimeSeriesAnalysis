from PySide6.QtCore import QAbstractTableModel, Qt, QModelIndex


class ProfileTableModel(QAbstractTableModel):
    headers = ["Столбец", "Тип", "Ненулевые", "Пропуски", "Мин", "Макс", "Среднее"]

    def __init__(self, profile):
        super().__init__()
        self.profile = profile

    def rowCount(self, parent=None):
        return len(self.profile)

    def columnCount(self, parent=None):
        return len(self.headers)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        if role != Qt.ItemDataRole.DisplayRole:
            return None

        item = self.profile[index.row()]
        keys = ["column", "dtype", "non_null", "nulls", "min", "max", "mean"]
        value = item[keys[index.column()]]

        if value is None:
            return "-"

        if isinstance(value, float):
            return f"{value:.4f}"

        return str(value)

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole
    ):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.headers[section]
        return None