from PySide6.QtCore import QAbstractTableModel, Qt

class PandasTableModel(QAbstractTableModel):
    def __init__(self, dataframe, max_rows=10):
        super().__init__()
        self._df = dataframe.head(max_rows)

    def rowCount(self, parent=None):
        return len(self._df)

    def columnCount(self, parent=None):
        return len(self._df.columns)

    def data(self, index, role):
        if role == Qt.DisplayRole:
            value = self._df.iloc[index.row(), index.column()]
            return str(value)
        return None

    def headerData(self, section, orientation, role):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self._df.columns[section]
        return section