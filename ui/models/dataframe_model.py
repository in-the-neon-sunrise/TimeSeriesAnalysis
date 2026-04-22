from PySide6.QtCore import QAbstractTableModel, Qt
from numbers import Real


class DataFrameModel(QAbstractTableModel):

    def __init__(self, df):
        super().__init__()
        self._df = df

    def rowCount(self, parent=None):
        return self._df.shape[0]

    def columnCount(self, parent=None):
        return self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            value = self._df.iloc[index.row(), index.column()]
            if isinstance(value, Real):
                return str(round(value, 4))
            return str(value)
        return None

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self._df.columns[section]
            if orientation == Qt.Vertical:
                return str(section)
