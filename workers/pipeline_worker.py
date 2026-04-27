from __future__ import annotations
from typing import Any, Callable
from PySide6.QtCore import QObject, QRunnable, Signal, Slot

class CancelToken:
    def __init__(self):
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled


class WorkerSignals(QObject):
    progress = Signal(int, str)
    result = Signal(object)
    error = Signal(str)
    finished = Signal(bool)


class PipelineWorker(QRunnable):
    def __init__(self, fn: Callable[..., Any], *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.cancel_token = CancelToken()
        self.signals = WorkerSignals()

    def cancel(self):
        self.cancel_token.cancel()

    @Slot()
    def run(self):
        cancelled = False
        try:
            result = self.fn(
                *self.args,
                **self.kwargs,
                progress_callback=self.signals.progress,
                is_cancelled=self.cancel_token.is_cancelled,
            )
            cancelled = self.cancel_token.is_cancelled()
            if not cancelled:
                self.signals.result.emit(result)
        except Exception as exc:
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit(cancelled)