import math
import json
from PyQt5.QtCore import QTimer, Qt, QRectF, QSize
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QFileDialog,
    QToolBar,
    QAction,
)
from PyQt5.QtGui import QPixmap, QImage, QPainterPath, QPen, QColor
from PIL import Image

TILE_SIZE = 1024


class ScribbleItem(QGraphicsPathItem):
    def __init__(self, points, color=QColor("red"), pen_width=3, parent=None):
        super().__init__(parent)
        self.points = points  # список (x, y) в абсолютных координатах
        self.color = QColor(color)
        self.pen_width = pen_width
        self._update_path()

    def _update_path(self):
        if not self.points:
            return
        path = QPainterPath()
        path.moveTo(*self.points[0])
        for p in self.points[1:]:
            path.lineTo(*p)
        self.setPath(path)
        self.setPen(QPen(self.color, self.pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))


class Tile(QGraphicsPixmapItem):
    def __init__(self, pixmap, x, y, size):
        super().__init__(pixmap)
        self.setPos(x, y)
        self.setOffset(0, 0)
        self.setShapeMode(QGraphicsPixmapItem.BoundingRectShape)
        self.setZValue(0)


class TiledImageViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHints(self.renderHints() |
                            QPainter.Antialiasing |
                            QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        self.tiles = {}
        self.tile_cache = {}
        self.image_path = None
        self.original_size = QSize(1, 1)
        self.full_image = None

        # Scribble
        self.scribble_mode = False
        self.current_scribble_points = []
        self.scribbles = []
        self.current_color = QColor("red")
        self.scribble_preview = None

        # Ограничения масштаба
        self.min_scale = 0.5
        self.max_scale = 10.0

    def load_image(self, path):
        # Сброс сцены / тайлов
        self.scene.clear()
        self.tiles = {}
        self.tile_cache = {}
        self.image_path = path
        self.full_image = QImage(path)

        if self.full_image.isNull():
            # Попробуем через PIL (например, если путь содержит спецсимволы)
            try:
                from PIL.ImageQt import ImageQt
                pil = Image.open(path)
                qimg = ImageQt(pil).copy()
                self.full_image = qimg
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось открыть изображение: {e}")
                return

        w = self.full_image.width()
        h = self.full_image.height()
        self.original_size = QSize(w, h)

        # Устанавливаем сцену и фон
        self.setSceneRect(QRectF(0, 0, w, h))

        full_pixmap = QPixmap.fromImage(self.full_image)
        self._full_pixmap_item = QGraphicsPixmapItem(full_pixmap)
        self._full_pixmap_item.setZValue(-1)
        self.scene.addItem(self._full_pixmap_item)

        # ✅ Один раз вписываем в окно
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

        QTimer.singleShot(50, self.update_visible_tiles)


    def wheelEvent(self, event):
        """ Масштабирование колесиком мыши с ограничениями """
        if event.angleDelta().y() > 0:
            factor = 1.25
        else:
            factor = 0.8

        # текущий масштаб
        current_scale = self.transform().m11()
        new_scale = current_scale * factor

        if new_scale < self.min_scale or new_scale > self.max_scale:
            return  # игнорируем слишком сильное увеличение/уменьшение

        self.scale(factor, factor)
        # ⚠️ убираем вызов update_visible_tiles отсюда,
        # тайлы будут подгружаться автоматически при скролле/зуме
        self.update_visible_tiles()

    def update_visible_tiles(self):
        """
        Загружает тайлы, которые попадают в текущую видимую область.
        Дополнительно грузим соседние тайлы (margin) чтобы не было пробелов при прокрутке.
        Защищаем расчёты границами изображения.
        """
        if not self.image_path or self.full_image is None:
            return

        view_rect = self.mapToScene(self.viewport().rect()).boundingRect()

        # Если видимая область пустая (иногда бывает при инициализации), используем всю сцену
        if view_rect.width() <= 0 or view_rect.height() <= 0:
            view_rect = self.scene.sceneRect()

        margin = 1  # загружаем соседние тайлы для сглаживания
        start_x = max(0, int(math.floor(view_rect.left() / TILE_SIZE)) - margin)
        end_x = min(int(math.ceil(view_rect.right() / TILE_SIZE)) + margin, 
                    max(0, math.ceil(self.original_size.width() / TILE_SIZE) - 1))
        start_y = max(0, int(math.floor(view_rect.top() / TILE_SIZE)) - margin)
        end_y = min(int(math.ceil(view_rect.bottom() / TILE_SIZE)) + margin, 
                    max(0, math.ceil(self.original_size.height() / TILE_SIZE) - 1))

        # Удаляем старые тайлы, которые сейчас вне расширенной области (чтобы не расти бесконечно)
        needed = set((i, j) for i in range(start_x, end_x + 1) for j in range(start_y, end_y + 1))
        for key in list(self.tiles.keys()):
            if key not in needed:
                try:
                    self.scene.removeItem(self.tiles[key])
                except Exception:
                    pass
                del self.tiles[key]

        # Загружаем требуемые тайлы
        for i in range(start_x, end_x + 1):
            for j in range(start_y, end_y + 1):
                if (i, j) not in self.tiles:
                    self.load_tile(i, j)

    def load_tile(self, i, j):
        """ Загружаем тайлы поверх фона, но под скриблами """
        if self.full_image is None:
            return

        max_i = math.ceil(self.original_size.width() / TILE_SIZE) - 1
        max_j = math.ceil(self.original_size.height() / TILE_SIZE) - 1
        if i < 0 or j < 0 or i > max_i or j > max_j:
            return

        if (i, j) in self.tile_cache:
            pixmap = self.tile_cache[(i, j)]
        else:
            x = i * TILE_SIZE
            y = j * TILE_SIZE
            w = min(TILE_SIZE, self.original_size.width() - x)
            h = min(TILE_SIZE, self.original_size.height() - y)
            tile_qimage = self.full_image.copy(x, y, w, h)
            pixmap = QPixmap.fromImage(tile_qimage)
            self.tile_cache[(i, j)] = pixmap

        item = Tile(pixmap, i * TILE_SIZE, j * TILE_SIZE, TILE_SIZE)
        item.setZValue(0)  # ✅ ниже скриблов
        self.scene.addItem(item)
        self.tiles[(i, j)] = item

    # ===== Scribble =====
    def set_scribble_mode(self, enabled: bool):
        self.scribble_mode = enabled

    def set_scribble_color(self, color: QColor):
        self.current_color = color

    def mousePressEvent(self, event):
        if self.scribble_mode and event.button() == Qt.LeftButton:
            pos_scene = self.mapToScene(event.pos())
            self.current_scribble_points = [(pos_scene.x(), pos_scene.y())]
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.scribble_mode and event.buttons() & Qt.LeftButton:
            pos_scene = self.mapToScene(event.pos())
            self.current_scribble_points.append((pos_scene.x(), pos_scene.y()))

            if self.scribble_preview:
                self.scene.removeItem(self.scribble_preview)
            self.scribble_preview = ScribbleItem(self.current_scribble_points, self.current_color)
            self.scene.addItem(self.scribble_preview)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.scribble_mode and event.button() == Qt.LeftButton:
            if self.current_scribble_points:
                item = ScribbleItem(self.current_scribble_points, self.current_color)
                item.setZValue(1)  # ✅ всегда поверх тайлов
                self.scene.addItem(item)
                self.scribbles.append(item)
            self.current_scribble_points = []
            if self.scribble_preview:
                self.scene.removeItem(self.scribble_preview)
                self.scribble_preview = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def export_scribbles(self, filepath):
        data = []
        for s in self.scribbles:
            points_norm = [(x / self.original_size.width(), y / self.original_size.height())
                           for x, y in s.points]
            data.append({
                "color": s.color.name(),
                "points": points_norm
            })
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def import_scribbles(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            points = [(x * self.original_size.width(), y * self.original_size.height())
                      for x, y in entry["points"]]
            item = ScribbleItem(points, entry["color"])
            item.setZValue(1)  # ✅ поверх тайлов
            self.scene.addItem(item)
            self.scribbles.append(item)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.viewer = TiledImageViewer()
        self.setCentralWidget(self.viewer)
        self.create_toolbar()

    def create_toolbar(self):
        toolbar = QToolBar("Main")
        self.addToolBar(toolbar)

        open_action = QAction("Открыть", self)
        open_action.triggered.connect(self.open_image)
        toolbar.addAction(open_action)

        scribble_action = QAction("Scribble", self)
        scribble_action.setCheckable(True)
        scribble_action.triggered.connect(
            lambda checked: self.viewer.set_scribble_mode(checked))
        toolbar.addAction(scribble_action)

        save_scribble_action = QAction("Сохранить метки", self)
        save_scribble_action.triggered.connect(self.save_scribbles)
        toolbar.addAction(save_scribble_action)

        load_scribble_action = QAction("Загрузить метки", self)
        load_scribble_action.triggered.connect(self.load_scribbles)
        toolbar.addAction(load_scribble_action)

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть изображение", "", "Images (*.png *.jpg *.jpeg *.tif *.bmp)")
        if path:
            self.viewer.load_image(path)

    def save_scribbles(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить метки", "", "JSON (*.json)")
        if path:
            self.viewer.export_scribbles(path)

    def load_scribbles(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить метки", "", "JSON (*.json)")
        if path:
            self.viewer.import_scribbles(path)


if __name__ == "__main__":
    app = QApplication([])
    w = MainWindow()
    w.show()
    app.exec_()
