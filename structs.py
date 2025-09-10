from abc import ABC
from dataclasses import dataclass, field

import datetime
from pathlib import Path
from typing import Optional, Dict, Set, Tuple, List
import numpy as np

import cv2

import shapely
from skimage.segmentation import slic, felzenszwalb, watershed
from skimage.measure import regionprops
from skimage.filters import sobel
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from scipy.signal import convolve2d
from shapely.geometry import Polygon
from skimage.segmentation import mark_boundaries, find_boundaries
import skimage
import matplotlib.pyplot as plt
from PIL import Image, ImageTk, ImageDraw
import json
import copy
from pathlib import Path
from numba import njit, prange
from scipy import ndimage

@dataclass
class ScribbleParams:
    radius: float
    code: int 
    def dict_to_save(self) -> Dict:
        res = dict()
        res["radius"] = float(self.radius)
        res["code"] = int(self.code) # now: type ind in ordered dict
        return res

@dataclass
class Scribble:
    id: int
    points: np.ndarray  # [n x 2] float array of absolute points
    params: ScribbleParams  # parameters of the scribble
    creation_time: datetime = None

    def __init__(self, id, points, params, creation_time = None):
        self.id = int(id)
        self.points = points
        self.params = params
        self.creation_time = creation_time
        self.creation_time = datetime.datetime.now()
        self._bbox = (
            None
            if not isinstance(self.points, np.ndarray) or self.points.size == 0
            else (
                min(self.points[:, 0]),
                min(self.points[:, 1]),
                max(self.points[:, 0]),
                max(self.points[:, 1])
            )
        )

    def __len__(self):
        return len(self.points)

    
    def dict_to_save(self):
        res = dict()
        res["id"] = int(self.id)
        res["points"] = self.points.tolist()
        res["params"] = self.params.dict_to_save()
        if self.creation_time is not None:
            res["creation_time"] = self.creation_time.isoformat()
        else:
            res["creation_time"] = ""
        return res
    
    def set_from_dict(self, loaded_dict):
        self.id = int(loaded_dict["id"])
        self.points = np.array(loaded_dict["points"], dtype=np.float32)
        self.params = ScribbleParams(loaded_dict["params"]["radius"], loaded_dict["params"]["code"])
        #self.creation_time = datetime.time.fromisoformat(loaded_dict["creation_time"])
        self.__post_init__()

    @property
    def bbox(self):
        return self._bbox
    
    def __hash__(self):
        return hash(self.points.sum()) + hash(self.creation_time)

    def __eq__(self, other):
        return np.array_equal(self.points, other.points)



@dataclass
class SuperPixel:
    id: int
    method: str  # method used to create the superpixel (e.g. 'SLIC_p1_p2_p3')
    border: np.ndarray  # [n x 2] float array of border absolute points
    parents: Optional[list[int]]  # list of ids of parent scribbles
    props: Optional[np.ndarray] # [6] float array for region props

    def __hash__(self):
        return hash(self.method + f"_{self.id}")

    def __eq__(self, other):
        return (self.method + f"_{self.id}") == (other.method + f"_{other.id}")

    def dict_to_save(self):
        res = dict()
        res["id"] = int(self.id)
        res["method"] = self.method
        res["border"] = [[round(float(y), 7) for y in x] for x in self.border]
        res["parents"] = [] if self.parents is None else self.parents
        res["props"] = [] if self.props is None else [float(i) for i in self.props]
        return res
    
    def set_from_dict(self, loaded_dict):
        self.id = int(loaded_dict["id"])
        self.method = loaded_dict["method"]
        self.border = np.array(loaded_dict["border"], dtype=np.float32)
        self.parents = loaded_dict["parents"]



@dataclass
class AnnotationInstance:
    id: int
    code: int
    border: np.ndarray  # [n x 2] float array of border absolute points
    parent_superpixel: int = -1
    parent_scribble: List[int] = field(default_factory=list)
    parent_intersect: Optional[int] = True

    def __hash__(self):
        return self.id

    def dict_to_save(self):
        res = dict()
        res["id"] = int(self.id)
        res["code"] = int(self.code)
        res["border"] = [[round(float(y), 7) for y in x] for x in self.border]
        res["parent_superpixel"] = int(self.parent_superpixel)
        res["parent_scribble"] = self.parent_scribble
        res["parent_intersect"] = self.parent_intersect
        return res
    
    def set_from_dict(self, loaded_dict):
        self.id = int(loaded_dict["id"])
        self.method = int(loaded_dict["method"])
        self.border = np.array(loaded_dict["border"], dtype=np.float32)


@dataclass
class ImageAnnotation:
    annotations: list[AnnotationInstance]

    def append(self, anno: AnnotationInstance):
        self.annotations.append(anno)

    def to_array(self) -> np.ndarray:
        pass

class SuperPixelMethod(ABC):
    def short_string(self) -> str:
        pass

    def __le__(self, other) -> bool:
        return self.short_string() <= other.short_string()
    
    def __ge__(self, other) -> bool:
        return self.short_string() >= other.short_string()
    
    def __lt__(self, other) -> bool:
        return self.short_string() < other.short_string()
    
    def __gt__(self, other) -> bool:
        return self.short_string() > other.short_string()

    def __hash__(self):
        return hash(self.short_string())

    def __eq__(self, other):
        return isinstance(other, SuperPixelMethod) and self.short_string() == other.short_string()

@dataclass
class WatershedSuperpixel(SuperPixelMethod):
    compactness: float
    n_components: int

    def short_string(self) -> str:
        return f"Watershed_{self.compactness:.2f}_{self.n_components}"  # Форматирование для избежания float-ошибок

    def __hash__(self):
        return hash(self.short_string())

    def __eq__(self, other):
        return isinstance(other, SuperPixelMethod) and self.short_string() == other.short_string()


@dataclass
class SLICSuperpixel(SuperPixelMethod):

    n_clusters: int
    compactness: float
    sigma: float

    def short_string(self) -> str:
        # Добавлены типы данных в строку
        return f"SLIC_{int(self.n_clusters)}_{float(self.compactness):.2f}_{float(self.sigma):.2f}"

    def __hash__(self):
        return hash(self.short_string())

    def __eq__(self, other):
        return isinstance(other, SuperPixelMethod) and self.short_string() == other.short_string()


@dataclass
class FelzenszwalbSuperpixel(SuperPixelMethod):

    min_size: int
    sigma: float
    scale: float

    def short_string(self) -> str:
        return f"Felzenszwalb_{self.min_size}_{self.sigma}_{self.scale}"

    def __hash__(self):
        return hash(self.short_string())

    def __eq__(self, other):
        return isinstance(other, SuperPixelMethod) and self.short_string() == other.short_string()


def find_sp_key_in_dict(sp_method: SuperPixelMethod, sp_dict: Dict[SuperPixelMethod, List]) -> bool:
    for key in sp_dict:
        if key.short_string() == sp_method.short_string():
            return True
    return False

def bbox_is_intersect(bbox_1: Tuple[int], bbox_2: Tuple[int]) -> bool:
    """
    bbox: (x_left, y_top, x_right, y_bottom)
    """
    x_intersect: bool = max(bbox_1[0], bbox_2[0]) <= min(bbox_1[2], bbox_2[2])
    y_intersect: bool = max(bbox_1[1], bbox_2[1]) <= min(bbox_1[3], bbox_2[3])
    return x_intersect and y_intersect

def bbox_intersect(bbox_1: Tuple[int], bbox_2: Tuple[int]) -> Tuple[int]:
    """
    bbox: (x_left, y_top, x_right, y_bottom)
    """
    res_bbox = [
        max(bbox_1[0], bbox_2[0]), max(bbox_1[1], bbox_2[1]),
        min(bbox_1[2], bbox_2[2]), min(bbox_1[3], bbox_2[3])
    ]
    return res_bbox

def simplify(polygon, tolerance = 1e-10) -> np.ndarray:
    """ Simplify a polygon with shapely.
    Polygon: ndarray
        ndarray of the polygon positions of N points with the shape (N,2)
    tolerance: float
        the tolerance
    """
    poly = shapely.geometry.Polygon(polygon)
    poly_s = poly.simplify(tolerance=tolerance)
    # convert it back to numpy
    return np.array(poly_s.boundary.coords[:], dtype=np.float32)

def check_scribbles_dont_intersect_one_region(
        scribbles: List[Scribble],
        regions: List[SuperPixel]
    ) -> bool:
    intersected_before: List[bool] = [False for _ in range(len(regions))]
    for scribble_ind in range(len(scribbles)):
        for region_ind in range(len(regions)):
            scribble_line: shapely.LineString = shapely.LineString(scribbles[scribble_ind].points)
            border: shapely.Polygon = shapely.Polygon(regions[region_ind].border)
            if border.intersects(scribble_line):
                if intersected_before[region_ind]:
                    return False
                else:
                   intersected_before[region_ind] = True 
    return True

@njit(parallel=True)
def parallel_stats_rgb(image, mask, max_label):
    """Параллельный расчет для больших RGB-изображений."""
    h, w, c = image.shape
    sums = np.zeros((max_label + 1, c), dtype=np.float64)
    sumsq = np.zeros((max_label + 1, c), dtype=np.float64)
    counts = np.zeros(max_label + 1, dtype=np.int64)
    
    for i in prange(h):
        for j in prange(w):
            label = mask[i, j]
            pixel = image[i, j]
            for k in range(c):
                val = pixel[k]
                sums[label, k] += val
                sumsq[label, k] += val ** 2
            counts[label] += 1
    
    valid = counts > 0
    means = np.zeros((np.sum(valid), c), dtype=np.float64)
    variances = np.zeros((np.sum(valid), c), dtype=np.float64)
    valid_labels = np.where(valid)[0]
    
    for idx in prange(len(valid_labels)):
        label = valid_labels[idx]
        cnt = counts[label]
        for k in prange(c):
            means[idx, k] = sums[label, k] / cnt
            variances[idx, k] = (sumsq[label, k] / cnt) - (means[idx, k] ** 2)
    
    return means, variances, valid_labels

def check_bbox_contain_scribble(polyline_points, rectangles):
    polygons = [Polygon([(a, b), (c, b), (c, d), (a, d)]) 
               for a, b, c, d in rectangles]
    tree = shapely.STRtree(polygons)
    
    # Проверяем каждый сегмент ломаной
    line = shapely.LineString(polyline_points)
    candidates = tree.query(line)
    for segment in zip(line.coords[:-1], line.coords[1:]):
        seg_line = shapely.LineString(segment)
        # Ищем пересекающие прямоугольники
        candidates = tree.query(seg_line)
        if not any(polygons[c].contains(seg_line) for c in candidates):
            return False
    return True

def find_holes(mask: np.ndarray) -> list[np.ndarray]:
    """Находит все дырки в бинарной маске."""
    inverted = ~mask
    labeled, num_features = ndimage.label(inverted)
    
    if num_features == 0:
        return []
    
    # Определяем компоненты, касающиеся границ
    borders = np.zeros_like(inverted, dtype=bool)
    borders[[0, -1], :] = True
    borders[:, [0, -1]] = True
    
    # Исправленный вызов labeled_comprehension
    touch_border = ndimage.labeled_comprehension(
        input=borders,
        labels=labeled,
        index=np.arange(1, num_features+1),
        func=np.any,
        out_dtype=bool,
        default=False
    )
    
    return [labeled == i for i in range(1, num_features+1) if not touch_border[i-1]]

def split_recursive(mask: np.ndarray) -> List[np.ndarray]:
    """Рекурсивно разделяет маску до полного устранения дырок."""
    holes = find_holes(mask)
    if not holes:
        return [mask.copy()]
    
    # Выбираем самую большую дырку для разделения
    hole_sizes = [np.count_nonzero(h) for h in holes]
    main_hole = holes[np.argmax(hole_sizes)]
    
    # Определяем направление разреза через PCA дырки
    y, x = np.where(main_hole)
    points = np.column_stack((x, y))
    centroid = np.mean(points, axis=0)
    
    perp_angle = np.pi/2
    
    # Создаем линию разреза
    yy, xx = np.indices(mask.shape)
    split_line = (xx - centroid[0])*np.cos(perp_angle) + (yy - centroid[1])*np.sin(perp_angle)
    
    # Разделяем маску
    masks = [
        mask & (split_line >= -0.5),  # + margin для перекрытия
        mask & (split_line < 0.5)
    ]
    
    # Рекурсивная обработка частей
    result = []
    for m in masks:
        m_clean = remove_small_components(m)
        if np.any(m_clean):
            result.extend(split_recursive(m_clean))
    
    return result

def remove_small_components(mask: np.ndarray, min_size=100) -> np.ndarray:
    """Удаляет мелкие компоненты и артефакты."""
    cleaned = ndimage.binary_opening(mask, structure=np.ones((3,3)))
    labeled, num_labels = ndimage.label(cleaned)
    
    if num_labels == 0:
        return np.zeros_like(mask)
    
    sizes = ndimage.sum(cleaned, labeled, range(1, num_labels+1))
    keep = sizes >= min_size
    return np.isin(labeled, np.where(keep)[0]+1)

def split_mask(mask: np.ndarray) -> List[np.ndarray]:
    """
    Возвращает список масок без дырок.
    Количество масок зависит от сложности исходной формы.
    """
    final_masks = split_recursive(mask)
    
    # Фильтрация полностью пустых масок
    return [m for m in final_masks if np.any(m)]

class SuperPixelAnnotationAlgo:

    def __init__(
        self,
        downscale_coeff: float,
        superpixel_methods: list[SuperPixelMethod],
        image_path: Path = None,
        image: Image = None
    ) -> None:
        """
        image
        """
        assert 0 < downscale_coeff <= 1
        self.image_path = image_path
        self.superpixel_methods: list[SuperPixelMethod] = superpixel_methods
        if image != None:
            self.image = image
        else:
            self.image = Image.fromarray(
                cv2.imread(str(self.image_path)).astype(np.uint8)
            )
        self._preprocess_image(downscale_coeff)
        self.superpixels: Dict[SuperPixelMethod, List[SuperPixel]] = {}
        self._annotations: Dict[SuperPixelMethod, ImageAnnotation] = {}
        # нужно для задание id в суперпикселях при доразбиении 
        self._superpixel_ind: Dict[SuperPixelMethod, int] = {}
        self._annotation_ind: Dict[SuperPixelMethod, int] = {}
        self.image_with_sp_border = {}
        self._create_superpixels()
        self.ind_scrible = 0
        self.prev_ind_scrible = 0

        self._scribbles: List[Scribble] = []

        # нужны для отмены действий сложных случаев в духе
        # [0, 1, 2] -> [0, 1] -> [0, 1, 3] -> [0] -> [0, 4]
        self.scribbles_id_sequence: List[int] = []
        # 4 числа, левый верхний угол и правый нижний угол
        #
        self.annotated_bbox: List[List[int]] = []
        self.bbox_size: int = 700
        self._property_dist = 5
        self._superpixel_radius = 0.08

    def _preprocess_image(self, downscale_coeff: float) -> None:
        sh = self.image.size
        self.image.thumbnail((int(sh[0] * downscale_coeff), int(sh[1] * downscale_coeff)), Image.Resampling.BILINEAR)
        self.image_lab = rgb2lab(np.array(self.image))

    def add_superpixel_method(self, superpixel_method: SuperPixelMethod) -> None:
        print("Recieved new superpixel method", superpixel_method.short_string())
        for exist_superpixel_method in self.superpixel_methods:
            if exist_superpixel_method.short_string() == superpixel_method.short_string():
                print("len of annos after added:", len(self._annotations))
                return
        self.superpixel_methods.append(superpixel_method)
        self._create_superpixel(superpixel_method)
        self._annotations[superpixel_method] = ImageAnnotation(annotations=[])
        print("len of annos after added:", len(self._annotations))
        print(len(self.superpixels[superpixel_method]))
        print(len(self.superpixels))

    def _exist_sp_mask(self, scribble: Scribble) -> bool:
        # res = False
        # for region in self.annotated_bbox:
        #     if region[0] <= scribble.bbox[0] and \
        #         region[1] <= scribble.bbox[1] and \
        #         region[2] > scribble.bbox[2] and \
        #         region[3] > scribble.bbox[3]:
        #         res = True
        #         break
        res = check_bbox_contain_scribble(scribble.points, self.annotated_bbox)
        return res
    
    def _create_superpixel_for_mask(self, superpixel_method, image_roi, mask, bbox):
        sp_mask = None
        if isinstance(superpixel_method, SLICSuperpixel):
            n_segm = mask.astype(np.bool_).astype(np.int32).sum() / 500
            sp_mask = slic(
                image_roi,
                n_segments=n_segm,
                compactness=superpixel_method.compactness,
                sigma=superpixel_method.sigma,
                start_label=1,
                mask=mask
            )
            # plt.imshow(mark_boundaries(np.array(self.image)[image_coord[1]:image_coord[3], image_coord[0]:image_coord[2]], sp_mask))
            # plt.show()
        elif isinstance(superpixel_method, FelzenszwalbSuperpixel):
            sp_mask = felzenszwalb(
                image_roi,
                scale=superpixel_method.scale, 
                sigma=superpixel_method.sigma, 
                min_size=superpixel_method.min_size
            )
        else:
            raise Exception("Unsupported superpixel type")
        segments = np.zeros((sp_mask.shape[0]+2, sp_mask.shape[1]+2), dtype=np.int32)
        segments -= 1
        segments[1:-1, 1:-1] = sp_mask
        # Создаем фигуру и оси
        # fig, ax = plt.subplots()
    
        # # Отображаем изображение с цветовой картой
        # im = ax.imshow(sp_mask == 0, cmap='viridis', aspect='auto')

        # # Добавляем цветовую шкалу слева
        # cbar = fig.colorbar(im, ax=ax, location='left')

        # # Настройки оформления
        # ax.set_title("Изображение с цветовой шкалой")
        # plt.show()
        
        unique_segments = sorted(np.unique(segments))[1:]
        # print(unique_segments)
        means, variances, valid_labels = parallel_stats_rgb(image_roi, sp_mask, np.max(sp_mask))
        # Loop through each unique segment
        for i, segment in enumerate(valid_labels):
            if segment <= 0:
                continue
            # Create a mask for the current segment
            binary_mask = (segments == segment).astype(np.uint8)
            # pixels = image_roi[binary_mask[1:-1, 1:-1]]
            contours = skimage.measure.find_contours(binary_mask)

            external_contour = contours[0][:, ::-1]
            # Преобразуйте контур в список координат
            polygon = (external_contour - 1).astype(np.float32)
            polygon[:, 0] /= sp_mask.shape[1]
            polygon[:, 1] /= sp_mask.shape[0]
            # print("polygon max:", polygon.max(), polygon.min())
            polygon[:, 0] *= (bbox[2] - bbox[0])
            polygon[:, 1] *= (bbox[3] - bbox[1])
            polygon[:, 0] += bbox[0]
            polygon[:, 1] += bbox[1]
            
            region_props = []
            region_props.extend(means[i])
            region_props.extend(variances[i])
            # print("region_props", region_props)
            self.superpixels[superpixel_method].append(
                SuperPixel(
                    id=self._superpixel_ind[superpixel_method],
                    method=superpixel_method.short_string(), 
                    border=np.around(simplify(polygon[::]), decimals=7), #len(polygon) // 20 if len(polygon) > 20 else 1
                    parents=None,
                    props=np.array(region_props)
                )
            )
            self._superpixel_ind[superpixel_method] += 1


    def _create_superpixel_for_scribble(self, 
                                        scribble: Scribble, 
                                        superpixel_method: SuperPixelMethod) -> None:
        if not find_sp_key_in_dict(superpixel_method, self._superpixel_ind):
            self._superpixel_ind[superpixel_method] = 0
        if not find_sp_key_in_dict(superpixel_method, self._annotations):
            self._annotations[superpixel_method] = ImageAnnotation(annotations = [])
        if not find_sp_key_in_dict(superpixel_method, self.superpixels):
            self.superpixels[superpixel_method] = []
        exist_sp = self._exist_sp_mask(scribble)
        if exist_sp:
            return
        bbox = (
            scribble.points[:, 0].min(), 
            scribble.points[:, 1].min(),
            scribble.points[:, 0].max(), 
            scribble.points[:, 1].max()
        )
        bbox = [
            max(bbox[0] - 1. * self.bbox_size / 2 / self.image_lab.shape[1], 0.0),
            max(bbox[1] - 1. * self.bbox_size / 2 / self.image_lab.shape[0], 0.0),
            min(bbox[2] + 1. * self.bbox_size / 2 / self.image_lab.shape[1], 1.0),
            min(bbox[3] + 1. * self.bbox_size / 2 / self.image_lab.shape[0], 1.0)
        ]
        
        image_coord = [
            round(bbox[0] * self.image_lab.shape[1]),
            round(bbox[1] * self.image_lab.shape[0]),
            round(bbox[2] * self.image_lab.shape[1]),
            round(bbox[3] * self.image_lab.shape[0])
        ]
        if image_coord[2] > self.image_lab.shape[1]:
            image_coord[2] = self.image_lab.shape[1]
        if image_coord[3] > self.image_lab.shape[0]:
            image_coord[3] = self.image_lab.shape[0]
        image_roi = self.image_lab[image_coord[1]:image_coord[3], image_coord[0]:image_coord[2]]
        mask = np.ones((image_roi.shape[0], image_roi.shape[1]), dtype=np.bool_)
        # plt.imshow(np.array(self.image)[image_coord[1]:image_coord[3], image_coord[0]:image_coord[2]])
        # plt.show()
        for existed_bbox in self.annotated_bbox:
            if not bbox_is_intersect(existed_bbox, bbox):
                continue
            intersected_bbox = list(bbox_intersect(existed_bbox, bbox))
            
            intersected_bbox[0] -= bbox[0]
            intersected_bbox[1] -= bbox[1]
            intersected_bbox[2] -= bbox[0]
            intersected_bbox[3] -= bbox[1]

            # intersected_bbox[2] /= (bbox[2] - bbox[0])
            # intersected_bbox[3] /= (bbox[3] - bbox[1])
            # здесь мы растянули пересечение от 0 до 1, чтобы можно было изменить маску

            intersected_bbox[0] *= self.image_lab.shape[1]
            intersected_bbox[1] *= self.image_lab.shape[0]
            intersected_bbox[2] *= self.image_lab.shape[1]
            intersected_bbox[3] *= self.image_lab.shape[0]
            
            intersected_bbox = [int(i) for i in intersected_bbox]
            mask[
                intersected_bbox[1]:intersected_bbox[3], 
                intersected_bbox[0]:intersected_bbox[2]
            ] = 0
        self.annotated_bbox.append(bbox)
        sp_mask = None
        if isinstance(superpixel_method, SLICSuperpixel):
            n_segm = int(mask.astype(np.bool_).astype(np.int32).sum() / 2500) + 1
            if n_segm  < 2:
                return
            sp_mask = slic(
                image_roi,
                n_segments=n_segm,
                compactness=superpixel_method.compactness,
                sigma=superpixel_method.sigma,
                start_label=1,
                mask=mask
            )
            # plt.imshow(mark_boundaries(np.array(self.image)[image_coord[1]:image_coord[3], image_coord[0]:image_coord[2]], sp_mask))
            # plt.show()
        elif isinstance(superpixel_method, FelzenszwalbSuperpixel):
            sp_mask = felzenszwalb(
                image_roi,
                scale=superpixel_method.scale, 
                sigma=superpixel_method.sigma, 
                min_size=superpixel_method.min_size
            )
            sp_mask += 1
            sp_mask[~mask] = 0
        elif isinstance(superpixel_method, WatershedSuperpixel):
            sp_mask = watershed(
                sobel(rgb2gray(image_roi)),
                markers=superpixel_method.n_components,
                compactness=superpixel_method.compactness
            )
            sp_mask += 1
            sp_mask[~mask] = 0
        else:    
            raise Exception("Unsupported superpixel type")
        segments = np.zeros((sp_mask.shape[0]+2, sp_mask.shape[1]+2), dtype=np.int32)
        segments -= 1
        segments[1:-1, 1:-1] = sp_mask
        # Создаем фигуру и оси
        # fig, ax = plt.subplots()
    
        # # Отображаем изображение с цветовой картой
        # im = ax.imshow(sp_mask == 0, cmap='viridis', aspect='auto')

        # # Добавляем цветовую шкалу слева
        # cbar = fig.colorbar(im, ax=ax, location='left')

        # # Настройки оформления
        # ax.set_title("Изображение с цветовой шкалой")
        # plt.show()
        
        unique_segments = sorted(np.unique(segments))[1:]
        # print(unique_segments)
        means, variances, valid_labels = parallel_stats_rgb(image_roi, sp_mask, np.max(sp_mask))
        # Loop through each unique segment
        for i, segment in enumerate(valid_labels):
            if segment <= 0:
                continue
            # Create a mask for the current segment
            binary_masks = (segments == segment).astype(np.uint8)
            # pixels = image_roi[binary_mask[1:-1, 1:-1]]
            for binary_mask in split_mask(binary_masks.astype(np.bool_)):
                contours = skimage.measure.find_contours(binary_mask.astype(np.uint8))

                external_contour = contours[0][:, ::-1]
                # Преобразуйте контур в список координат
                polygon = (external_contour - 1).astype(np.float32)
                polygon[:, 0] /= sp_mask.shape[1]
                polygon[:, 1] /= sp_mask.shape[0]
                # print("polygon max:", polygon.max(), polygon.min())
                polygon[:, 0] *= (bbox[2] - bbox[0])
                polygon[:, 1] *= (bbox[3] - bbox[1])
                polygon[:, 0] += bbox[0]
                polygon[:, 1] += bbox[1]
                
                region_props = []
                region_props.extend(means[i])
                region_props.extend(variances[i])
                # print("region_props", region_props)
                self.superpixels[superpixel_method].append(
                    SuperPixel(
                        id=self._superpixel_ind[superpixel_method],
                        method=superpixel_method.short_string(), 
                        border=np.around(simplify(polygon[::]), decimals=7), #len(polygon) // 20 if len(polygon) > 20 else 1
                        parents=None,
                        props=np.array(region_props)
                    )
                )
                self._superpixel_ind[superpixel_method] += 1


    def _create_superpixel(self, superpixel_method: SuperPixelMethod) -> None:
        self._superpixel_ind[superpixel_method] = 0
        self.superpixels[superpixel_method] = []
        self._annotation_ind[superpixel_method] = 0
        return
        sp_mask = None
        if isinstance(superpixel_method, SLICSuperpixel):
            sp_mask = slic(
                self.image_lab,
                n_segments=superpixel_method.n_clusters,
                compactness=superpixel_method.compactness,
                sigma=superpixel_method.sigma,
                start_label=1
            )
        elif isinstance(superpixel_method, FelzenszwalbSuperpixel):
            sp_mask = felzenszwalb(
                self.image_lab,
                scale=superpixel_method.scale, 
                sigma=superpixel_method.sigma, 
                min_size=superpixel_method.min_size
            )
        else:
            raise Exception("Unsupported superpixel type")
        print("Image shapes", np.array(self.image).shape, sp_mask.shape)
        temp = mark_boundaries(np.array(self.image), sp_mask)
        if np.all(temp < 1.1):
            temp = (temp * 255).astype(np.uint8)
        temp_img = Image.fromarray(temp)
        segments = np.zeros((sp_mask.shape[0]+2, sp_mask.shape[1]+2), dtype=np.int32)
        segments -= 1
        segments[1:-1, 1:-1] = sp_mask
        
        unique_segments = np.unique(segments)[1:]
        # Loop through each unique segment
        for segment in unique_segments:
            # Create a mask for the current segment
            binary_mask = (segments == segment).astype(np.uint8)
            contours = skimage.measure.find_contours(binary_mask)

            external_contour = contours[0][:, ::-1]
            # Преобразуйте контур в список координат
            polygon = (external_contour - 1).astype(np.float32)
            polygon[:, 0] /= sp_mask.shape[1]
            polygon[:, 1] /= sp_mask.shape[0]
            self.superpixels[superpixel_method].append(
                SuperPixel(
                    id=self._superpixel_ind[superpixel_method],
                    method=superpixel_method.short_string(), 
                    border=np.around(simplify(polygon[::]), decimals=7), #len(polygon) // 20 if len(polygon) > 20 else 1
                    parents=None
                )
            )
            self._superpixel_ind[superpixel_method] += 1
        print("Polygon coords max", polygon.max())

    def _create_superpixels(self) -> None:
        for superpixel_method in self.superpixel_methods:
            self._create_superpixel(superpixel_method)
            self._annotations[superpixel_method] = ImageAnnotation(annotations=[])

    def cancel_prev_act(self):
        scribble_to_del = self.scribbles_id_sequence.pop()
        for anno_key in list(self._annotations.keys()):
            # print("self._annotations[anno_key].annotations")
            # print(len(self._annotations[anno_key].annotations))
            anno_to_del = []
            for anno_ind in range(len(self._annotations[anno_key].annotations)):
                for i in range(len(self._annotations[anno_key].annotations[anno_ind].parent_scribble)):
                    if self._annotations[anno_key].annotations[anno_ind].parent_scribble[i] == scribble_to_del:
                        self._annotations[anno_key].annotations[anno_ind].parent_scribble.pop(i)
                        if len(self._annotations[anno_key].annotations[anno_ind].parent_scribble) == 0:
                            anno_to_del.append(anno_ind)
            print(len(anno_to_del))
            for j in anno_to_del[::-1]:
                self._annotations[anno_key].annotations.pop(j)
            self._scribbles.pop()
            # print(len(self._annotations[anno_key].annotations))

        
    
    def serialize(self, path: str) -> None:
        """Сериализация данных с автоматическим созданием файла"""
        try:
            # Создаем структуру данных для сохранения
            data = {
                "scribbles": self._serialize_scribbles(),
                "superpixels": self._serialize_superpixels(),
                "annotations": self._serialize_annotations(),
                "bbox": self._serialize_bbox()
            }
            
            # Записываем в файл с созданием директорий
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except PermissionError:
            raise Exception("Permission denied to write file")
        except IsADirectoryError:
            raise Exception("Specified path is a directory")
        except Exception as e:
            raise Exception(f"Serialization error: {str(e)}")

    def _serialize_scribbles(self) -> list:
        return [s.dict_to_save() for s in self._scribbles]

    def _serialize_bbox(self) -> list:
        return [[x[0], x[1], x[2], x[3]] for x in self.annotated_bbox]

    def _serialize_superpixels(self) -> dict:
        return {
            method.short_string(): [sp.dict_to_save() for sp in sp_list]
            for method, sp_list in self.superpixels.items()
        }

    def _serialize_annotations(self) -> dict:
        return {
            method.short_string(): [anno.dict_to_save() for anno in anno_list.annotations]
            for method, anno_list in self._annotations.items()
        }
    
    
    def deserialize(self, path: str) -> None:
        """Десериализация данных с валидацией"""
        try:
            with open(path, "r") as f:
                loaded_dict = json.load(f)

            self._clear_existing_data()
            self._load_superpixels(loaded_dict)
            self._load_annotations(loaded_dict)
            self._load_scribbles(loaded_dict)
            self._load_bbox(loaded_dict)

        except FileNotFoundError:
            raise Exception("File not found")
        except json.JSONDecodeError:
            raise Exception("Invalid JSON format")
        except KeyError as e:
            raise Exception(f"Missing required field: {str(e)}")
        except Exception as e:
            raise Exception(f"Loading error: {str(e)}")

    def _clear_existing_data(self):
        """Очистка текущих данных"""
        self.superpixel_methods.clear()
        self.superpixels.clear()
        self._annotations.clear()
        self._scribbles.clear()
        self._superpixel_ind.clear()
        self.annotated_bbox.clear()

    def _load_superpixels(self, data: dict):
        """Загрузка данных о суперпикселях"""
        for method_str, sp_list in data["superpixels"].items():
            method = self._parse_method_from_string(method_str)
            self.superpixels[method] = [
                SuperPixel(
                    id=int(sp["id"]),
                    method=sp["method"],
                    border=np.array(sp["border"], dtype=np.float32),
                    parents=sp["parents"],
                    props=np.array(sp["props"], dtype=np.float32)
                ) for sp in sp_list
            ]
            self.superpixel_methods.append(method)
            self._annotation_ind[method] = 0
            self._superpixel_ind[method] = 0
            for sp in sp_list:
                self._superpixel_ind[method] = max(self._superpixel_ind[method], sp["id"]+1)

    def _load_annotations(self, data: dict):
        """Загрузка аннотаций"""
        for method_str, anno_list in data["annotations"].items():
            method = next(m for m in self.superpixel_methods if m.short_string() == method_str)
            self._annotations[method] = ImageAnnotation(
                annotations=[
                    AnnotationInstance(
                        id=int(anno["id"]),
                        code=anno["code"],
                        border=np.array(anno["border"], dtype=np.float32),
                        parent_superpixel=anno["parent_superpixel"],
                        parent_scribble=[int(i) for i in anno["parent_scribble"]]
                    ) for anno in anno_list
                ]
            )
            self._annotation_ind[method] = -1
            for anno in anno_list:
                self._annotation_ind[method] = max(self._annotation_ind[method], int(anno["id"]) + 1)

    def _load_scribbles(self, data: dict):
        """Загрузка скрайблов"""
        self._scribbles = [
            Scribble(
                id=scribble["id"],
                points=np.array(scribble["points"], dtype=np.float32),
                params=ScribbleParams(
                    radius=scribble["params"]["radius"],
                    code=scribble["params"]["code"]
                )
            ) for scribble in data["scribbles"]
        ]
        self.scribbles_id_sequence = [int(scribble["id"]) for scribble in data["scribbles"]]
    
    def _load_bbox(self, data: dict):
        """Загрузка скрайблов"""
        self.annotated_bbox = [[float(x) for x in y] for y in data["bbox"]]

    @staticmethod
    def _parse_method_from_string(method_str: str) -> SuperPixelMethod:
        """Статический метод для преобразования строки в объект SuperPixelMethod"""
        parts = method_str.split('_')
        
        try:
            method_type = parts[0]
            
            if method_type == "SLIC":
                if len(parts) != 4:
                    raise ValueError(f"Invalid SLIC format: {method_str}")
                return SLICSuperpixel(
                    n_clusters=int(parts[1]),
                    compactness=float(parts[2]),
                    sigma=float(parts[3])
                )
                
            elif method_type == "Felzenszwalb":
                if len(parts) != 4:
                    raise ValueError(f"Invalid Felzenszwalb format: {method_str}")
                return FelzenszwalbSuperpixel(
                    min_size=int(parts[1]),
                    sigma=float(parts[2]),
                    scale=float(parts[3])
                )
                
            elif method_type == "Watershed":
                return WatershedSuperpixel(
                    threshold=float(parts[1])
                )
                
            else:
                raise ValueError(f"Unknown method type: {method_type}")
                
        except (IndexError, ValueError) as e:
            raise ValueError(f"Failed to parse method string '{method_str}': {str(e)}")

    def add_scribble(self, scribble: Scribble) -> None:
        scribble.id = self.ind_scrible
        self.scribbles_id_sequence.append(scribble.id)
        self.ind_scrible += 1
        print("Adding scribble")
        print(f"Current number of scribbles: {len(self._scribbles)}")
        print(f"Scribble ID: {scribble.id}")
        
        self._scribbles.append(scribble)
        self._update_annotations(scribble)
    
    def sp_annoted_before(self, sp_idx):
        annotated_before = False
        superpixel_method = list(self._annotations.keys())[0]
        for anno_ind, anno in enumerate(self._annotations[superpixel_method].annotations):
            if anno.parent_superpixel == sp_idx:
                annotated_before = True
                break
        return annotated_before
    
    def use_sensetivity_for_region(self, sp_idx: int, sens: float, scribble: Scribble) -> None:
        sp_method = list(self.superpixels.keys())[0]
        polygons = []
        for i in self.superpixels[sp_method]:
            polygons.append(shapely.Polygon(i.border))
        target_poly = polygons[sp_idx]
        sp_id = self.superpixels[sp_method][sp_idx].id
        target_props = self.superpixels[sp_method][sp_idx].props
        tree = shapely.STRtree(polygons)
        buffer_zone = target_poly.buffer(self._superpixel_radius * sens)
        candidates = tree.query(buffer_zone)
        result = []
        index_map = {id(poly): idx for idx, poly in enumerate(polygons)}
        for candidate_idx in candidates:
            # Пропускаем негеометрические объекты
            candidate = polygons[candidate_idx]
            candidate_id = self.superpixels[sp_method][candidate_idx].id
            if not isinstance(candidate, Polygon):
                # print(1)
                continue
            if candidate == target_poly:
                # print(2)
                continue
            distance = target_poly.distance(candidate)
            cand_props = self.superpixels[sp_method][candidate_idx].props
            cur_superpixel = self.superpixels[sp_method][candidate_idx]
            if distance <= 0.0001 * sens and \
                np.all(np.abs(cand_props - target_props) < self._property_dist * sens):
                if self.sp_annoted_before(candidate_id):
                    continue
                result.append(index_map[id(candidate)])
                self._annotations[sp_method].annotations.append(
                    AnnotationInstance(
                        id=self._annotation_ind[sp_method],
                        code=scribble.params.code,
                        border=cur_superpixel.border.astype(np.float32),
                        parent_superpixel=int(cur_superpixel.id),
                        parent_scribble=[scribble.id],
                        parent_intersect=False
                    )
                )
                self._annotation_ind[sp_method] += 1
                self.use_sensetivity_for_region(int(candidate_idx), sens, scribble)

    def _update_annotations(self, last_scribble: Scribble) -> None:
        self.prev_ind_scribble = self.ind_scrible
        scribble_line = shapely.LineString(last_scribble.points)

        superpixel_method = self.superpixel_methods[0]
        superpixel_ind_to_del = []
        superpixel_to_append = []
        scribbles_to_check = []
        annotations_to_del = []
        polygons = [Polygon(sp.border) for sp in self.superpixels[superpixel_method]]
        tree = shapely.STRtree(polygons)
        # buffer_zone = scribble_line.buffer(0)
        candidates = tree.query(scribble_line)
        result = []
        index_map = {id(poly): idx for idx, poly in enumerate(polygons)}
        for cur_superpixel_ind in candidates:
            # Пропускаем негеометрические объекты
            candidate = polygons[cur_superpixel_ind]
            cur_superpixel = self.superpixels[superpixel_method][cur_superpixel_ind]
            if not isinstance(candidate, Polygon):
                # print(1)
                continue
        # for cur_superpixel_ind, cur_superpixel in enumerate(self.superpixels[superpixel_method]):
        #     scrible_bbox = (last_scribble.points[:, 0].min(), last_scribble.points[:, 1].min(), last_scribble.points[:, 0].max(), last_scribble.points[:, 1].max())
        #     sp_bbox = (cur_superpixel.border[:, 0].min(), cur_superpixel.border[:, 1].min(), cur_superpixel.border[:, 0].max(), cur_superpixel.border[:, 1].max())
        #     can_intersect = bbox_is_intersect(scrible_bbox, sp_bbox)
        #     if can_intersect == False:
        #         continue
            # Проверка пересечения с суперпикселем
            # superpixel_polygon = Polygon(cur_superpixel.border)
            if len(last_scribble.points) > 1 and candidate.intersects(scribble_line):
                annotated_before = False
                # print(f"Checking scribble with superpixel ID: {cur_superpixel.id}")
                # Проверка, аннотирован ли суперпиксель
                for anno_ind, anno in enumerate(self._annotations[superpixel_method].annotations):
                    if anno.parent_superpixel == cur_superpixel.id:
                        annotated_before = True
                        if anno.parent_intersect == False:
                            # разметка распространилось по чувствительности
                            if anno.code != last_scribble.params.code:
                                anno.parent_scribble = [last_scribble.id]
                                anno.code=last_scribble.params.code
                            else:
                                anno.parent_scribble.append(last_scribble.id)
                            break
                        anno_created_scribble = None
                        for scribble in self._scribbles:
                            if anno.parent_scribble[0] == scribble.id:
                                anno_created_scribble = scribble
                                break
                        if anno_created_scribble.params.code != last_scribble.params.code:
                            # print(f"Checking scribble with superpixel ID: {cur_superpixel.id}")
                            annotations_to_del.append(anno_ind)
                            scribbles_to_check.extend(anno.parent_scribble)
                            superpixel_ind_to_del.append(cur_superpixel_ind)
                            region = copy.deepcopy(cur_superpixel.border)
                            region[:, 0] *= self.image.size[0]
                            region[:, 1] *= self.image.size[1]
                            polygon = [(x[0], x[1]) for x in region]
                            overlay = Image.new("RGB", self.image.size, (0, 0, 0))
                            draw = ImageDraw.Draw(overlay)
                            draw.polygon(polygon, fill="white", outline="white")
                            overlay_rgb = overlay.convert("RGB").convert("L")
                            overlay_rgb = np.array(overlay_rgb)
                            print("conv started")
                            # overlay_rgb = convolve2d(overlay_rgb, np.ones((3,3), dtype=np.int32), "same")
                            print("conv finished")
                            is_new_superpixel_correct = False
                            temp_divided_sp: List[SuperPixel] = []
                            cur_n_segments = 3
                            while not is_new_superpixel_correct:
                                sp_mask = slic(
                                    self.image_lab,
                                    n_segments=cur_n_segments,
                                    compactness=20,
                                    sigma=1,
                                    start_label=1,
                                    mask=overlay_rgb
                                )
                                temp_divided_sp.clear()
                                print("sp finished")
                                segments = np.zeros((sp_mask.shape[0]+2, sp_mask.shape[1]+2), dtype=np.int32)
                                segments -= 1
                                segments[1:-1, 1:-1] = sp_mask
                                means, variances, valid_labels = parallel_stats_rgb(np.array(self.image_lab), sp_mask, np.max(sp_mask))
                                unique_segments = np.unique(sp_mask)[1:]
                                for i, segment in enumerate(valid_labels):
                                    if segment <= 0:
                                        continue
                                    region_props = []
                                    region_props.extend(means[i])
                                    region_props.extend(variances[i])
                                    # print("process segment in supepixel divide", segment)
                                    # Create a mask for the current segment
                                    binary_mask = (sp_mask == segment).astype(np.uint8)
                                    
                                    contours = skimage.measure.find_contours(binary_mask)
                                    external_contour = contours[0][:, ::-1]
                                    # Преобразуйте контур в список координат
                                    polygon = external_contour.astype(np.float32)
                                    #image_to_vis = self.image.copy()
                                    #draw = ImageDraw.Draw(image_to_vis)
                                    #draw.polygon(polygon, outline="red")
                                    #draw.polygon(region, outline="yellow")
                                    #image_to_vis.show()
                                    #plt.imshow(image_to_vis)
                                    #plt.show()
                                    polygon[:, 0] /= overlay_rgb.shape[1]
                                    polygon[:, 1] /= overlay_rgb.shape[0]
                                    polygon1 = shapely.Polygon(polygon).intersection(shapely.Polygon(cur_superpixel.border))
                                    if isinstance(polygon1, shapely.GeometryCollection):
                                        polygon1 = polygon1.geoms[0]
                                    elif isinstance(polygon1, shapely.MultiPolygon):
                                        polygon1 = polygon1.geoms[0]

                                    polygon = np.array(polygon1.boundary.coords)
                                    temp_divided_sp.append(
                                        SuperPixel(
                                            id=self._superpixel_ind[superpixel_method],
                                            method=superpixel_method.short_string(), 
                                            border=np.around(polygon, decimals=7), #len(polygon) // 20 if len(polygon) > 20 else 1
                                            parents=cur_superpixel.id,
                                            props=np.array(region_props)
                                        )
                                    )
                                is_new_superpixel_correct = check_scribbles_dont_intersect_one_region(
                                    [anno_created_scribble, last_scribble],
                                    temp_divided_sp
                                )
                                cur_n_segments *= 2
                                if cur_n_segments > 50:
                                    break
                            for sp in temp_divided_sp:
                                superpixel_to_append.append(sp)
                                self._superpixel_ind[superpixel_method] += 1
                        else:
                            anno.parent_scribble.append(int(last_scribble.id))


                        # print(f"Superpixel {cur_superpixel.id} is already annotated")
                if annotated_before:
                    continue
                self._annotations[superpixel_method].annotations.append(
                    AnnotationInstance(
                        id=self._annotation_ind[superpixel_method],
                        code=last_scribble.params.code,
                        border=cur_superpixel.border.astype(np.float32),
                        parent_superpixel=int(cur_superpixel.id),
                        parent_scribble=[int(last_scribble.id)],
                        parent_intersect=True
                    )
                )
                self._annotation_ind[superpixel_method] += 1
                self.use_sensetivity_for_region(cur_superpixel_ind, 1, last_scribble)
        superpixel_ind_to_del = sorted(list(set(superpixel_ind_to_del)))
        for sp_to_del in superpixel_ind_to_del[::-1]:
            self.superpixels[superpixel_method].pop(sp_to_del)
        annotations_to_del = sorted(list(set(annotations_to_del)))
        for anno_to_del in annotations_to_del[::-1]:
            self._annotations[superpixel_method].annotations.pop(anno_to_del)
        for superpixel in superpixel_to_append:
            self.superpixels[superpixel_method].append(superpixel)
            superpixel_polygon = Polygon(superpixel.border)
            # check with last scribble
            if len(last_scribble.points) > 1 and superpixel_polygon.intersects(scribble_line):
                self._annotations[superpixel_method].annotations.append(
                    AnnotationInstance(
                        id=self._annotation_ind[superpixel_method],
                        code=last_scribble.params.code,
                        border=superpixel.border.astype(np.float32),
                        parent_superpixel=superpixel.id,
                        parent_scribble=[last_scribble.id],
                        parent_intersect=True
                    )
                )
                self._annotation_ind[superpixel_method] += 1
                self.use_sensetivity_for_region(
                        len(self.superpixels[superpixel_method]) - 1, 
                        1, 
                        last_scribble)
                continue
                
            for scribble_id in scribbles_to_check:
                scribble = None
                # find scribble with needed id
                for scr in self._scribbles:
                    if scr.id == scribble_id:
                        scribble = scr
                        break
                scribble_line_temp = shapely.LineString(scribble.points)
                if len(scribble.points) > 1 and superpixel_polygon.intersects(scribble_line_temp):
                    self._annotations[superpixel_method].annotations.append(
                        AnnotationInstance(
                            id=self._annotation_ind[superpixel_method],
                            code=scribble.params.code,
                            border=superpixel.border.astype(np.float32),
                            parent_superpixel=superpixel.id,
                            parent_scribble=[scribble.id],
                            parent_intersect=True
                        )
                    )
                    self._annotation_ind[superpixel_method] += 1
                    self.use_sensetivity_for_region(
                        len(self.superpixels[superpixel_method]) - 1, 
                        1, 
                        scribble)
                    break
                
        
        print(f"Number of annotations after update: {len(self._annotations[superpixel_method].annotations)}")
        print(f"Total number of superpixels: {len(self.superpixels[superpixel_method])}")


    def get_annotation(self, method: SuperPixelMethod) -> ImageAnnotation:
        annos = self._annotations[method]
        # postprocessing
        annos2 = annos
        return ImageAnnotation(annos2)
