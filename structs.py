from abc import ABC
from dataclasses import dataclass

import datetime
from pathlib import Path
from typing import Optional, Dict, Set, Tuple
import numpy as np

from PIL import Image
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


@dataclass
class ScribbleParams:
    radius: float
    code: int


@dataclass
class Scribble:
    id: int
    points: np.ndarray  # [n x 2] float array of absolute points
    params: ScribbleParams  # parameters of the scribble
    creation_time: datetime = None

    def __len__(self):
        return len(self.points)

    def __post_init__(self):
        self.creation_time = datetime.datetime.now()
        self._bbox = (
            None
            if self.points.size == 0
            else (
                (
                    min(self.points[:, 0]),
                    min(self.points[:, 1]),
                ),
                (
                    max(self.points[:, 0]),
                    max(self.points[:, 1]),
                ),
            )
        )

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

    def __hash__(self):
        return hash(self.method + f"_{self.id}")

    def __eq__(self, other):
        return (self.method + f"_{self.id}") == (other.method + f"_{other.id}")


@dataclass
class AnnotationInstance:
    id: int
    code: int
    border: np.ndarray  # [n x 2] float array of border absolute points

    def __hash__(self):
        return self.id


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
    
    def __lt__(self, other) -> bool:
        return self.short_string() > other.short_string()

    def __hash__(self):
        return hash(self.short_string())

    def __eq__(self, other):
        return isinstance(other, SuperPixel) and self.short_string() == other.short_string()

@dataclass
class WatershedSuperpixel(SuperPixelMethod):
    threshold: float

    def short_string(self) -> str:
        return f"Watershed_{self.threshold}"
    
    def __hash__(self):
        return hash(self.short_string())

    def __eq__(self, other):
        return isinstance(other, SuperPixel) and self.short_string() == other.short_string()



@dataclass
class SLICSuperpixel(SuperPixelMethod):

    n_clusters: int
    compactness: float
    scale: float

    def short_string(self) -> str:
        return f"SLIC_{self.n_clusters}_{self.compactness}_{self.scale}"

    def __hash__(self):
        return hash(self.short_string())

    def __eq__(self, other):
        return isinstance(other, SuperPixel) and self.short_string() == other.short_string()

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
        return isinstance(other, SuperPixel) and self.short_string() == other.short_string()


def bbox_is_intersect(bbox_1: Tuple[int], bbox_2: Tuple[int]) -> bool:
    """
    bbox: (x_left, y_top, x_right, y_bottom)
    """
    x_intersect: bool = max(bbox_1[0], bbox_2[0]) <= min(bbox_1[2], bbox_2[2])
    y_intersect: bool = max(bbox_1[1], bbox_2[1]) <= min(bbox_1[3], bbox_2[3])
    return x_intersect and y_intersect

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
        self.superpixel_methods = superpixel_methods
        if image != None:
            self.image = image
        else:
            self.image = Image.fromarray(
                cv2.imread(str(self.image_path)).astype(np.uint8)
            )
        self._preprocess_image(downscale_coeff)
        self._create_superpixels()
        self.ind_scrible = 0
        self.prev_ind_scrible = 0

        self._scribbles = []

    def _preprocess_image(self, downscale_coeff: float) -> None:
        sh = self.image.size
        self.image.thumbnail((sh[0] * downscale_coeff, sh[1] * downscale_coeff), Image.Resampling.BILINEAR)
        self.image_lab = rgb2lab(np.array(self.image))

    def add_superpixel_method(self, superpixel_method: SuperPixelMethod) -> None:
        print("Recieved new superpixel method", superpixel_method.short_string())
        for exist_superpixel_method in self.superpixel_methods:
            if exist_superpixel_method.short_string() == superpixel_method.short_string():
                print("len of annos after added:", len(self._annotations))
                return
        self.superpixel_methods.append(superpixel_method)
        self._create_superpixel(superpixel_method)
        self._annotations[superpixel_method] = set()
        print("len of annos after added:", len(self._annotations))
        print(len(self.superpixels[superpixel_method]))
        print(len(self.superpixels))


    def _create_superpixel(self, superpixel_method: SuperPixelMethod) -> None:
        sp_mask = None
        if isinstance(superpixel_method, SLICSuperpixel):
            sp_mask = slic(
                self.image_lab,
                n_segments=superpixel_method.n_clusters,
                compactness=superpixel_method.compactness,
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
        self.superpixels[superpixel_method] = set()
        print("Image shapes", np.array(self.image).shape, sp_mask.shape)
        temp = mark_boundaries(np.array(self.image), sp_mask)
        if np.all(temp < 1.1):
            temp = (temp * 255).astype(np.uint8)
        temp_img = Image.fromarray(temp)
        self.image_with_sp_border[superpixel_method.short_string()] = sp_mask
        segments_src = sp_mask
        segments = np.zeros((segments_src.shape[0]+2, segments_src.shape[1]+2), dtype=np.int32)
        segments -= 1
        segments[1:-1, 1:-1] = segments_src
        
        print(segments[:10, :10])
        unique_segments = np.unique(segments)[1:]
        # Loop through each unique segment
        for segment in unique_segments:
            # Create a mask for the current segment
            binary_mask = (segments == segment).astype(np.uint8)
            contours = skimage.measure.find_contours(binary_mask)

            external_contour = contours[0][:, ::-1]
            # Преобразуйте контур в список координат
            polygon = (external_contour).astype(np.float32)
            """
            coords = np.where(doubled_image == segment)
            y_min, y_max, x_min, x_max = coords[0].min() - 1, coords[0].max() + 2, coords[1].min() - 1, coords[1].max() + 2
            sp_submask = doubled_image[y_min: y_max, x_min: x_max] == segment
            sp_submask = convolve2d(sp_submask, np.ones((3, 3)), mode="same")
            bounds = (sp_submask.astype(np.bool_) & borders_results[y_min: y_max, x_min: x_max]).astype(np.int32)

            # Find contours of the current segment
            contours, _ = cv2.findContours(bounds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            external_contour = contours[0]

            # Преобразуйте контур в список координат
            polygon = (external_contour[:, 0, :] - 1).astype(np.float32)
            polygon[:, 0] += y_min
            polygon[:, 1] += x_min
            """
            polygon[:, 0] /= segments_src.shape[1]
            polygon[:, 1] /= segments_src.shape[0]
            #print("Polygon coords max", polygon.max())
            self.superpixels[superpixel_method].add(
                SuperPixel(
                    id=segment,
                    method=superpixel_method.short_string(), 
                    border=polygon[::], #len(polygon) // 20 if len(polygon) > 20 else 1
                    parents=None
                )
            )
        

    def _create_superpixels(self) -> None:
        self.superpixels: Dict[SuperPixelMethod, Set[SuperPixel]] = {}
        self._annotations: Dict[SuperPixel, ImageAnnotation] = {}
        self.image_with_sp_border = {}
        for superpixel_method in self.superpixel_methods:
            self._create_superpixel(superpixel_method)
            self._annotations[superpixel_method] = ImageAnnotation(annotations=[])

    def cancel_prev_act(self):
        print("Cancelled:")
        for superpixel_method in self.superpixel_methods:
            print(len(self._annotations[superpixel_method]))
            self._annotations[superpixel_method] = set(i for i in self._annotations[superpixel_method] if i.id < self.prev_ind_scrible)
            print(len(self._annotations[superpixel_method]))

    def add_scribble(self, scribble: Scribble) -> None:
        print("Adding scribble")
        print(f"Current number of scribbles: {len(self._scribbles)}")
        print(f"Scribble ID: {scribble.id}")
        
        self._scribbles.append(scribble)
        self._update_annotations(scribble)

    def _update_annotations(self, last_scribble: Scribble) -> None:
        self.prev_ind_scribble = self.ind_scrible
        scribble_line = shapely.LineString(last_scribble.points)

        for superpixel_method in self.superpixel_methods:
            annotated_before = False

            for cur_superpixel in self.superpixels[superpixel_method]:
                scrible_bbox = (last_scribble.points[:, 0].min(), last_scribble.points[:, 1].min(), last_scribble.points[:, 0].max(), last_scribble.points[:, 1].max())
                sp_bbox = (cur_superpixel.border[:, 0].min(), cur_superpixel.border[:, 1].min(), cur_superpixel.border[:, 0].max(), cur_superpixel.border[:, 1].max())
                can_intersect = bbox_is_intersect(scrible_bbox, sp_bbox)
                if can_intersect == False:
                    continue
                """
                img = np.array(self.image_with_sp_border[superpixel_method.short_string()])
                img = Image.fromarray(img)
                ImageDraw.Draw
                plt.imshow()
                plt.show()
                """
                print(f"Checking scribble with superpixel ID: {cur_superpixel.id}")

                # Проверка, аннотирован ли суперпиксель
                if any(np.array_equal(anno.border, cur_superpixel.border) for anno in self._annotations[superpixel_method]):
                    annotated_before = True
                    print(f"Superpixel {cur_superpixel.id} is already annotated")
                    continue

                # Проверка пересечения с суперпикселем
                superpixel_polygon = Polygon(cur_superpixel.border)
                if len(last_scribble.points) > 1 and superpixel_polygon.intersects(scribble_line):
                    self._annotations[superpixel_method].add(
                        AnnotationInstance(
                            id=self.ind_scrible,
                            code=last_scribble.params.code,
                            border=cur_superpixel.border
                        )
                    )
                    self.ind_scrible += 1
            
            print(f"Number of annotations after update: {len(self._annotations[superpixel_method])}")
            print(f"Total number of superpixels: {len(self.superpixels[superpixel_method])}")


    def get_annotation(self, method: SuperPixelMethod) -> ImageAnnotation:
        annos = self._annotations[method]
        # postprocessing
        annos2 = annos
        return ImageAnnotation(annos2)


if __name__ == "__main__":
    scr1 = Scribble(
        id=0,
        points=np.asarray([[0.05, 0.07], [0.12, 0.1], [0.4, 0.2]]),
        params=ScribbleParams(radius=5, code=0),
    )
    print(scr1)
    print(scr1.bbox)

    algo = SuperPixelAnnotationAlgo(
        image_path=Path("/home/yalekseevich/dev/term_work_5_year/data/image1.jpeg"),
        downscale_coeff=1,
        superpixel_methods=[
            SLICSuperpixel(n_clusters=10, compactness=0.5)
        ],
    )

    algo.add_scribble(scr1)
