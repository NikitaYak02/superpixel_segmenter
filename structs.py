from abc import ABC
from dataclasses import dataclass

import datetime
from pathlib import Path
from typing import Optional
import numpy as np

from PIL import Image
import cv2

import shapely
from skimage.segmentation import slic, felzenszwalb, watershed
from skimage.measure import regionprops
from skimage.filters import sobel
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from shapely.geometry import Polygon
from skimage.segmentation import mark_boundaries

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
        print("Recived new superpixel method", superpixel_method.short_string())
        for exist_superpixel_method in self.superpixel_methods:
            if exist_superpixel_method.short_string() == superpixel_method.short_string():
                print("len of annos after added:", len(self._annotations))
                return
        self.superpixel_methods.append(superpixel_method)
        self._create_superpixel(superpixel_method)
        self._annotations[superpixel_method] = set()
        print("len of annos after added:", len(self._annotations))


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
        self.image_with_sp_border[superpixel_method.short_string()] = temp_img
        for i, region in enumerate(regionprops(sp_mask)):
            # Get the coordinates of the superpixel
            coords = region.coords.astype(np.float32)
            coords[:, 0] /= np.float32(self.image.height)
            coords[:, 1] /= np.float32(self.image.width)
            # Create a polygon from the coordinates
            polygon = np.array(Polygon(coords))
            
            self.superpixels[superpixel_method].add(
                SuperPixel(
                    id=i, 
                    method=superpixel_method.short_string(), 
                    border=coords,
                    parents=None
                )
            )

    def _create_superpixels(self) -> None:
        self.superpixels = {}
        self._annotations = {}
        self.image_with_sp_border = {}
        for superpixel_method in self.superpixel_methods:
            self._create_superpixel(superpixel_method)
            self._annotations[superpixel_method] = set()

    def cancel_prev_act(self):
        print("Cancelled:")
        for superpixel_method in self.superpixel_methods:
            print(len(self._annotations[superpixel_method]))
            self._annotations[superpixel_method] = set(i for i in self._annotations[superpixel_method] if i.id < self.prev_ind_scrible)
            print(len(self._annotations[superpixel_method]))

    def add_scribble(self, scribble: Scribble) -> None:
        self._scribbles.append(scribble)
        self._update_annotations()

    def _update_annotations(self) -> None:
        self.prev_ind_scrible = self.ind_scrible
        last_scribble = self._scribbles[-1]
        # look for superpixels that intersect with the scribble
        # ...
        # check if these superpixels are not labeled yet
        # if yes, split scribble into small superpixels until ...
        for superpixel_method in self.superpixel_methods:
            # do calculations
            for cur_superpixel in self.superpixels[superpixel_method]:
                annotated_before = False
                for anno in self._annotations[superpixel_method]:
                    if np.array_equal(anno.border, cur_superpixel.border):
                        # [TO DO] добавить подразбиение и проверить, что они разного
                        annotated_before = True
                        break
                if annotated_before == True:
                    break
                sp_polygon = Polygon(cur_superpixel.border)
                # проверить может здесь подойдет метод интерсект
                for point in last_scribble.points:
                    point_sh = shapely.Point(point[0], point[1])
                    if sp_polygon.contains(point_sh):
                        self._annotations[superpixel_method].add(
                            AnnotationInstance(
                                id=self.ind_scrible,
                                code=last_scribble.params.code,
                                border=cur_superpixel.border
                            )
                        )
                        self.ind_scrible += 1
                        break
            print("Superpixel annos len after update:", len(self._annotations[superpixel_method]))
            print("Total num of pixels:", len(self.superpixels[superpixel_method]))


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
