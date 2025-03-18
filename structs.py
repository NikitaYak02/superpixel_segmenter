from abc import ABC
from dataclasses import dataclass, field

import datetime
from pathlib import Path
from typing import Optional, Dict, Set, Tuple, List
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
import json
import copy
from scipy.signal import convolve2d


@dataclass
class ScribbleParams:
    radius: float
    code: int 
    def dict_to_save(self) -> Dict:
        res = dict()
        res["radius"] = self.radius
        res["code"] = self.code # now: type ind in ordered dict
        return res

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
            if not isinstance(self.points, np.ndarray) or self.points.size == 0
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
    
    def dict_to_save(self):
        res = dict()
        res["id"] = self.id
        res["points"] = self.points.tolist()
        res["params"] = self.params.dict_to_save()
        res["creation_time"] = self.creation_time.isoformat()
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

    def __hash__(self):
        return self.id

    def dict_to_save(self):
        res = dict()
        res["id"] = int(self.id)
        res["code"] = int(self.code)
        res["border"] = [[round(float(y), 7) for y in x] for x in self.border]
        res["parent_superpixel"] = int(self.parent_superpixel)
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
    sigma: float

    def short_string(self) -> str:
        return f"SLIC_{self.n_clusters}_{self.compactness}_{self.sigma}"

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


    def _create_superpixel(self, superpixel_method: SuperPixelMethod) -> None:
        self._superpixel_ind[superpixel_method] = 0
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
        self.superpixels[superpixel_method] = []
        print("Image shapes", np.array(self.image).shape, sp_mask.shape)
        temp = mark_boundaries(np.array(self.image), sp_mask)
        if np.all(temp < 1.1):
            temp = (temp * 255).astype(np.uint8)
        temp_img = Image.fromarray(temp)
        self.image_with_sp_border[superpixel_method.short_string()] = sp_mask
        self._annotation_ind[superpixel_method] = 0
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
        print("Cancelled:")
        for superpixel_method in self.superpixel_methods:
            print(len(self._annotations[superpixel_method]))
            self._annotations[superpixel_method] = set(i for i in self._annotations[superpixel_method] if i.id < self.prev_ind_scrible)
            print(len(self._annotations[superpixel_method]))

    def serialize_scribble(self) -> List:
        res = []
        for scribble in self._scribbles:
            res.append(scribble.dict_to_save())
        return res
    
    def serialize_superpixels(self) -> Dict:
        res = dict()
        for sp_method in self.superpixel_methods:
            subres = []
            for superpixel in self.superpixels[sp_method]:
                subres.append(superpixel.dict_to_save())
            res[sp_method.short_string()] = subres
        return res
    
    def serialize_annotations(self) -> Dict:
        res = dict()
        for sp_method in self.superpixel_methods:
            subres = []
            for superpixel in self._annotations[sp_method].annotations:
                subres.append(superpixel.dict_to_save())
            res[sp_method.short_string()] = subres
        return res

    def serialize(self, path="/home/nikita/superpixel_segmenter/output/dump.json"):
        scribbles_list = self.serialize_scribble()
        superpixels_dict = self.serialize_superpixels()
        annotations_dict = self.serialize_annotations()
        res = {
            "scribbles": scribbles_list,
            "superpixels": superpixels_dict,
            "annotations": annotations_dict
        }
        with open(path, "w") as f:
            print(json.dumps(res), file=f)
        #np.save(path + ".npy", res)
    
    def deserialize(self, path="/home/nikita/superpixel_segmenter/output/dump.json"):
        with open(path, "r") as f:
            loaded_dict = json.load(f)
        #data2=np.load(path + ".npy", allow_pickle=True)
        #loaded_dict = data2[()]
        self._scribbles.clear()
        
        self.superpixel_methods.clear()
        for sp_method in list(loaded_dict["superpixels"].keys()):
            parts = sp_method.split("_")
            sp_method_class = None
            if parts[0] == "SLIC":
                self.superpixel_methods.append(
                    SLICSuperpixel(
                        n_clusters=int(parts[1]),
                        compactness=float(parts[2]),
                        sigma=float(parts[3])
                    )
                )
            elif parts[1] == "Felzenszwalb":
                self.superpixel_methods.append(
                    FelzenszwalbSuperpixel(
                        n_clusters=int(parts[1]),
                        compactness=float(parts[2]),
                        scale=float(parts[3])
                    )
                )
            sp_method_class = self.superpixel_methods[-1]
            self._superpixel_ind[sp_method_class] = 0
            superpixels = []
            for superpixel in loaded_dict["superpixels"][sp_method]:
                sp = SuperPixel(superpixel["id"], superpixel["method"], np.array(superpixel["border"], dtype=np.float32), superpixel["parents"])
                self._superpixel_ind[sp_method_class] = max(self._superpixel_ind[sp_method_class], int(superpixel["id"]) + 1)
                superpixels.append(sp)
            self.superpixels[sp_method_class] = superpixels
        for sp_method_class in self.superpixel_methods:
            self._annotation_ind[sp_method_class] = 0
            annos = []
            for anno in loaded_dict["annotations"][sp_method_class.short_string()]:
                sp = AnnotationInstance(int(anno["id"]), int(anno["code"]), np.array(anno["border"], dtype=np.float32), int(anno["parent_superpixel"]))
                annos.append(sp)
                self._annotation_ind[sp_method_class] = max(self._annotation_ind[sp_method_class], int(anno["id"]) + 1)
            self._annotations[sp_method_class] = ImageAnnotation(annotations=annos)
        for scribble_dict in loaded_dict["scribbles"]:
            scribble_to_init = Scribble(0, 0, 0)
            scribble_to_init.set_from_dict(scribble_dict)
            self._scribbles.append(scribble_to_init)
        
        print("Loaded finished")

    def add_scribble(self, scribble: Scribble) -> None:
        scribble.id = self.ind_scrible
        self.ind_scrible += 1
        print("Adding scribble")
        print(f"Current number of scribbles: {len(self._scribbles)}")
        print(f"Scribble ID: {scribble.id}")
        
        self._scribbles.append(scribble)
        self._update_annotations(scribble)

    def _update_annotations(self, last_scribble: Scribble) -> None:
        self.prev_ind_scribble = self.ind_scrible
        scribble_line = shapely.LineString(last_scribble.points)

        superpixel_method = self.superpixel_methods[0]
        superpixel_ind_to_del = []
        superpixel_to_append = []
        scribbles_to_check = []
        annotations_to_del = []
        for cur_superpixel_ind, cur_superpixel in enumerate(self.superpixels[superpixel_method]):
            scrible_bbox = (last_scribble.points[:, 0].min(), last_scribble.points[:, 1].min(), last_scribble.points[:, 0].max(), last_scribble.points[:, 1].max())
            sp_bbox = (cur_superpixel.border[:, 0].min(), cur_superpixel.border[:, 1].min(), cur_superpixel.border[:, 0].max(), cur_superpixel.border[:, 1].max())
            can_intersect = bbox_is_intersect(scrible_bbox, sp_bbox)
            if can_intersect == False:
                continue
            # Проверка пересечения с суперпикселем
            superpixel_polygon = Polygon(cur_superpixel.border)
            if len(last_scribble.points) > 1 and superpixel_polygon.intersects(scribble_line):
                annotated_before = False
                print(f"Checking scribble with superpixel ID: {cur_superpixel.id}")
                # Проверка, аннотирован ли суперпиксель
                for anno_ind, anno in enumerate(self._annotations[superpixel_method].annotations):
                    if anno.parent_superpixel == cur_superpixel.id:
                        annotated_before = True
                        anno_created_scribble = None
                        for scribble in self._scribbles:
                            if anno.parent_scribble[0] == scribble.id:
                                anno_created_scribble = scribble
                                break
                        if anno_created_scribble.params.code != last_scribble.params.code:
                            print(f"Checking scribble with superpixel ID: {cur_superpixel.id}")
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
                            overlay_rgb = convolve2d(overlay_rgb, np.ones((3,3), dtype=np.int32), "same")
                            print("conv finished")
                            sp_mask = slic(
                                self.image_lab,
                                n_segments=10,
                                compactness=superpixel_method.compactness,
                                sigma=superpixel_method.sigma,
                                start_label=1,
                                mask=overlay_rgb
                            )
                            print("sp finished")
                            segments = np.zeros((sp_mask.shape[0]+2, sp_mask.shape[1]+2), dtype=np.int32)
                            segments -= 1
                            segments[1:-1, 1:-1] = sp_mask
                            unique_segments = np.unique(sp_mask)[1:]
                            for segment in unique_segments:
                                print("process segment in supepixel divide", segment)
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
                                

                                superpixel_to_append.append(
                                    SuperPixel(
                                        id=self._superpixel_ind[superpixel_method],
                                        method=superpixel_method.short_string(), 
                                        border=np.around(polygon[::], decimals=7), #len(polygon) // 20 if len(polygon) > 20 else 1
                                        parents=cur_superpixel.id
                                    )
                                )
                                self._superpixel_ind[superpixel_method] += 1
                        else:
                            anno.parent_scribble.append(last_scribble.id)


                        print(f"Superpixel {cur_superpixel.id} is already annotated")
                if annotated_before:
                    continue
                self._annotations[superpixel_method].annotations.append(
                    AnnotationInstance(
                        id=self._annotation_ind[superpixel_method],
                        code=last_scribble.params.code,
                        border=cur_superpixel.border.astype(np.float32),
                        parent_superpixel=cur_superpixel.id,
                        parent_scribble=[last_scribble.id]
                    )
                )
                self._annotation_ind[superpixel_method] += 1
        for sp_to_del in superpixel_ind_to_del[::-1]:
            self.superpixels[superpixel_method].pop(sp_to_del)
        for superpixel in superpixel_to_append:
            self.superpixels[superpixel_method].append(superpixel)
        annotations_to_del = sorted(list(set(annotations_to_del)))
        for anno_to_del in annotations_to_del[::-1]:
            self._annotations[superpixel_method].annotations.pop(anno_to_del)
        for superpixel in superpixel_to_append:
            superpixel_polygon = Polygon(superpixel.border)
            # check with last scribble
            if len(last_scribble.points) > 1 and superpixel_polygon.intersects(scribble_line):
                self._annotations[superpixel_method].annotations.append(
                    AnnotationInstance(
                        id=self._annotation_ind[superpixel_method],
                        code=last_scribble.params.code,
                        border=superpixel.border.astype(np.float32),
                        parent_superpixel=superpixel.id,
                        parent_scribble=[last_scribble.id]
                    )
                )
                self._annotation_ind[superpixel_method] += 1
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
                            parent_scribble=[scribble.id]
                        )
                    )
                    self._annotation_ind[superpixel_method] += 1
                
        
        print(f"Number of annotations after update: {len(self._annotations[superpixel_method].annotations)}")
        print(f"Total number of superpixels: {len(self.superpixels[superpixel_method])}")


    def get_annotation(self, method: SuperPixelMethod) -> ImageAnnotation:
        annos = self._annotations[method]
        # postprocessing
        annos2 = annos
        return ImageAnnotation(annos2)
