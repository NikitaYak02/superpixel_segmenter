import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import numpy as np
from typing import OrderedDict, List
from tkinter import ttk
import copy
import structs

def get_downscaled_image(image: Image, max_size: int):
    """
    image: PIL.Image
    max_size: int - ma resolution for image
    return: resized image and downscale coef (int)
    """
    down_coeff = 1
    max_side = max(image.size[0], image.size[1])
    while max_side // down_coeff > max_size:
        down_coeff += 1
    image = image.resize((int(image.size[0] / down_coeff), int(image.size[1] / down_coeff)))
    return image, down_coeff

# Функция для получения ключа по значению
def get_key_by_value(ordered_dict, value):
    for key, val in ordered_dict.items():
        if val[1] == value:
            return key
    return None  # Если значение не найдено

def get_key_by_value_markers(ordered_dict, value):
    for key, val in ordered_dict.items():
        if val[0] == value:
            return key
    return None  # Если значение не найдено

class ScribbleApp:
    def __init__(self, master, image_path):
        self.master = master
        self.master.title("Scribble on Image")
        self.scrible_counter = 0
        # цвета маркеров
        self.markers = OrderedDict(
            {
                "background": "#1c1818",
                "chalcopyrite": "#ff0000",
                "galena": "#cbff00",
                "magnetite": "#00ff66",
                "bornite": "#0065ff",
                "pyrrhotite": "#cc00ff",
                "pyrite/marcasite": "#ff4c4c",
                "pentlandite": "#dbff4c",
                "sphalerite": "#4cff93",
                "arsenopyrite": "#4c93ff",
                "hematite": "#db4cff",
                "tenantite-tetrahedrite group": "#ff9999",
                "covelline": "#eaff99",
            }
        )
        """
        self.markers_idx = OrderedDict(
            {
                "background": 0,
                "chalcopyrite": 1,
                "galena": 2,
                "magnetite": 3,
                "bornite": 4,
                "pyrrhotite": 5,
                "pyrite/marcasite": 6,
                "pentlandite": 7,
                "sphalerite": 8,
                "arsenopyrite": 9,
                "hematite": 10,
                "tenantite-tetrahedrite group": 11,
                "covelline": 12,
            }
        )
        """
        self.curr_marker, self._color = list(self.markers.items())[0]

        # dict["marker_name": (marker_idx, marker_hex_color)]
        self.markers = {
            item[0]: (i, item[1])
            for i, item in enumerate(self.markers.items(), start=1)
        }
        
        # Load the image
        self.original_image = Image.open(image_path)
        self.image, self.downscale_coeff = get_downscaled_image(self.original_image.copy(), 1500)
        self.resized_image = copy.deepcopy(self.image)
        self.superpixel_anno_algo = structs.SuperPixelAnnotationAlgo(
            downscale_coeff=1,
            superpixel_methods=[],
            image_path="",
            image=self.image
        )

        # Create a frame for the canvas and scrollbars
        self.frame = tk.Frame(master)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Create a canvas
        self.canvas = tk.Canvas(self.frame, width=self.image.width, height=self.image.height)
        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create scrollbars
        self.v_scrollbar = tk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scrollbar = tk.Scrollbar(master, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.h_scrollbar.pack(fill=tk.X)

        # Configure the canvas to use the scrollbars
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)

        # Convert to PhotoImage for Tkinter
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.lines = []
        self.prev_line = []
        self.lines_color = []   

        # Create a frame for controls on the left side
        self.control_frame = tk.Frame(master)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Создание кнопок для добавление суперпиксельного алгоритма и для отмены действия
        self.add_superpixel_anno_algo_button = tk.Button(self.control_frame, text="Add superpixel anno algo", command=self.add_superpixel_anno_method)
        self.add_superpixel_anno_algo_button.pack(side=tk.LEFT, pady=30)
        
        self.cancel_action_button = tk.Button(self.control_frame, text="Save", command=self.save)
        self.cancel_action_button.pack(side=tk.LEFT, pady=30)

        self.cancel_action_button = tk.Button(self.control_frame, text="Load", command=self.load)
        self.cancel_action_button.pack(side=tk.LEFT, pady=30)

        # Алгоритмы сегментации
        self.segmentation_algorithms = OrderedDict({
            "SLIC": "slic",
            "Felzenszwalb": "fwb",
        })

        # ComboBox для выбора цвета маркера
        self.color_combobox = ttk.Combobox(self.control_frame, values=list(self.markers.keys()))
        self.color_combobox.current(0)  # Устанавливаем начальное значение
        self.color_combobox.bind("<<ComboboxSelected>>", self.marker_changed)
        self.color_combobox.pack(side=tk.LEFT, pady=10)

        self.cur_superpixel_method_combobox = ttk.Combobox(self.control_frame, values=["default"])
        self.cur_superpixel_method_combobox.bind("<<ComboboxSelected>>", self.method_changed)
        self.cur_superpixel_method_combobox.pack(side=tk.LEFT, pady=10)
        self.added_superpixels_method: List[structs.SuperPixelMethod] = []

        # Create a zoom slider
        self.zoom_slider = tk.Scale(self.control_frame, from_=0.5, to=5, resolution=0.1, 
                                    orient=tk.HORIZONTAL, label="Zoom")
        self.zoom_slider.bind("<ButtonRelease-1>", self.update_zoom)
        self.zoom_slider.set(1)  # Set initial zoom level
        self.zoom_slider.pack(side=tk.LEFT, pady=10)  # Убедитесь, что слайдер добавлен в интерфейс

        # Слайдеры для алгоритмов сегментации
        # Слайдеры для SLIC
        self.slider_n_segments = tk.Scale(self.control_frame, from_=1, to=500, resolution=1, 
                                        orient=tk.HORIZONTAL, label="n_segments (SLIC)")
        self.slider_compactness = tk.Scale(self.control_frame, from_=0.1, to=100, resolution=0.1, 
                                            orient=tk.HORIZONTAL, label="compactness (SLIC)")
        self.slider_sigma_slic = tk.Scale(self.control_frame, from_=0.1, to=10, resolution=0.1, 
                                        orient=tk.HORIZONTAL, label="sigma (SLIC)")

        # Слайдеры для felzenszwalb
        self.slider_scale_felzenszwalb = tk.Scale(self.control_frame, from_=50, to=500, resolution=1, 
                                    orient=tk.HORIZONTAL, label="scale (Felzenszwalb)")
        self.slider_sigma_felzenszwalb = tk.Scale(self.control_frame, from_=1, to=20, resolution=0.1, 
                                                orient=tk.HORIZONTAL, label="sigma (Felzenszwalb)")
        self.slider_min_size = tk.Scale(self.control_frame, from_=30, to=500, resolution=1, 
                                        orient=tk.HORIZONTAL, label="min_size (Felzenszwalb)")

        # Устанавливаем начальные значения для слайдеров
        self.slider_n_segments.set(500)
        self.slider_compactness.set(10.0)
        self.slider_sigma_slic.set(2.0)
        self.slider_scale_felzenszwalb.set(10)
        self.slider_sigma_felzenszwalb.set(1.0)
        self.slider_min_size.set(10)

        # Добавляем слайдеры в интерфейс
        self.slider_n_segments.pack(side=tk.LEFT, pady=10)
        self.slider_compactness.pack(side=tk.LEFT, pady=10)
        self.slider_sigma_slic.pack(side=tk.LEFT, pady=10)

        self.slider_scale_felzenszwalb.pack(side=tk.LEFT, pady=10)
        self.slider_sigma_felzenszwalb.pack(side=tk.LEFT, pady=10)
        self.slider_min_size.pack(side=tk.LEFT, pady=10)

        # Скрываем слайдеры по умолчанию
        self.hide_all_sliders()

        # ComboBox для выбора алгоритма сегментации
        self.algorithm_combobox = ttk.Combobox(self.control_frame, values=list(self.segmentation_algorithms.keys()))
        self.algorithm_combobox.current(0)  # Устанавливаем начальное значение
        self.algorithm_combobox.bind("<<ComboboxSelected>>", self.algorithm_changed)
        self.algorithm_combobox.pack(side=tk.LEFT, pady=10)

        self.algorithm_changed(None)

        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        self.last_x, self.last_y = None, None
        self.scale = 1.0  # Zoom level

    def save(self):
        self.superpixel_anno_algo.serialize()

    def load(self):
        self.superpixel_anno_algo.deserialize()
        for i in range(len(self.superpixel_anno_algo.superpixel_methods)):
            self.added_superpixels_method.append(self.superpixel_anno_algo.superpixel_methods[i])
            new_method = self.superpixel_anno_algo.superpixel_methods[i].short_string()
            current_values = list(self.cur_superpixel_method_combobox['values'])
            if not (new_method in current_values):
                current_values.append(new_method)
                self.cur_superpixel_method_combobox['values'] = current_values
        self.update_zoom(0)
    
    def cancel_action(self):
        self.superpixel_anno_algo.cancel_prev_act()

    def marker_changed(self, event):
        self.curr_marker = event.widget.get()
        self._color = self.markers[self.curr_marker][1]
        return

    def method_changed(self, _):
        print("Method combobox changed")
        self.update_zoom(self.zoom_slider.get())

    def paint(self, event):
        # Get current scroll position
        x_scroll = self.canvas.xview()[0] * self.resized_image.width #* float(self.zoom_slider.get())
        y_scroll = self.canvas.yview()[0] * self.resized_image.height

        # Adjust coordinates based on current scale and scroll position
        # print(event.x, x_scroll, event.y, y_scroll, self.resized_image.size)
        x = (max(event.x, 1) + x_scroll) 
        y = (max(event.y, 1) + y_scroll) 
        if self.last_x is not None and self.last_y is not None:
            # Draw line on canvas
            self.canvas.create_line((self.last_x, self.last_y,
                                     x , y), fill=self._color, width=2)
        self.prev_line.append((
            x / self.resized_image.width,
            y / self.resized_image.height
        ))
        self.last_x, self.last_y = x, y

    def reset(self, event):
        self.last_x, self.last_y = None, None
        scribble = np.array(self.prev_line, dtype=np.float32)
        if len(scribble) < 2:
            return
        self.lines.append(scribble)
        self.lines_color.append(self._color)
        print("scribble sum:", scribble.sum())
        key = get_key_by_value(self.markers, self._color)
        self.superpixel_anno_algo.add_scribble(
            structs.Scribble(
                id=self.scrible_counter,
                points=scribble,
                params=structs.ScribbleParams(
                    radius=1,
                    code=self.markers[key][0]
                )
            )
        )
        self.scrible_counter += 1
        self.prev_line = []
        self.update_zoom(self.zoom_slider.get())

    def hide_all_sliders(self):
        """Скрывает все слайдеры для параметров сегментации."""
        self.slider_n_segments.pack_forget()
        self.slider_compactness.pack_forget()
        self.slider_sigma_slic.pack_forget()
        self.slider_scale_felzenszwalb.pack_forget()
        self.slider_sigma_felzenszwalb.pack_forget()
        self.slider_min_size.pack_forget()

    def algorithm_changed(self, event):
        self.selected_algorithm = self.algorithm_combobox.get()
        print(f"Selected Segmentation Algorithm: {self.selected_algorithm}")

        # Скрываем все слайдеры и показываем только нужные
        self.hide_all_sliders()

        if self.selected_algorithm == "SLIC":
            self.slider_n_segments.pack(side=tk.LEFT, pady=20)
            self.slider_compactness.pack(side=tk.LEFT, pady=20)
            self.slider_sigma_slic.pack(side=tk.LEFT, pady=20)
        elif self.selected_algorithm == "Felzenszwalb":
            self.slider_scale_felzenszwalb.pack(side=tk.LEFT, pady=20)
            self.slider_sigma_felzenszwalb.pack(side=tk.LEFT, pady=20)
            self.slider_min_size.pack(side=tk.LEFT, pady=20)



    def update_zoom(self, value):
        self.canvas.delete("all")
        self.scale = float(self.zoom_slider.get())
        # Resize the image based on the current scale
        new_width = int(self.original_image.width  / self.downscale_coeff * self.scale)
        new_height = int(self.original_image.height / self.downscale_coeff * self.scale)
        self.resized_image = self.original_image.resize((new_width, new_height), Image.LANCZOS)
        self.cur_superpixel_method_short_string = self.cur_superpixel_method_combobox.get()
        print(self.cur_superpixel_method_short_string)
        overlay = Image.new("RGBA", self.resized_image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        if not (self.cur_superpixel_method_short_string in ["default", ""]):
            tgt_sh = self.resized_image.size
            for sp_method in self.added_superpixels_method:
                if sp_method.short_string() == self.cur_superpixel_method_short_string:
                    break
            #print(self.superpixel_anno_algo._annotations)
            annos = self.superpixel_anno_algo._annotations[sp_method]
            for superpixel in annos.annotations:
                proc_borders = copy.deepcopy(superpixel.border)
                proc_borders[:, 0] *= tgt_sh[0]
                proc_borders[:, 1] *= tgt_sh[1]
                polygon = [(x[0], x[1]) for x in proc_borders]
                marker_name = get_key_by_value_markers(self.markers, superpixel.code)
                color = str(self.markers[marker_name][1])
                color = (int(color[1:3], base=16), int(color[3:5], base=16), int(color[5:7], base=16), 125)
                draw.polygon(polygon, fill=color, outline="yellow")
                #self.canvas.create_polygon(*(proc_borders.reshape(-1)), fill="#00FF00", alpha=0.5, outline="red", width=2)
            #print(self.superpixel_anno_algo.superpixels)
            superpixels = self.superpixel_anno_algo.superpixels[sp_method]
            for superpixel in superpixels:
                proc_borders = copy.deepcopy(superpixel.border)
                proc_borders[:, 0] *= tgt_sh[0]
                proc_borders[:, 1] *= tgt_sh[1]
                polygon = [(x[0], x[1]) for x in proc_borders]
                draw.polygon(polygon, outline="yellow")
        overlay_rgb = overlay.convert("RGB")
        overlay_alpha = overlay.split()[-1]  # Alpha channel
        image = self.resized_image.copy()
        image.paste(overlay_rgb, (0, 0), mask=overlay_alpha)

        self.tk_image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

        # Redraw all lines on the resized image
        for scribble in self.superpixel_anno_algo._scribbles:
            line = scribble.points
            color = None
            for (idx, color_in_markers) in self.markers.values():
                if scribble.params.code == idx:
                    color = color_in_markers
            for i in range(len(line) - 1):
                self.canvas.create_line((new_width * line[i][0], new_height * line[i][1],
                                new_width * line[i+1][0], new_height * line[i+1][1]), fill=color, width=2)
        #for line, color in zip(self.lines, self.lines_color):
        #    for i in range(len(line) - 1):
        #        self.canvas.create_line((new_width * line[i][0], new_height * line[i][1],
        #                        new_width * line[i+1][0], new_height * line[i+1][1]), fill=color, width=2)

        # Update scroll region
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
    
    def add_superpixel_anno_method(self):
        # Get the current scrollbar positions
        self.selected_algorithm = self.algorithm_combobox.get()
        print(f"Selected Segmentation Algorithm: {self.selected_algorithm}")

        if self.selected_algorithm == "SLIC":
            self.superpixel_anno_algo.add_superpixel_method(
                structs.SLICSuperpixel(
                    n_clusters=self.slider_n_segments.get(),
                    compactness=self.slider_compactness.get(),
                    sigma=self.slider_sigma_slic.get()
                )
            )
        elif self.selected_algorithm == "Watershed":
            # Здесь можно добавить параметры для Watershed, если они нужны
            pass
        elif self.selected_algorithm == "Felzenszwalb":
            self.superpixel_anno_algo.add_superpixel_method(
                structs.FelzenszwalbSuperpixel(
                    min_size = self.slider_min_size.get(),
                    sigma = self.slider_sigma_felzenszwalb.get(),
                    scale = self.slider_scale_felzenszwalb.get()
                )
            )
        self.added_superpixels_method.append(self.superpixel_anno_algo.superpixel_methods[-1])
        new_method = self.superpixel_anno_algo.superpixel_methods[-1].short_string()
        current_values = list(self.cur_superpixel_method_combobox['values'])
        if not (new_method in current_values):
            current_values.append(new_method)
            self.cur_superpixel_method_combobox['values'] = current_values 

    def save_image(self, filename):
        self.image.save(filename)
        
if __name__ == "__main__":
    root = tk.Tk()
    app = ScribbleApp(root, "/home/nikita/superpixel_segmenter/S1_v1/imgs/test/01.jpg")
    root.mainloop()