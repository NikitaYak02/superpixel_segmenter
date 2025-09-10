import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

from PIL import Image, ImageTk, ImageDraw
import numpy as np
from typing import OrderedDict, List
import copy
import structs
from pathlib import Path
from tkinter import messagebox
import json
from shapely import LineString

MAX_DOWNSCALE_COEFF = 10

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
        if down_coeff >= MAX_DOWNSCALE_COEFF:
            break
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

def is_new_line_valid(new_line: LineString, color: str, 
                      existing_lines: list[LineString], existing_lines_colors: list[str]) -> bool:
    """
    Проверяет, пересекается ли новый LineString с любым из существующих.

    :param new_line: Новый LineString для проверки
    :param existing_lines: Список существующих LineString
    :return: True, если пересечений нет, иначе False
    """
    for existing_color, existing in zip(existing_lines_colors, existing_lines):
        if new_line.intersects(existing) and color != existing_color:
            return False
    return True

class ScribbleApp:
    def __init__(self, master):
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

        # Create a frame for the canvas and scrollbars
        # Create frames
        self.control_frame = tk.Frame(root)
        self.control_frame.grid(row=0, column=0, sticky="nsew")

        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.grid(row=0, column=1, sticky="nsew")

        # Convert to PhotoImage for Tkinter
        # self.tk_image = ImageTk.PhotoImage(self.image)
        # self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        # self.lines = []
        # self.prev_line = []
        # self.lines_color = []   

        # Button to open image
        self.open_button = tk.Button(self.control_frame, text="Load image", command=self.open_image)
        self.open_button.pack(side=tk.TOP)

        self.calncel_button = tk.Button(self.control_frame, text="Cancel prev scribble", command=self.cancel_action)
        self.calncel_button.pack(side=tk.TOP, pady=15)
        # Создание кнопок для добавление суперпиксельного алгоритма и для отмены действия
        self.add_superpixel_anno_algo_button = tk.Button(self.control_frame, text="Add new superpixel method", command=self.add_superpixel_anno_method)
        self.add_superpixel_anno_algo_button.pack(side=tk.TOP, pady=15)
        
        self.cancel_action_button = tk.Button(self.control_frame, text="Save", command=self.save)
        self.cancel_action_button.pack(side=tk.TOP, pady=15)

        self.cancel_action_button = tk.Button(self.control_frame, text="Load", command=self.load)
        self.cancel_action_button.pack(side=tk.TOP, pady=15)

        # Алгоритмы сегментации
        self.segmentation_algorithms = OrderedDict({
            "SLIC": "slic",
            "Felzenszwalb": "fwb",
            "Watershed": "watershed"
        })

        # ComboBox для выбора цвета маркера
        self.color_combobox = ttk.Combobox(self.control_frame, values=list(self.markers.keys()))
        self.color_combobox.current(0)  # Устанавливаем начальное значение
        self.color_combobox.bind("<<ComboboxSelected>>", self.marker_changed)
        self.color_combobox.pack(side=tk.TOP, pady=10)

        self.cur_superpixel_method_combobox = ttk.Combobox(self.control_frame, values=["default"])
        self.cur_superpixel_method_combobox.bind("<<ComboboxSelected>>", self.method_changed)
        self.cur_superpixel_method_combobox.pack(side=tk.TOP, pady=10)
        self.added_superpixels_method: List[structs.SuperPixelMethod] = []

        # Create a zoom slider
        self.zoom_slider = tk.Scale(self.control_frame, from_=0.5, to=15, resolution=0.1, 
                                    orient=tk.HORIZONTAL, label="Zoom")
        self.zoom_slider.bind("<ButtonRelease-1>", self.update_zoom)
        self.zoom_slider.set(1)  # Set initial zoom level
        self.zoom_slider.pack(side=tk.TOP, pady=10)

        self.sens_dist_slider = tk.Scale(self.control_frame, from_=0, to=500, resolution=1, 
                                    orient=tk.HORIZONTAL, label="Superpixels dists'")
        self.sens_dist_slider.bind("<ButtonRelease-1>", self.update_sens_dist)
        self.sens_dist_slider.set(1)  # Set initial zoom level
        self.sens_dist_slider.pack(side=tk.TOP, pady=10)

        self.sens_prop_slider = tk.Scale(self.control_frame, from_=0, to=50, resolution=1, 
                                    orient=tk.HORIZONTAL, label="Superpixels props' dist")
        self.sens_prop_slider.bind("<ButtonRelease-1>", self.update_sens_prop)
        self.sens_prop_slider.set(1)  # Set initial zoom level
        self.sens_prop_slider.pack(side=tk.TOP, pady=10)

        self.draw_borders_var = tk.BooleanVar()
        self.check_button = tk.Checkbutton(
            master=self.control_frame,
            text="draw borders",
            variable=self.draw_borders_var,
            offvalue=False,
            onvalue=True,
            command=self.draw_borders_button_changed,
        )
        self.check_button.pack(side=tk.TOP, pady=10)

        self.draw_annos_var = tk.BooleanVar()
        self.annos_button = tk.Checkbutton(
            master=self.control_frame,
            text="draw anno",
            variable=self.draw_annos_var,
            offvalue=False,
            onvalue=True,
            command=self.draw_annos_button_changed,
        )
        self.annos_button.pack(side=tk.TOP, pady=10)

        self.draw_scrib_var = tk.BooleanVar()
        self.scrib_button = tk.Checkbutton(
            master=self.control_frame,
            text="draw anno",
            variable=self.draw_scrib_var,
            offvalue=False,
            onvalue=True,
            command=self.draw_scrib_button_changed,
        )
        self.scrib_button.pack(side=tk.TOP, pady=10)


        # Слайдеры для алгоритмов сегментации
        # Слайдеры для SLIC
        self.slider_n_segments = tk.Scale(self.control_frame, from_=1, to=500, resolution=1, 
                                        orient=tk.HORIZONTAL, label="n_segments (SLIC)")
        self.slider_compactness = tk.Scale(self.control_frame, from_=0.1, to=100, resolution=0.1, 
                                            orient=tk.HORIZONTAL, label="compactness (SLIC)")
        self.slider_sigma_slic = tk.Scale(self.control_frame, from_=0.1, to=10, resolution=0.1, 
                                        orient=tk.HORIZONTAL, label="sigma (SLIC)")

        # Слайдеры для felzenszwalb
        self.slider_scale_felzenszwalb = tk.Scale(self.control_frame, from_=1000, to=10000, resolution=1, 
                                    orient=tk.HORIZONTAL, label="scale (Felzenszwalb)")
        self.slider_sigma_felzenszwalb = tk.Scale(self.control_frame, from_=1, to=20, resolution=0.1, 
                                                orient=tk.HORIZONTAL, label="sigma (Felzenszwalb)")
        self.slider_min_size = tk.Scale(self.control_frame, from_=30, to=500, resolution=1, 
                                        orient=tk.HORIZONTAL, label="min_size (Felzenszwalb)")
        
        self.slider_n_segm_watershed = tk.Scale(self.control_frame, from_=10, to=1000, resolution=1, 
                                    orient=tk.HORIZONTAL, label="n_segm (watershed)")
        self.slider_comp_watershed = tk.Scale(self.control_frame, from_=1e-5, to=20, resolution=1e-5, 
                                                orient=tk.HORIZONTAL, label="comp (watershed)")

        # Устанавливаем начальные значения для слайдеров
        self.slider_n_segments.set(500)
        self.slider_compactness.set(17.0)
        self.slider_sigma_slic.set(1.0)
        self.slider_scale_felzenszwalb.set(10)
        self.slider_sigma_felzenszwalb.set(1.0)
        self.slider_min_size.set(10)
        self.slider_n_segm_watershed.set(500)
        self.slider_comp_watershed.set(1e-4)

        # Добавляем слайдеры в интерфейс
        self.slider_n_segments.pack(side=tk.TOP, pady=10)
        self.slider_compactness.pack(side=tk.TOP, pady=10)
        self.slider_sigma_slic.pack(side=tk.TOP, pady=10)

        self.slider_scale_felzenszwalb.pack(side=tk.TOP, pady=10)
        self.slider_sigma_felzenszwalb.pack(side=tk.TOP, pady=10)
        self.slider_min_size.pack(side=tk.TOP, pady=10)
        
        self.slider_n_segm_watershed.pack(side=tk.TOP, pady=10)
        self.slider_comp_watershed.pack(side=tk.TOP, pady=10)

        # Скрываем слайдеры по умолчанию
        self.hide_all_sliders()

        # ComboBox для выбора алгоритма сегментации
        self.algorithm_combobox = ttk.Combobox(self.control_frame, values=list(self.segmentation_algorithms.keys()))
        self.algorithm_combobox.current(0)  # Устанавливаем начальное значение
        self.algorithm_combobox.bind("<<ComboboxSelected>>", self.algorithm_changed)
        self.algorithm_combobox.pack(side=tk.TOP, pady=10)

        self.algorithm_changed(None)


        self.last_x, self.last_y = None, None
        self.scale = 1.0  # Zoom level

    def draw_borders_button_changed(self):
        self.update_zoom(None)

    def draw_annos_button_changed(self):
        self.update_zoom(None)

    def draw_scrib_button_changed(self):
        self.update_zoom(None)

    def create_canvas(self):
        # Create a canvas
        self.canvas = tk.Canvas(
            self.canvas_frame, 
            width=1200, #min(self.image.width, 1200), 
            height=800, #min(self.image.height, 800), 
            bg='white'
        )

        # Create scrollbars
        self.h_scrollbar = tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.h_scrollbar.pack(side=tk.TOP, fill=tk.X)

        self.canvas.pack(side=tk.TOP, expand=True)

        self.v_scrollbar = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Configure the canvas to use the scrollbars
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)
        
        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            if hasattr(self, 'canvas'):
                self.canvas.destroy()
                self.h_scrollbar.destroy()
                self.v_scrollbar.destroy()
            self.original_image = Image.open(file_path)
            self.image, self.downscale_coeff = get_downscaled_image(self.original_image.copy(), 15000)
            self.create_canvas()
            self.superpixel_anno_algo = structs.SuperPixelAnnotationAlgo(
                downscale_coeff=1,
                superpixel_methods=[],
                image_path="",
                image=self.image
            )
            self.lines = []
            self.prev_line = []
            self.lines_color = []   
            self.display_image()


    def display_image(self):
        img_resized = self.image.resize((int(self.image.width * self.scale), int(self.image.height * self.scale)))
        self.tk_image = ImageTk.PhotoImage(img_resized)

        self.canvas.delete("all")
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))


    def save(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Annotation File"
        )
        if file_path:  # Проверяем что пользователь не отменил диалог
            try:
                # Создаем директорию если необходимо
                Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                self.superpixel_anno_algo.serialize(file_path)
                messagebox.showinfo("Success", f"Annotations saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file:\n{str(e)}")

    def load(self):
        file_path = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Open Annotation File"
        )
        if file_path:
            try:
                self.superpixel_anno_algo.deserialize(file_path)
                self._update_ui_after_loading()
                messagebox.showinfo("Success", f"Annotations loaded from:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")

    def _update_ui_after_loading(self):
        """Обновление интерфейса после загрузки данных"""
        self.added_superpixels_method.clear()
        for method in self.superpixel_anno_algo.superpixel_methods:
            self.added_superpixels_method.append(method)
            new_method = method.short_string()
            current_values = list(self.cur_superpixel_method_combobox['values'])
            if new_method not in current_values:
                current_values.append(new_method)
                self.cur_superpixel_method_combobox['values'] = current_values
        self.update_zoom(self.zoom_slider.get())
    
    def cancel_action(self):
        self.superpixel_anno_algo.cancel_prev_act()
        self.lines.pop()
        self.lines_color.pop()
        self.update_zoom(None)

    def marker_changed(self, event):
        self.curr_marker = event.widget.get()
        self._color = self.markers[self.curr_marker][1]
        return

    def method_changed(self, _):
        print("Method combobox changed")
        self.update_zoom(self.zoom_slider.get())

    def paint(self, event):
        # Get current scroll position
        x_scroll = self.canvas.xview()[0] * self.image.width #* float(self.zoom_slider.get())
        y_scroll = self.canvas.yview()[0] * self.image.height

        # Adjust coordinates based on current scale and scroll position
        # print(event.x, x_scroll, event.y, y_scroll, self.image.size)
        x = (max(event.x, 1) + x_scroll) 
        y = (max(event.y, 1) + y_scroll) 
        if self.last_x is not None and self.last_y is not None:
            # Draw line on canvas
            self.canvas.create_line((self.last_x, self.last_y,
                                     x , y), fill=self._color, width=6)
        self.prev_line.append((
            x / self.image.width,
            y / self.image.height
        ))
        self.last_x, self.last_y = x, y

    def reset(self, event):
        self.last_x, self.last_y = None, None
        scribble = np.array(self.prev_line, dtype=np.float32)
        prev_lines = [LineString(scrib) for scrib in self.lines]
        if not is_new_line_valid(LineString(scribble), self._color, prev_lines, self.lines_color) or len(scribble) < 2:
            self.prev_line = []
            self.update_zoom(self.zoom_slider.get())
            return

        self.lines.append(scribble)
        self.lines_color.append(self._color)
        print("scribble sum:", scribble.sum())
        key = get_key_by_value(self.markers, self._color)
        sp_method = None
        if self.selected_algorithm == "SLIC":
            sp_method = structs.SLICSuperpixel(
                n_clusters=self.slider_n_segments.get(),
                compactness=self.slider_compactness.get(),
                sigma=self.slider_sigma_slic.get()
            )
        elif self.selected_algorithm == "Watershed":
            sp_method = structs.WatershedSuperpixel(
                compactness = self.slider_comp_watershed.get(),
                n_components = self.slider_n_segm_watershed.get()
            )
        elif self.selected_algorithm == "Felzenszwalb":
            sp_method = structs.FelzenszwalbSuperpixel(
                min_size = self.slider_min_size.get(),
                sigma = self.slider_sigma_felzenszwalb.get(),
                scale = self.slider_scale_felzenszwalb.get()
            )
        self.scrible_counter += 1
        self.prev_line = []
        self.superpixel_anno_algo._create_superpixel_for_scribble(
            structs.Scribble(
                id=self.scrible_counter,
                points=scribble,
                params=structs.ScribbleParams(
                    radius=1,
                    code=self.markers[key][0]
                )
            ),
            sp_method
        )
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
        self.update_zoom(self.zoom_slider.get())

    def hide_all_sliders(self):
        """Скрывает все слайдеры для параметров сегментации."""
        self.slider_n_segments.pack_forget()
        self.slider_compactness.pack_forget()
        self.slider_sigma_slic.pack_forget()
        self.slider_scale_felzenszwalb.pack_forget()
        self.slider_sigma_felzenszwalb.pack_forget()
        self.slider_min_size.pack_forget()
        self.slider_n_segm_watershed.pack_forget()
        self.slider_comp_watershed.pack_forget()

    def algorithm_changed(self, event):
        self.selected_algorithm = self.algorithm_combobox.get()
        print(f"Selected Segmentation Algorithm: {self.selected_algorithm}")

        # Скрываем все слайдеры и показываем только нужные
        self.hide_all_sliders()

        if self.selected_algorithm == "SLIC":
            self.slider_n_segments.pack(side=tk.TOP, pady=20)
            self.slider_compactness.pack(side=tk.TOP, pady=20)
            self.slider_sigma_slic.pack(side=tk.TOP, pady=20)
        elif self.selected_algorithm == "Felzenszwalb":
            self.slider_scale_felzenszwalb.pack(side=tk.TOP, pady=20)
            self.slider_sigma_felzenszwalb.pack(side=tk.TOP, pady=20)
            self.slider_min_size.pack(side=tk.TOP, pady=20)
        elif self.selected_algorithm == "Watershed":
            self.slider_n_segm_watershed.pack(side=tk.TOP, pady=10)
            self.slider_comp_watershed.pack(side=tk.TOP, pady=10)


    def update_sens_prop(self, _):
        if hasattr(self, "superpixel_anno_algo"):
            self.superpixel_anno_algo._property_dist = float(self.sens_prop_slider.get())

    def update_sens_dist(self, _):
        if hasattr(self, "superpixel_anno_algo"):
            self.superpixel_anno_algo._superpixel_radius = float(self.sens_dist_slider.get())
            self.superpixel_anno_algo._superpixel_radius /= self.superpixel_anno_algo.image_lab.shape[0]

    def update_zoom(self, value, signal_from_paint=False):
        if not hasattr(self, "superpixel_anno_algo"):
            return
        if not signal_from_paint:
            self.canvas.delete("all")
            self.scale = float(self.zoom_slider.get())
            # Resize the image based on the current scale
            new_width = int(self.original_image.width  / self.downscale_coeff * self.scale)
            new_height = int(self.original_image.height / self.downscale_coeff * self.scale)
            self.image = self.original_image.resize((new_width, new_height), Image.LANCZOS)
            self.cur_superpixel_method_short_string = self.cur_superpixel_method_combobox.get()
            print(self.cur_superpixel_method_short_string)
            self.overlay = Image.new("RGBA", self.image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(self.overlay)
        if not (self.cur_superpixel_method_short_string in ["default", ""]):
            tgt_sh = self.image.size
            for sp_method in self.added_superpixels_method:
                if sp_method.short_string() == self.cur_superpixel_method_short_string:
                    break
            #print(self.superpixel_anno_algo._annotations)
            if self.draw_annos_var.get():
                annos = self.superpixel_anno_algo._annotations[sp_method]
                for superpixel in annos.annotations:
                    proc_borders = copy.deepcopy(superpixel.border)
                    proc_borders[:, 0] *= tgt_sh[0]
                    proc_borders[:, 1] *= tgt_sh[1]
                    polygon = [(x[0], x[1]) for x in proc_borders]
                    marker_name = get_key_by_value_markers(self.markers, superpixel.code)
                    color = str(self.markers[marker_name][1])
                    color = (int(color[1:3], base=16), int(color[3:5], base=16), int(color[5:7], base=16), 125)
                    draw.polygon(polygon, fill=color)
                #self.canvas.create_polygon(*(proc_borders.reshape(-1)), fill="#00FF00", alpha=0.5, outline="red", width=2)
            #print(self.superpixel_anno_algo.superpixels)
            if not signal_from_paint and self.draw_borders_var.get():
                superpixels = self.superpixel_anno_algo.superpixels[sp_method]
                for superpixel in superpixels:
                    proc_borders = copy.deepcopy(superpixel.border)
                    proc_borders[:, 0] *= tgt_sh[0]
                    proc_borders[:, 1] *= tgt_sh[1]
                    polygon = [(x[0], x[1]) for x in proc_borders]
                    draw.polygon(polygon, outline="yellow")
        overlay_rgb = self.overlay.convert("RGB")
        overlay_alpha = self.overlay.split()[-1]  # Alpha channel
        image = self.image.copy()
        image.paste(overlay_rgb, (0, 0), mask=overlay_alpha)

        self.tk_image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

        # Redraw all lines on the resized image
        if self.draw_scrib_var.get():
            for scribble in self.superpixel_anno_algo._scribbles:
                line = scribble.points
                color = None
                for (idx, color_in_markers) in self.markers.values():
                    if scribble.params.code == idx:
                        color = color_in_markers
                        break
                for i in range(len(line) - 1):
                    self.canvas.create_line((new_width * line[i][0], new_height * line[i][1],
                                    new_width * line[i+1][0], new_height * line[i+1][1]), fill=color, width=6)
        # for line, color in zip(self.lines, self.lines_color):
        #    for i in range(len(line) - 1):
        #        self.canvas.create_line((new_width * line[i][0], new_height * line[i][1],
        #                        new_width * line[i+1][0], new_height * line[i+1][1]), fill=color, width=2)

        # # Update scroll region
        # self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
    
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
            self.superpixel_anno_algo.add_superpixel_method(
                structs.WatershedSuperpixel(
                    compactness = self.slider_comp_watershed.get(),
                    n_components = self.slider_n_segm_watershed.get()
                )
            )
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
    app = ScribbleApp(root)
    root.mainloop()