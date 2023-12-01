import tkinter as tk
from tkinter import Tk, Frame, Button, filedialog
from tkinter import simpledialog
from PIL import Image, ImageTk
import numpy as np
import os
from tkinter import messagebox
import cv2


class Window:
    tkWindow = None
    parent = None

    def __init__(self, tkWindow, parent):
        self.parent = parent
        self.tkWindow = tkWindow
        self.children = []

    def add_child(self, window):
        self.children.append(window)

    def close(self):
        [window.close() for window in self.children]
        self.tkWindow.destroy()

    def isRoot(self):
        if self.parent is None:
            return True
        else:
            return False

class Binary:
    def __init__(self, image):
        self.image = image
        unique_colors = set(self.image.getdata())
        self.is_gray_scale = True if len(unique_colors) <= 256 else False

    def negate(self):
        image = self.preprocess_image()
        pixel_values = list(image.getdata())
        negated_values = [255 - value for value in pixel_values]
        negated_image = Image.new('L', self.image.size)
        negated_image.putdata(negated_values)
        return negated_image

    def threshold(self, threshold_value, preserve=False):
        image = self.preprocess_image()
        pixel_values = list(image.getdata())
        threshold_values = [0 if value < threshold_value else 255 if not preserve else value for value in pixel_values]
        threshold_image = Image.new('L', self.image.size)
        threshold_image.putdata(threshold_values)
        return threshold_image

    def reduce_grayscale_levels(self, levels):
        image = self.preprocess_image()
        pixel_values = list(image.getdata())
        new_pixel_values = [int(value / 255 * levels) * (255 // (levels - 1)) for value in pixel_values]
        new_image = Image.new('L', self.image.size)
        new_image.putdata(new_pixel_values)
        return new_image

    def preprocess_image(self):
        image = self.image if self.image.mode == "L" else self.image.convert("L")
        return image

# LAB3
    def logical_not(self):
        image = self.preprocess_image()
        pixel_values = list(image.getdata())
        negated_values = [255 - value for value in pixel_values]
        negated_image = Image.new('L', self.image.size)
        negated_image.putdata(negated_values)
        return negated_image

    def logical_and(self, image1):
        image = self.preprocess_image()
        image1 = image1.preprocess_image()
        result_image = Image.new('L', image1.size)

        for i in range(image1.width):
            for j in range(image1.height):
                # Perform logical AND operation for each pixel
                pixel_value1 = image.getpixel((i, j))
                pixel_value2 = image1.getpixel((i, j))
                result_pixel = min(pixel_value1, pixel_value2)
                result_image.putpixel((i, j), result_pixel)

        return result_image

    def logical_or(self, image1):
        image = self.preprocess_image()
        image1 = image1.preprocess_image()

        pixel_values1 = list(image1.getdata())
        pixel_values2 = list(image.getdata())

        or_values = [max(value1, value2) for value1, value2 in zip(pixel_values1, pixel_values2)]

        or_image = Image.new('L', self.image.size)
        or_image.putdata(or_values)
        return or_image

    def logical_xor(self, image_2):
        image1 = self.preprocess_image()
        image2 = image_2.preprocess_image()
        pixel_data1 = np.array(image1)
        pixel_data2 = np.array(image2)
        xor_array = np.bitwise_xor(pixel_data1, pixel_data2)
        xor_image = Image.fromarray(xor_array)
        return xor_image

class Histogram:
    def __init__(self, image):
        self.image = image
        unique_colors = set(self.image.getdata())
        self.is_gray_scale = True if len(unique_colors) <= 256 else False


    def calculate_histogram(self):
        if self.is_gray_scale:
            histogram = self.image.histogram()
            return histogram, None, None
        else:
            histogram = self.image.histogram()
            r = histogram[0:256]
            g = histogram[256:512]
            b = histogram[512:768]
            return r, g, b

    def linear_histogram_stretching(self):
        pixel_values = list(self.image.getdata())
        if self.is_gray_scale:  # Check if the image is in grayscale mode
            # Get the pixel values as a list

            stretched_values = self.linear_stretch_channel(pixel_values)

            # Create a new image with the stretched values
            stretched_image = Image.new('L', self.image.size)
            stretched_image.putdata(stretched_values)
            return stretched_image
        else:
            # Separate the pixel values into individual color channels
            red_channel, green_channel, blue_channel = zip(*pixel_values)

            # Perform linear histogram stretching for each color channel
            stretched_red = self.linear_stretch_channel(red_channel)
            stretched_green = self.linear_stretch_channel(green_channel)
            stretched_blue = self.linear_stretch_channel(blue_channel)
            stretched_red_image = Image.new('L', self.image.size)
            stretched_red_image.putdata(stretched_red)
            stretched_green_image = Image.new('L', self.image.size)
            stretched_green_image.putdata(stretched_green)
            stretched_blue_image = Image.new('L', self.image.size)
            stretched_blue_image.putdata(stretched_blue)

            # Combine the stretched color channels into a new image
            stretched_image = Image.merge('RGB', (stretched_red_image, stretched_green_image, stretched_blue_image))
            return stretched_image

    def linear_stretch_channel(self, channel):
            min_value = min(channel)
            max_value = max(channel)
            stretched_values = [int((pixel - min_value) / (max_value - min_value) * 255) for pixel in channel]
            return stretched_values

    def nonlinear_stretch_image(self, gamma, sat=False):
        img_data = np.array(self.image, dtype=np.float32)
        img_data /= 255.0
        img_data = np.power(img_data, gamma)
        img_data *= 255
        img_data = np.clip(img_data, 0, 255)

        if sat:
            low_percentile, high_percentile = np.percentile(img_data, [2.5, 97.5])
            img_data = np.interp(img_data, (low_percentile, high_percentile), (0, 255))

        stretched_image = Image.fromarray(np.uint8(img_data))

        return stretched_image

    def histogram_equalization(self):
        img_data = np.array(self.image)
        if self.image.mode in ["RGB", "RGBA"]:
            for channel in range(3):
                img_data[..., channel] = self.equalize_channel(img_data[..., channel])
        else:
            img_data = self.equalize_channel(img_data)
        return Image.fromarray(img_data)

    def histogram_stretching(self, p1, p2, q3, q4):
        img_data = np.array(self.image)
        if self.image.mode == "RGB":
            for channel in range(3):
                img_data[..., channel] = np.interp(img_data[..., channel], (p1, p2), (q3, q4))
        else:
            img_data = np.interp(img_data, (p1, p2), (q3, q4))
        return Image.fromarray(np.uint8(img_data))

    def equalize_channel(self, img_channel):
        hist, bins = np.histogram(img_channel.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        img_equalized = cdf[img_channel]
        return img_equalized

    def linear_stretch_channel(self, channel):
        min_value = min(channel)
        max_value = max(channel)
        stretched_values = [
            int((pixel - min_value) / (max_value - min_value) * 255)
            for pixel in channel
        ]

        return stretched_values

    def create_histogram_window(self, root, hist_data, color):
        root_hist = tk.Toplevel(root)
        root_hist.title(f"{color} Histogram")

        canvas = tk.Canvas(root_hist, width=800, height=600, bg='white')
        canvas.pack()

        max_value = max(hist_data)

        for i in range(256):
            if hist_data[i] > 0:
                canvas.create_line(50 + i * 3, 550, 50 + i * 3, 550 - (hist_data[i] * 500 / max_value), fill=color)

        for i in range(0, 256, 10):
            canvas.create_text(50 + i * 3, 570, anchor=tk.N, text=str(i), fill="black")

        for i in range(0, max_value + 1, int(max_value / 10)):
            canvas.create_text(50, 550 - (i * 500 / max_value), anchor=tk.E, text=str(i), fill="black")

        def show_value(event):
            x = canvas.canvasx(event.x)
            y = canvas.canvasy(event.y)
            val = max_value - int((y - 50) / 500 * max_value)
            canvas.delete("value_text")
            canvas.create_text(x, y - 10, anchor=tk.S, text=str(val), tag="value_text", fill="black")

        canvas.bind("<Motion>", show_value)


class HistogramWindow:
    def __init__(self, image, path, parent):
        self.image = image
        self.histogram = Histogram(image)
        self.path = path
        self.hist_window = tk.Toplevel(parent.tkWindow)
        self.create_histogram()

        #super().__init__(self.hist_window, parent)

    def create_histogram(self):
        result = self.histogram.calculate_histogram()
        if self.histogram.is_gray_scale:
            r = result[0]
        else:
            r, g, b = result
        self.hist_window.title("Histogram")

        canvas = tk.Canvas(self.hist_window, width=800, height=600, bg='white')
        canvas.pack()

        if self.histogram.is_gray_scale:
            max_value = max(r)
            for i in range(256):
                value = r[i]
                if value > 0:
                    canvas.create_line(50 + i * 3, 550, 50 + i * 3, 550 - (value * 500 / max_value), fill="black")
                if i == 0:
                    print(value)
            for i in range(0, 256, 10):
                canvas.create_text(50 + i * 3, 570, anchor=tk.N, text=str(i), fill="black")

            canvas.update()

        else:
            canvas = self.create_scrollableCanvas(2500)
            max_value = max(max(r), max(g), max(b))

            colors = [["red", 0], ["green", 1], ["blue", 2]]

            for [color, position] in colors:
                for i in range(256):
                    shift = 0 if position == 0 else 60 + (255 * 3) * position
                    value = r[i] if color == "red" else (g[i] if color == "green" else b[i])
                    if value > 0:
                        canvas.create_line(shift + 50 + i * 3, 550, shift + 50 + i * 3, 550 - (value * 500 / max_value),
                                           fill=color)
                for i in range(0, 256, 10):
                    canvas.create_text(shift + 50 + i * 3, 570, anchor=tk.N, text=str(i), fill="black")
            for i in range(0, max_value + 1, int(max_value / 10)):
                canvas.create_text(50, 550 - (i * 500 / max_value), anchor=tk.E, text=str(i), fill="black")
        canvas.update()

        def display_value(event):
            x = canvas.canvasx(event.x)
            y = canvas.canvasy(event.y)
            val = max_value - int((y - 50) / 500 * max_value)
            canvas.delete("value_text")
            canvas.create_text(x, y - 10, anchor=tk.S, text=str(val), tag="value_text", fill="black")

        canvas.bind("<Motion>", display_value)

    def create_scrollableCanvas(self, width):
        canvas = tk.Canvas(self.hist_window, width=850, height=600, bg='white', scrollregion=(0, 0, width, 600))
        scrollbar = tk.Scrollbar(self.hist_window, orient="horizontal", command=canvas.xview)
        scrollbar.pack(side="bottom", fill="x")
        canvas.config(xscrollcommand=scrollbar.set)
        canvas.pack(expand=tk.YES, fill=tk.BOTH)
        canvas.config(scrollregion=canvas.bbox("all"))
        return canvas


class ImageWindow(Window):
    def __init__(self, image, path, parent):
        self.image_path = path
        self.image = image
        self.photo = ImageTk.PhotoImage(self.image)
        self.image_window = tk.Toplevel(parent.tkWindow)
        self.image_window.minsize(400, 500)
        self.image_window.maxsize(1400, 1200)
        self.image_window.title("Edit Image: " + path)

        self.create_menu()

        # Create the bottom panel for the image
        self.bottom_panel = tk.Frame(self.image_window, width=400)
        self.bottom_panel.pack(fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(self.bottom_panel, image=self.photo)
        self.image_label.photo = self.photo
        self.image_label.pack(fill=tk.BOTH, expand=True)

        super().__init__(self.image_window, parent)
    def create_menu(self):

        top_panel = tk.Frame(self.image_window, height=50, width=400)
        top_panel.pack(fill=tk.BOTH)


        menu_button = tk.Menubutton(top_panel, text="Lab1", underline=0, padx=5)
        menu_button.pack(side=tk.LEFT)


        menu = tk.Menu(menu_button, tearoff=0)
        menu_button.configure(menu=menu)


        menu.add_command(label="Zapisz obraz", command=self.save_image)
        menu.add_command(label="Duplikuj", command=self.duplicate_image)
        menu.add_command(label="Pokaż Histogram", command=self.show_histogram)
        menu.add_command(label="Pokaż LUT", command=self.show_lut_table)



        menu_button = tk.Menubutton(top_panel, text="Lab2", underline=0, padx=5)
        menu_button.pack(side=tk.LEFT)

        # Create a menu
        menu = tk.Menu(menu_button, tearoff=0)
        menu_button.configure(menu=menu)

        # Add options to the menu
        menu.add_command(label="Rozciąganie Liniowe", command=self.show_linear_stretch)
        menu.add_command(label="Rozciąganie Nieliniowe (Funkcja gamma)", command=self.show_nonlinear_stretch)
        menu.add_command(label="Equalizacja", command=self.show_equalization)
        menu.add_command(label="Rozciąganie Histogramu", command=self.show_histogram_stretching)
        menu.add_command(label="Negacja", command=self.show_negation)
        menu.add_command(label="Progowanie Binarne", command=self.show_threshold)
        menu.add_command(label="Redukcja poziomów szarości", command=self.show_reduce_grayscale)
        menu.add_command(label="Rozciąganie Histogramu z zakresem p1-p2 do q3-q4", command=self.show_threshold_preserve)

        menu_button = tk.Menubutton(top_panel, text="Lab3", underline=0, padx=5)
        menu_button.pack(side=tk.LEFT)

        # Create a menu
        menu = tk.Menu(menu_button, tearoff=0)
        menu_button.configure(menu=menu)

        # Add options to the menu
        menu.add_command(label="Dodaj obraz", command=self.show_add_image)
        menu.add_command(label="Odejmij obraz", command=self.show_sub_image)
        menu.add_command(label="Dodaj liczbę", command=self.show_add)
        menu.add_command(label="Odejmij liczbę", command=self.show_subtract)
        menu.add_command(label="Mnożenie przez liczbę", command=self.show_multiply)
        menu.add_command(label="Dzielenie przez liczbę", command=self.show_divide)
        menu.add_command(label="NOT", command=self.show_logical_not)
        menu.add_command(label="AND", command=self.show_logical_and)
        menu.add_command(label="OR", command=self.show_logical_or)
        menu.add_command(label="XOR", command=self.show_logical_xor)

        menu_button = tk.Menubutton(top_panel, text="Lab4", underline=0, padx=5)
        menu_button.pack(side=tk.LEFT)

        menu = tk.Menu(menu_button, tearoff=0)
        menu_button.configure(menu=menu)

        menu.add_command(label="Smooth Image average", command=lambda: self.show_smooth_image_opencv('average'))
        menu.add_command(label="Smooth Image weighted", command=lambda: self.show_smooth_image_opencv('weighted'))
        menu.add_command(label="Smooth Image gaussian", command=lambda: self.show_smooth_image_opencv('gaussian'))
        menu.add_command(label="Sharp Image ", command=lambda: self.show_sharp_image_opencv)
        menu.add_command(label="sobel", command=lambda: self.apply_edge_detection)
        menu.add_command(label="ass", command=lambda: self.apply_median_filter)




    def open_new_image(self):
        file_path = filedialog.askopenfilename()

        if file_path:
            self.image_path = file_path
            image = Image.open(file_path)
            self.update_image(image)

    def show_histogram(self):
        hist_window = HistogramWindow(parent=self, path=self.image_path, image=self.image)
        self.add_child(hist_window)

    def show_lut_table(self):
        lut_window = LutWindow(parent=self, path=self.image_path, image=self.image)
        self.add_child(lut_window)

    def duplicate_image(self):
        copied_image = self.image.copy()
        copied_image_window = ImageWindow(parent=self, path=self.image_path, image=copied_image)
        self.add_child(copied_image_window)

    def save_image(self):
        _, file_extension = os.path.splitext(self.image_path)
        file_path = filedialog.asksaveasfilename(initialfile=self.image_path)

        if file_path:
            self.image.save(file_path)

    def update_image(self, image):
        self.image = image
        self.photo = ImageTk.PhotoImage(self.image)
        self.image_label.configure(image=self.photo)
        self.image_label.photo = self.photo

    def close(self):
        self.image_window.destroy()

    def show_linear_stretch(self):
        histogram = Histogram(self.image)
        image = histogram.linear_histogram_stretching()
        self.update_image(image)

    def show_nonlinear_stretch(self):
        histogram = Histogram(self.image)
        gamma = simpledialog.askfloat("Gamma", "Wybierz wartość gammy od 0.1 do 5.0:", minvalue=0.1, maxvalue=5.0)

        saturation = messagebox.askyesno("Wybierz opcje", "Z przesyceniem?")

        image = histogram.nonlinear_stretch_image(gamma, saturation)
        self.update_image(image)

    def show_equalization(self):
        histogram = Histogram(self.image)
        image = histogram.histogram_equalization()
        self.update_image(image)

    def show_histogram_stretching(self):
        histogram = Histogram(self.image)
        p1 = simpledialog.askfloat("Rozciąganie histogramu", "Wybierz wartość p1 od 0 do 255:", minvalue=0, maxvalue=255)
        if p1 is None:
            return

        p2 = simpledialog.askfloat("Rozciąganie histogramu", "Wybierz wartość p2 od 0 do 255:", minvalue=0, maxvalue=255)
        if p2 is None:
            return

        q3 = simpledialog.askfloat("Rozciąganie histogramu", "Wybierz wartość q3 od 0 do 255:", minvalue=0, maxvalue=255)
        if q3 is None:
            return

        q4 = simpledialog.askfloat("Rozciąganie histogramu", "Wybierz wartość q4 od 0 do 255:", minvalue=0, maxvalue=255)
        if q4 is None:
            return

        image = histogram.histogram_stretching(p1, p2, q3, q4)
        self.update_image(image)

    def show_negation(self):
        binary = Binary(self.image)
        image = binary.negate()
        self.update_image(image)

    def show_threshold(self):
        binary = Binary(self.image)
        threshold = simpledialog.askfloat("Progowanie", "Wybierz wartość progowania:", minvalue=1, maxvalue=255)
        image = binary.threshold(threshold)
        self.update_image(image)

    def show_threshold_preserve(self):
        binary = Binary(self.image)
        threshold = simpledialog.askfloat("Progowanie", "Wybierz wartość progowania:", minvalue=1, maxvalue=255)
        image = binary.threshold(threshold, preserve=True)
        self.update_image(image)

    def show_reduce_grayscale(self):
        binary = Binary(self.image)
        levels = simpledialog.askfloat("Redukcja poziomów szarości", "Wybierz poziom od 1 do 20:", minvalue=1,
                                       maxvalue=20)
        image = binary.reduce_grayscale_levels(levels)
        self.update_image(image)

#Lab 3
    def show_add_image(self):
        file_path = filedialog.askopenfilename()

        if file_path:
            saturation = messagebox.askyesno("Wybierz opcje", "Z przesyceniem?")
            image = Image.open(file_path)
            multi = Multi(self.image)
            image = multi.addImage(image, saturation)
            self.update_image(image)

    def show_sub_image(self):
        file_path = filedialog.askopenfilename()

        if file_path:
            image = Image.open(file_path)
            multi = Multi(self.image)
            image = multi.sub_image(image)
            self.update_image(image)

    def show_add(self):
        multi = Multi(self.image)
        value = simpledialog.askfloat("Dodaj", "Wybierz wartość:", minvalue=1, maxvalue=255)
        image = multi.operate_on_image(value, "+")
        self.update_image(image)
    def show_subtract(self):
        multi = Multi(self.image)
        value = simpledialog.askfloat("Odejmowanie", "Wybierz wartość:", minvalue=1, maxvalue=255)
        image = multi.operate_on_image(value, "-")
        self.update_image(image)

    def show_multiply(self):
        multi = Multi(self.image)
        value = simpledialog.askfloat("Mnożenie", "Wybierz wartość:", minvalue=1, maxvalue=255)
        image = multi.operate_on_image(value, "*")
        self.update_image(image)

    def show_divide(self):
        multi = Multi(self.image)
        value = simpledialog.askfloat("Różnica bezwzględna", "Wybierz wartość:", minvalue=1, maxvalue=255)
        image = multi.operate_on_image(value, "/")
        self.update_image(image)

    def show_logical_not(self):
        binary = Binary(self.image)
        image = binary.logical_not()
        self.update_image(image)

    def show_logical_and(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image1 = Image.open(file_path)
            binary = Binary(self.image)
            result_image = binary.logical_and(Binary(image1))
            self.update_image(result_image)

    def show_logical_or(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image1 = Image.open(file_path)
            binary = Binary(self.image)
            image = binary.logical_or(Binary(image1))
            self.update_image(image)

    def show_logical_xor(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image1 = Image.open(file_path)
            binary = Binary(self.image)
            result_image = binary.logical_xor(Binary(image1))
            self.update_image(result_image)

#Lab 4
    def show_smooth_image_opencv(self, mask_type='average'):
        image_cv = np.array(self.image)

        if mask_type == 'average':
            kernel = np.ones((3, 3), dtype=np.float32) / 9.0
        elif mask_type == 'weighted':
            kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
        elif mask_type == 'gaussian':
            kernel = cv2.getGaussianKernel(3, 0) @ cv2.getGaussianKernel(3, 0).T

        smoothed_image_cv = cv2.filter2D(image_cv, -1, kernel)
        smoothed_image = Image.fromarray(smoothed_image_cv)

        self.update_image(smoothed_image)



    def show_sharp_image_opencv(self):
        masks = [
            [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
            [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
            [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]

        ]
        user_choise = simpledialog.askinteger("Maska", "wybierz od 1 do 3")
        i = user_choise - 1
        neighbour = Neighbour(self.image)
        if 0 <= i < len(masks):
            image = neighbour.linear_sharpening(mask=masks[i])
            self.update_image(image)

    def apply_edge_detection(image, method, edge_type='sobel'):
        if edge_type == 'sobel':
            scale = 1
            delta = 0
            ddepth = cv2.CV_16S
            if method == 'horizontal':
                grad = cv2.Sobel(image, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
            elif method == 'vertical':
                grad = cv2.Sobel(image, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
            else:
                raise ValueError("Unknown edge detection method")
            abs_grad = cv2.convertScaleAbs(grad)
            return abs_grad
        elif edge_type == 'prewitt':
            # Prewitt implementation can be added here
            kernelX = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], np.float32)
            kernelY = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], np.float32)
            if method == 'horizontal':
                grad = cv2.filter2D(image, -1, kernelX)
            elif method == 'vertical':
                grad = cv2.filter2D(image, -1, kernelY)
        else:
            raise ValueError("Unknown edge detection type")

    def apply_median_filter(image, kernel_size, border_type):
        if border_type == 'constant':
            border = cv2.BORDER_CONSTANT
        elif border_type == 'reflect':
            border = cv2.BORDER_REFLECT
        elif border_type == 'wrap':
            border = cv2.BORDER_WRAP
        else:
            raise ValueError("Unknown border type")
        return cv2.medianBlur(image, kernel_size)





class LutWindow:
    def __init__(self, image, path, parent):
        self.image = image
        self.path = path
        self.histogram = Histogram(image)
        hist_window = self.show_LUT(parent.tkWindow)

        #super().__init__(hist_window, parent)

    def show_LUT(self, parent_window):
        lut_window = tk.Toplevel(parent_window)
        lut_window.title("LUT Table")
        lut_list = tk.Listbox(lut_window, height=20, width=40)
        lut_list.pack()
        lut_list.insert(tk.END, "Value\t|\tPixel Count\n")
        lut_list.insert(tk.END, "-" * 24 + "\n")

        histogram = self.histogram.calculate_histogram()

        if self.histogram.is_gray_scale:
            self.print_LUT_channel(histogram[0], lut_list)
        else:
            r, g, b = histogram
            self.print_LUT_channel(r, lut_list, "Red")
            self.print_LUT_channel(g, lut_list, "Green")
            self.print_LUT_channel(b, lut_list, "Blue")

        #lut_scroll = tk.Scrollbar(lut_window, orient="Horizontal")
        #lut_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        #lut_list.config(yscrollcommand=lut_scroll.set)
        #lut_scroll.config(command=lut_list.yview)

        return lut_window

    def print_LUT_channel(self, data, lut_list, label=None):

        if label is not None:
            lut_list.insert(tk.END, "\n")
            lut_list.insert(tk.END, label)
            lut_list.insert(tk.END, "\n")

        lut_arr = [0] * 256

        for index in range(0, 256):
            lut_arr[index] = data[index]

        for i in range(len(lut_arr)):
            lut_list.insert(tk.END, f"{i}\t|\t{lut_arr[i]}\n")


class Multi:
    def __init__(self, image):
        self.image = self.preprocess_image(image)
        unique_colors = set(self.image.getdata())
        self.is_gray_scale = True if len(unique_colors) <= 256 else False

    def preprocess_image(self, image=None):
        if image is None:
            image = self.image
        image = image if image.mode == "L" else image.convert("L")
        return image

    def addImage(self, image_2, limitSaturation=False):
        image = self.image
        image_2 = self.preprocess_image(image_2)
        if limitSaturation:
            image = self.reduce_grayscale_levels(image)
            image_2 = self.reduce_grayscale_levels(image_2)
        pixel_values = list(image.getdata())
        pixel_values2 = list(image_2.getdata())
        added_values = [np.clip(value2 + value, 0, 255) for value, value2 in zip(pixel_values, pixel_values2)]

        added_image = Image.new('L', self.image.size)
        added_image.putdata(added_values)
        return added_image

    def sub_image(self, image_2):
        image = self.image
        image_2 = self.preprocess_image(image_2)

        pixel_values = list(image.getdata())
        pixel_values2 = list(image_2.getdata())
        sub_values = [abs(value - value2) for value, value2 in zip(pixel_values, pixel_values2)]

        sub_image = Image.new('L', self.image.size)
        sub_image.putdata(sub_values)
        return sub_image

    def operate_on_image(self, value=1, operand="+", ):
        pixel_values = list(self.image.getdata())
        if operand == "+":
            new_pixel_values = [np.clip(pixel_value + value, 0, 255) for pixel_value in pixel_values]
        elif operand == "-":
            new_pixel_values = [np.clip(pixel_value - value, 0, 255) for pixel_value in pixel_values]
        elif operand == "*":
            new_pixel_values = [np.clip(pixel_value * value, 0, 255) for pixel_value in pixel_values]
        elif operand == "/":
            new_pixel_values = [np.clip(int(pixel_value / value), 0, 255) for pixel_value in pixel_values]

        new_image = Image.new('L', self.image.size)
        new_image.putdata(new_pixel_values)
        return new_image

    def reduce_grayscale_levels(self, image):
        pixel_values = list(image.getdata())
        new_pixel_values = [np.clip(int(value / 2), 1, 127) for value in pixel_values]

        new_image = Image.new('L', image.size)
        new_image.putdata(new_pixel_values)
        return new_image

def update_image(self, image):
    self.image = image
    self.photo = ImageTk.PhotoImage(self.image)
    self.image_label.configure(image=self.photo)
    self.image_label.photo = self.photo


def show_linear_stretch(self):
    histogram = Histogram(self.image)
    image = histogram.linear_histogram_stretching()
    self.update_image(image)


def show_nonlinear_stretch(self):
    histogram = Histogram(self.image)
    gamma = simpledialog.askfloat("Korekta Gamma", "Wybierz poziom od 0.1 do 5.0:", minvalue=0.1, maxvalue=5.0)
    if gamma is None:
        return
    image = histogram.nonlinear_stretch_image(gamma)
    self.update_image(image)


def show_equalization(self):
    histogram = Histogram(self.image)
    image = histogram.histogram_equalization()
    self.update_image(image)


def show_negation(self):
    binary = Binary(self.image)
    image = binary.negate()
    self.update_image(image)


def show_threshold(self):
    binary = Binary(self.image)
    threshold = simpledialog.askfloat("Threshold", "Wybierz wartość progowania:", minvalue=1, maxvalue=255)
    image = binary.threshold(threshold)
    self.update_image(image)


def show_threshold_preserve(self):
    binary = Binary(self.image)
    threshold = simpledialog.askfloat("Threshold", "Wybierz wartość progowania:", minvalue=1, maxvalue=255)
    image = binary.threshold(threshold, preserve=True)
    self.update_image(image)


def show_reduce_grayscale(self):
    binary = Binary(self.image)
    levels = simpledialog.askfloat("Redukcja poziomów szarości", "Wybierz poziom od 1 do 20:", minvalue=1,
                                   maxvalue=20)
    image = binary.reduce_grayscale_levels(levels)
    self.update_image(image)


class MainWindow(Window):
    def __init__(self):
        rootWindow = Tk()
        rootWindow.title("Projekt na Obrazy")
        rootWindow.geometry("400x50")

        panel = Frame(rootWindow, width=200, height=50)  # Use Frame from tkinter
        panel.pack()
        button1 = Button(panel, text="Otwórz Obraz pierwszy", command=self.open_new_image1)
        button1.pack(side="left", padx=10)
        button2 = Button(panel, text="Otwórz Obraz drugi", command=self.open_new_image2)
        button2.pack(side="right", padx=10)

        super().__init__(rootWindow, None)
        self.tkWindow = rootWindow

    def start(self):
        self.tkWindow.mainloop()

    def open_new_image1(self):
        file_path = filedialog.askopenfilename()

        if file_path:
            image_window = ImageWindow(path=file_path, image=Image.open(file_path), parent=self)
            self.add_child(image_window)

    def open_new_image2(self):
        file_path = filedialog.askopenfilename()

        if file_path:
            image_window = ImageWindow(path=file_path, image=Image.open(file_path), parent=self)
            self.add_child(image_window)

class openCV:
    def sharp_image_opencv(self, mask):
        image_array = np.array(self.image)
        if len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

        custom_filter = np.array(mask, dtype=np.float32)
        sharpened = cv2.filter2D(image_array, -1, custom_filter)
        sharpened = np.uint8(np.absolute(sharpened))
        sharpened_image = cv2.addWeighted(image_array, 1.5, sharpened, -0.5, 0)
        return Image.fromarray(sharpened_image)


class App:
    @staticmethod
    def run():
        main_window = MainWindow()
        main_window.start()


if __name__ == "__main__":
    App.run()
