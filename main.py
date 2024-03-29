# Importowanie niezbędnych bibliotek
import tkinter as tk
from tkinter import messagebox
from tkinter import Tk, Frame, Button, filedialog
from tkinter import simpledialog
from PIL import Image, ImageTk
import numpy as np
import os
from tkinter import messagebox
import cv2

# Klasa Window służy jako podstawa dla wszystkich okien w aplikacji.
class Window:
    tkWindow = None
    parent = None

    def __init__(self, tkWindow, parent):
        self.parent = parent
        self.tkWindow = tkWindow
        self.children = []

    # Metoda dodająca okno potomne do listy
    def add_child(self, window):
        self.children.append(window)

    # Metoda zamykająca okno i wszystkie okna potomne
    def close(self):
        [window.close() for window in self.children]
        self.tkWindow.destroy()

    # Metoda sprawdzająca, czy okno jest głównym oknem aplikacji
    def isRoot(self):
        if self.parent is None:
            return True
        else:
            return False

# Klasa Binary służy do operacji na obrazach binarnych, takich jak negacja, progowanie, redukcja poziomów szarości.
class Binary:
    def __init__(self, image):
        self.image = image
        unique_colors = set(self.image.getdata())
        self.is_gray_scale = True if len(unique_colors) <= 256 else False

    # Metoda negująca obraz
    def negate(self):
        image = self.preprocess_image()
        pixel_values = list(image.getdata())
        negated_values = [255 - value for value in pixel_values]
        negated_image = Image.new('L', self.image.size)
        negated_image.putdata(negated_values)
        return negated_image

    # Metoda do progowania obrazu
    def threshold(self, threshold_value, preserve=False):
        image = self.preprocess_image()
        pixel_values = list(image.getdata())
        threshold_values = [0 if value < threshold_value else 255 if not preserve else value for value in pixel_values]
        threshold_image = Image.new('L', self.image.size)
        threshold_image.putdata(threshold_values)
        return threshold_image

    # Metoda do redukcji liczby poziomów szarości
    def reduce_grayscale_levels(self, levels):
        image = self.preprocess_image()
        pixel_values = list(image.getdata())
        new_pixel_values = [int(value / 255 * levels) * (255 // (levels - 1)) for value in pixel_values]
        new_image = Image.new('L', self.image.size)
        new_image.putdata(new_pixel_values)
        return new_image

    # Metoda do wstępnego przetwarzania obrazu
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

# Klasa Histogram służy do obliczania i operowania na histogramie obrazu.
class Histogram:
    def __init__(self, image):
        self.image = image
        # Sprawdzenie, czy obraz jest w skali szarości
        unique_colors = set(self.image.getdata())
        self.is_gray_scale = True if len(unique_colors) <= 256 else False

    # Metoda obliczająca histogram obrazu
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
        if self.is_gray_scale:
            stretched_values = self.linear_stretch_channel(pixel_values)

            stretched_image = Image.new('L', self.image.size)
            stretched_image.putdata(stretched_values)
            return stretched_image
        else:

            red_channel, green_channel, blue_channel = zip(*pixel_values)

            stretched_red = self.linear_stretch_channel(red_channel)
            stretched_green = self.linear_stretch_channel(green_channel)
            stretched_blue = self.linear_stretch_channel(blue_channel)
            stretched_red_image = Image.new('L', self.image.size)
            stretched_red_image.putdata(stretched_red)
            stretched_green_image = Image.new('L', self.image.size)
            stretched_green_image.putdata(stretched_green)
            stretched_blue_image = Image.new('L', self.image.size)
            stretched_blue_image.putdata(stretched_blue)

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

# Klasa HistogramWindow służy do wyświetlania histogramu obrazu w nowym oknie.
class HistogramWindow:
    def __init__(self, image, path, parent):
        self.image = image # Obraz źródłowy
        self.histogram = Histogram(image)
        self.path = path
        self.hist_window = tk.Toplevel(parent.tkWindow)
        self.create_histogram()

    # Metoda do tworzenia i wyświetlania histogramu
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

# Klasa ImageWindow służy do wyświetlania i edycji obrazu w nowym oknie.
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
        menu.add_command(label="Negacja", command=self.show_negation)
        menu.add_command(label="Redukcja poziomów szarości", command=self.show_reduce_grayscale)
        menu.add_command(label="Progowanie Binarne", command=self.show_threshold)
        menu.add_command(label="Progowanie z zachowanie poziomów szarości", command=self.show_threshold_preserve)
        menu.add_command(label="Rozciąganie Histogramu z zakresem p1-p2 do q3-q4",command=self.show_histogram_stretching)

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

        menu.add_command(label="Wygładanie liniowe", command=lambda: self.show_smooth_image_opencv('average'))
        menu.add_command(label="Wygładanie liniowe z wagami", command=lambda: self.show_smooth_image_opencv('weighted'))
        menu.add_command(label="Wygładzenie liniowe bez wag", command=lambda: self.show_smooth_image_opencv('Bez'))
        menu.add_command(label="Wygładanie liniowe metodą Gausowsską", command=lambda: self.show_smooth_image_opencv('gaussian'))
        menu.add_command(label="Wyostrzenie obrazu maską [0, -1, 0], [-1, 4, -1], [0, -1, 0]", command=lambda: self.show_sharp_image('Pierwsza'))
        menu.add_command(label="Wyostrzenie obrazu maską [-1, -1, -1], [-1, 8, -1], [-1, -1, -1]", command=lambda: self.show_sharp_image('Druga'))
        menu.add_command(label="Wyostrzenie obrazu maską [1, -2, 1], [-2, 4, -2], [1, -2, 1]", command=lambda: self.show_sharp_image('Trzecia'))
        menu.add_command(label="Sobel kierunkowy E", command=lambda: self.show_sobel_directional('E'))
        menu.add_command(label="Sobel kierunkowy SE", command=lambda: self.show_sobel_directional('SE'))
        menu.add_command(label="Sobel kierunkowy S", command=lambda: self.show_sobel_directional('S'))
        menu.add_command(label="Sobel kierunkowy WS", command=lambda: self.show_sobel_directional('WS'))
        menu.add_command(label="Sobel kierunkowy W", command=lambda: self.show_sobel_directional('W'))
        menu.add_command(label="Sobel kierunkowy WN", command=lambda: self.show_sobel_directional('WN'))
        menu.add_command(label="Sobel kierunkowy N", command=lambda: self.show_sobel_directional('N'))
        menu.add_command(label="Sobel kierunkowy NE", command=lambda: self.show_sobel_directional('NE'))
        menu.add_command(label="Sobel", command=lambda: self.show_sobel())
        menu.add_command(label="User Border Constant", command=lambda: self.show_user_border_const())
        menu.add_command(label="User Border Replicate", command=lambda: self.show_user_border_replicate())
        menu.add_command(label="Border Reflect", command=lambda: self.show_border_reflect())
        menu.add_command(label="Border Wrap", command=lambda: self.show_border_wrap())
        menu.add_command(label="Filtr medianowy", command=lambda: self.show_median_filter())

        menu_button = tk.Menubutton(top_panel, text="Lab5", underline=0, padx=5)
        menu_button.pack(side=tk.LEFT)

        menu = tk.Menu(menu_button, tearoff=0)
        menu_button.configure(menu=menu)

        menu.add_command(label="Canno", command=lambda: self.show_canny_edge_detection())
        menu.add_command(label="Segmentacja z dwoma progami", command=lambda: self.show_segment_image_with_input())
        menu.add_command(label="Segmentacja metodą Otsu", command=lambda: self.show_segment_image_otsu())
        menu.add_command(label="Segmentacja metodą adaptacyjną", command=lambda: self.show_adaptive_thresholding())

        menu_button = tk.Menubutton(top_panel, text="Lab6", underline=0, padx=5)
        menu_button.pack(side=tk.LEFT)

        menu = tk.Menu(menu_button, tearoff=0)
        menu_button.configure(menu=menu)
        menu.add_command(label="Erozja (Dysk)", command=lambda: self.show_erode("disk"))
        menu.add_command(label="Erozja (Krzyż)", command=lambda: self.show_erode("cross"))
        menu.add_command(label="Dylacja (Dysk)", command=lambda: self.show_dilate("disk"))
        menu.add_command(label="Dylacja (Krzyż)", command=lambda: self.show_dilate("cross"))
        menu.add_command(label="Open (Dysk)", command=lambda: self.show_opening("disk"))
        menu.add_command(label="Open (Krzyż)", command=lambda: self.show_opening("cross"))
        menu.add_command(label="Close (Dysk)", command=lambda: self.show_closing("disk"))
        menu.add_command(label="Close (Krzyż)", command=lambda: self.show_closing("cross"))
        menu.add_command(label="Momenty", command=lambda: self.show_binary_moments())
        menu.add_command(label="Pole powierzchni", command=lambda: self.show_surface_area())
        menu.add_command(label="Obwód", command=lambda: self.show_circuit())
        menu.add_command(label="AspectRatio (Współczynnik proporcji)", command=lambda: self.show_aspect_ratio())
        menu.add_command(label="Extent (Stopień wypełnienia)", command=lambda: self.show_extent())
        menu.add_command(label="Solidity (Stożkowatość)", command=lambda: self.show_Solidity())
        menu.add_command(label="Equivalent Diameter (Średnica równoważna)", command=lambda: self.show_equivalent_diameter())
        menu.add_command(label="Exportuj do txt", command=lambda: self.show_export_data_to_txt())


    # Lab 1
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

    # Lab 2
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
        p1 = simpledialog.askfloat("Rozciąganie histogramu", "Wybierz wartość p1 od 0 do 255:", minvalue=0,
                                   maxvalue=255)
        if p1 is None:
            return

        p2 = simpledialog.askfloat("Rozciąganie histogramu", "Wybierz wartość p2 od 0 do 255:", minvalue=0,
                                   maxvalue=255)
        if p2 is None:
            return

        q3 = simpledialog.askfloat("Rozciąganie histogramu", "Wybierz wartość q3 od 0 do 255:", minvalue=0,
                                   maxvalue=255)
        if q3 is None:
            return

        q4 = simpledialog.askfloat("Rozciąganie histogramu", "Wybierz wartość q4 od 0 do 255:", minvalue=0,
                                   maxvalue=255)
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

    # Lab 3
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

    # Lab 4
    def show_smooth_image_opencv(self, mask_type='average'):
        image_cv = np.array(self.image)

        if mask_type == 'average':
            kernel = np.ones((3, 3), dtype=np.float32) / 9.0
        elif mask_type == 'weighted':
            user_input = simpledialog.askinteger("Ważona", "Podaj wartość K:", minvalue=0, maxvalue=9)
            kernel = np.array([[1, 2, 1], [2, user_input, 2], [1, 2, 1]], dtype=np.float32) / 8 + user_input
        elif mask_type == 'gaussian':
            kernel = cv2.getGaussianKernel(3, 0) @ cv2.getGaussianKernel(3, 0).T
        elif mask_type == 'Bez':
            kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)

        smoothed_image_cv = cv2.filter2D(image_cv, -1, kernel)
        smoothed_image = Image.fromarray(smoothed_image_cv)

        self.update_image(smoothed_image)

    def show_sharp_image(self, mask_type='Pierwsza'):
        image_cv = np.array(self.image)

        if mask_type == 'Pierwsza':
            kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32) / 8.0
        elif mask_type == 'Druga':
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32) / 16.0
        elif mask_type == 'Trzecia':
            kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32) / 4

        sharp_image_cv = cv2.filter2D(image_cv, -1, kernel)
        sharp_image = Image.fromarray(sharp_image_cv)

        self.update_image(sharp_image)

    def show_sobel_directional(self, mask_type='E'):
        image_cv = np.array(self.image)

        if mask_type == 'E':
            sobel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32) / 8.0
        elif mask_type == 'SE':
            sobel = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype=np.float32) / 8.0
        elif mask_type == 'S':
            sobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32) / 8.0
        elif mask_type == 'WS':
            sobel = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=np.float32) / 8.0
        elif mask_type == 'W':
            sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 8.0
        elif mask_type == 'WN':
            sobel = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=np.float32) / 8.0
        elif mask_type == 'N':
            sobel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) / 8.0
        elif mask_type == 'NE':
            sobel = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype=np.float32) / 8.0

        sobel_image_cv = cv2.filter2D(image_cv, -1, sobel)
        sobel_image = Image.fromarray(sobel_image_cv)

        self.update_image(sobel_image)

    def show_sobel(self):
        image_cv = np.array(self.image)

        sobel_x = cv2.Sobel(image_cv, 6, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image_cv, 6, 0, 1, ksize=3)

        sobel_image_cv = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
        sobel_image = Image.fromarray(sobel_image_cv)

        self.update_image(sobel_image)

    def show_prewwit(self):
        image_cv = np.array(self.image)

        prewit_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32) / 6.0
        prewit_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32) / 6.0

        prewit_image_cv = cv2.addWeighted(cv2.filter2D(image_cv, -1, prewit_x), 0.5,
                                          cv2.filter2D(image_cv, -1, prewit_y), 0.5, 0)
        prewit_image = Image.fromarray(prewit_image_cv)

        self.update_image(prewit_image)

    def show_user_border_const(self):
        image_cv = np.array(self.image)

        border_value = simpledialog.askinteger("Border", "Wybierz wartość progowania:", minvalue=1, maxvalue=255)

        border_image_cv = cv2.copyMakeBorder(image_cv, border_value, border_value, border_value, border_value, cv2.BORDER_CONSTANT, value=[border_value, border_value, border_value])
        border_image = Image.fromarray(border_image_cv)

        self.update_image(border_image)

    def show_user_border_replicate(self):
        image_cv = np.array(self.image)
        border_value = simpledialog.askinteger("Border", "Wybierz wartość progowania:", minvalue=1, maxvalue=255)
        fill_value = simpledialog.askinteger("Fill Value", "Wybierz wartość stałą do wypełnienia krawędzi obrazu:",minvalue=0, maxvalue=255)
        border_image_cv = cv2.copyMakeBorder(image_cv, border_value, border_value, border_value, border_value, cv2.BORDER_CONSTANT, value=[fill_value, fill_value, fill_value])
        border_image = Image.fromarray(border_image_cv)

        self.update_image(border_image)

    def show_border_reflect(self, border=None):
        image_cv = np.array(self.image)

        border_image_cv = cv2.copyMakeBorder(image_cv, border, border, border, border, cv2.BORDER_REFLECT)
        border_image = Image.fromarray(border_image_cv)
        self.update_image(border_image)

    def show_border_wrap(self):
        image_cv = np.array(self.image)
        border = simpledialog.askinteger("Border", "Wybierz wartość progowania:", minvalue=1, maxvalue=255)
        border_image_cv = cv2.copyMakeBorder(image_cv, border, border, border, border, cv2.BORDER_WRAP)
        border_image = Image.fromarray(border_image_cv)

        self.update_image(border_image)

    def apply_median_filter(self, image, kernel_size):
        result_image = cv2.medianBlur(image, kernel_size)
        return result_image

    def show_median_filter(self):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

        while True:
            kernel_size = simpledialog.askinteger("Filtr medianowy",
                                                  "Wybierz rozmiar maski, wpisz kolejno dla:\n3x3 wpisz 3\n5x5 wpisz 5\n7x7 wpisz 7\n9x9 wpisz 9\n",
                                                  minvalue=3, maxvalue=9)

            if kernel_size is not None:  # Jeżeli użytkownik nie anulował okna dialogowego
                if kernel_size not in [4, 6, 8]:
                    break
                else:
                    messagebox.showinfo("Błąd", "Nie można wybrać liczb 4, 6, 8. Wybierz inną wartość.")
            else:
                messagebox.showinfo("Informacja", "Anulowano filtr medianowy.")
                return

        result_image = self.apply_median_filter(image, kernel_size)
        self.update_image(Image.fromarray(result_image))



    # Lab 5

    def show_canny_edge_detection(self):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        threshold1 = simpledialog.askfloat("Progowanie", "Wybierz wartość progowania:", minvalue=1, maxvalue=255)
        threshold2 = simpledialog.askfloat("Progowanie", "Wybierz wartość progowania:", minvalue=1, maxvalue=255)

        canned_image_cv = cv2.Canny(image, threshold1, threshold2)
        canned_image = Image.fromarray(canned_image_cv)

        self.update_image(canned_image)

    def show_segment_image_with_input(self):
        image_array = np.array(self.image)
        lower_thresh = simpledialog.askfloat("Progowanie", "Wybierz wartość progowania:", minvalue=1, maxvalue=255)
        ret_low, lower = cv2.threshold(image_array, lower_thresh, 255, cv2.THRESH_BINARY)
        upper_thresh = simpledialog.askfloat("Progowanie", "Wybierz wartość progowania:", minvalue=1, maxvalue=255)
        ret_up, upper = cv2.threshold(image_array, upper_thresh, 255, cv2.THRESH_BINARY_INV)
        combined = cv2.bitwise_and(lower, upper)

        self.update_image(Image.fromarray(combined))

    def show_segment_image_otsu(self):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        _, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_threshold_value = _

        segmented_image_otsu_cv = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        segmented_image_otsu = Image.fromarray(segmented_image_otsu_cv)

        self.update_image(segmented_image_otsu)
        rootWindow = Tk()
        rootWindow.title(otsu_threshold_value)
        rootWindow.geometry("200x10")

    def show_adaptive_thresholding(self):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        segmented_image_adaptive_cv = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11, 2)
        segmented_image_adaptive = Image.fromarray(segmented_image_adaptive_cv)

        self.update_image(segmented_image_adaptive)

#Lab 6

    def show_erode(self, kernel_type='disk'):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

        if kernel_type == 'disk':
            kernel = np.ones((3, 3), np.uint8)
        elif kernel_type == 'cross':
            kernel = np.array([[0, 3, 0], [3, 3, 3], [0, 3, 0]], np.uint8)
        else:
            raise ValueError("Nieprawidłowy rodzaj kernela")

        erode = cv2.erode(image, kernel, iterations=1)
        erode_image = Image.fromarray(erode)

        self.update_image(erode_image)

    def show_dilate(self, kernel_type='disk'):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

        if kernel_type == 'disk':
            kernel = np.ones((3, 3), np.uint8)
        elif kernel_type == 'cross':
            kernel = np.array([[0, 3, 0], [3, 3, 3], [0, 3, 0]], np.uint8)
        else:
            raise ValueError("Nieprawidłowy rodzaj kernela")

        dilate = cv2.dilate(image, kernel, iterations=1)
        dilate_image = Image.fromarray(dilate)

        self.update_image(dilate_image)

    def show_opening(self, kernel_type='disk'):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

        if kernel_type == 'disk':
            kernel = np.ones((3, 3), np.uint8)
        elif kernel_type == 'cross':
            kernel = np.array([[0, 3, 0], [3, 3, 3], [0, 3, 0]], np.uint8)
        else:
            raise ValueError("Nieprawidłowy rodzaj kernela")

        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        opening_image = Image.fromarray(opening)

        self.update_image(opening_image)

    def show_closing(self, kernel_type='disk'):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

        if kernel_type == 'disk':
            kernel = np.ones((3, 3), np.uint8)
        elif kernel_type == 'cross':
            kernel = np.array([[0, 3, 0], [3, 3, 3], [0, 3, 0]], np.uint8)
        else:
            raise ValueError("Nieprawidłowy rodzaj kernela")

        closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        closing_image = Image.fromarray(closing)

        self.update_image(closing_image)

    def show_binary_moments(self):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        ret, thresh = cv2.threshold(image, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, 1, 2)

        moment_results = []

        for i, cnt in enumerate(contours):
            moments = cv2.moments(cnt)
            moment_results.append(f"Moments for object {i + 1}: {' '.join(f'{key}={value};' for key, value in moments.items())}")

        for result in moment_results:
            print(result)

        return str(moment_results)


    def show_surface_area(self):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = 0
        for contour in contours:
            area += cv2.contourArea(contour)

        print(area)
        return str(area)

    def show_circuit(self):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(biggest_contour, True)

        print(round(perimeter, 0))
        return str(round(perimeter, 0))

    def show_aspect_ratio(self):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        _, binary_image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_area = 0
        total_pixels = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            total_area += area
            x, y, w, h = cv2.boundingRect(contour)
            object_pixels = w * h
            total_pixels += object_pixels

        aspect_ratio = total_area / total_pixels

        print(round(aspect_ratio, 2))
        return str(round(aspect_ratio, 2))

    def show_extent(self):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = cv2.contourArea(contours[0])
        x, y, w, h = cv2.boundingRect(contours[0])
        rect_area = w * h
        extent = float(area) / rect_area if rect_area != 0 else 0
        extent = round(extent, 2)

        print(extent)
        return str(extent)

    def show_Solidity(self):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = cv2.contourArea(contours[0])
        hull = cv2.convexHull(contours[0])
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area != 0 else 0
        solidity = round(solidity, 2)

        print(solidity)
        return str(solidity)

    def show_equivalent_diameter(self):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        area = 0
        for contour in contours:
            area += cv2.contourArea(contour)

        equivalent_diameter = np.sqrt(4 * area / np.pi)

        print(round(equivalent_diameter, 2))
        return str(round(equivalent_diameter, 2))

    def show_export_data_to_txt(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile="Data_LAB6.txt",
            title="Wybierz miejsce do zapisu pliku"
        )

        if file_path:
            try:
                with open(file_path, 'w') as file:
                    file.write("Binary Moments: " + self.show_binary_moments() + "\n")
                    file.write("Surface Area: " + self.show_surface_area() + "\n")
                    file.write("Circiut: " + self.show_circuit() + "\n")
                    file.write("Aspect Ratio: " + self.show_aspect_ratio() + "\n")
                    file.write("Extent: " + self.show_extent() + "\n")
                    file.write("Solidity: " + self.show_Solidity() + "\n")
                    file.write("Equivalent Diameter: " + self.show_equivalent_diameter() + "\n")

                print(f"Dane zostały zapisane w pliku: {file_path}")
            except Exception as e:
                print(f"Wystąpił błąd podczas zapisywania danych do pliku: {e}")


class ImageComparator:
    def __init__(self, tolerance=30):
        self.image1 = None
        self.image2 = None
        self.tolerance = tolerance

    def load_images(self, path1, path2):
        self.image1 = cv2.imread(path1)
        self.image2 = cv2.imread(path2)
        if self.image1 is None or self.image2 is None:
            return False
        return True

    def compare_images(self):
        if self.image1.shape != self.image2.shape:
            return False
        difference = cv2.absdiff(self.image1, self.image2)
        mask = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, self.tolerance, 255, cv2.THRESH_BINARY)
        differences = cv2.countNonZero(mask)
        self.image2[mask != 0] = [0, 0, 255]
        return differences

    def show_difference(self):
        cv2.imshow("Roznice w pikselach", self.image2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def compare_lines(self):
        if self.image1.shape != self.image2.shape:
            return False
        identical_lines = 0
        different_lines = 0
        for i in range(self.image1.shape[0]):
            line1 = self.image1[i, :]
            line2 = self.image2[i, :]
            difference = cv2.absdiff(line1, line2)
            if len(line1.shape) > 2 and line1.shape[2] == 3:
                difference = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(difference, self.tolerance, 255, cv2.THRESH_BINARY)
            if cv2.countNonZero(mask) == 0:
                identical_lines += 1
            else:
                different_lines += 1
        return identical_lines, different_lines

    def display_line_comparison_results(self, parent=None):
        identical_lines, different_lines = self.compare_lines()
        if identical_lines is False:
            tk.messagebox.showerror("Błąd", "Obrazy mają różne wymiary!", parent=parent)
        else:
            tk.messagebox.showinfo("Wyniki porównania linii", f"Identyczne linie: {identical_lines}\nRóżne linie: {different_lines}", parent=parent)
class LutWindow:
    def __init__(self, image, path, parent):
        # Inicjalizacja okna LUT z podanym obrazem, ścieżką do pliku i referencją do obiektu nadrzędnego.
        self.image = image
        self.path = path
        self.histogram = Histogram(image)
        hist_window = self.show_LUT(parent.tkWindow)

    def show_LUT(self, parent_window):
        # Wyświetlenie okna z tabelą LUT (Lookup Table) dla obrazu.
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

        return lut_window

    def print_LUT_channel(self, data, lut_list, label=None):
    # Drukowanie tabeli LUT dla pojedynczego kanału (lub skali szarości).
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
        # Inicjalizacja obiektu klasy z obrazem, przetworzenie obrazu na skalę szarości i sprawdzenie, czy obraz jest w skali szarości.
        self.image = self.preprocess_image(image)
        unique_colors = set(self.image.getdata())
        self.is_gray_scale = True if len(unique_colors) <= 256 else False

    def preprocess_image(self, image=None):
        # Przetworzenie obrazu na skalę szarości, jeśli nie jest już w tej skali.
        if image is None:
            image = self.image
        image = image if image.mode == "L" else image.convert("L")
        return image

    def addImage(self, image_2, limitSaturation=False):
        # Dodanie dwóch obrazów. Możliwość ograniczenia nasycenia przez zredukowanie liczby poziomów szarości.
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
        # Odejmowanie jednego obrazu od drugiego.
        image = self.image
        image_2 = self.preprocess_image(image_2)

        pixel_values = list(image.getdata())
        pixel_values2 = list(image_2.getdata())
        sub_values = [abs(value - value2) for value, value2 in zip(pixel_values, pixel_values2)]

        sub_image = Image.new('L', self.image.size)
        sub_image.putdata(sub_values)
        return sub_image

    def operate_on_image(self, value=1, operand="+", ):
        # Operacje matematyczne na obrazie: dodawanie, odejmowanie, mnożenie, dzielenie.
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
        # Redukcja liczby poziomów szarości w obrazie, aby ograniczyć nasycenie.
        pixel_values = list(image.getdata())
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

# Klasa MainWindow służy jako główne okno aplikacji, z którego można otwierać obrazy i wykonywać na nich operacje.
class MainWindow(Window):
    def __init__(self):
        rootWindow = Tk()
        rootWindow.title("Projekt na Obrazy")
        rootWindow.geometry("400x50")

        panel = Frame(rootWindow, width=400, height=100)
        panel.pack()
        button1 = Button(panel, text="Otwórz Obraz pierwszy", command=self.open_new_image1)
        button1.pack(side="left", padx=10)
        button2 = Button(panel, text="Otwórz Obraz drugi", command=self.open_new_image2)
        button2.pack(side="right", padx=10)

        compare_button = Button(panel, text="Porównaj obrazy", command=self.open_comparison_window)
        compare_button.pack(side="left", padx=10)

        super().__init__(rootWindow, None)
        self.tkWindow = rootWindow
        self.image1_path = None
        self.image2_path = None

    def start(self):
        self.tkWindow.mainloop()

    def open_new_image1(self):
        file_path = filedialog.askopenfilename()

        if file_path:
            self.image1_path = file_path
            image_window = ImageWindow(path=file_path, image=Image.open(file_path), parent=self)
            self.add_child(image_window)

    def open_new_image2(self):
        file_path = filedialog.askopenfilename()

        if file_path:
            self.image2_path = file_path
            image_window = ImageWindow(path=file_path, image=Image.open(file_path), parent=self)
            self.add_child(image_window)

    def open_comparison_window(self):
        if self.image1_path is None or self.image2_path is None:
            tk.messagebox.showerror("Błąd", "Najpierw wybierz obrazy do porównania.")
            return

        comparator = ImageComparator(tolerance=25)
        if comparator.load_images(self.image1_path, self.image2_path):
            differences = comparator.compare_images()
            # Porównanie linii i uzyskanie wyników
            identical_lines, different_lines = comparator.compare_lines()

            # Pokazanie liczby różniących się pikseli
            messagebox.showinfo("Różnice", f"Liczba różniących się pikseli: {differences}")

            # Pokazanie informacji o liniach
            messagebox.showinfo("Porównanie linii","Identyczne linie: {identical_lines}\nRóżne linie: {different_lines}")

            # Pokazanie obrazu z różnicami
            self.show_image_differences(comparator.image2)  # Zakładając, że comparator.image2 to zmodyfikowany obraz

    def show_image_differences(self, image):
        # Tworzenie nowego okna Tkinter do wyświetlenia obrazu
        difference_window = tk.Toplevel(self.tkWindow)
        difference_window.title("Różnice w obrazach")

        # Konwersja obrazu OpenCV na format zrozumiały dla Tkinter
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Konwersja z BGR na RGB
        image = Image.fromarray(image)  # Konwersja na obiekt PIL.Image
        photo = ImageTk.PhotoImage(image=image)

        # Wyświetlanie obrazu w nowym oknie
        label = tk.Label(difference_window, image=photo)
        label.image = photo  # Utrzymanie referencji, aby obraz nie został usunięty przez garbage collector
        label.pack()

# Klasa App służy do uruchomienia aplikacji.
class App:
    @staticmethod
    def run():
        main_window = MainWindow()
        main_window.start()

# Uruchomienie aplikacji
if __name__ == "__main__":
    App.run()
