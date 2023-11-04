import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

file_path1 = ""
file_path2 = ""

def open_image1():
    global file_path1
    file_path1 = filedialog.askopenfilename(filetypes=[("Image files", "*.bmp;*.tif;*.png;*.jpg")])
    if file_path1:
        image = Image.open(file_path1)
        photo = ImageTk.PhotoImage(image)
        label1.config(image=photo)
        label1.image = photo

def open_image2():
    global file_path2
    file_path2 = filedialog.askopenfilename(filetypes=[("Image files", "*.bmp;*.tif;*.png;*.jpg")])
    if file_path2:
        image = Image.open(file_path2)
        photo = ImageTk.PhotoImage(image)
        label2.config(image=photo)
        label2.image = photo

def save_image():
    global file_path1
    if file_path1:
        image = Image.open(file_path1)
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if save_path:
            image.save(save_path)

def duplicate_image():
    global file_path1
    if file_path1:
        img_duplicate = Image.open(file_path1)

        new_window = tk.Toplevel()
        new_window.title("Duplicate Image")
        img = ImageTk.PhotoImage(img_duplicate)
        panel_duplicate = tk.Label(new_window, image=img)
        panel_duplicate.image = img
        panel_duplicate.pack(padx=10, pady=10)

def create_LUT_from_image(image_path=None):
    if image_path is None:
        image_path = file_path1
    try:
        img = Image.open(image_path)
        img_data = list(img.getdata())

        lut_values = [0] * 256  # Initialize the LUT array with zeros

        for pixel in img_data:
            value = pixel[0]  # Extract the first value for each pixel
            lut_values[value] += 1  # Increment the count for the corresponding pixel value

        lut_window = tk.Toplevel()
        lut_window.title("LUT Table")

        lut_values_list = tk.Listbox(lut_window, height=20, width=40)  # Updated size of the listbox
        lut_values_list.pack()

        lut_values_list.insert(tk.END, "Value\t|\tPixel Count\n")  # Adding column headers
        lut_values_list.insert(tk.END, "-" * 30 + "\n")  # Adding a separator

        for i in range(256):
            lut_values_list.insert(tk.END, f"{i}\t|\t{lut_values[i]}\n")  # Adding values to the table

        lut_scroll = tk.Scrollbar(lut_window, orient="vertical")
        lut_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        lut_values_list.config(yscrollcommand=lut_scroll.set)
        lut_scroll.config(command=lut_values_list.yview)

        return lut_values
    except IOError:
        print(f"Failed to load image from path: {image_path}")
        return None

def calculate_histogram(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')

    r, g, b = image.split()
    r_data = list(r.getdata())
    g_data = list(g.getdata())
    b_data = list(b.getdata())

    r_hist = [0] * 256
    g_hist = [0] * 256
    b_hist = [0] * 256

    for pixel in r_data:
        r_hist[pixel] += 1

    for pixel in g_data:
        g_hist[pixel] += 1

    for pixel in b_data:
        b_hist[pixel] += 1

    return r_hist, g_hist, b_hist

def create_histogram_window(hist_data, color):
    root_hist = tk.Toplevel()
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

def create_histogram(image_path, selected_image):
    if not image_path:
        return

    try:
        loaded_img = Image.open(image_path)
    except IOError:
        print(f"Failed to load image from path: {image_path}")
        return

    r, g, b = calculate_histogram(loaded_img)

    if selected_image == "Obraz 1":
        create_histogram_window(r, "red")
        create_histogram_window(g, "green")
        create_histogram_window(b, "blue")
    elif selected_image == "Obraz 2":
        create_histogram_window(r, "red")
        create_histogram_window(g, "green")
        create_histogram_window(b, "blue")

root = tk.Tk()
root.geometry("1200x500")
root.title("Projekt Obrazy")

s = ttk.Style()
s.configure('my.TButton', font=('Helvetica', 12), foreground='black')

button1 = ttk.Button(root, text="Załaduj obraz 1", style='my.TButton', command=open_image1)
button1.grid(row=0, column=0, padx=10, pady=10)

button2 = ttk.Button(root, text="Załaduj obraz 2", style='my.TButton', command=open_image2)
button2.grid(row=0, column=1, padx=10, pady=10)

button3 = ttk.Button(root, text="Zapisz obraz", style='my.TButton', command=save_image)
button3.grid(row=0, column=2, padx=10, pady=10)

button4 = ttk.Button(root, text="Duplikuj obraz", style='my.TButton', command=duplicate_image)
button4.grid(row=0, column=3, padx=10, pady=10)

button5 = ttk.Button(root, text="Wyświetl LUT", style='my.TButton', command=lambda: create_LUT_from_image(file_path1 if selected_option.get() == "Obraz 1" else (file_path2 if selected_option.get() == "Obraz 2" else file_path1)))
button5.grid(row=0, column=6, padx=10, pady=10)

button6 = ttk.Button(root, text="Wyświetl histogram", style='my.TButton', command=lambda: create_histogram(file_path1 if selected_option.get() == "Obraz 1" else (file_path2 if selected_option.get() == "Obraz 2" else file_path1)))
button6.grid(row=0, column=7, padx=10, pady=10)

button7 = ttk.Button(root, text="Utwórz histogram", style='my.TButton', command=lambda: create_histogram(file_path1 if selected_option.get() == "Obraz 1" else (file_path2 if selected_option.get() == "Obraz 2" else file_path1), selected_option.get()))
button7.grid(row=0, column=8, padx=10, pady=10)

label1 = tk.Label(root)
label1.grid(row=1, column=0, padx=10, pady=10)

label2 = tk.Label(root)
label2.grid(row=1, column=1, padx=10, pady=10)

label_choice = ttk.Label(root, text="Wybierz obraz:")
label_choice.grid(row=0, column=4, padx=10, pady=10)

option_list = ["Obraz 1", "Obraz 2", "Duplikowany obraz"]
selected_option = tk.StringVar(root)
selected_option.set(option_list[0])
dropdown = ttk.OptionMenu(root, selected_option, *option_list)
dropdown.grid(row=0, column=5, padx=10, pady=10)

root.mainloop()
