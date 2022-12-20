from pathlib import Path

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd

from PIL import Image, ImageTk


class GeneratorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Car generator")

        # window size
        window_width = 700 + 164
        window_height = 700

        # screen dimension
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # center coordinates
        center_x = int(screen_width / 2 - window_width / 2)
        center_y = int(screen_height / 2 - window_height / 2)

        # set the position of the window to the center of the screen
        self.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        self.iconbitmap(Path(__file__).parent / 'icons/app_icon.ico')

    def mainloop(self):
        frame = MainFrame(self)
        frame.pack(fill=tk.BOTH, expand=True)

        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        finally:
            super().mainloop()


class MainFrame(ttk.Frame):

    def __init__(self, container):
        super().__init__(container)
        self.widget_setup()
        self.style_setup()

    def style_setup(self):
        s = ttk.Style()
        s.theme_use("clam")

        s.configure("TButton", font=("Helvetica", 16),
                    foreground="#F7FBEF",
                    background="#292929",
                    lightcolor="none",
                    darkcolor="none",
                    focuscolor="#333333",
                    borderwidth=0,
                    bordercolor="none")

        s.configure("TLabel", font=("Helvetica", 64),
                    foreground="#F7FBEF",
                    background="#292929",
                    lightcolor="none",
                    darkcolor="none",
                    focuscolor="#333333",
                    borderwidth=0,
                    bordercolor="none")

        s.configure("TFrame",
                    foreground="#F7FBEF",
                    background="#1F1F1F",
                    lightcolor="none",
                    darkcolor="none",
                    focuscolor="#333333",
                    borderwidth=0,
                    bordercolor="none")

        s.map('TButton',
              background=[("pressed", "#292929"),
                          ("active", "#333333")],
              borderwidth=[("active", 0)],
              bordercolor=[("active", "none")],
              lightcolor=[("active", "none")],
              darkcolor=[("active", "none")],
              foreground=[("pressed", "#F7FBEF"),
                          ("active", "#F7FBEF")]
              )

    def widget_setup(self):
        image_canvas = ImageCanvas(self)
        image_canvas.pack(expand=True, fill=tk.BOTH, side=tk.LEFT)
        image_canvas.bind('<Configure>', self.action_resize_image)
        self.image_canvas: ImageCanvas = image_canvas

        button_frame: ButtonFrame = ButtonFrame(
            self, self.action_generate)
        button_frame.pack(fill=tk.BOTH, side=tk.TOP)

    # TODO: Delete this variable when model is functional
    index = 0

    def action_generate(self, event=None):
        # TODO: integrate model into the GUI
        model_output = ...  # generate model output
        image = ...  # convert model output to Image

        # TODO: Delete this block when model is functional
        self.index = self.index % 3 + 1
        image = Image.open(Path(__file__).parent /
                           ('images/example' + str(self.index) + '.jpg'))
        self.index += 1

        self.image_canvas.set_image(image)
        self.image_canvas.update()

    def action_resize_image(self, event):
        canvas = event.widget
        origin = (0, 0)
        size = (event.width, event.height)
        if canvas.bbox("bg") != origin + size:
            canvas.display_image(origin, size)


class ImageCanvas(tk.Canvas):
    def __init__(self, container):
        super().__init__(container)
        self.load_image(Path(__file__).parent / 'images/start.png')
        self["highlightthickness"] = 0

    def set_image(self, image, path=""):
        self.image_path = path
        self.image = image
        self.img_copy = self.image.copy()
        self.photo = ImageTk.PhotoImage(self.image)
        self.display_image()

    def load_image(self, path):
        self.set_image(Image.open(path), path)

    def display_image(self, origin=(0, 0), size=None):
        if size is None:
            size = (self.winfo_width(), self.winfo_height())

        self.delete("bg")
        self.image = self.img_copy.resize(size)
        self.photo = ImageTk.PhotoImage(self.image)
        self.create_image(*origin, anchor="nw", image=self.photo, tags="bg")
        self.tag_lower("bg", "all")


class ButtonFrame(ttk.Frame):
    def __init__(self, container, action):
        super().__init__(container)
        self.widget_setup(action)

    def widget_setup(self, action):
        ipadding = {'ipadx': 10, 'ipady': 10}

        btn = ttk.Button(self, text='Generate')
        btn.bind('<Button-1>', action, add="+")
        btn.pack(**ipadding, fill=tk.X)


if __name__ == "__main__":
    gui = GeneratorApp()
    gui.mainloop()
