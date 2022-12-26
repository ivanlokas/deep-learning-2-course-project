import tkinter as tk
from pathlib import Path
from tkinter import ttk

import torch
from PIL import Image, ImageTk
from torchvision import transforms

from models.convolutional import ConvolutionalGenerator


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
    # Image transform
    mean, std = (-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5), (1 / 0.5, 1 / 0.5, 1 / 0.5)
    transform = transforms.Compose([
        transforms.Normalize(mean, std),
        transforms.ToPILImage(),
    ])

    # Hyper parameters
    batch_size = 80
    learning_rate = 1e-3
    betas = (0.5, 0.999)
    n_epochs = 100
    noise_dimension = 128
    image_size = 256
    n_channels = 3
    epoch = 16

    # Model
    generator = ConvolutionalGenerator(image_size=image_size, noise_dimension=noise_dimension)

    # Load state dict
    path = Path(__file__).parent.parent / 'states' \
           / f'convolutional_bs_{batch_size}_ne_{n_epochs}_lr_{learning_rate}_sz_{image_size}' \
           / f'generator_epoch_{epoch}'
    generator.load_state_dict(torch.load(path))

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

    def action_generate(self, event=None):
        noise = torch.randn(1, MainFrame.noise_dimension, 1, 1)
        model_output = MainFrame.generator(noise).float()
        model_output = torch.reshape(model_output, (MainFrame.n_channels, MainFrame.image_size, MainFrame.image_size))
        image = MainFrame.transform(model_output)

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
