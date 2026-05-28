import os
import tkinter as tk
from PIL import Image, ImageTk

IMG_FOLDER = "data/OCR data/cropped_plates"
LABEL_FILE = "data/OCR data/rec_gt_test.txt"

class SimpleLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR Labeling Tool")
        self.root.geometry("600x400")

        self.images = [f for f in os.listdir(IMG_FOLDER) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.labeled_images = set()
        
        if os.path.exists(LABEL_FILE):
            with open(LABEL_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if parts:
                        self.labeled_images.add(parts[0])
                    
        self.images = [img for img in self.images if img not in self.labeled_images]
        self.current_idx = 0

        if not self.images:
            tk.Label(root, text="All images labeled!", font=("Arial", 20)).pack(pady=100)
            return

        self.lbl_progress = tk.Label(root, text="", font=("Arial", 12))
        self.lbl_progress.pack(pady=10)

        self.canvas = tk.Canvas(root, width=400, height=200, bg="gray")
        self.canvas.pack()

        self.entry_var = tk.StringVar()
        self.entry = tk.Entry(root, textvariable=self.entry_var, font=("Arial", 24), justify='center')
        self.entry.pack(pady=20)
        self.entry.focus()
        self.root.bind('<Return>', self.save_and_next)

        self.load_image()

    def load_image(self):
        if self.current_idx >= len(self.images):
            self.canvas.delete("all")
            self.entry.destroy()
            self.lbl_progress.config(text="Labeling completed!")
            return

        img_name = self.images[self.current_idx]
        self.lbl_progress.config(text=f"Processing: {img_name} ({self.current_idx + 1}/{len(self.images)})")

        img_path = os.path.join(IMG_FOLDER, img_name)
        image = Image.open(img_path)
        
        image = image.resize((400, min(200, int((400 / image.width) * image.height))), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        
        self.canvas.delete("all")
        self.canvas.create_image(200, 100, image=self.photo)
        self.entry_var.set("")

    def save_and_next(self, event=None):
        label = self.entry_var.get().strip()
        if not label:
            return
            
        img_name = self.images[self.current_idx]
        
        with open(LABEL_FILE, "a", encoding="utf-8") as f:
            f.write(f"{img_name}\t{label}\n")

        self.current_idx += 1
        self.load_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleLabeler(root)
    root.mainloop()