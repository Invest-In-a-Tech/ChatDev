import tkinter as tk
from tkinter import filedialog
class Editor(tk.Text):
    '''
    This class represents the editor component of the MyPad application.
    '''
    def __init__(self, root):
        super().__init__(root)
        self.configure(font=("Arial", 12))
        self.pack(fill=tk.BOTH, expand=True)