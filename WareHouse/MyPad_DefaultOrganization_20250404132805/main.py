'''
This is the main file of the MyPad application. It initializes the GUI and starts the application.
'''
import tkinter as tk
from editor import Editor
class MyPad:
    def __init__(self, root):
        self.root = root
        self.root.title("MyPad")
        self.editor = Editor(root)
        self.editor.pack()
if __name__ == "__main__":
    root = tk.Tk()
    app = MyPad(root)
    root.mainloop()