from tkinter import filedialog as _filedialog
import tkinter as _tk


def choose_file(initialdir='.', filetypes=(('all files', '*.*'),), save=False):
    window = _tk.Tk()
    window.wm_withdraw()
    filename = ''
    if not save:
        filename = _filedialog.askopenfilename(initialdir=initialdir, title='Choose file', filetypes=filetypes)
    else:
        filename = _filedialog.asksaveasfilename(initialdir=initialdir, title='Choose file', filetypes=filetypes)
    window.destroy()
    return filename

def choose_dir(initialdir='.'):
    window = _tk.Tk()
    window.wm_withdraw()
    directory = _filedialog.askdirectory(initialdir=initialdir, title='Choose directory')
    window.destroy()
    return directory
