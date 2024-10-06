# gui.py

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os
from main_app import OrchestraConductorApp

def start_app(music_path, bpm):
    try:
        # Validate BPM
        bpm = float(bpm)
        if bpm <= 0:
            raise ValueError("BPM must be a positive number.")
    except ValueError as ve:
        messagebox.showerror("Invalid BPM", str(ve))
        return

    if not os.path.exists(music_path):
        messagebox.showerror("Invalid File", "The selected music file does not exist.")
        return

    # Disable the start button to prevent multiple instances
    start_button.config(state=tk.DISABLED)

    # Start the main application in a separate thread
    app_thread = threading.Thread(target=run_application, args=(music_path, bpm))
    app_thread.start()

    messagebox.showinfo("Application Started", "Orchestra Conductor is now running.")

def run_application(music_path, bpm):
    app = OrchestraConductorApp(music_path, bpm)
    app.run()

def browse_file():
    file_path = filedialog.askopenfilename(
        title="Select Music File",
        filetypes=(("MP3 Files", "*.mp3"), ("WAV Files", "*.wav"), ("All Files", "*.*"))
    )
    if file_path:
        music_path_var.set(file_path)

# Create the main window
root = tk.Tk()
root.title("Orchestra Conductor - Setup")
root.geometry("500x250")
root.resizable(False, False)

# Apply a style
style = ttk.Style()
style.theme_use('clam')  # You can choose 'clam', 'alt', 'default', 'classic'

# Create a frame for padding
main_frame = ttk.Frame(root, padding="20")
main_frame.pack(fill=tk.BOTH, expand=True)

# Music Path
music_path_label = ttk.Label(main_frame, text="Select Music File:")
music_path_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))

music_path_var = tk.StringVar()

music_path_entry = ttk.Entry(main_frame, textvariable=music_path_var, width=40)
music_path_entry.grid(row=0, column=1, pady=(0, 10), padx=(0, 10))

browse_button = ttk.Button(main_frame, text="Browse", command=browse_file)
browse_button.grid(row=0, column=2, pady=(0, 10))

# BPM Entry
bpm_label = ttk.Label(main_frame, text="Enter Music BPM:")
bpm_label.grid(row=1, column=0, sticky=tk.W, pady=(0, 10))

bpm_var = tk.StringVar()

bpm_entry = ttk.Entry(main_frame, textvariable=bpm_var, width=20)
bpm_entry.grid(row=1, column=1, sticky=tk.W, pady=(0, 10))

# Start Button
start_button = ttk.Button(main_frame, text="Start", command=lambda: start_app(music_path_var.get(), bpm_var.get()))
start_button.grid(row=2, column=1, pady=(20, 0), sticky=tk.E)

# Instructions
instructions = (
    "Instructions:\n"
    "1. Select a music file (MP3/WAV).\n"
    "2. Enter the BPM of the music.\n"
    "3. Click 'Start' to begin."
)
instructions_label = ttk.Label(main_frame, text=instructions, foreground="gray")
instructions_label.grid(row=3, column=0, columnspan=3, pady=(20, 0))

# Center the GUI on the screen
root.update_idletasks()
window_width = root.winfo_width()
window_height = root.winfo_height()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width // 2) - (window_width // 2)
y = (screen_height // 2) - (window_height // 2)
root.geometry(f'{window_width}x{window_height}+{x}+{y}')

# Start the GUI event loop
root.mainloop()
