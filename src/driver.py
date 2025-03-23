import tkinter as tk
from tkinter import scrolledtext, ttk, filedialog
import subprocess
import threading
import os
import re

# makes the code run in virtual environment with the requied libraries
VENV_PYTHON = os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe")

class ModelRunnerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Model Runner")
        self.root.geometry("800x500")
        self.root.minsize(600, 400)

        
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(5, weight=1)

        
        self.folder_label = tk.Label(root, text="Select Folder:")
        self.folder_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        self.folder_path = tk.StringVar()
        self.folder_entry = tk.Entry(root, textvariable=self.folder_path)
        self.folder_entry.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

        self.btn_browse = tk.Button(root, text="Browse", command=self.browse_folder)
        self.btn_browse.grid(row=0, column=2, padx=10, pady=5)

        
        self.model_label = tk.Label(root, text="Select Model Script:")
        self.model_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(root, textvariable=self.model_var, state="readonly")
        self.model_dropdown.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        
        self.btn_run = tk.Button(root, text="Run Model", command=self.run_model, state=tk.DISABLED)
        self.btn_run.grid(row=1, column=2, padx=10, pady=5)

        
        self.output_text = scrolledtext.ScrolledText(root, height=15)
        self.output_text.grid(row=2, column=0, columnspan=3, padx=10, pady=5, sticky="nsew")

        
        self.btn_clear = tk.Button(root, text="Clear Output", command=self.clear_output)
        self.btn_clear.grid(row=3, column=1, columnspan=3, pady=5)

        
        self.accuracy_label = tk.Label(root, text="Accuracy: N/A", font=("Arial", 12, "bold"))
        self.accuracy_label.grid(row=3, column=0, columnspan=3, pady=5)

        
        for i in range(3):
            self.root.columnconfigure(i, weight=1)
        self.root.rowconfigure(2, weight=1)

        
        self.model_dropdown.bind("<<ComboboxSelected>>", self.on_model_selected)

    def browse_folder(self):
        """select the folder where the models are sotred"""
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.folder_path.set(folder_selected)
            self.refresh_model_list(folder_selected)

    def refresh_model_list(self, folder):
        
        try:
            models = [f for f in os.listdir(folder) if f.endswith(".py")]
            self.model_dropdown["values"] = models

            if models:
                self.model_var.set(models[0])  # Select the first model by default
                self.btn_run.config(state=tk.NORMAL)
            else:
                self.btn_run.config(state=tk.DISABLED)

        except Exception as e:
            self.output_text.insert(tk.END, f"\nError loading models: {e}\n", "error")
            self.output_text.see(tk.END)

    def on_model_selected(self, event=None):
        
        self.btn_run.config(state=tk.NORMAL)

    def run_model(self):

        selected_folder = self.folder_path.get()
        selected_model = self.model_var.get()

        if not selected_folder or not selected_model:
            return

        self.output_text.insert(tk.END, f"Running model: {selected_model}\n")
        self.output_text.see(tk.END)

        
        thread = threading.Thread(target=self.execute_script, args=(selected_folder, selected_model))
        thread.start()

    def execute_script(self, folder, script_name):
        
        try:
            script_path = os.path.join(folder, script_name)  
            process = subprocess.Popen(
                [VENV_PYTHON, script_path],  
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            accuracy = None  

            for line in iter(process.stdout.readline, ''):
                self.output_text.insert(tk.END, line)
                self.output_text.see(tk.END)  
                self.root.update_idletasks()

                match = re.search(r'Accuracy:\s*([\d.]+)', line)
                if match:
                    accuracy = match.group(1)

            process.stdout.close()
            process.wait()


            stderr_output = process.stderr.read()
            if stderr_output:
                self.output_text.insert(tk.END, f"\nERROR:\n{stderr_output}\n", "error")
                self.output_text.see(tk.END)

            if accuracy:
                self.accuracy_label.config(text=f"Accuracy: {accuracy}")

        except Exception as e:
            self.output_text.insert(tk.END, f"\nException: {e}\n", "error")
            self.output_text.see(tk.END)

    def clear_output(self):
        """Clears the output text area."""
        self.output_text.delete("1.0", tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    app = ModelRunnerApp(root)
    root.mainloop()
