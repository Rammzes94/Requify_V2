#!/usr/bin/env python3
"""
image_analyzer_gui.py

A simple GUI application for analyzing images with Ollama vision models.
It provides a drag-and-drop interface to analyze any image on your computer.
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from pathlib import Path

# Try to import required packages
try:
    from agno.agent import Agent
    from agno.media import Image
    from agno.models.ollama import Ollama
    AGNO_AVAILABLE = True
except ImportError:
    AGNO_AVAILABLE = False

# Get the current directory where the script is located
SCRIPT_DIR = Path(__file__).parent

# Available Ollama vision models
VISION_MODELS = [
    "llava",
    "bakllava",
    "gemma3",
    "llama3.2-vision"
]

class ImageAnalyzerApp:
    """Simple GUI application for analyzing images with Ollama."""
    
    def __init__(self, root):
        """Initialize the application UI."""
        self.root = root
        self.root.title("Ollama Image Analyzer")
        self.root.geometry("800x700")
        
        # Configure grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.columnconfigure(1, weight=1)
        
        # Create widgets
        self.create_widgets(main_frame)
        
        # Check if Agno is installed
        if not AGNO_AVAILABLE:
            messagebox.showerror(
                "Missing Dependencies",
                "The Agno package is not installed. Please install it with:\n\npip install agno"
            )
    
    def create_widgets(self, parent):
        """Create all widgets for the application."""
        
        # Image selection frame
        img_frame = ttk.LabelFrame(parent, text="Image Selection", padding="10")
        img_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        # Image path entry and browse button
        ttk.Label(img_frame, text="Image Path:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.img_path_var = tk.StringVar()
        self.img_path_entry = ttk.Entry(img_frame, textvariable=self.img_path_var, width=60)
        self.img_path_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        browse_btn = ttk.Button(img_frame, text="Browse...", command=self.browse_image)
        browse_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # Drag and drop instructions
        drop_label = ttk.Label(img_frame, text="Or drag and drop an image file here:")
        drop_label.grid(row=1, column=0, columnspan=3, padx=5, pady=5)
        
        # Drop area
        self.drop_area = tk.Canvas(img_frame, bg="lightgray", height=100)
        self.drop_area.grid(row=2, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        self.drop_area.create_text(
            400, 50, text="Drop your image file here", font=("Arial", 14)
        )
        
        # Set up drag and drop events
        self.drop_area.bind("<ButtonPress-1>", self.on_drop_area_click)
        
        # Model selection and prompt frame
        options_frame = ttk.LabelFrame(parent, text="Analysis Options", padding="10")
        options_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        # Model selection
        ttk.Label(options_frame, text="Model:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.model_var = tk.StringVar(value=VISION_MODELS[0])
        model_combo = ttk.Combobox(options_frame, textvariable=self.model_var, values=VISION_MODELS)
        model_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        # Prompt
        ttk.Label(options_frame, text="Prompt:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.prompt_var = tk.StringVar(value="Describe in detail what you see in this image.")
        prompt_entry = ttk.Entry(options_frame, textvariable=self.prompt_var, width=60)
        prompt_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        
        # Analyze button
        analyze_btn = ttk.Button(
            options_frame, 
            text="Analyze Image", 
            command=self.analyze_image,
            style="Accent.TButton"
        )
        analyze_btn.grid(row=2, column=0, columnspan=2, padx=5, pady=10)
        
        # Results frame
        results_frame = ttk.LabelFrame(parent, text="Analysis Results", padding="10")
        results_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        results_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, height=15)
        self.results_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.results_text.insert(tk.END, "Analysis results will appear here...")
        self.results_text.config(state=tk.DISABLED)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(parent, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=3, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
    
    def browse_image(self):
        """Open a file dialog to select an image."""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.gif *.bmp *.tiff"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(
            title="Select Image",
            filetypes=filetypes
        )
        
        if filepath:
            self.img_path_var.set(filepath)
    
    def on_drop_area_click(self, event):
        """Handle click on drop area - open file dialog."""
        self.browse_image()
    
    def analyze_image(self):
        """Analyze the selected image using Ollama."""
        # Check if Agno is installed
        if not AGNO_AVAILABLE:
            messagebox.showerror(
                "Missing Dependencies",
                "The Agno package is not installed. Please install it with:\n\npip install agno"
            )
            return
        
        # Get image path
        image_path = self.img_path_var.get()
        if not image_path:
            messagebox.showerror("Error", "Please select an image to analyze")
            return
        
        if not os.path.exists(image_path):
            messagebox.showerror("Error", f"Image file not found: {image_path}")
            return
        
        # Get model and prompt
        model_name = self.model_var.get()
        prompt = self.prompt_var.get()
        
        # Update status
        self.status_var.set(f"Analyzing image with {model_name}...")
        
        # Clear results
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Processing...\n")
        self.results_text.config(state=tk.DISABLED)
        
        # Run analysis in a separate thread to avoid freezing the UI
        threading.Thread(target=self._run_analysis, args=(image_path, prompt, model_name)).start()
    
    def _run_analysis(self, image_path, prompt, model_name):
        """Run the actual analysis in a background thread."""
        try:
            # Create an agent with the specified Ollama vision model
            agent = Agent(
                model=Ollama(id=model_name),
                markdown=True,
            )
            
            # Process the image
            response = agent.run(
                prompt,
                images=[Image(filepath=image_path)],
            )
            
            # Update the UI with the results
            self.root.after(0, self._update_results, response.content, "Success")
            
        except Exception as e:
            error_msg = f"Error analyzing image: {str(e)}\n\n"
            error_msg += "Troubleshooting tips:\n"
            error_msg += "1. Make sure Ollama is installed and running\n"
            error_msg += f"2. Verify that the '{model_name}' model is installed (run 'ollama list')\n"
            error_msg += f"3. If the model is missing, install it with 'ollama pull {model_name}'\n"
            error_msg += "4. Check if your image format is supported (JPG, PNG, etc.)\n"
            
            self.root.after(0, self._update_results, error_msg, "Error")
    
    def _update_results(self, text, status):
        """Update the results text area with the analysis results."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, text)
        self.results_text.config(state=tk.DISABLED)
        
        self.status_var.set(f"Status: {status}")


def simulate_analysis():
    """Simulate image analysis for testing when Agno is not available."""
    import time
    import random
    
    # Simulate processing time
    time.sleep(2)
    
    # Sample responses
    responses = [
        "The image shows a beautiful landscape with mountains in the background and a lake in the foreground. "
        "The sky is clear blue with some scattered clouds. The scene appears to be in a national park or wilderness area.",
        
        "I can see a photograph of a natural scene with trees and possibly water. "
        "The image has good lighting and appears to be taken during daytime. "
        "The composition suggests this is a popular scenic location.",
        
        "This is an image of what appears to be a landscape. There are natural elements visible "
        "including vegetation and possibly water features. The lighting suggests it was taken outdoors "
        "during daylight hours."
    ]
    
    return random.choice(responses)


def main():
    """Main function to start the application."""
    # Set up the root window
    root = tk.Tk()
    root.title("Ollama Image Analyzer")
    
    # Create a style
    style = ttk.Style()
    if "clam" in style.theme_names():
        style.theme_use("clam")
    
    # Create the application
    app = ImageAnalyzerApp(root)
    
    # Start the main loop
    root.mainloop()


if __name__ == "__main__":
    main() 