import sys
import os

# Add the current directory to Python path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from your packages
from cameras.wired_android_camera import WiredAndroidCamera, WiredAndroidOpenCV
from inference.clip_base_inference import InfiniteCLIPInference
from data_visualization_tools.data_visualization import SomeVisualizationClass  # Adjust based on actual class name

# If you need development modules
from development.clip_base import SomeBaseClass  # Adjust based on actual class name
from development.clip_fine_tuned import FineTunedCLIP  # Adjust based on actual class name
from development.clip_modified import ModifiedCLIP  # Adjust based on actual class name

# If you need benchmarking
from inference.clip_base_benchmark import BenchmarkClass  # Adjust based on actual class name

def main():
    # Choose your inference type
    inference_type = "synthetic"  # Change to "camera" or "directory" as needed
    
    if inference_type == "synthetic":
        # Basic infinite inference with synthetic images
        inference = InfiniteCLIPInference(model_name="ViT-B/32", device="cuda")
        
    elif inference_type == "directory":
        # Inference on images from a directory
        image_dir = "path/to/your/images"  # Change this path
        inference = DirectoryInference(image_directory=image_dir, model_name="ViT-B/32", device="cuda")
        
    elif inference_type == "camera":
        # Real-time camera inference
        inference = CameraInference(camera_index=0, model_name="ViT-B/32", device="cuda")
    
    # Set target FPS (0 for maximum speed)
    target_fps = 30
    
    # Record start time
    inference.start_time = time.time()
    
    # Run infinite inference
    inference.run_infinite_inference(target_fps=target_fps)

if __name__ == "__main__":
    main()