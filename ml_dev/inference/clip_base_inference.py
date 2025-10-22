import torch
import clip
import time
import signal
import sys
import numpy as np
from PIL import Image
import os
from pathlib import Path

class InfiniteCLIPInference:
    def __init__(self, model_name="ViT-B/32", device="cuda"):
        self.device = device
        self.model_name = model_name
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.running = True
        self.inference_count = 0
        self.total_processing_time = 0
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        print(f"Loaded {model_name} on {device}")
        
    def signal_handler(self, sig, frame):
        """Handle interrupt signals gracefully"""
        print(f"\nReceived signal {sig}. Shutting down...")
        self.running = False
        
    def setup_text_prompts(self, labels=None):
        """Setup text prompts for classification"""
        if labels is None:
            labels = ["cat", "dog", "car", "person", "tree", "building", 
                     "food", "animal", "landscape", "computer"]
        
        self.text_prompts = [f"a photo of a {label}" for label in labels]
        self.text_tokens = clip.tokenize(self.text_prompts).to(self.device)
        
        # Precompute text embeddings once
        with torch.no_grad():
            self.text_features = self.model.encode_text(self.text_tokens)
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
            
        print(f"Setup {len(self.text_prompts)} text prompts")
        
    def get_next_image(self):
        """Get the next image for processing - override this for your use case"""
        # This is where you'd get your real image source
        # Examples: camera feed, video stream, image directory, etc.
        
        # For demo purposes, we'll create synthetic images
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        return Image.fromarray(img_array), f"synthetic_image_{self.inference_count}"
    
    def process_single_image(self, image, image_id="unknown"):
        """Process a single image and return results"""
        start_time = time.time()
        
        try:
            # Preprocess image
            processed_image = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                # Encode image
                image_features = self.model.encode_image(processed_image)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity
                similarity = (100.0 * image_features @ self.text_features.T)
                probs = similarity.softmax(dim=-1)
                
            # Get results
            probs = probs.cpu().numpy()[0]
            best_idx = np.argmax(probs)
            best_label = self.text_prompts[best_idx]
            confidence = probs[best_idx]
            
            processing_time = time.time() - start_time
            
            return {
                'image_id': image_id,
                'best_label': best_label,
                'confidence': confidence,
                'all_probs': probs,
                'processing_time': processing_time
            }
            
        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
            return None
    
    def process_results(self, results):
        """Process and display inference results"""
        if results is None:
            return
            
        self.inference_count += 1
        self.total_processing_time += results['processing_time']
        avg_time = self.total_processing_time / self.inference_count
        
        # Display results
        if self.inference_count % 10 == 0:
            print(f"\n--- Inference #{self.inference_count} ---")
            print(f"Image: {results['image_id']}")
            print(f"Prediction: {results['best_label']} (confidence: {results['confidence']:.3f})")
            print(f"Processing time: {results['processing_time']*1000:.1f}ms")
            print(f"Average time: {avg_time*1000:.1f}ms")
            print(f"Throughput: {1/avg_time:.1f} FPS")
        else:
            # Brief output for most frames
            print(f"#{self.inference_count}: {results['best_label']} ({results['confidence']:.2f}) - {results['processing_time']*1000:.1f}ms", end='\r')
    
    def run_infinite_inference(self, target_fps=30):
        """Run infinite inference loop"""
        print("\n" + "="*60)
        print("STARTING INFINITE CLIP INFERENCE")
        print("="*60)
        print("Press Ctrl+C to stop")
        
        # Setup text prompts
        self.setup_text_prompts()
        
        # Performance tracking
        frame_times = []
        start_time = time.time()
        
        try:
            while self.running:
                iteration_start = time.time()
                
                # Get next image
                image, image_id = self.get_next_image()
                
                # Process image
                results = self.process_single_image(image, image_id)
                
                # Handle results
                self.process_results(results)
                
                # Frame rate control
                processing_time = time.time() - iteration_start
                frame_times.append(processing_time)
                
                # Maintain target FPS
                if target_fps > 0:
                    target_frame_time = 1.0 / target_fps
                    sleep_time = target_frame_time - processing_time
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
                # Keep only recent frame times for stats
                if len(frame_times) > 100:
                    frame_times.pop(0)
                    
        except KeyboardInterrupt:
            print("\nStopped by user")
        except Exception as e:
            print(f"\nUnexpected error: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Clean shutdown with statistics"""
        print("\n" + "="*60)
        print("INFERENCE SHUTDOWN")
        print("="*60)
        
        if self.inference_count > 0:
            avg_time = self.total_processing_time / self.inference_count
            total_time = time.time() - self.start_time if hasattr(self, 'start_time') else 0
            
            print(f"Total inferences: {self.inference_count}")
            print(f"Average processing time: {avg_time*1000:.2f} ms")
            print(f"Average throughput: {1/avg_time:.2f} FPS")
            print(f"Total processing time: {self.total_processing_time:.2f} seconds")
        
        print("CLIP inference stopped gracefully")

# Example usage with different image sources
class CameraInference(InfiniteCLIPInference):
    def __init__(self, camera_index=0, **kwargs):
        super().__init__(**kwargs)
        self.camera_index = camera_index
        # You would initialize your camera here
        # self.camera = cv2.VideoCapture(camera_index)
        
    def get_next_image(self):
        """Get image from camera"""
        # Example with OpenCV (uncomment if you have camera)
        # ret, frame = self.camera.read()
        # if ret:
        #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     return Image.fromarray(frame_rgb), f"camera_frame_{self.inference_count}"
        # else:
        #     return self.get_fallback_image()
        
        # Fallback to synthetic images
        return super().get_next_image()

class DirectoryInference(InfiniteCLIPInference):
    def __init__(self, image_directory, **kwargs):
        super().__init__(**kwargs)
        self.image_directory = Path(image_directory)
        self.image_files = list(self.image_directory.glob("*.jpg")) + list(self.image_directory.glob("*.png"))
        self.current_index = 0
        print(f"Found {len(self.image_files)} images in directory")
        
    def get_next_image(self):
        """Get next image from directory (cycles through images)"""
        if not self.image_files:
            return self.get_fallback_image()
            
        image_file = self.image_files[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.image_files)
        
        try:
            image = Image.open(image_file)
            return image, image_file.name
        except Exception as e:
            print(f"Error loading {image_file}: {e}")
            return self.get_next_image()