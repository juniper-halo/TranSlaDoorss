import torch
import clip
from datasets import load_dataset
from PIL import Image
import time
import numpy as np
from torch.utils.data import DataLoader
import argparse

class CLIPInferenceBenchmark:
    def __init__(self, model_name="ViT-B/32", device="cuda"):
        self.device = device
        self.model_name = model_name
        self.model, self.preprocess = clip.load(model_name, device=device)
        print(f"Loaded {model_name} on {device}")
        
    def load_huggingface_dataset(self, dataset_name, split="test", num_samples=1000):

        """Load a dataset from HuggingFace"""
        print(f"Loading {dataset_name} dataset...")
        
        try:
            if dataset_name == "cifar10":
                dataset = load_dataset("cifar10", split=split)
                print(f"CIFAR-10 features: {dataset.features}")
                
            elif dataset_name == "food101":
                dataset = load_dataset("food101", split=split)
                
            elif dataset_name == "cats_vs_dogs":
                dataset = load_dataset("cats_vs_dogs", split=split)
                
            else:
                # try to load generic image dataset
                dataset = load_dataset(dataset_name, split=split)
                
            #limit dataset size
            if num_samples and len(dataset) > num_samples:
                dataset = dataset.select(range(num_samples))
                
            print(f"Loaded {len(dataset)} samples from {dataset_name}")
            return dataset
            
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            return None

    def preprocess_dataset(self, dataset):
        """Preprocess dataset for CLIP"""
        processed_data = []
        
        for i, example in enumerate(dataset):
            try:
                # Handle different dataset structures
                if "img" in example:  # CIFAR-10 uses 'img'
                    image_array = example["img"]
                    if isinstance(image_array, np.ndarray):
                        image = Image.fromarray(image_array)
                    else:
                        image = image_array
                        
                elif "image" in example:
                    image = example["image"]
                    if not isinstance(image, Image.Image):
                        image = Image.fromarray(image)
                        
                else:
                    # Try to find any image-like data
                    for key, value in example.items():
                        if isinstance(value, (Image.Image, np.ndarray)):
                            if isinstance(value, np.ndarray) and value.ndim in [2, 3]:
                                image = Image.fromarray(value)
                                break
                    else:
                        print(f"No image found in example {i}, keys: {list(example.keys())}")
                        continue
                
                # Apply CLIP preprocessing
                processed_image = self.preprocess(image)
                processed_data.append({
                    "image": processed_image,
                    "label": example.get("label", -1),
                    "original_image": image
                })
                
            except Exception as e:
                print(f"Error processing example {i}: {e}")
                print(f"Example keys: {list(example.keys()) if hasattr(example, 'keys') else 'No keys'}")
                continue
                
        print(f"Successfully processed {len(processed_data)}/{len(dataset)} images")
        return processed_data

    def get_text_prompts(self, dataset_name):
        """Get appropriate text prompts for different datasets"""
        if dataset_name == "cifar10":
            labels = ["airplane", "automobile", "bird", "cat", "deer", 
                     "dog", "frog", "horse", "ship", "truck"]
            prompts = [f"a photo of a {label}" for label in labels]
            
        elif dataset_name == "food101":
            prompts = [f"a photo of {label}" for label in [
                "apple pie", "pizza", "hamburger", "sushi", "pasta",
                "salad", "steak", "ice cream", "cake", "coffee"
            ]]
            
        elif dataset_name == "cats_vs_dogs":
            prompts = ["a photo of a cat", "a photo of a dog"]
            
        else:
            # Generic prompts
            prompts = [
                "a photo of an object", "a photo of an animal", 
                "a photo of a person", "a photo of food",
                "a photo of a vehicle", "a photo of a building"
            ]
            
        return prompts

    def benchmark_single_image(self, processed_data, text_prompts, num_trials=100):
        """Benchmark single image inference"""
        print("\n" + "="*50)
        print("SINGLE IMAGE INFERENCE BENCHMARK")
        print("="*50)
        
        if not processed_data:
            print("No processed data available for benchmarking")
            return None
            
        # Precompute text embeddings (do this once)
        text_tokens = clip.tokenize(text_prompts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Warm up GPU
        print("Warming up GPU...")
        warmup_image = processed_data[0]["image"].unsqueeze(0).to(self.device)
        for _ in range(10):
            with torch.no_grad():
                _ = self.model.encode_image(warmup_image)
        
        # Benchmark single image
        single_image = processed_data[0]["image"].unsqueeze(0).to(self.device)
        latencies = []
        
        print(f"Running {num_trials} trials...")
        with torch.no_grad():
            for i in range(num_trials):
                start_time = time.time()
                
                # Image encoding
                image_features = self.model.encode_image(single_image)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Similarity calculation
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                
                latency = (time.time() - start_time) * 1000  # Convert to milliseconds
                latencies.append(latency)
                
                if (i + 1) % 20 == 0:
                    print(f"  Completed {i + 1}/{num_trials} trials")
        
        # Calculate statistics
        latencies = np.array(latencies)
        stats = {
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'throughput_fps': 1000 / np.mean(latencies)
        }
        
        print(f"\nSingle Image Results:")
        print(f"  Average Latency: {stats['mean_latency_ms']:.2f} ms")
        print(f"  Throughput: {stats['throughput_fps']:.1f} FPS")
        print(f"  Latency Std: {stats['std_latency_ms']:.2f} ms")
        print(f"  95th Percentile: {stats['p95_latency_ms']:.2f} ms")
        
        return stats

    def run_complete_benchmark(self, dataset_name="cifar10", num_samples=500):
        """Run complete benchmark suite"""
        print("="*60)
        print(f"CLIP INFERENCE BENCHMARK - {self.model_name}")
        print(f"Dataset: {dataset_name}")
        print(f"Device: {self.device}")
        print("="*60)
        
        # Load dataset
        dataset = self.load_huggingface_dataset(dataset_name, num_samples=num_samples)
        if dataset is None:
            print(f"Failed to load dataset {dataset_name}")
            return
        
        # Preprocess data - FIXED
        processed_data = self.preprocess_dataset(dataset)
        
        if not processed_data:
            print("No images were successfully processed. Trying alternative approach...")
            # Try a simpler dataset
            return self.test_with_synthetic_data()
            
        # Get text prompts
        text_prompts = self.get_text_prompts(dataset_name)
        print(f"Using {len(text_prompts)} text prompts")
        
        # Run benchmarks
        single_results = self.benchmark_single_image(processed_data, text_prompts)
        
        # Compile results
        results = {
            'model': self.model_name,
            'dataset': dataset_name,
            'device': self.device,
            'single_image': single_results,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return results
 

def main():
    parser = argparse.ArgumentParser(description='CLIP Inference Benchmark')

    parser.add_argument('--model', type=str, default='ViT-B/32', 
                       choices=['ViT-B/32', 'ViT-B/16', 'RN50'],
                       help='CLIP model to benchmark')
    
    parser.add_argument('--dataset', type=str, default='cifar10',
                       help='HuggingFace dataset name')
    
    parser.add_argument('--num_samples', type=int, default=500,
                       help='Number of samples to use for benchmarking')
    
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = CLIPInferenceBenchmark(
        model_name=args.model,
        device=args.device
    )
    
    # Run benchmark
    results = benchmark.run_complete_benchmark(
        dataset_name=args.dataset,
        num_samples=args.num_samples
    )
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()