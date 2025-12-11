"""
asl image preprocessing pipeline (resize/crop to clip-friendly rgb square)
"""
from PIL import Image
from typing import Tuple, Optional


class ASLPreprocessor:
    """preprocess asl sign images for clip"""
    
    def __init__(self, target_size: int = 224):
        """init with target size (default 224 for clip)"""
        self.target_size = target_size
    
    def preprocess(self, image: Image.Image) -> Image.Image:
        """convert to rgb, center-crop square, resize to target"""
        # convert to rgb if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # center crop to square handle aspect ratio
        image = self._center_crop_to_square(image)
        
        # resize to target size with high quality resampling
        image = image.resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)
        
        return image
    
    def _center_crop_to_square(self, image: Image.Image) -> Image.Image:
        """center crop to square aspect ratio"""
        width, height = image.size
        
        # if already square return as is
        if width == height:
            return image
        
        # calculate crop dimensions
        size = min(width, height)
        left = (width - size) // 2
        top = (height - size) // 2
        
        # perform center crop
        cropped_image = image.crop((left, top, left + size, top + size))
        
        return cropped_image
    
    def preprocess_batch(self, images: list) -> list:
        """
        Preprocess a batch of images
        
        Args:
            images: List of PIL Images to preprocess
            
        Returns:
            List of preprocessed PIL Images
        """
        return [self.preprocess(image) for image in images]
    
    def get_preprocessing_info(self, image: Image.Image) -> dict:
        """
        Get information about the preprocessing steps applied
        
        Args:
            image: Original PIL Image
            
        Returns:
            Dictionary with preprocessing information
        """
        original_size = image.size
        original_mode = image.mode
        
        # apply preprocessing
        processed_image = self.preprocess(image)
        processed_size = processed_image.size
        
        return {
            'original_size': original_size,
            'original_mode': original_mode,
            'processed_size': processed_size,
            'target_size': self.target_size,
            'was_cropped': original_size[0] != original_size[1],
            'was_converted': original_mode != 'RGB'
        }


# test the preprocessor
if __name__ == "__main__":

    from PIL import Image
    import numpy as np
    
    print("Testing ASL Preprocessor...")
    
    test_image = Image.fromarray(np.random.randint(0, 255, (300, 200, 3), dtype=np.uint8))
    print(f"Original image: {test_image.size}, mode: {test_image.mode}")
    

    preprocessor = ASLPreprocessor(target_size=224)

    processed_image = preprocessor.preprocess(test_image)
    print(f"Processed image: {processed_image.size}, mode: {processed_image.mode}")
    
    info = preprocessor.get_preprocessing_info(test_image)
    print(f"Preprocessing info: {info}")
    
    print("ASL Preprocessor test completed!")
