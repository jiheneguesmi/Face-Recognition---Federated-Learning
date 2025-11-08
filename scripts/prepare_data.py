"""
Data preprocessing script for face images
Detects and aligns faces in images
"""

import argparse
import os
from pathlib import Path
from facenet_pytorch import MTCNN
from PIL import Image
import torch
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_and_align_faces(input_dir: str, output_dir: str, member_name: str):
    """
    Detect and align faces in images
    
    Args:
        input_dir: Directory containing raw images
        member_name: Name of the member
        output_dir: Directory to save processed images
    """
    # Initialize MTCNN for face detection
    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20)
    
    # Create output directory
    input_path = Path(input_dir) / member_name
    output_path = Path(output_dir) / member_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_path.glob(f'*{ext}')))
    
    if len(image_files) == 0:
        logger.warning(f"No images found in {input_path}")
        return
    
    logger.info(f"Processing {len(image_files)} images for {member_name}")
    
    processed_count = 0
    failed_count = 0
    
    for img_path in image_files:
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            
            # Detect and align face
            face = mtcnn(img, save_path=None)
            
            if face is not None:
                # Convert tensor to numpy array
                if isinstance(face, torch.Tensor):
                    face_np = face.permute(1, 2, 0).numpy()
                    face_np = ((face_np + 1) / 2 * 255).astype(np.uint8)
                    face_img = Image.fromarray(face_np)
                else:
                    face_img = Image.fromarray(face)
                
                # Save processed image
                output_file = output_path / img_path.name
                face_img.save(output_file)
                processed_count += 1
            else:
                logger.warning(f"No face detected in {img_path}")
                failed_count += 1
        
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            failed_count += 1
    
    logger.info(f"Processed {processed_count} images, {failed_count} failed for {member_name}")


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='Prepare face data')
    parser.add_argument('--member', type=str, required=True, help='Member name (e.g., omarmej, abir, omarbr, jihene)')
    parser.add_argument('--input-dir', type=str, default='data/raw', help='Input directory')
    parser.add_argument('--output-dir', type=str, default='data/processed', help='Output directory')
    
    args = parser.parse_args()
    
    detect_and_align_faces(args.input_dir, args.output_dir, args.member)
    logger.info("Data preparation completed!")


if __name__ == "__main__":
    main()

