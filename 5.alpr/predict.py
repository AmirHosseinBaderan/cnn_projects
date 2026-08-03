import cv2
import torch
import os
import random
import matplotlib

# Check if we have a display available for interactive plotting
if os.environ.get('DISPLAY'):
    matplotlib.use('TkAgg')  # Interactive backend for display
else:
    matplotlib.use('Agg')    # Non-interactive backend for headless

import matplotlib.pyplot as plt
from config import Config

from inference.predictor import Predictor
from inference.decoder import DetectionDecoder
from inference.visualizer import Visualizer

from models.detector import Detector
from trainer.checkpoint import CheckpointManager


def main():

    device = Config.DEVICE

    model = Detector()

    checkpoint = CheckpointManager(
        Config.CHECKPOINT_DIR
    )

    predictor = Predictor(
        model=model,
        checkpoint=checkpoint,
        device=device,
        image_size=Config.IMAGE_SIZE,
    )

    decoder = DetectionDecoder(
        image_size=Config.IMAGE_SIZE,
        stride=Config.STRIDE,
    )

    visualizer = Visualizer()

    # Get 10 random images from test directory
    image_dir = "./data/car_images/test"
    all_images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    # Select up to 10 random images
    image_files = random.sample(all_images, min(10, len(all_images)))
    
    # Create a figure for displaying results
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file)
        
        prediction = predictor.predict(
            image_path,
        )

        detections = decoder.decode(
            prediction,
        )

        image = cv2.imread(
            image_path,
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        result = visualizer.draw(
            image,
            detections,
        )
        
        # Display on the corresponding subplot
        ax = axes[idx]
        ax.imshow(result)
        ax.set_title(f'Image {idx+1}: {image_file}', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "random_predictions.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Results saved to {output_path}")
    
    # Show the figure if we have a display
    if os.environ.get('DISPLAY'):
        plt.show()
    else:
        print("No display available, skipping plt.show()")
    
    plt.close()


if __name__ == "__main__":
    main()