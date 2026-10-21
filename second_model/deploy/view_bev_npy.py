import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def view_npy(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return

    data = np.load(filepath)
    print(f"Loaded {os.path.basename(filepath)}")
    print(f"Original shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Min value: {data.min():.4f}, Max value: {data.max():.4f}")
    
    # Squeeze out batch dimension if present (e.g. [1, C, H, W] -> [C, H, W])
    if data.ndim == 4 and data.shape[0] == 1:
        data = data[0]
        print(f"Squeezed batch dimension. New shape: {data.shape}")

    plt.figure(figsize=(10, 8))

    # Complex-YOLOv4 BEV features typically have 3 channels (RGB) [3, H, W]
    # SECOND BEV features typically have many channels (e.g. 256) [C, H, W]
    if data.ndim == 3:
        if data.shape[0] == 3:
            print("Detected 3-channel (CHW) BEV map. Converting to RGB image.")
            # Convert CHW to HWC for matplotlib
            img = np.transpose(data, (1, 2, 0))
            
            # Normalize to [0, 1] for visualization
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
            
            plt.imshow(img)
            plt.title(f"RGB BEV Image: {os.path.basename(filepath)}")
            
        elif data.shape[-1] == 3:
            print("Detected 3-channel (HWC) BEV map. Converting to RGB image.")
            img = data
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
                
            plt.imshow(img)
            plt.title(f"RGB BEV Image: {os.path.basename(filepath)}")
            
        else:
            print(f"Data has {data.shape[0]} channels. Plotting maximum intensity projection.")
            # For 256-channel features (like SECOND), a max projection lets you see all the activations
            if data.shape[0] > data.shape[-1]: # CHW format
                img = np.max(data, axis=0)
            else: # HWC format
                img = np.max(data, axis=-1)
                
            plt.imshow(img, cmap='viridis')
            plt.title(f"Max Projection BEV: {os.path.basename(filepath)}")
            plt.colorbar(label='Intensity')
            
    elif data.ndim == 2:
        print("Detected 2D BEV map.")
        plt.imshow(data, cmap='viridis')
        plt.title(f"2D BEV: {os.path.basename(filepath)}")
        plt.colorbar(label='Intensity')
    else:
        print("Cannot automatically visualize this shape directly.")
        return
        
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize BEV .npy feature maps")
    parser.add_argument('--npy', type=str, required=True, help='Path to the .npy file')
    args = parser.parse_args()
    
    view_npy(args.npy)
