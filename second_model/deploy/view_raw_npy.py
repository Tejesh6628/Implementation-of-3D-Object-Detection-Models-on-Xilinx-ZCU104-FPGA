import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="View arbitrary .npy files and generate visualizations.")
    parser.add_argument('--file', type=str, required=True, help="Path to the .npy file")
    parser.add_argument('--out', type=str, default=None, help="Path to save the visualization (optional)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found.")
        sys.exit(1)

    try:
        data = np.load(args.file)
    except Exception as e:
        print(f"Error loading {args.file}: {e}")
        sys.exit(1)

    print(f"=== File Info: {args.file} ===")
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print(f"Min:   {data.min():.4f}")
    print(f"Max:   {data.max():.4f}")
    print(f"Mean:  {data.mean():.4f}")
    print(f"Non-zero elements: {np.count_nonzero(data)} ({(np.count_nonzero(data)/data.size)*100:.2f}%)")

    # Visualization
    plt.figure(figsize=(10, 8))
    
    if data.ndim == 2:
        plt.imshow(data, cmap='viridis')
        plt.colorbar()
        plt.title(f"2D Array: {data.shape}")
        
    elif data.ndim == 3:
        # Check if it's a multi-channel image (e.g., HxWxC or CxHxW)
        if data.shape[0] in [1, 3, 4]:
            print("Assuming CxHxW format for visualization.")
            img = data.transpose(1, 2, 0)
            if img.shape[2] == 1:
                plt.imshow(img[:, :, 0], cmap='gray')
            else:
                # Normalize for RGB display if out of [0, 1]
                if img.max() > 1.0 or img.min() < 0.0:
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                plt.imshow(img)
            plt.title(f"Image format: {data.shape}")
            
        elif data.shape[-1] in [1, 3, 4]:
            print("Assuming HxWxC format for visualization.")
            img = data
            if img.shape[2] == 1:
                plt.imshow(img[:, :, 0], cmap='gray')
            else:
                if img.max() > 1.0 or img.min() < 0.0:
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                plt.imshow(img)
            plt.title(f"Image format: {data.shape}")
            
        else:
            print("3D volume detected. Displaying max projections along each axis.")
            plt.close() # Close the single figure, we need subplots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            p0 = axes[0].imshow(np.max(data, axis=0), cmap='viridis')
            axes[0].set_title("Max Projection (Axis 0)")
            fig.colorbar(p0, ax=axes[0], fraction=0.046, pad=0.04)
            
            p1 = axes[1].imshow(np.max(data, axis=1), cmap='viridis')
            axes[1].set_title("Max Projection (Axis 1)")
            fig.colorbar(p1, ax=axes[1], fraction=0.046, pad=0.04)
            
            p2 = axes[2].imshow(np.max(data, axis=2), cmap='viridis')
            axes[2].set_title("Max Projection (Axis 2)")
            fig.colorbar(p2, ax=axes[2], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
    else:
        print(f"Cannot visualize array with {data.ndim} dimensions.")
        plt.close()
        return

    if args.out:
        plt.savefig(args.out)
        print(f"Saved visualization to {args.out}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
