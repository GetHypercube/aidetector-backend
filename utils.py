import psutil
import os
from PIL import Image, UnidentifiedImageError

def compress_and_resize_image(image_path, max_size=(1024, 1024)):
    """
    Compresses and resizes an image to a manageable size.

    Args:
        image_path (str): Path to the image file.
        max_size (tuple): Maximum width and height of the resized image.

    Returns:
        str: Path to the processed image.
    """
    try:
        # Validate the file format
        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            raise ValueError("Unsupported file format. Accepts only JPEG, PNG, and WebP.")

        # Open and process the image
        with Image.open(image_path) as img:
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                # Resize the image only if it's larger than max_size
                img.thumbnail(max_size, Image.LANCZOS)
            # Save the processed image in a lossless format
            processed_image_path = os.path.splitext(image_path)[0] + "_processed.png"
            img.save(processed_image_path, format='PNG', optimize=True)
            return processed_image_path

    except UnidentifiedImageError as exc:
        # Explicitly re-raising with context from the original exception
        raise ValueError("Invalid image file or path.") from exc

def print_memory_usage():
    """
    Prints the current memory usage of the process.
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory used: {mem_info.rss / (1024 * 1024):.2f} MB")
