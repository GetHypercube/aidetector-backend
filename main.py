import argparse
from dmdetector import process_image as dm_process_image
from gandetector import process_image as gan_process_image

def main():
    parser = argparse.ArgumentParser(description="Run both DM and GAN detection on an image.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for more verbose output")
    args = parser.parse_args()

    # Run DM Detector
    dm_results = dm_process_image(args.image_path, debug=args.debug)
    print("DM Detector Results:")
    print(dm_results)

    # Run GAN Detector
    gan_results = gan_process_image(args.image_path, debug=args.debug)
    print("GAN Detector Results:")
    print(gan_results)

if __name__ == "__main__":
    main()
