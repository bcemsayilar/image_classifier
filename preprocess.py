import os  # dizin okuma
import logging  # detaylı bir print / log yazdırma kütüphanesi
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import argparse

from image_utils import check_is_dir, \
    filter_images, \
    load_image, \
    read_images_from_dir, \
    max_resolution_rescale, \
    min_resulation_filter, \
    center_crop, \
    save_image, \
    save_images_to_dir

from smartcrop import detect


def process_images(raw_dir, processed_dir, max_size=1024, min_size=224, use_center_crop=False):
    logging.info("Reading images from directory.")
    loaded_images = read_images_from_dir(raw_dir)

    logging.info("Resizing images.")
    resized_images = list(map(lambda x: max_resolution_rescale(x, max_size, max_size), tqdm(loaded_images)))

    logging.info("Filtering images.")
    filtered_images = list(filter(lambda x: min_resulation_filter(x, min_size, min_size), tqdm(loaded_images)))

    logging.info(f"Length of filtered images: {len(filtered_images)}")

    if use_center_crop:
        logging.info("Center cropping images.")
        cropped_images = list(map(lambda x: center_crop(x, min_size, min_size), tqdm(filtered_images)))
    else:
        logging.info("Smart cropping images.")
        cropped_images = list(map(lambda x: detect(x, square=True, crop=True), tqdm(filtered_images)))

    logging.info("Saving images to directory.")
    save_images_to_dir(cropped_images, processed_dir)


def main():
    parser = argparse.ArgumentParser(description="Process images by resizing, filtering, and cropping.")
    parser.add_argument('--raw_dir', type=str, required=True, help='Directory containing raw images.')
    parser.add_argument('--processed_dir', type=str, required=True, help='Directory to save processed images.')
    parser.add_argument('--max_size', type=int, default=1024, help='Maximum size for image resizing.')
    parser.add_argument('--min_size', type=int, default=224, help='Minimum size for image filtering.')
    parser.add_argument('--use_center_crop', action='store_true', help='Use center crop instead of smart crop.')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    process_images(args.raw_dir, args.processed_dir, args.max_size, args.min_size, args.use_center_crop)


if __name__ == "__main__":
    main()
