import logging  # detaylı bir print / log yazdırma kütüphanesi
import os  # dizin okuma

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

# loggingin farklı seviyeleri var. örneğin, INFO, DEBUG, ERROR, WARNING
# Bu seviyelerde ekrana print etmeyi ve log dosyalarını bir yere kayıt altına almayı sağlıyor.
# logging settings
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# Dizinlerle çalışacağımız için, önce böyle bir dizin var mı yok mu onu kontrol edebileceğimiz bir fonksiyon yazalım.


def check_is_dir(path):
    if not os.path.isdir(path):
        raise ValueError(f"Provided path {path} is not a directory")
    return True


# Dizin altındaki tüm fotoğrafları tarayıp, istenilen uzantıdakileri toplasın.
def filter_images(list_of_files):
    valid_extension = {".jpeg", ".png", ".jpg", ".webp"}
    return [
        file
        for file in list_of_files
        if any(file.endswith(ext) for ext in valid_extension)
    ]


# Okunan dosyaları bir pilow formatında kayıt altına almamız gerekiyor.
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Image fonksiyonu görseli okuyor.
    return image


def read_images_from_dir(dir_path):
    """
    Bu fonksiyon, check_is_dir, filter_images, load_image fonksiyonları ile okuma işlemi gerçekleştirir.
    :param dir_path:
    :return:
    """
    check_is_dir(dir_path)  # path kontrolü
    files = os.listdir(dir_path)  # path içerisinde yer alan tüm dosyaları listeler
    image_files = filter_images(files)  # dosyaları uzantılarına göre filtreler.
    image_paths = [
        os.path.join(dir_path, file) for file in image_files
    ]  # listdir sadece dosya isimlerini aldığı için burada bir absolute path alıyoruz.

    images = [
        load_image(image_path) for image_path in tqdm(image_paths)
    ]  # tqdm ile total okuma süresi vs, progression görmüş olacağız.
    logging.info(f"Loaded {len(images)} images from {dir_path}")
    return images

def get_images_from_dir(dir_path):
    """
    Bu fonksiyon, check_is_dir, filter_images, load_image fonksiyonları ile okuma işlemi gerçekleştirir.
    :param dir_path:
    :return:
    """
    check_is_dir(dir_path)  # path kontrolü
    files = os.listdir(dir_path)  # path içerisinde yer alan tüm dosyaları listeler
    image_files = filter_images(files)  # dosyaları uzantılarına göre filtreler.
    image_paths = [
        os.path.join(dir_path, file) for file in image_files
    ]  # listdir sadece dosya isimlerini aldığı için burada bir ab solute path alıyoruz.
    return image_paths


def max_resolution_rescale(image, max_width, max_height):
    width, height = image.size
    if width > max_width or height > max_height:
        ratio = min(max_width / width, max_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        image = image.resize((new_width, new_height), Image.LANCZOS)
    return image


def min_resulation_filter(image, min_width, min_height):
    width, height = image.size
    return width >= min_width and height >= min_height


def center_crop(image, new_width, new_height):
    width, height = image.size
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    cropped_image = image.crop((left, top, right, bottom))
    logging.info(f"Center cropped image to {new_width} x {new_height}")
    return cropped_image

def save_image(image, save_path):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if not isinstance(image, Image.Image):
        raise ValueError("Input image must be a numpy array or PIL image")
    if image.mode == "RGB":
        # Bazen imagelar RGBE olabiliyor, ya da background ya da alfası olan CMYK imagelar olabiliyor. Onu kontrol ediyoruz.
        image = image.convert("RGB")

    image.save(save_path)
    logging.info(f"Saved image to {save_path}")


def create_drectory(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        print(f"Directory created: {dir_path}")
    else:
        print(f"Directory already exists {dir_path}")


def save_images_to_dir(images, dir_path):
    create_drectory(dir_path)
    check_is_dir(dir_path)

    for idx, image in tqdm(enumerate(images, 1)):
        save_path = os.path.join(dir_path, f"Image_{idx}.png")
        save_image(image, save_path)
    return True
