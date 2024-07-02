import pandas as pd
from img2vec_pytorch import Img2Vec
from PIL import Image
import argparse
import logging
from image_utils import create_directory

import image_utils
img2vec = Img2Vec()


def get_embeddings(processed_image_dir:str, dir_name:str):
    paths = image_utils.get_images_from_dir(processed_image_dir)
    # Sadece pathleri tutacağız. Çünkü embeddinglerin .csv dosyasını oluşturduğuğmuzda bir kolonda
    # filepathleri, diğer kolonda vektörleri tutacağız.
    images = [image_utils.load_image(path) for path in paths]

    embeddings = img2vec.get_vec(images)
    logging.info(f"Shape of My Embeddings {embeddings.shape} ~ Sting")

    # Bu bize numpy arrayi ya da tensorflow tensoru olarak dönecek.
    df = pd.DataFrame(embeddings)
    df["filepaths"] = paths
    create_directory("./embeddings_dog_muffin")
    df.to_csv(f"./embeddings_dog_muffin/{dir_name}_embeddings.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Embedding processed images to vectors")
    parser.add_argument('--processed_image_dir', type=str, required=True, help='Directory containing processed images.')
    parser.add_argument('--dir_name', type=str, required=True, help='Name of the directory or image you want to embed')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    get_embeddings(args.processed_image_dir, args.dir_name)


if __name__ == "__main__":
    main()

