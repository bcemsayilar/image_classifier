import os
import shutil
from itertools import compress

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from sklearn.decomposition import PCA
from tqdm import tqdm
import argparse
import logging

from image_utils import calculate_pca, \
    calculate_kmeans, \
    load_embeddings, \
    load_embeddings, \
    create_directory


def clustering(project_name, pca_dim, cluster_range):
    embeddings, image_paths = load_embeddings(f"./embeddings/{project_name}_embeddings.csv")
    pca_embeddings = calculate_pca(embeddings=embeddings, dim=pca_dim)
    centroid, labels = calculate_kmeans(pca_embeddings, k=cluster_range)

    for label_number in tqdm(range(cluster_range)):
        label_mask = labels == label_number

        path_images = list(compress(image_paths, label_mask))
        target_directory = f"clusters/{project_name}/cluster_{label_number}"
        create_directory(target_directory)

        for image_path in path_images:
            shutil.copy2(image_path, target_directory)


def main():
    parser = argparse.ArgumentParser(description="Cluster images with embedding vectors")
    parser.add_argument('--project_name', type=str, required=True, help='Directory containing raw images.')
    parser.add_argument('--pca_dim', type=int, default=16, help='Maximum size for image resizing.')
    parser.add_argument('--cluster_range', type=int, default=4, help='Minimum size for image filtering.')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    clustering(args.project_name, args.pca_dim, args.cluster_range)


if __name__ == "__main__":
    main()
