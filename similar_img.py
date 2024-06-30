import gradio as gr
import pandas as pd
import numpy as np
from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity




# Img2Vec oluşturun
img2vec = Img2Vec(cuda=False)  # CUDA'yı kapalı olarak ayarladık


def find_most_similar(image, label):
    """

    :param image: input image
    :return: most similar image according to cosine similarity, in a given path
    """
    embeddings_df = pd.read_csv(f"./embeddings/{label}_embeddings.csv")
    embeddings = embeddings_df.iloc[:, :-1].values
    filepaths = embeddings_df["filepaths"].values

    # Yüklenen görselin embedding'ini hesaplayın
    img_embedding = img2vec.get_vec(image, tensor=False)

    # Cosine similarity hesaplayın
    similarities = cosine_similarity([img_embedding], embeddings)

    # En yakın embedding'i bulun
    most_similar_idx = np.argmax(similarities)
    most_similar_filepath = filepaths[most_similar_idx]

    # En yakın görseli döndürün
    most_similar_image = Image.open(most_similar_filepath)
    return most_similar_image


# Gradio
gr.Interface(
    fn=find_most_similar,
    inputs=[gr.Image(type="pil"),
            gr.Dropdown(choices=["bird", "horse", "cat", "dog"], label="Select a Label")],
    outputs=gr.Image(type="pil"),
    title="Find Most Similar Image",
    description="Upload an image and find the most similar image in the dataset."
).launch()


