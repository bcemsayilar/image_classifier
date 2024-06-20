import pandas as pd
from img2vec_pytorch import Img2Vec
from PIL import Image

import image_utils
img2vec = Img2Vec()
animal = "horse"

paths = image_utils.get_images_from_dir(f'./processed_images/{animal}')
# Sadece pathleri tutacağız. Çünkü embeddinglerin .csv dosyasını oluşturduğuğmuzda bir kolonda
# filepathleri, diğer kolonda vektörleri tutacağız.
images = [image_utils.load_image(path) for path in paths]


embeddings = img2vec.get_vec(images)

print(embeddings.shape)
# Bu bize numpy arrayi ya da tensorflow tensoru olarak dönecek.
df = pd.DataFrame(embeddings)
df["filepaths"] = paths
df.to_csv(f"./embeddings/{animal}_embeddings.csv", index=False)