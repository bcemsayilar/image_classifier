# Classifier with Torch

This project is designed to classify images using convolutional neural networks (CNNs), with PyTorch.  It also includes simple and effective preprocessing, generic clustering with PCA and kmeans, embedding with Img2Vec and splitting functions. The project is useful for data scientists and machine learning engineers looking to implement image classification and clustering models and deploy it for test to gradio with ease.
Huge thanks to @cobanov for [Superpeer course](https://superpeer.com/cobanov/collection/kendi-siniflandiricinizi-egitin)

## Project Structure

- `embeddings/`: Contains CSV files with embeddings for muffins and chihuahuas or birds, cats, dogs, and horses.
  - `dog_embeddings.csv`
  - `muffin_embeddings.csv`

- `processed_images/`: Contains preprocessed resized, smart cropped images.
  - `chihuahua/`
  - `dog/`

- `raw_images/`: Contains raw images.
  - `chihuahua/`
  - `dog/`


- `clustering.py`: Script for clustering images based on their embeddings.
- `data_model.py`: Script for defining the data model and handling data.
- `get_embeddings.py`: Script for generating embeddings from images.
- `gradio_infer.py`: Script for creating a web interface using Gradio for inference.
- `image_utils.py`: Utility functions for image processing.
- `inference.py`: Script for performing inference on images.
- `model.py`: Script for defining and managing the PyTorch model.
- `preprocess.ipynb`: Jupyter notebook for data preprocessing.
- `smartcrop.py`: Script for cropping images smartly.
- `splitter.py`: Script for splitting the dataset into training and test sets.
- `train.py`: Script for training the image classification model.
- `visualisation.py`: Script for visualizing results and performance metrics.

## Run Locally

Clone the project

```bash
  git clone https://https://github.com/bcemsayilar/image_classifier
```

Go to the project directory

```bash
  cd my-project
```
Create virtual enviroment
```bash
  python -m venv venv
```

Activate the virtual environment

On windows

```bash
  .\venv\Scripts\activate
```

On linux/macOS

```bash
  source venv/bin/activate
```

Install dependencies

```bash
  pip install -r requirements.txt
```


Getting Dataset with Kaggle API
-
[Kaggle API doc](https://www.kaggle.com/docs/api)
```bash
 pip install -q kaggle
 mkdir ~/.kaggle
 cp kaggle.json ~/.kaggle/
 chmod 600 ~/.kaggle/kaggle.json
```

```bash
kaggle datasets download -d 'muffin-vs-chihuahua-image-classification'
```

Preprocess images
-
```bash
  python preprocess.py --raw_dir ./dog_muffin --processed_dir ./processed_dog_muffin
```

Get embeddings
-
```bash
  python get_embeddings --processed_images_dir ./processed_dog_muffin/dog --dir_name ./dog
  python get_embeddings --processed_images_dir ./processed_dog_muffin/muffin --dir_name ./muffin
```

Clustering for removal
-
```bash
  python clustering.py --embedding_dir ./embeddings_dog_muffin --csv_name dog
  python clustering.py --embedding_dir ./embeddings_dog_muffin --csv_name muffin
```

Split dataset into train and test
-
```bash
  python splitter.py --dataset_path ./dog_muffin
```

Start training
-
```bash
 python train.py --model_dataset_name dogmuffin_model_dataset --dir_name dogmuffin
```


## Authors

- [@bcemsayilar](https://www.github.com/bcemsayilar)

## 🚀 About Me
I'm a machine learning engineer currently at PEAKUP
