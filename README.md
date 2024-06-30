# Classifier with Torch 

This project is designed to classify images of birds, cats, dogs, and horses using convolutional neural networks (CNNs). It includes data preprocessing, feature extraction, and clustering techniques. The project is useful for data scientists and machine learning engineers looking to implement image classification and clustering models.

## Project Structure

- `embeddings/`: Contains CSV files with embeddings for birds, cats, dogs, and horses.
  - `bird_embeddings.csv`
  - `cat_embeddings.csv`
  - `dog_embeddings.csv`
  - `horse_embeddings.csv`

- `processed_images/`: Contains preprocessed images categorized by animals.
  - `bird/`
  - `cat/`
  - `dog/`
  - `horse/`

- `raw_images/`: Contains raw images categorized by animals.
  - `bird/`
  - `cat/`
  - `dog/`
  - `horse/`

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
## Authors

- [@bcemsayilar](https://www.github.com/bcemsayilar)

## 🚀 About Me
I'm a machine learning engineer currently at PEAKUP

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

## License

[MIT](https://choosealicense.com/licenses/mit/)
