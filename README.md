# Visual Classifier with Convolutional Neural Networks 

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
- `get_embeddings.py`: Script for generating embeddings from images.
- `image_utils.py`: Utility functions for image processing.
- `preprocess.ipynb`: Jupyter notebook for data preprocessing.
- `smartcrop.py`: Script for cropping images smartly.

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
