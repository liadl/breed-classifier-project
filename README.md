# Pet Breed Classifier

This project uses Deep Learning to classify dog and cat breeds. It features an end-to-end pipeline that includes model training, a FastAPI backend, and a web-based frontend.

## Features
- **Transfer Learning:** Uses a pre-trained ResNet50 model to achieve high accuracy.
- **Data Augmentation:** Applied techniques like random cropping, flipping, and color jittering to improve model robustness.
- **Database Integration:** Uses SQLAlchemy (SQLite) to log every prediction and store user feedback.
- **Analytics:** The web interface displays a distribution chart of the classified breeds.
- **Containerized:** Includes a `Dockerfile` for easy deployment.

## Preview
[Pet Breed Classifier UI](Screenshot%202026-05-25%20190208.png)

## How to run it

## Data & Model Weights (Important!)

To keep this repository clean and lightweight, the dataset and model weights are not included here. Please download them using the links below:

## Data & Model Weights (Important!)

To keep this repository clean and lightweight, the dataset and model weights are not included here. Please download them using the links below:

- **Oxford-IIIT Pet Dataset:** Download the dataset [here](https://www.kaggle.com/datasets/tanlikesmath/the-oxfordiiit-pet-dataset).

- **Trained Model Weights:** Download the `resnet50_pets_augmented.pth` file [here](https://drive.google.com/file/d/1zX0EeaRop0cG4d7kFAU_j99SSSkti7-n/view?usp=sharing) 
and place it in the project root folder.

### 1. Prerequisites
Make sure you have Python 3.12+ installed.

### 2. Install dependencies
```bash
pip install -r requirements.txt
