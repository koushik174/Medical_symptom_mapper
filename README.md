# Medical_symptom_mapper using Uncertanity Estimation


A deep learning model for classifying medical symptoms and providing uncertainty estimates.

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/medical-symptom-classifier.git
cd medical-symptom-classifier
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate
pip install -r requirements.txt
```

3. Prepare the environment:
```bash
mkdir -p models/saved_models
```

## Training

To train the model:
```bash
python src/train.py
```

## Prediction

To make predictions:
```bash
python src/predict.py --text "Patient presents with fever and cough"
```

## Model Weights

The trained model weights are not included in the repository due to size constraints. You can:
1. Train the model yourself using the provided code
2. Download pre-trained weights from [releases](link-to-releases)

