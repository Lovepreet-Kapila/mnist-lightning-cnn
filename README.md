# MNIST Classification with PyTorch Lightning 

This project trains a Convolutional Neural Network (CNN) on the MNIST dataset using [PyTorch Lightning](https://www.pytorchlightning.ai/). It demonstrates a clean, modular deep learning workflow with reproducible training, validation, and testing.

Highlights
- Achieved **98.7% test accuracy** on handwritten digit classification
- Built with **PyTorch Lightning** for scalable and readable training loops
- Includes **EarlyStopping**, metric logging, and modular architecture
- Uses **torchmetrics** for clean accuracy tracking

## Model Architecture
- `Conv2d(1 → 32)` → ReLU → MaxPool  
- `Conv2d(32 → 64)` → ReLU → MaxPool  
- Flatten → `Linear(3136 → 128)` → ReLU  
- `Linear(128 → 10)` → Softmax (via CrossEntropyLoss)

## Results
| Metric       | Value     |
|--------------|-----------|
| Train Acc    | 100%      |
| Val Acc      | 98.6%     |
| Test Acc     | 98.7%     |
| Test Loss    | 0.0449    |
pip install -r requirements.txt
python src/mnist_lightning.py
