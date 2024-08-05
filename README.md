# The Day After Tomorrow

## Predicting tropical storm behaviour through Deep Learning

### Documentation
---
The API for the helper function can be found on the following website:
https://ese-msc-2023.github.io/acds-the-day-after-tomorrow-lilian/

### Code
---
Jupyter notebooks:
- `wind_speed_clean.ipynb`: 
    - Includes EDA and wind speed prediction models. Two main methods are used. 
    - Method 1: Convolutional model that directly predicts the wind speeds from the images. This method is 
    presented in this notebook.
    - Method 2: LSTM model that uses only the labels from the previous times. This methods is
    packaged and can be callable in our notebooks.
- `generate_images_clean.ipynb`: 
    - Includes image generation model development process.
    - The best model, which utilizes the difference matrix implemented with convolutional LSTM, is presented.
    - The model is able to get turning motion in our generated images with high resolution. Only drawback is that they
    are brighter than normal images.
    - At the end, results from different trials such as classical convolutional LSTM and non-machine learning
    methods are presented.
- `surprise_storm.ipynb`: 
    - Includes surprise storm prediction processes.
    - Use Method 1 to predict 10 wind speeds, Method 2 to predict the next 3 images. This is because when
    we predict the wind speeds from the predicted images, we got very high speeds due to brighteness.
    - For future work, denoising can be applied to our image generation model to improve this problem.

The source code is in the `forecaster` package. Most of the functions are called 
from this package and used in the jupyter notebooks.

Forecaster package main content:
- `__init__.py`
- `read_data.py`: Reading tabular data.
- `data_preparation.py`: Creating custom dataset.
- `eda.py`: EDA functions.
- `wind_speed_lstm.py`: Feature based LSTM.
- `differencing_matrix.py`

The predictions for the surprise storm can be found in the following files:
1. Wind speed predictions: `lilian_windpredictions.csv`
2. Generated images: `lilian_generatedimages/`

### Tests
---
Forecaster package is automatically tested. To run the pytest test suite,
from the base directory run:
```bash
pytest tests/
```

### Installation
---
To reproduce the results, follow the below steps:
1. Clone the repository. 
```bash
git clone https://github.com/ese-msc-2023/acds-the-day-after-tomorrow-lilian.git
```
2. Install the following requirements:
```bash
pip install -r requirements.txt
```
3. Install the package forecaster:
```bash
pip install .
```
4. Get the training data from the following link: https://drive.google.com/drive/folders/1tFqoQl-sdK6qTY1vNWIoAdudEsG6Lirj?usp=drive_link
5. Run the desired notebook.
