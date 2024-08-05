Synopsis:
---------

In 2023, several storms have happened in the world causing a lot of damage.
Therefore, weather forecasting is very important to prevent the damage. In this project,
we will use the data from the past to predict the weather in the future. The data includes
the satellite images of the 30 storms in the past and the wind speeds of every image.

Problem definition
------------------

This project has two main tasks:

- Task 1: Generate next images.

- Task 2: Predict wind speed.

Also, our models will be tested on an example storm where the last 3 images and 13 wind speeds
need to be predicted.

Generate next 3 images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Several methods can be used to generate the next images. Several approaches exist to 
generate the next images such as looking at the differences between each step. In this project, 
we will use the Convolutional LSTM network to generate the next 3 images. The Convolutional 
LSTM network is a combination of Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM).

Predict wind speed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To predict wind speed, several methods were tried. The best performing models work such that they
they take one image as input and predicts the wind speed of that image. The models are trained on
the images of the 30 storms. The models are tested on an example storm.

Additional sections
~~~~~~~~~~~~~~~~~~~

Installation information can be found in the README.md file.

Function API
============

.. automodule:: forecaster
  :members:
  :imported-members:
