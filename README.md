# TinyTensor: Deep Learning Library in Python
# What is TinyTensor?

TinyTensor is an open-source deep learning library built from scratch in Python, inspired by Joel Grus' approach. This project includes 2 examples for creating a Non-Linear Neural Network and training it in order to predict the common XOR and FizzBuzz problems.

## Getting Started
To use this deep learning library locally, you have 2 methods available:

1. Install locally
2. Use the Docker Image

### Method 1: Install Locally

#### 1. Clone the Repository:
```bash
git clone https://github.com/michaelradu/tinytensor.git
```

#### 2. Import the Library files:
```python
# Example of basic imports for training a Neural Network
from tinytensor.train import train
from tinytensor.nn import NeuralNet
from tinytensor.layers import Linear, Tanh
from tinytensor.optim import SGD
```

#### Optional. Export your model:
```python
import pickle

with open('./models/trained_pipeline-0.1.0.pk1', 'wb') as f:
    pickle.dump(net, f)
```

#### 3. Build Awesome Things!

### Method 2: Use the Docker Image

#### 1. Clone the Repository:
```bash
git clone https://github.com/michaelradu/tinytensor.git
```

#### 2. Build Docker Image:
```bash
cd tinytensor
docker build -t tinytensor-app .
```

#### 3. Run Docker Container:
```bash
docker run tinytensor-app
```

##### 2. Import the Library files:
```python
# Example of basic imports for training a Neural Network
from tinytensor.train import train
from tinytensor.nn import NeuralNet
from tinytensor.layers import Linear, Tanh
from tinytensor.optim import SGD
```

#### Optional. Export your model:
```python
import pickle

with open('./models/trained_pipeline-0.1.0.pk1', 'wb') as f:
    pickle.dump(net, f)
```

#### 3. Build Awesome Things!

# Dependencies

- Numpy
- Docker (Optional)

# Project Structure

- **tinytensor:** This directory contains the deep learning library files.
    - **tinytensor/\_init_.py:** Default init file for python libraries.
    - **tinytensor/data.py:** Tools for iterating over data in batches.
    - **tinytensor/layers.py:** Layers for training Neural Networks.
    - **tinytensor/loss.py:** File containing available loss functions.
    - **tinytensor/nn.py:** Neural Network base class for instantiating objects. In essence, a collection of layers.
    - **tinytensor/optim.py:** Optimization functions available for the training process.
    - **tinytensor/tensor.py:** Cheat tensor class implementation.
    - **tinytensor/train.py:** Default Training function for fitting models.
- **models:** This directory contains exported pre-trained models, optional to the actual library.
- **examples:** Directory showcasing example neural networks built with tinytensor.
    - **fizzbuzz.py:** Example tinytensor Neural Network trained on the popular fizzbuzz problem with a default number of 5000 epochs.
    - **xor.py:** Example non-linear tinytensor Neural Network trained on predicting valid xor values. 
    - **xor-export.py:** `xor.py` example with pickle model export included for use within other apps.


# Model Fitting Process

Fitting or training a model is a straightforward process. To achieve this:

1. Import or create a new dataset.
2. Edit your file to incorporate the new dataset.
3. Implement any model pipeline according to your requirements.
4. (Optional) Export your model with pickle.
5. Run your python file to train the model. 
6. Update the version, and a new model, named trained_pipeline-0.1.1.pkl, will be saved in the models directory.

For more code-wise information see the provided examples and play around with them.

# Contributions Welcome

Contributions to this project are highly encouraged! If you have ideas for improvements, bug fixes, or new features, feel free to open an issue or submit a pull request. Let's collaborate to make this library even better and learn new concepts together!

# License

This code is licensed under the GNU General Public License, version 3 (GPL-3.0). See the LICENSE file for more details.

# Acknowledgments

Special thanks to [Joel Grus](https://github.com/joelgrus) and the open-source community for their amazing contributions to Machine Learning and for their copious amounts of educational content, making projects like this possible.

Feel free to explore, experiment, modify, rewrite, and integrate this library into your applications. Happy coding!

# Current Features
1. Tensors
    ---
    - Cheat Numpy Implementation 
2. Loss Functions
    ---
    - MSE (Mean Squared Error), although the current implementation is more similar to Total Squared Erorr.
3. Layers
    ---
    - Linear
    - Tanh Activation
4. Neural Nets
    ---
    - Convolutional
5. Optimizers
    ---
    - SGD (Stochastic Gradient Descent)
6. Data
    ---
    - Batch Iterators
7. Training
    ---
    - Helper Training Function
8. Provided Examples
    ---
    - XOR Example
    - XOR + Model Export Example
    - FizzBuzz Example

