
# Models
The following is a brief summary of different models that represent different checkpoints in development process.

To run the code or a specific model, please refer to the [running a model](#running-a-model) section.

The logs of the respective models can be found by clicking the links below the respective model to browse the repositories at the respective state and opening the [best_result.log](best_result.log) or [best_result.txt](best_result.log) file (depending on how old the model is).

## First model
[3f5521c](https://github.com/RaoulLuque/image-recognition-neural-network/tree/3f5521c3a99c06911f46d639afd329db93781204)
- Stochastic gradient descent (batch size of 1)
- Mean square error function
- Model layout:
  ```
  model.add_layer(FCLayer(28 * 28, 100))  # input_shape=(1, 28*28)   ;   output_shape=(1, 100)
  model.add_layer(ActivationLayer(ActivationFunction.tanh))
  model.add_layer(FCLayer(100, 50))  # input_shape=(1, 100)          ;   output_shape=(1, 50)
  model.add_layer(ActivationLayer(ActivationFunction.tanh))
  model.add_layer(FCLayer(50, 10))  # input_shape=(1, 50)            ;   output_shape=(1, 10)
  model.add_layer(ActivationLayer(ActivationFunction.tanh))
  ```
- 9-10% error rate
- 100 epochs
- Fixed learning rate of 0.1

## Second model
[9eac97e](https://github.com/RaoulLuque/ImageRecognitionFromScratch/tree/9eac97e44408121367c2a4befaad8b49598b5123)
- Mini batch gradient descent (batch size of 32)
- Mean square error function
- Model layout:
  ```
  model.add_layer(FCLayer(28 * 28, 128))  # input_shape=(1, 28*28)   ;   output_shape=(1, 128)
  model.add_layer(ActivationLayer(ActivationFunction.tanh, 128))
  model.add_layer(FCLayer(128, 10))       # input_shape=(1, 128)     ;   output_shape=(1, 10)
  model.add_layer(ActivationLayer(ActivationFunction.tanh, 10))
  ```
- 6.75% error rate
- 100 epochs
- Fixed learning rate of 0.1

## Third model
[73111ee](https://github.com/RaoulLuque/ImageRecognitionFromScratch/tree/73111ee333557ac0d6c4aefa3cfc2a775a0cccdd)
- Mini batch gradient descent (batch size of 32)
- Cross entropy loss function
- Softmax activation function on last layer
- Model layout:
  ```
  model.add_layer(FCLayer(28 * 28, 128))  # input_shape=(1, 28*28)   ;   output_shape=(1, 128)
  model.add_layer(ActivationLayer(ActivationFunction.tanh, 128))
  model.add_layer(FCLayer(128, 10))       # input_shape=(1, 128)     ;   output_shape=(1, 10)
  model.add_layer(ActivationLayer(ActivationFunction.softmax, 10))
  ```
- 3.1% error rate
- 100 epochs
- Fixed learning rate of 0.1

## Fourth model
[d578b4b](https://github.com/RaoulLuque/ImageRecognitionFromScratch/tree/d578b4b0c7c053d292ae270f1e7d40fed14926c5)
- Mini batch gradient descent (batch size of 32)
- Cross entropy loss function
- Softmax activation function on last layer
- Adam optimizer
- Model layout:
  ```
  model.add_layer(FCLayer(28 * 28, 128, optimizer=Optimizer.Adam))  # input_shape=(1, 28*28)   ;   output_shape=(1, 128)
  model.add_layer(ActivationLayer(ActivationFunction.tanh, 128))
  model.add_layer(FCLayer(128, 10, optimizer=Optimizer.Adam))       # input_shape=(1, 128)     ;   output_shape=(1, 10)
  model.add_layer(ActivationLayer(ActivationFunction.softmax, 10))
  ```
- 2.64% error rate
- 30 epochs
- Fixed learning rate of 0.01

## Fifth model
[1a608e1](https://github.com/RaoulLuque/ImageRecognitionFromScratch/tree/1a608e1aa6394129d516857bde713eeddd258f84)
- Mini batch gradient descent (batch size of 32)
- Cross entropy loss function
- Softmax activation function on last layer
- Adam optimizer
- Dropout layers
- Model layout:
  ```
  model.add_layer(FCLayer(28 * 28, 128, optimizer=Optimizer.Adam))  # input_shape=(1, 28*28)   ;   output_shape=(1, 128)
  model.add_layer(ActivationLayer(ActivationFunction.tanh, 128))
  model.add_layer(DropoutLayer(0.2, 128))
  model.add_layer(FCLayer(128, 10, optimizer=Optimizer.Adam))       # input_shape=(1, 128)     ;   output_shape=(1, 10)
  model.add_layer(ActivationLayer(ActivationFunction.softmax, 10))
  ```
- 2.19% error rate
- 50 epochs
- Fixed learning rate of 0.001

## Sixth model
[251738c](https://github.com/RaoulLuque/ImageRecognitionFromScratch/tree/251738c9ff68e2344f4ee6ded2dfd62f122815c1)
- Mini batch gradient descent (batch size of 16)
- Cross entropy loss function
- Softmax activation function on last layer
- Adam optimizer
- Dropout layers
  - Model layout:
    ```
    model.add_layer(
      FCLayer(28 * 28, 128, optimizer=Optimizer.Adam))             # input_shape=(1, 28*28)    ;   output_shape=(1, 128)
      model.add_layer(ActivationLayer(ActivationFunction.ReLu, 128))
      model.add_layer(DropoutLayer(0.2, 128))
      model.add_layer(FCLayer(128, 50, optimizer=Optimizer.Adam))  # input_shape=(1, 128)      ;   output_shape=(1, 50)
      model.add_layer(ActivationLayer(ActivationFunction.ReLu, 50))
      model.add_layer(DropoutLayer(0.2, 50))
      model.add_layer(FCLayer(50, 10, optimizer=Optimizer.Adam))   # input_shape=(1, 50)       ;   output_shape=(1, 10)
      model.add_layer(ActivationLayer(ActivationFunction.softmax, 10))
    ```
- 2.02% error rate
- 200 epochs
- Fixed learning rate of 0.0005

## Seventh model
[0ae6882](https://github.com/RaoulLuque/ImageRecognitionFromScratch/tree/0ae68824cf0889e8a7dcfc6b965cf504ea153767)
- Mini batch gradient descent (batch size of 16)
- Cross entropy loss function
- Softmax activation function on last layer
- Adam optimizer
- Dropout layers
- (Default) Data augmentation (0.25 Chance to do so)
- Model layout:
    ```
    model.add_layer(
      model.add_layer(FCLayer(28 * 28, 128, optimizer=Optimizer.Adam))  # input_shape=(1, 28*28)    ;   output_shape=(1, 128)
    model.add_layer(ActivationLayer(ActivationFunction.ReLu, 128))
    model.add_layer(DropoutLayer(0.2, 128))

    model.add_layer(FCLayer(128, 50, optimizer=Optimizer.Adam))         # input_shape=(1, 128)      ;   output_shape=(1, 50)
    model.add_layer(ActivationLayer(ActivationFunction.ReLu, 50))
    model.add_layer(DropoutLayer(0.2, 50))

    model.add_layer(FCLayer(50, 10, optimizer=Optimizer.Adam))          # input_shape=(1, 50)       ;   output_shape=(1, 10)
    model.add_layer(ActivationLayer(ActivationFunction.softmax, 10))
    ```
- 1.93% error rate
- 100 epochs
- Fixed learning rate of 0.0005

# Eight model

- Mini batch gradient descent (batch size of 16)
- Cross entropy loss function
- Softmax activation function on last layer
- Adam optimizer
- Dropout layers
- (Default) Data augmentation (0.25 Chance to do so)
- Early stopping (min relative delta 0.005 and patience of 15)
- He weight initialization
- Model layout:
    ```
    model.add_layer(
      model.add_layer(FCLayer(28 * 28, 128, optimizer=Optimizer.Adam))  # input_shape=(1, 28*28)    ;   output_shape=(1, 128)
    model.add_layer(ActivationLayer(ActivationFunction.ReLu, 128))
    model.add_layer(DropoutLayer(0.2, 128))

    model.add_layer(FCLayer(128, 50, optimizer=Optimizer.Adam))         # input_shape=(1, 128)      ;   output_shape=(1, 50)
    model.add_layer(ActivationLayer(ActivationFunction.ReLu, 50))
    model.add_layer(DropoutLayer(0.2, 50))

    model.add_layer(FCLayer(50, 10, optimizer=Optimizer.Adam))          # input_shape=(1, 50)       ;   output_shape=(1, 10)
    model.add_layer(ActivationLayer(ActivationFunction.softmax, 10))
    ```
- 1.58% error rate
- 175 epochs (early stopping after 91)
- Fixed learning rate of 0.0005

# Running a model
To start up the application, one will have to install the dependencies first. [uv](https://github.com/astral-sh/uv) is recommended to be installed. An installation guide can be found [here](https://docs.astral.sh/uv/getting-started/). If [pipx](https://pipx.pypa.io/stable/) is already installed on the machine, it is as easy as
````commandline
pipx install uv
````

After having installed uv, to create a venv and install the necessary dependencies, run:
```commandline
uv python install
uv sync --all-extras --dev
```
The above will install all dependencies. To finish the setup of the python environment, please also run:
```commandline
set -a
source .env
```

Now the project could be run with
```commandline
uv run src/main.py
```
However, the project uses [poethepoet](https://github.com/nat-n/poethepoet) as a task runner. To install poethepoet, run with pipx installed
````commandline
pipx install poethepoet
````

Now the application can be started by running
```commandline
poe run
```

To run a specific model, click on the link provided below the model in this README, and download the source code of that specific commit and proceed as described above.
