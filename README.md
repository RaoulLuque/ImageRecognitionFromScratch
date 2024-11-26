# Models

## First model
[3f5521c](https://github.com/RaoulLuque/image-recognition-neural-network/tree/3f5521c3a99c06911f46d639afd329db93781204)
- Stochastic gradient descent
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
- Fixed learning rate of 0,1%

## Second model
- Mini batch gradient descent
  - Model layout:
    ```
    model.add_layer(FCLayer(28 * 28, 128))  # input_shape=(1, 28*28)   ;   output_shape=(1, 128)
    model.add_layer(ActivationLayer(ActivationFunction.tanh, 128))
    model.add_layer(FCLayer(128, 10))       # input_shape=(1, 128)     ;   output_shape=(1, 10)
    model.add_layer(ActivationLayer(ActivationFunction.tanh, 10))
    ```
- 6,75% error rate
- 100 epochs
- Fixed learning rate of 0,1%
