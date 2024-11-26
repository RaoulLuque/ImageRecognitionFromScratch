# Models

## First model
[3f5521c](https://github.com/RaoulLuque/image-recognition-neural-network/tree/3f5521c3a99c06911f46d639afd329db93781204)
- Stochastic gradient descent (batch size of 32)
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
- Fixed learning rate of 0,1%

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
- 6,75% error rate
- 100 epochs
- Fixed learning rate of 0,1%

## Third model
[73111ee](https://github.com/RaoulLuque/ImageRecognitionFromScratch/tree/73111ee333557ac0d6c4aefa3cfc2a775a0cccdd)
- Mini batch gradient descent (batch size of 32)
- Cross entropy loss function
- Model layout:
  ```
  model.add_layer(FCLayer(28 * 28, 128))  # input_shape=(1, 28*28)   ;   output_shape=(1, 128)
  model.add_layer(ActivationLayer(ActivationFunction.tanh, 128))
  model.add_layer(FCLayer(128, 10))       # input_shape=(1, 128)     ;   output_shape=(1, 10)
  model.add_layer(ActivationLayer(ActivationFunction.softmax, 10))
  ```
- 3,1% error rate
- 100 epochs
- Fixed learning rate of 0,1%
