LeNet-5 from scratch. Originally developed by LeCun et al. (1998).

Interesting facts about [LeNet](https://en.wikipedia.org/wiki/LeNet):

- Progenitor of modern convolutional neural networks.
- Early development started in 1988 by Yann LeCun et al.
- First successful application of backpropagation in neural networks.
- Widespread practical applications of LeNet-5 began in 1998.
- Recognised for recognising handwritten digits, especially U.S. postal ZIP codes.

Results:

- 0.94 test accuracy trained with [MNIST dataset from Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) (59400 training examples, 600 test examples).
- 10 epochs with batch size of 64 and scheduled learning rate decay (0.001 for first 5 epochs, 0.0001 for last 5 epochs).

Program Files:
| File | Purpose |
| :---: | :--- |
| main.py | Runs main script for LeNet-5. |
| model.py | Contains functions for LeNet-5. |
| unpack.py | Unpacks MNIST dataset for LeNet-5. |
| utils.py | Contains utility functions for LeNet-5. |
| pytorch.py | LeNet-5 implementation with PyTorch. |
| params.npz | Saved parameters for LeNet-5 (Current). |
| params.pth | Saved parameters for LeNet-5 (PyTorch). |

![image](https://github.com/obdwinston/Convolutional-Network/assets/104728656/e2cff080-f76f-4e59-b86b-63d184a46803)
