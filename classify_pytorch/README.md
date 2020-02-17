Download the MNIST dataset from [Baidu Drive](https://pan.baidu.com/s/1X-FB-SKUvVvWkXdo_b8SHA), the password is mu8h.
## Train
As an example, use the following command to train a CNN on Mnist

    python main.py --datapath (mnist training data folder)
    　　　　　　　　--batch_size 256
    　　　　　　　　--epochs 10
    　　　　　　　　--use_cuda False

## Test
Run `test.py`  it can predict the category of the sample image
