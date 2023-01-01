# CAVL(ICLR2022)

This is the MindSpore implementation of CAVL in the following paper.

ICLR 2022: Exploiting Class Activation Value for Partial-Label Learning

# [CAVL Description](#contents)

CAVL (Class Activation Value Learning) is a novel PLL method that selects the true label by the class with the maximum CAV for model training:

1) The class activation map(CAM) , a simple teachnique for discriminating the learning patterns of each class in images , could be utilized to make accurate predictions on selecting the true label from candidate labels.

2) Propose the class activation value(CAV), which owns similar properties of CAM, while CAV is versatile in various types of inputs and models.

# [Dataset](#contents)

Our experiments are conducted on four popular benchmark datasets to test the performance of our CAVL, which are MNIST, Fashion-MNIST, KuzushijiMNIST and CIFAR-10.

Note: This experiment is a demo on MNIST.

# [Environment Requirements](#contents)

Framework

- [MindSpore](https://gitee.com/mindspore/mindspore)

For more information, please check the resources below：

- [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
- [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```bash
python main.py
```
