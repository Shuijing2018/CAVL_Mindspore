import os
import argparse

import mindspore
import numpy as np
from mindspore import context
from mindspore import ops, Tensor, Parameter, amp
# args = parser.parse_known_args()[0]
context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
from mindspore.dataset.vision import Inter
from mindspore import dtype as mstype
from mindspore.nn import LossBase

from mindspore.ops import functional as F

class DatasetGenerator:
    def __init__(self, data, label1, label2):
        self.data = data
        self.label1 = label1
        self.label2 = label2

    def __getitem__(self, index):
        return self.data[index], self.label1[index], self.label2[index], index

    def __len__(self):
        return len(self.data)

def create_dataset(data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    # Define the dataset
    mnist_ds = ds.MnistDataset(data_path)
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # Define the map for the required operations
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # Use map function to apply data operation to the dataset
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_nml_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)

    # Shuffle and batch operation
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)

    return mnist_ds


import mindspore.nn as nn
from mindspore.common.initializer import Normal


class LeNet5(nn.Cell):
    """
    Network structure: Lenet
    """

    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        # Define the operations required
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        # Use defined operations to build the forward network
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class MLP(nn.Cell):
    """
    Network structure: Lenet
    """

    def __init__(self, num_class=10, num_channel=1):
        super(MLP, self).__init__()
        # Define the operations required
        self.fc1 = nn.Dense(32*32, 400)
        self.fc2 = nn.Dense(400, num_class, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def construct(self, x):
        # Use defined operations to build the forward network
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Import the libraries required for model training
from mindspore.nn import Accuracy
from mindspore.train.callback import LossMonitor
from mindspore import Model
from scipy.special import comb

def generate_uniform_cv_candidate_labels(train_labels):
    K = np.max(train_labels) - np.min(train_labels) + 1
    n = train_labels.shape[0]
    cardinality = float((2 ** K - 2))
    number = np.array([comb(K, i + 1) for i in range(K - 1)]).astype(float)
    frequency_dis = number / cardinality
    prob_dis = np.zeros(K - 1)  # tensor of K-1
    for i in range(K - 1):
        if i == 0:
            prob_dis[i] = frequency_dis[i]
        else:
            prob_dis[i] = frequency_dis[i] + prob_dis[i - 1]

    random_n = np.random.uniform(0, 1, n).astype(float) # tensor: n
    mask_n = np.ones(n)  # n is the number of train_data
    partialY = np.zeros((n, K))
    partialY[np.arange(n), train_labels] = 1.0

    temp_num_partial_train_labels = 0  # save temp number of partial train_labels

    for j in range(n):  # for each instance
        for jj in range(K - 1):  # 0 to K-2
            if random_n[j] <= prob_dis[jj] and mask_n[j] == 1:
                temp_num_partial_train_labels = jj + 1  # decide the number of partial train_labels
                mask_n[j] = 0

        temp_num_fp_train_labels = temp_num_partial_train_labels - 1
        candidates = np.random.permutation(K.item()).astype(np.long)  # because K is tensor type
        candidates = candidates[candidates != train_labels[j]]
        temp_fp_train_labels = candidates[:temp_num_fp_train_labels]

        partialY[j, temp_fp_train_labels] = 1.0  # fulfill the partial label matrix
    print("Finish Generating Candidate Label Sets!\n")
    return partialY

def prepare_train_loaders_for_uniform_cv_candidate_labels(ds_train):
    img = []
    label = []
    for data in ds_train.create_dict_iterator():
        img.append(data['image'].asnumpy())
        label.append(data['label'].asnumpy())
    img = np.concatenate(img, axis=0)
    label = np.concatenate(label, axis=0)
    K = np.max(label) + 1  # K is number of classes, full_train_loader is full batch
    partialY = generate_uniform_cv_candidate_labels(label)
    return img, label, partialY


class L1LossForMultiLabel(LossBase):
    """Define multi-label loss function"""
    def __init__(self, reduction="mean"):
        super(L1LossForMultiLabel, self).__init__(reduction)
        self.abs = ops.Abs()
        self.logsoftmax = ops.LogSoftmax(axis=1)

    def construct(self, outputs, confidence, index):
        """There are 3 inputs，the predicted value base，the true value target1 and target2"""
        logsm_outputs = self.logsoftmax(outputs)
        final_outputs = logsm_outputs * confidence[index, :]
        average_loss = - ((final_outputs).sum(axis=1)).mean()
        return average_loss


def test_loop(model, dataset):
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for data, label in dataset.create_tuple_iterator():
        pred = model(data)
        total += len(data)
        correct += (pred.argmax(1) == label).asnumpy().sum()
    correct /= total
    print(f"Test: \n Accuracy: {(100*correct):>0.1f}%\n")

if __name__ == '__main__':
    # Instantiating the network
    net = MLP()

    # Define loss function
    # net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # Define optimizer
    net_opt = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)
    # net_opt = nn.Adam(net.trainable_params(), learning_rate=0.001, weight_decay=0.001)

    from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

    # Set model saving parameters
    config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
    # Apply model saving parameters
    ckpoint = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)

    loss = L1LossForMultiLabel()

    train_epoch = 250
    mnist_path = "./datasets/MNIST_Data"
    dataset_size = 1

    ds_train = create_dataset(os.path.join(mnist_path, "train"), 100, dataset_size)
    ds_test = create_dataset(os.path.join(mnist_path, "test"), 100, dataset_size)
    img, label, partialY = prepare_train_loaders_for_uniform_cv_candidate_labels(ds_train)
    tempY = np.sum(partialY,axis=1)
    tempY = np.repeat(tempY[:, np.newaxis], partialY.shape[1], axis=1)
    confidence = partialY.astype(float)/tempY
    confidence = Tensor(confidence)
    # print(confidence)
    ds_train2 = ds.GeneratorDataset(DatasetGenerator(img, label, partialY),['data','label_true','label','index'])
    ds_train2 = ds_train2.batch(100)
    abs = ops.Abs()
    def train_loop(model, dataset, loss_fn, optimizer, confidence):
        # Define forward function
        def forward_fn(data, label, index):
            logits = model(data)
            loss = loss_fn(logits, label, index)
            return loss, logits

        # Get gradient function
        grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

        # Define function of one-step training
        def train_step(data, confidence, index):
            (loss, output), grads = grad_fn(data, confidence, index)
            loss = ops.depend(loss, optimizer(grads))

            return loss, output

        size = dataset.get_dataset_size()
        model.set_train()
        for batch, (data, label_true, label, index) in enumerate(dataset.create_tuple_iterator()):
            loss, output = train_step(data, confidence, index)
            cav = output * abs(1 - output) * label
            cav_pred = cav.argmax(axis=1)
            gt_label = F.one_hot(cav_pred, output.shape[1], on_value=Tensor(1.0, mindspore.float32),
                                 off_value=Tensor(0.0, mindspore.float32))
            confidence[index, :] = gt_label
            if batch % 100 == 0:
                loss, current = loss.asnumpy(), batch
                print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")
        return confidence

    for t in range(train_epoch):
        print(f"Epoch {t + 1}\n-------------------------------")
        confidence = train_loop(net, ds_train2, loss, net_opt, confidence)
        test_loop(net, ds_test)
    # test_net(net, model, mnist_path)
    print(confidence)