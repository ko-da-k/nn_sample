#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from chainer import cuda, Variable, Chain, optimizers
import chainer.functions as F
import chainer.links as L


class MLP(Chain):
    """Multi Layer Perceptron"""

    def __init__(self, n_units_1: int, n_units_2: int, n_units_3: int, n_units_4: int):
        """
        4層のパーセプトロン
        :n_units_1: 入力層
        :n_units_2: 中間層
        :n_units_3: 中間層
        :n_units_4: 出力層
        """
        super(MLP, self).__init__(
            l1=L.Linear(n_units_1, n_units_2),
            l2=L.Linear(n_units_2, n_units_3),
            l3=L.Linear(n_units_3, n_units_4),
        )

    def __call__(self, x: Variable, t: Variable):
        self.h1 = F.dropout(F.relu(self.l1(x)))
        self.h2 = F.dropout(F.relu(self.l2(self.h1)))
        y = self.l3(self.h2)
        return F.softmax_cross_entropy(y, t), y

    def predict(self, test_x: np.ndarray):
        test_x = Variable(test_x)
        self.h1 = F.dropout(F.relu(self.l1(test_x)))
        self.h2 = F.dropout(F.relu(self.l2(self.h1)))
        y = self.l3(self.h2)
        predict_list = list(map(np.argmax, F.softmax(y).data))
        return predict_list


class AutoEncoder(Chain):
    """autoencoder"""

    def __init__(self, data_size: int, hidden_size: int):
        super(AutoEncoder, self).__init__(
            l1=L.Linear(data_size, hidden_size),
            l2=L.Linear(hidden_size, data_size),
        )

    def __call__(self, x: Variable, t: Variable):
        h1 = F.dropout(F.relu(self.l1(x)))
        y = self.l2(h1)
        return F.mean_squared_error(y, t), y


def training(model, train: np.ndarray, train_label: np.ndarray, test: np.ndarray, test_label: np.ndarray,
            loop: int, batchsize: int, testbatch: int):
    """
    学習を行う
    :param model: chainerのクラス
    :param train: 学習データ
    :param train_label: 学習データのラベル
    :param test: テストデータ
    :param test_label: テストラベル
    :param loop: epoch回数
    :param batchsize: 一回の学習で使うデータ数
    :param testbatch: テストデータのバッチサイズ
    """

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    datasize = len(train)  # 学習データのサイズ
    if datasize % batchsize != 0:
        print("バッチサイズとデータサイズが適切でない可能性あり")
    train_mean_loss = []
    train_mean_acc = []
    test_mean_loss = []
    test_mean_acc = []

    for epoch in range(0, loop):
        print("epoch: ", epoch)
        index = np.random.permutation(datasize)  # 順番をバラす
        sum_loss = 0
        sum_accuracy = 0
        for i in range(0, datasize, batchsize):
            x = Variable(train[index][i:i + batchsize])
            t = Variable(train_label[index[i:i + batchsize]])
            loss, y = model(x, t)

            accuracy = F.accuracy(y, t)

            model.zerograds()
            loss.backward()
            optimizer.update()
            # optimizer.update(model, x, t)

            sum_loss += float(cuda.to_cpu(loss.data)) * len(x.data)
            sum_accuracy += float(cuda.to_cpu(accuracy.data)) * len(x.data)
            # sum_accuracy += float(cuda.to_cpu(model.accuracy.data)) * len(x.data)

        print("train mean loss : ", sum_loss / datasize)
        print("train accuracy : ", sum_accuracy / datasize)
        train_mean_loss.append(sum_loss / datasize)
        train_mean_acc.append(sum_accuracy / datasize)

        test_sum_loss = 0
        test_sum_accuracy = 0
        testsize = len(test)
        for i in range(0, testsize, testbatch):
            test_x = Variable(test[i:i + testbatch])
            test_t = Variable(test_label[i:i + testbatch])
            test_loss, test_y = model(test_x, test_t)

            test_accuracy = F.accuracy(test_y, test_t)
            test_sum_loss += float(cuda.to_cpu(test_loss.data)) * len(test_x.data)
            test_sum_accuracy += float(cuda.to_cpu(test_accuracy.data)) * len(test_x.data)
            # test_sum_accuracy += float(cuda.to_cpu(model.accuracy.data)) * testbatch
        print("test mean loss : ", test_sum_loss / testsize)
        print("test accuracy : ", test_sum_accuracy / testsize)
        test_mean_loss.append(test_sum_loss / testsize)
        test_mean_acc.append(test_sum_accuracy / testsize)

    plt.plot(train_mean_loss, "b")
    plt.plot(test_mean_loss, "r")
    plt.title("loss")
    plt.show()

    plt.plot(train_mean_acc, "b")
    plt.plot(test_mean_acc, "r")
    plt.title("accuracy")
    plt.ylim(0, 1)
    plt.show()

    return model


def ae_training(model, data: np.ndarray, loop: int, batchsize: int):
    """
    AutoEncoderのトレーニング
    :param model: モデル
    :param data: データ
    :param loop: 学習回数
    :param batchsize: バッチサイズ
    :return: 学習後のmodel
    """

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    datasize = len(data)  # 学習データのサイズ
    if datasize % batchsize != 0:
        print("バッチサイズとデータサイズが適切でない可能性あり")
    train_mean_loss = []

    for epoch in range(0, loop):
        print("epoch: ", epoch)
        index = np.random.permutation(datasize)  # 順番をバラす
        sum_loss = 0
        for i in range(0, datasize, batchsize):
            x = Variable(data[index][i:i + batchsize])
            t = Variable(data[index][i:i + batchsize])
            loss, y = model(x, t)

            model.zerograds()
            loss.backward()
            optimizer.update()
            # optimizer.update(model, x, t)

            sum_loss += float(cuda.to_cpu(loss.data)) * len(x.data)  # batchsizeにすると、最後がずれる可能性あり

        print("train mean loss : ", sum_loss / datasize)
        train_mean_loss.append(sum_loss / datasize)

    plt.plot(train_mean_loss, "b")
    plt.title("loss")
    plt.show()

    return model


def draw_digit(data):
    size = 28
    plt.figure(figsize=(2.5, 3))

    X, Y = np.meshgrid(range(size), range(size))
    Z = data.reshape(size, size)  # convert from vector to 28x28 matrix
    Z = Z[::-1, :]  # flip vertical
    plt.xlim(0, 27)
    plt.ylim(0, 27)
    plt.pcolor(X, Y, Z)
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")

    plt.show()


if __name__ == '__main__':
    print('fetch MNIST dataset')
    mnist = fetch_mldata('MNIST original', data_home=".")  # mnistは70000件のデータ,784次元
    mnist.data = mnist.data = mnist.data.astype(np.float32)
    mnist.data /= 255
    mnist.target = mnist.target.astype(np.int32)

    # draw_digit(mnist.data[5])

    train, test, train_label, test_label = train_test_split(mnist.data, mnist.target, test_size=0.2, random_state=0)

    n_units_1 = len(train[0])  # 次元数
    n_units_2 = 300  # 中間層
    n_units_3 = 100
    n_units_4 = len(list(set(train_label)))

    # model = L.Classifier(MLP(n_units_1, n_units_2, n_units_3))  # 損失値を返すClassifierクラス
    model = MLP(n_units_1, n_units_2, n_units_3, n_units_4)
    model = training(model, train, train_label, test, test_label, 100, 50, 50)

    ae = AutoEncoder(n_units_1, 500)
    ae = ae_training(ae, mnist.data, 100, 50)
