'''
Author: AzarAI
Email: 3420396703@qq.com
Date: 2023-03-16 22:12:43
LastEditTime: 2023-03-21 22:27:04
'''
import numpy as np
import pandas as pd
from scipy.special import expit as S

class NeuralNetwork:
# This neuralnetwork is initially designed as a input-hidden-output model
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate) -> None:
        # Ready the network configurations
        self.lr = learningrate
        self.inputnodes = inputnodes
        self.hiddennodes = hiddennodes
        self.outputnodes = outputnodes

        self.w_ih = np.random.normal(0.0, pow(self.hiddennodes, -0.5), (self.hiddennodes, self.inputnodes))
        self.w_ho = np.random.normal(0.0, pow(self.hiddennodes, -0.5), (self.outputnodes, self.hiddennodes))

        self.activation_func = S

    def train_single(self, input_list, target_list):
        # Train the neural-network with given data
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        # print('inputs_shape:', inputs.shape, '\ttargets_shape:', targets.shape)
        # print('w_ih_shape:', self.w_ih.shape, '\tw_ho_shape:', self.w_ho.shape)

        hidden_inputs = np.dot(self.w_ih, input_list)
        hidden_outputs = self.activation_func(hidden_inputs).reshape((100, 1))

        # print("hidden_inputs_shape:", hidden_inputs.shape, "\thidden_outputs_shape:", hidden_outputs.shape)

        final_inputs = np.dot(self.w_ho, hidden_outputs)
        final_outputs = self.activation_func(final_inputs).reshape((10, 1))

        # print("final_inputs_shape:", final_inputs.shape, "\tfinal_outputs_shape:", final_outputs.shape)
        # print("\t\tfinal_outputs:", final_outputs)

        # Get the error data
        error_outputs = targets-final_outputs
        error_hidden = np.dot(self.w_ho.T, error_outputs)

        # print("error_outputs_shape:", error_outputs.shape, "\terror_hidden_shape:", error_hidden.shape)

        # Modify the weight bettween neuros
        self.w_ho += self.lr * np.dot((error_outputs * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.w_ih += self.lr * np.dot((error_hidden * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

    def train(self, input_lists, target_lists, cycle=1, show_status:bool = False):
        for x in range(cycle):
            for i in range(len(target_lists)):
                self.train_single(data_train_value[i], data_train_target[i])
            print("\tTraining Prcessing: (cycle: ", x+1, "/", cycle, ")")


    def query(self, input_list):
        inputs = np.array(input_list, ndmin=2).T

        inputs_hidden = np.dot(self.w_ih, inputs)
        outputs_hidden = self.activation_func(inputs_hidden)

        inputs_final = np.dot(self.w_ho, outputs_hidden)
        outputs_final = self.activation_func(inputs_final)

        return outputs_final

def get_train_data():
    # Version 1.0
    # with open('./data/mnist_train.csv', 'r') as file:
    #     data = file.readlines()

    # train_data = []
    # target_data = []
    # for image in data:
    #     d = image.split(',')
    #     train = np.asfarray(d[1:]) / 255.0 * 0.99 + 0.01
    #     train_data.append(train)

    #     target = np.zeros(output_nodes) + 0.01
    #     target[int(d[0])] = 0.99
    #     target_data.append(target)

    # The Primary method of getting csv files is too ugly , I'd like to use pandas
    file_train = pd.read_csv('./data/mnist_train.csv', header=None, dtype='float').values

    # Normalize the data into the interval (0, 1)
    data_value = np.asfarray(file_train[:, 1:]) / 255.0 * 0.99 +0.01

    data_target_pre = np.asfarray(file_train[:, 0])
    data_target = []
    for i in range(len(data_target_pre)):
        a = np.zeros(output_nodes) + 0.01
        a[int(data_target_pre[i])] = 0.99
        data_target.append(a)

    return data_target, data_value

def get_test_data():
    # File input method is same as done above.

    file_test = pd.read_csv('./data/mnist_test.csv', header=None, dtype='float').values
    # Normalize the data into the interval (0, 1)
    data_value = np.asfarray(file_test[:, 1:]) / 255.0 * 0.99 +0.01
    data_target = np.asfarray(file_test[:, 0])

    return data_target, data_value


if __name__ == "__main__":

    # Configurate the date of network
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.3

    network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # Get the MNIST data
    data_train_target, data_train_value = get_train_data()
    data_test_target, data_test_value = get_test_data()


    # Data of model-training
    CYCLE = 20

    # Train the model
    print("Start training:")
    network.train(data_train_value, data_train_target, 20, show_status=True)
    print("\tTraining is finisheed.")

    # Test the model
    print("Start testing:")
    correct = 0
    whole = len(data_test_target)
    for i in range(whole):
        answer_list = network.query(data_test_value[i])
        answer = int(answer_list.argmax())
        if data_test_target[i] == answer:
            correct += 1
    print('\tTesting is finished.')

    print("The model accuracy rate is ", correct / whole * 100, '%')
