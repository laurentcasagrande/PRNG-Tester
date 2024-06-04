from abc import ABC, abstractmethod
import numpy as np
import random
from sklearn.model_selection import train_test_split
from keras import models, layers
from keras.utils import to_categorical
from rich.console import Console
import pickle
import os


class PRNG_tester():
    def __init__(self, seed: int, num_of_datapoints: int, length_of_data: int, console: Console, folder_name):
        self.seed = seed
        self.num_of_datapoints = num_of_datapoints
        self.length_of_data = length_of_data
        self.folder_name = folder_name
        self.console=console
        self.test_size = 0.2
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_lables = None
        self.network = None

    @abstractmethod
    def prng(self):
        raise NotImplementedError

    def save_data_to_folder(self):
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)
        with open(f'{self.folder_name}/train_data','wb') as f: pickle.dump(self.train_data, f)
        with open(f'{self.folder_name}/train_labels','wb') as f: pickle.dump(self.train_labels, f)
        with open(f'{self.folder_name}/test_data','wb') as f: pickle.dump(self.test_data, f)
        with open(f'{self.folder_name}/test_labels','wb') as f: pickle.dump(self.test_labels, f)

    def load_data(self):
        with self.console.status("[bold green]loading data...") as status:
            with open(f'{self.folder_name}/train_data', 'rb') as f:
                self.train_data = pickle.load(f)
            with open(f'{self.folder_name}/train_labels', 'rb') as f:
                self.train_labels = pickle.load(f)
            with open(f'{self.folder_name}/test_data', 'rb') as f:
                self.test_data = pickle.load(f)
            with open(f'{self.folder_name}/test_labels', 'rb') as f:
                self.test_labels = pickle.load(f)

    def create_data(self):
        with self.console.status("[bold green]creating data...") as status:
            np_data = np.empty(
                self.num_of_datapoints*self.length_of_data, dtype=int)
            np_labels = np.empty(self.num_of_datapoints, dtype=int)

            for i in range(self.num_of_datapoints):
                for j in range(self.length_of_data):
                    num = self.prng()
                    np_data[i*self.num_of_datapoints + j] = num

                num2 = self.prng()

                np_labels[i] = num2

            np_labels = to_categorical(np_labels)

            np_data_reshaped = np.array(np_data).reshape(
                self.num_of_datapoints, self.length_of_data)
            
            self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(np_data_reshaped, np_labels, test_size=self.test_size)
            self.save_data_to_folder()

    def train(self):
        """
        """

        if self.train_data is None or self.train_labels is None:
            self.load_data()
        
        # Creating the network using Sequential layers
        self.network = models.Sequential()

        # Adding layers
        self.network.add(layers.Dense(512, activation='relu',
                    input_shape=(self.length_of_data,)))
        self.network.add(layers.Dense(256, activation='relu'))
        self.network.add(layers.Dense(50, activation='relu'))
        self.network.add(layers.Dense(100, activation='relu'))
        self.network.add(layers.Dense(50, activation='relu'))
        self.network.add(layers.Dense(10, activation='softmax'))

        # Selecting a loss function, optimizer and metrics to monitor during training and testing
        self.network.compile(optimizer='rmsprop',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

        # Fit the model to its training data
        #         Epochs = 5
        #         Batch Size = 128
        self.network.fit(self.train_data, self.train_labels, epochs=100, batch_size=512)
        self.network.save(f'{self.folder_name}/model')

    def test(self):
        if self.test_data is None or self.test_labels is None:
            self.load_data()
        if self.network is None:
            self.network = models.load_model(f'{self.folder_name}/model')
        # Calculate Test loss and Test Accuracy
        test_loss, test_acc = self.network.evaluate(self.test_data, self.test_labels)

        # Print Test loss and Test Accuracy
        print(f"Test Loss: {test_loss}\nTest Accuracy : {test_acc * 100} %")

class PythonDefault(PRNG_tester):
    def __init__(self, seed: int, n_datapoints: int, length_of_data: int, console :Console):
        PRNG_tester.__init__(self, seed, n_datapoints, length_of_data, console, "PythonDefault")
        random.seed(seed)

    def prng(self):
        return random.randrange(10)



class CountUp(PRNG_tester):
    def __init__(self, seed: int, n_datapoints: int, length_of_data: int, console: Console):
        PRNG_tester.__init__(self, seed, n_datapoints,
                   length_of_data, console, "CountUp")

    def prng(self):
        self.seed += 1
        return self.seed % 10


def train_and_save_model(prng_data: PRNG_tester):
    prng_data.create_data()


if __name__ == '__main__':
    num_of_datapoints = 1000000
    lenght_of_data = 1000
    console = Console()

    default = CountUp(42, num_of_datapoints, lenght_of_data, console)

    #default.create_data()
    #default.train()
    default.test()



