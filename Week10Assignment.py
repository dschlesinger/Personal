#Import statements, some not in use right now
import os
import pandas as pd
import random
from getpass import getpass
import keras.layers
import numpy as np
import torch
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, callbacks
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

#Gpu configuration
gpus = tf.config.list_physical_devices('GPU')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#print("right program")
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=100000)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
# Change this to the location of the database directories

#Dirrector and more imports
DB_DIR = os.path.dirname(os.path.realpath(r"C:\Users\Theon\Downloads\installation_guide"))
sys.path.insert(1, DB_DIR)
from db_utils import get_imdb_dataset, get_speech_dataset, get_single_digit_dataset

#alphabetical list used in final implementation to convtert input and dirrect it to the right model also used to validate input
list_alp = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","y","x","z",]

#validates input for final implementation
def run_check(list):
    for i in list:
        if len(i) > 1:
            return False
        elif i.lower() not in list_alp:
            return False
        else:
            return True

def choose_dataset(dataset_type):
    """Select dataset based on string variable."""
    #NLP dataset
    if dataset_type == "nlp":
        return get_imdb_dataset(dir=DB_DIR)
    #Mnist
    elif dataset_type == "computer_vision":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    #Speech rocgnition Data
    elif dataset_type == "speech_recognition":
        (X_train, y_train), (X_test, y_test), (_, _) = get_speech_dataset()
    #A-Z dataset from Kaggle
    elif dataset_type == "Handwritten_letters":
        data = pd.read_csv(r"C:\Users\Theon\Downloads\Installation_guide\A_Z_Data\A_Z _Data.csv")
        print(data.dtypes)

        #Converts csv into X_train data
        X_train_one_i = list(data.drop('0', axis='columns').values)
        X_train_one = X_train_one_i
        del X_train_one[300000:]
        X_train = tf.Variable(X_train_one)

        #takes y_values out of x_values
        Y_train_one_i = list(data['0'].values)

        #Converts csv into y_train data
        Y_train_one = Y_train_one_i
        del Y_train_one[300000:]
        y_train = Y_train_one

        #creates x/y test
        X_train_one_2 = X_train_one_i
        Y_train_one_2 = X_train_one_i
        del X_train_one_2[0:300000]
        del Y_train_one_2[0:300000]
        X_test = tf.Variable(X_train_one)
        y_test = tf.Variable(Y_train_one)
    else:
        raise ValueError("Couldn't find dataset.")
    #(X_train, X_test) = normalize_dataset(dataset_type, X_train, X_test)

    #reshapes and returns requested data
    (X_train, y_train), (X_test, y_test) = reshape_dataset(X_train, y_train, X_test, y_test)
    return (X_train, y_train), (X_test, y_test)

#Decided to leave data as imported, so did not use this function
def normalize_dataset(string, X_train, X_test):
    """Normalize speech recognition and computer vision datasets."""
    if string == "computer vision":
        X_train = X_train / 255
        X_test = X_test / 255
    else:
        mean = np.mean(X_train)
        std = np.std(X_train)
        X_train = (X_train-std)/mean
        X_test = (X_test-std)/mean

    return (X_train, X_test)

#catagorizes and reshapes data as needed, varries by dataset, I manually changed it when needed
def reshape_dataset(X_train, y_train, X_test, y_test):
    """Reshape Computer Vision and Speech datasets."""
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return (X_train, y_train), (X_test, y_test)



class GAN():
    def init(self, input_shape=(28,28,1), rand_vector_shape=(100,),
             lr=0.0002, beta=0.5):
        #"computer_vision"/"Handwritten_letters"
        (X_train, y_train), (X_test, y_test) = choose_dataset("Handwritten_letters")
        self.X_train = X_train
        self.y_train = y_train
        # Input sizes
        self.img_shape = input_shape
        self.input_size = rand_vector_shape
        # optimizer
        self.opt = tf.keras.optimizers.Adam(lr, beta)
        self.rand_num = 100

    def discriminator_model(self):
            """Create discriminator model."""
            model = tf.keras.models.Sequential(name='Discriminator')
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Input(shape=(28,28,1)))
            model.add(keras.layers.Dense(512, activation='relu'))
            model.add(keras.layers.Dense(256, activation='relu'))
            """
            #increased dicriminator power due to bad dicriminator accurcary and image quality 
            model.add(keras.layers.Dense(512, activation='relu'))
            model.add(keras.layers.Dense(1024, activation='relu'))
            model.add(keras.layers.Dense(1024, activation='relu'))
            model.add(keras.layers.Dense(512, activation='relu'))
            """
            # Output
            model.add(keras.layers.Dense(1, activation='sigmoid'))
            return model

    def generator_model(self):
            """Create generator model."""
            model = tf.keras.models.Sequential(name='Generative')
            model.add(keras.layers.Input(shape=(self.rand_num,)))
            model.add(keras.layers.Dense(256, activation='relu'))
            model.add(keras.layers.Dense(512, activation='relu'))
            #Tried to balance model after increasing discriminator power, did not use do to computational expense
            #model.add(keras.layers.Dense(1024, activation='relu'))
            #model.add(keras.layers.Dense(1024, activation='relu'))
            model.add(keras.layers.Dense(784, activation='tanh'))
            model.add(keras.layers.Reshape((28, 28)))
            return model

    def call(self):
        lr = 0.0002
        beta = 0.5
        self.opt = tf.keras.optimizers.Adam(lr, beta)
        self.generator = self.generator_model()
        self.generator.compile(loss='binary_crossentropy',
                               optimizer=self.opt,
                               metrics=['accuracy'])
        #print(self.generator.summary())

        # Create Discriminator model
        self.discriminator = self.discriminator_model()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=self.opt,
                                   metrics=['accuracy'])
        #print(self.discriminator.summary())
        # Set the Discriminator as non trainable in the combined GAN model
        self.discriminator.trainable = False

        # Define model input and output
        input = tf.keras.Input(self.input_size)
        generated_img = self.generator(input)
        output = self.discriminator(generated_img)

        # Define and compile combined GAN model
        self.GAN = keras.Model(input, output, name="GAN")
        self.GAN.compile(loss='binary_crossentropy', optimizer=self.opt,
                         metrics=['accuracy'])

    def train(self, lett, batch_size, epochs=1001, save_interval=200, print_interval=500):
            """Train GAN Model."""
            half_batch = int(batch_size / 2)

            #experimented with changing real to generated image ratio
            quarter_batch = int(batch_size/4)
            three_fouths_batch = int(quarter_batch * 3)
            real_img_amount = half_batch
            gen_img_amount = half_batch

            y_pos_train_dis = tf.ones(half_batch, 1)
            y_neg_train_dis = tf.zeros(half_batch, 1)
            y_train_GAN = tf.ones(batch_size, 1)

            for epoch in range(epochs):
                # Generate training data for Discriminator
                list_xshape = []
                list_xshape = np.random.choice(self.X_train.shape[0], half_batch)
                X_pos_train_dis = []

                #used to create the selective or single image Gans
                cor = 0
                #get_num = int(input("Num?: "))
                bol = True
                while bol == True:
                    #get random X_train position
                    rand = np.random.choice(self.X_train.shape[0])
                    num_times = -1
                    print(self.y_train[rand])
                    #checks if class is correct
                    for e in self.y_train[rand]:
                        num_times = num_times + 1
                        if e == 1:
                            if num_times == lett+20:
                                #print(num_times)
                                cor = cor + 1
                                #fill batch with single image, remove for loop for selective
                                for i in range(half_batch):
                                    new = self.X_train[rand]
                                    X_pos_train_dis.extend(new)
                                #matplot image for debuging
                                #img = tf.Variable(new)
                                #img = tf.reshape(img, shape=(28,28))
                                #r, c = 4, 4
                                #fig, axs = plt.subplots()
                                #axs.imshow(img, cmap='gray')
                                #axs.axis('off')
                                #print(rand)
                                #plt.show()
                    if cor == 1:
                        #switching between selective and single image gan ==1 for single > half_batch-1 for selective
                        #if cor > half_batch-1:
                        bol = False
                #X_pos_train_dis = []
                """
                #generates randomized training data
                for u in list_xshape:
                    new = self.X_train[u]
                    #print(new)
                    X_pos_train_dis.extend(new)
                    #print(X_pos_train_dis)
                    #print("")
                """
                X_pos_train_dis = tf.Variable(X_pos_train_dis, dtype='float32')
                X_pos_train_dis = tf.reshape(X_pos_train_dis, shape=(real_img_amount,784))
                ##print(X_pos_train_dis) random half_batch amount of real images
                #X_neg_train_dis = random half_batch amount of generated images
                X_neg_train_dis = []

                #generates generated images
                for r in range(gen_img_amount):
                    neg_data = tf.random.uniform(shape=(1,self.rand_num), minval=0, maxval=1)
                    transfer_var = self.generator(neg_data)
                    transfer_var = tf.Variable(transfer_var, dtype='float32')
                    X_neg_train_dis.extend(list(transfer_var))

                #data manipulation and mixing the generated and real images
                X_neg_train_dis = tf.reshape(X_neg_train_dis,shape=(gen_img_amount,784))
                X_neg_train_dis = list(X_neg_train_dis)
                X_pos_train_dis = list(X_pos_train_dis)
                y_pos_train_dis = list(y_pos_train_dis)
                y_neg_train_dis = list(y_neg_train_dis)
                X_pos_train_dis.extend(X_neg_train_dis)
                #print(X_pos_train_dis)
                y_pos_train_dis.extend(y_neg_train_dis)
                c = list(zip(X_pos_train_dis,y_pos_train_dis))
                #print(c)
                random.shuffle(c)
                #print(c)
                X_train_dis,y_train_dis = zip(*c)
                #print(y_pos_train_dis)
                #print(tf.shape(X_train_dis))

                #Creates GAN training data, randomized 100,1 tensor
                X_train_GAN = []
                for o in range(batch_size):
                    newGAN = list(tf.random.uniform(shape=(self.rand_num,),minval=0,maxval=1))
                    X_train_GAN.extend(newGAN)
                #print(len(X_train_dis))
                #print(len(y_train_dis))

                #data manipulation to tensor
                X_train_dis = tf.Variable(X_train_dis)
                X_train_dis = tf.reshape(X_train_dis, shape=(batch_size, 28, 28))
                y_train_dis = tf.Variable(y_train_dis)
                y_train_dis = tf.reshape(y_train_dis, shape=(batch_size, 1))

                #train discriminator
                loss_dis = self.discriminator.train_on_batch(X_train_dis, y_train_dis)
                #print(loss_dis)
                #print(len(X_train_GAN))

                #running log of completed epcohs
                print(str(epoch) + "_" + str(lett))

                X_train_GAN = tf.reshape(X_train_GAN, shape=(batch_size,self.rand_num))
                #print(len(y_train_GAN))

                #trains generator
                loss_gen = self.GAN.train_on_batch(X_train_GAN,y_train_GAN)

                #controls save intervals, used 500 save and print img 200 show losss
                if (epoch%save_interval)==0:
                    print("Discrimintator loss = " + str(loss_dis[0]) + " & Generator loss = " + str(loss_gen[0]))
                    print("Discrimintator accuracy = " + str(loss_dis[1]) + " & Generator accuracy = " + str(loss_gen[1]))
                if (epoch%print_interval)==0:
                    print("print")
                    self.plot_img(epoch, lett, batch_size)

            #for final implementation needs to return trained model for storage
            return self.generator

    def plot_img(self, epoch, selnum, batch):
        r, c = 4,4
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                noise = np.random.normal(0, 1, (1, self.rand_num))
                img = self.generator.predict(noise)[0,:]
                img = tf.reshape(img, shape=(28,28))
                #print(img)
                axs[i,j].imshow(img, cmap='gray')
                axs[i,j].axis('off')
                cnt = cnt+1
        #name = str(selnum) + "_" + str(epoch) + "_" + str(batch) + "_test"
        name = "A_select_test_" + str(epoch)
        fig.savefig(r"C:\Users\Theon\Downloads\Select_GAN_img_save\Letters/img_epoch_{0}.png".format(name))

        plt.title("Epoch " + str(epoch))
        #if epoch == 19999:
        plt.title("Final epoch "+str(epoch))
        #print("runs?")
        #plt.show()
        return None

#calls

#instantates
call = GAN()

#declares variables
call.init()



list_of_al_train = []
list_of_img = []

#modified matplot funct for 1D array
def plot_new_img(list_img, num):
    print(len(list_img))
    print(type(list_img))
    c = len(list_img)
    fig, axs = plt.subplots(c)
    for j in range(len(list_img)):
        img = list_img[j]
        img = tf.Variable(img, shape=(28,28))
        img = tf.reshape(img, shape=(28,28))
        print(tf.shape(img))
        axs[j].imshow(img, cmap='gray')
        axs[j].axis('off')
    name = "Writting_test" + str(num)
    fig.savefig(r"C:\Users\Theon\Downloads\Select_GAN_img_save\test/img_epoch_{0}.png".format(name))
    plt.show()

list_of_al_train = []
t_num = 0

#final implementation base function
def run_fun(list, num):
    for run in list:
        find_num = 0
        for find in list_alp:
            if run == find:
                print(run)
                noise = np.random.normal(0, 1, (1, 100))
                img = list_of_al_train[find_num].predict(noise)
                #img = list(img)
                list_of_img.extend(img)
                print(len(list_of_img))
            else:
                find_num = find_num + 1
    plot_new_img(list_of_img, num)
    num = num + 1

#gets user input for how to proceed
train_ovn = str(input("Train overnight?: "))

num_i = 0
#used have multiple paths coded for different sitations

#used to train over long periods of time and iterates over all classes
if train_ovn == "y":
    for i in range(3):
        #final implentation call, for loop determine the letters avaible had issue with running over 19 letters
        call.call()
        list_of_al_train.append(call.train(i,16))
    #repetively asks for input and generates 1D array
    while True:
        list_of_img = []
        get_input = str(input("input: "))
        list = []
        list = get_input.split("-")
        check = run_check(list)

        #checks if input is valid, I don't have the models saved and if the input isnt valid it could crash the program
        if check == True:
            print(str(list) + "_True")
            run_fun(list,num_i)
        elif check == False:
            print(str(list) + "_False")
        num_i = num_i + 1

#used for to train for one number
elif train_ovn == "n":
    train_ovn_num = int(input("What number?: "))
    batch_i = 32
    call.call()
    call.train(train_ovn_num,batch_i)
