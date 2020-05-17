import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from math import radians
import datetime

import cv2, time
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten 
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import save_model, load_model

tf.__version__

class VGG19_C:
    def __init__(self):
        self.covid_path = 'dataset/covid_dataset.csv'
        self.covid_image_path = 'dataset/covid_adjusted/'
        self.normal_path = 'dataset/normal_xray_dataset.csv'
        self.normal_image_path = 'dataset/normal_dataset/'
        self.head_count = 99
        self.test_ratio = 0.15
        self.shape = (224, 224, 3)
        self.folds = 5
        self.batch_size = 32
        self.epochs = 500
        self.verbose = 2 
        self.activation_optimizer = Adam(lr=0.0001, decay=1e-6)
        self.early_stop_criteria = EarlyStopping(patience=100, restore_best_weights=True)
        self.prior_model_path = 'prior_model.h5'

    def Generate_Model(self, params = 'default'):
        if params == 'default':
            shape = self.shape
            
        start_generate = datetime.datetime.now()
  
        model = tf.keras.Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3), input_shape=self.shape, activation='relu', padding='same')) 
        model.add(Conv2D(64, kernel_size=(3, 3), input_shape=self.shape, activation='relu', padding='same'))
        
        model_to_transfer = self.load_model(self.prior_model_path)
        self.model_to_transfer = model_to_transfer

        for i, layer in enumerate(model_to_transfer.layers[0].layers[2:]):
            model.add(layer)

        
        print('model_to_transfer-feedforward')
        for i, layer in enumerate(model_to_transfer.layers[1:]):
            model.add(layer)
        
        for i, layer in enumerate(model.layers):
            if i > 1:
                layer.trainable = False

        opt = self.activation_optimizer
        model.compile(
                loss='categorical_crossentropy', 
                optimizer=opt, 
                metrics=['accuracy']
        )

        train_aug = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )
        
        
        self.train_aug = train_aug
        
        end_generate = datetime.datetime.now()
        self.generate_time = str(end_generate - start_generate)

        print('calculation time for ALL: {}'.format(self.generate_time))
        
        return model
    
    def Run_Model(self, params = 'default'):
        if params == 'default':
            folds = self.folds; batch_size = self.batch_size; epochs = self.epochs; 
            X_train = self.X_train; y_train = self.y_train; X_test = self.X_test; y_test = self.y_test
            model = self.model; train_aug = self.train_aug
            
        lst_perf_folds = []
        lst_perf_folds_evaluate = []
        lst_perf_folds_history = []
        lst_perf_folds_report = []
        start_run = datetime.datetime.now()

        kf = KFold(n_splits = folds, random_state = 1, shuffle = True)
        kf.get_n_splits(X_train)

        for fold, (train_index, validation_index) in enumerate(kf.split(X_train)):
            print('\n Fold %d' % (fold))
            start_iter = datetime.datetime.now()
            X_train_fold, X_validation = X_train[train_index], X_train[validation_index]
            y_train_fold, y_validation = y_train[train_index], y_train[validation_index]

            early_stop_criteria = self.early_stop_criteria
            history = model.fit(train_aug.flow(X_train_fold, y_train_fold, batch_size=batch_size),
                                validation_data=(X_validation, y_validation),
                                validation_steps=len(X_validation) / batch_size,
                                steps_per_epoch=len(X_train_fold) / batch_size,
                                epochs=epochs, verbose=self.verbose,
                                callbacks=[
                                early_stop_criteria
                                ]
            )  

            y_pred_test = model.predict(X_test, batch_size = batch_size)
            y_pred_train = model.predict(X_train, batch_size = batch_size)
            y_pred_train_fold = model.predict(X_train_fold, batch_size = batch_size)
            y_pred_validation = model.predict(X_validation, batch_size = batch_size)

            rep1 = classification_report(np.argmax(y_test, axis = 1), np.argmax(y_pred_test, axis = 1), output_dict = True)
            rep2 = classification_report(np.argmax(y_train, axis = 1), np.argmax(y_pred_train, axis = 1), output_dict = True)
            rep3 = classification_report(np.argmax(y_train_fold, axis = 1), np.argmax(y_pred_train_fold, axis = 1), output_dict = True)
            rep4 = classification_report(np.argmax(y_validation, axis = 1), np.argmax(y_pred_validation, axis = 1), output_dict = True)

            lst_perf_folds.append((rep1['accuracy'], rep2['accuracy'], rep3['accuracy'], rep4['accuracy']))
            lst_perf_folds_history.append(model.history.history)
            lst_perf_folds_report.append((rep1, rep2, rep3, rep4))

            evaluate_TEST = model.evaluate(X_test, y_test, verbose=0)
            evaluate_TRAIN = model.evaluate(X_train, y_train, verbose=0)
            evaluate_TRAIN_Fold = model.evaluate(X_train_fold, y_train_fold, verbose=0)
            evaluate_VALIDATION = model.evaluate(X_validation, y_validation, verbose=0)

            lst_perf_folds_evaluate.append((evaluate_TEST, evaluate_TRAIN, evaluate_TRAIN_Fold, evaluate_VALIDATION))

            end_iter = datetime.datetime.now()
            print('calculation time for iteration-{}: {}'.format(str(fold), str(end_iter - start_iter)))

        mean_Accuracy_TEST = round(np.mean(np.array(lst_perf_folds)[:, 0]), 4)
        self.mean_Accuracy_TEST = mean_Accuracy_TEST

        mean_Accuracy_TRAIN = round(np.mean(np.array(lst_perf_folds)[:, 1]), 4)
        self.mean_Accuracy_TRAIN = mean_Accuracy_TRAIN

        mean_Accuracy_TRAIN_Fold = round(np.mean(np.array(lst_perf_folds)[:, 2]), 4)

        mean_Accuracy_VALIDATION = round(np.mean(np.array(lst_perf_folds)[:, 3]), 4)
        self.mean_Accuracy_VALIDATION = mean_Accuracy_VALIDATION
        print('Avg-TEST Acc: {} ... Avg-VALIDATION Acc: {}'.format(mean_Accuracy_TEST, mean_Accuracy_VALIDATION))
        print('Avg-TRAIN Acc: {} ... Avg-TRAIN_Fold Acc: {}'.format(mean_Accuracy_TRAIN, mean_Accuracy_TRAIN_Fold))
        print('lst_perf_folds_evaluate: {}'.format(lst_perf_folds_evaluate))
        
        print('Support Check:')
        print('TEST: # of Images Normal: {} vs Covid-19: {}'.format(rep1['1']['support'], rep1['0']['support']))
        print('VALIDATION: # of Images Normal: {} vs Covid-19: {}'.format(rep4['1']['support'], rep4['0']['support']))
        print('TRAIN: # of Images Normal: {} vs Covid-19: {}'.format(rep2['1']['support'], rep2['0']['support']))
        print('TRAIN-Fold: # of Images Normal: {} vs Covid-19: {}'.format(rep3['1']['support'], rep3['0']['support']))   
        
        end_run = datetime.datetime.now()
        self.run_time = str(end_run - start_run)

        print('calculation time for RUN: {}'.format(self.run_time))
        print('Finished at {}'.format(datetime.datetime.now()))
        
    def Create_DataFrames(self, params = 'default'):
        if params == 'default':
            covid_path = self.covid_path; normal_path = self.normal_path; head_count = self.head_count
            
        covid_df = pd.read_csv(covid_path, usecols=['filename', 'finding'])
        normal_df = pd.read_csv(normal_path, usecols=['filename', 'finding'])
        normal_df = normal_df.head(head_count)
        
        self.covid_df = covid_df
        self.normal_df = normal_df

        return covid_df, normal_df

    def Fetch_Images(self, params = 'default' ):
        if params == 'default':
            covid_df = self.covid_df; normal_df = self.normal_df
            
        covid_images_lst = []
        covid_labels = []
        
        covid_image_path = self.covid_image_path
        normal_image_path = self.normal_image_path

        for index, row in covid_df.iterrows():
            filename = row['filename']
            label = row['finding']
            path = covid_image_path + filename

            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            covid_images_lst.append(image)
            covid_labels.append(label)

        normal_images_lst = []
        normal_labels = []

        for index, row in normal_df.iterrows():
            filename = row['filename']
            label = row['finding']
            path = normal_image_path + filename

            # temporary fix while we preprocess ALL the images
            if filename == '4c268764-b5e5-4417-85a3-da52916984d8.jpg':
                break

            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            normal_images_lst.append(image)
            normal_labels.append(label)

            # normalize to interval of [0,1]
            covid_images = np.array(covid_images_lst) / 255

            # normalize to interval of [0,1]
            normal_images = np.array(normal_images_lst) / 255
            
            self.covid_images = covid_images
            self.normal_images = normal_images
            self.covid_labels = covid_labels
            self.normal_labels = normal_labels

        return covid_images, normal_images, covid_labels, normal_labels
    
    def plot_images(self, images, title):  
        nrows, ncols = 10, 10
        figsize = [5, 5]

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, facecolor=(1, 1, 1))

        for i, axi in enumerate(ax.flat):
            axi.imshow(images[i])
            axi.set_axis_off()

        plt.suptitle(title, fontsize=24)
        plt.tight_layout(pad=0.2, rect=[0, 0, 1, 0.9])
        plt.show()

    def Split_Train_Test(self, params = 'default'):
        if params == 'default':
            covid_images = self.covid_images; normal_images = self.normal_images 
            covid_labels = self.covid_labels; normal_labels = self.normal_labels; test_ratio = self.test_ratio
            
        # covid_images=92, normal_images=99 >> 191 = 152_Train + 39_Test
        # split into training and testing   # , shuffle = True
        covid_x_train, covid_x_test, covid_y_train, covid_y_test = \
        train_test_split(covid_images, covid_labels, test_size = test_ratio)

        normal_x_train, normal_x_test, normal_y_train, normal_y_test =\
        train_test_split(normal_images, normal_labels, test_size = test_ratio)

        X_train = np.concatenate((normal_x_train, covid_x_train), axis=0)
        X_test = np.concatenate((normal_x_test, covid_x_test), axis=0)
        y_train = np.concatenate((normal_y_train, covid_y_train), axis=0)
        y_test = np.concatenate((normal_y_test, covid_y_test), axis=0)

        # make labels into categories - either 0 or 1
        y_train = LabelBinarizer().fit_transform(y_train)
        y_train = to_categorical(y_train)

        y_test = LabelBinarizer().fit_transform(y_test)
        y_test = to_categorical(y_test)
        
        self.X_train = X_train; self.y_train = y_train
        self.X_test = X_test; self.y_test = y_test
        
        return X_train, y_train, X_test, y_test
    
    def Time_Stamp(self):
        date_time = datetime.datetime.now()

        D = str(date_time.day)
        M = str(date_time.month)
        Y = str(date_time.year)

        h = str(date_time.hour)
        m = str(date_time.minute)
        s = str(date_time.second)

        lst_date = [D, M, Y, h, m, s]
        
        return lst_date
    
    def FileNameUnique(self, prefix = "Grp16_", suffix = '.csv'):
        file_name = prefix

        lst_date = self.Time_Stamp()
        
        for idx, i in enumerate(lst_date):
            if idx == 2:
                file_name += i + '_'
            elif idx == 5:
                file_name += i + suffix
            else:
                file_name += i + '.'

        return file_name
    
    def model_parameters(self):
        list_param_name = ['test_ratio', 'folds', 'batch_size', 'epochs', 'verbose', 'shape', 
                            'activation_optimizer', 'early_stop_criteria',
                            'covid_path', 'covid_image_path', 'normal_path', 'normal_image_path', 'head_count'] 
        
        list_param_values = [self.test_ratio, self.folds, self.batch_size, self.epochs, self.verbose, self.shape, 
                              self.activation_optimizer, self.early_stop_criteria,
               self.covid_path, self.covid_image_path, self.normal_path, self.normal_image_path, self.head_count]
        
        dict_params = {'parameter': list_param_name, 'value': list_param_values}
        df_params = pd.DataFrame(dict_params)

        return df_params
        
    def model_parameters_save(self):
        list_param_name = ['mean_Accuracy_TEST', 'mean_Accuracy_VALIDATION', 'mean_Accuracy_TRAIN', 'Run_Time',
                            'test_ratio', 'folds', 'batch_size', 'epochs', 'verbose', 'shape', 
                            'activation_opt_keys', 'activation_opt_vals', 'early_stop_criteria',
                            'covid_path', 'covid_image_path', 'normal_path', 'normal_image_path', 'head_count'] 
        
        opt_config = opt_config = self.activation_optimizer.get_config()
        list_optimizer_keys = [ k for k in opt_config ]
        list_optimizer_values = [ v for v in opt_config.values() ]

        list_param_values = [self.mean_Accuracy_TEST, self.mean_Accuracy_VALIDATION, self.mean_Accuracy_TRAIN, 
        self.run_time, self.test_ratio, self.folds, self.batch_size, self.epochs, self.verbose, self.shape, 
        list_optimizer_keys, list_optimizer_values, self.early_stop_criteria, self.covid_path, self.covid_image_path, 
        self.normal_path, self.normal_image_path, self.head_count]
        
        dict_params = {'parameter': list_param_name, 'value': list_param_values}
        df_params = pd.DataFrame(dict_params)

        return df_params

    def save_model(self, model_t0_save, file_name):
        save_model(model_t0_save, file_name)
        print('model saved as: {}'.format(file_name))
 
    def load_model(self, file_name):
        loaded_model = load_model(file_name)
        return loaded_model