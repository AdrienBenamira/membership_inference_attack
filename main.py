from utils import config
from experience_mnist import *
from experience_cifar10 import *
from experience_cifar100 import *
import os
import shutil
import datetime

config = config()

now = str(datetime.datetime.now())[:19]
now = now.replace(":","_")
now = now.replace("-","_")
now = now.replace(" ","_")

src_dir = config.path.data_path
path = config.path.result_path + str(config.statistics.dataset) + "_" + str(config.statistics.type) + "_" + str(now)
os.mkdir(path)
dst_dir = path+"/config.yaml"
shutil.copy(src_dir,dst_dir)



if config.statistics.dataset == "MNIST":
    if config.statistics.type == "training_size":
        print("START STATS ON TRAINING SIZE WITH MNIST : ", config.statistics.training_size_value)
        res_precision = np.zeros(len(config.statistics.training_size_value))
        res_recall = np.zeros(len(config.statistics.training_size_value))
        res_accuracy = np.zeros(len(config.statistics.training_size_value))
        #res_best_acc_target = np.zeros(len(config.statistics.training_size_value))
        #res_best_acc_shadows = np.zeros(len(config.statistics.training_size_value))
        res_precision_per_class = np.zeros((len(config.statistics.training_size_value), 10))
        res_recall_per_class = np.zeros((len(config.statistics.training_size_value), 10))
        res_accuracy_per_class = np.zeros((len(config.statistics.training_size_value), 10))
        for index_value_to_test, value_to_test in enumerate(config.statistics.training_size_value):
            config.set_subkey("general", "train_target_size", value_to_test)
            precision_general, recall_general, accuracy_general, precision_per_class, recall_per_class, accuracy_per_class = experience_mnist(config, path, index_value_to_test)
            res_precision[index_value_to_test] = precision_general
            res_recall[index_value_to_test] = recall_general
            res_accuracy[index_value_to_test] = accuracy_general
            #res_best_acc_target[index_value_to_test] = best_acc_target
            #res_best_acc_shadows[index_value_to_test] = best_acc_shadows
            res_precision_per_class[index_value_to_test] = precision_per_class
            res_recall_per_class[index_value_to_test] = recall_per_class
            res_accuracy_per_class[index_value_to_test] = accuracy_per_class
            np.save(path + "/res_precision.npy", res_precision)
            np.save(path + "/res_recall.npy", res_recall)
            np.save(path + "/res_accuracy.npy", res_accuracy)
            #np.save(path + "/res_best_acc_target.npy", res_best_acc_target)
            np.save(path + "/res_recall_per_class.npy", res_recall_per_class)
            np.save(path + "/res_precision_per_class.npy", res_precision_per_class)
            np.save(path + "/res_accuracy_per_class.npy", res_accuracy_per_class)
            #np.save(path + "/res_best_acc_shadows.npy", res_best_acc_shadows)
    if config.statistics.type == "number_shadow":
        print("START STATS ON NUMBER SHADOW  WITH MNIST : ", config.statistics.number_shadow_value)
        res_precision = np.zeros(len(config.statistics.number_shadow_value))
        res_recall = np.zeros(len(config.statistics.number_shadow_value))
        res_accuracy = np.zeros(len(config.statistics.number_shadow_value))
        #res_best_acc_target = np.zeros(len(config.statistics.number_shadow_value))
        #res_best_acc_shadows = np.zeros(len(config.statistics.number_shadow_value))
        res_precision_per_class = np.zeros((len(config.statistics.number_shadow_value), 10))
        res_recall_per_class = np.zeros((len(config.statistics.number_shadow_value), 10))
        res_accuracy_per_class = np.zeros((len(config.statistics.number_shadow_value), 10))
        for index_value_to_test, value_to_test in enumerate(config.statistics.number_shadow_value):
            config.set_subkey("general", "number_shadow_model", value_to_test)
            precision_general, recall_general, accuracy_general, precision_per_class, recall_per_class, accuracy_per_class = experience_mnist(config, path, index_value_to_test)
            res_precision[index_value_to_test] = precision_general
            res_recall[index_value_to_test] = recall_general
            res_accuracy[index_value_to_test] = accuracy_general
            #res_best_acc_target[index_value_to_test] = best_acc_target
            #res_best_acc_shadows[index_value_to_test] = best_acc_shadows
            res_precision_per_class[index_value_to_test] = precision_per_class
            res_recall_per_class[index_value_to_test] = recall_per_class
            res_accuracy_per_class[index_value_to_test] = accuracy_per_class
            np.save(path + "/res_precision.npy", res_precision)
            np.save(path + "/res_recall.npy", res_recall)
            np.save(path + "/res_accuracy.npy", res_accuracy)
            #np.save(path + "/res_best_acc_target.npy", res_best_acc_target)
            np.save(path + "/res_recall_per_class.npy", res_recall_per_class)
            np.save(path + "/res_precision_per_class.npy", res_precision_per_class)
            np.save(path + "/res_accuracy_per_class.npy", res_accuracy_per_class)
            #np.save(path + "/res_best_acc_shadows.npy", res_best_acc_shadows)
    if config.statistics.type == "overfitting":
        print("START STATS ON OVERFITTING WITH MNIST : ", config.statistics.epoch_value)
        res_precision = np.zeros(len(config.statistics.epoch_value))
        res_recall = np.zeros(len(config.statistics.epoch_value))
        res_accuracy = np.zeros(len(config.statistics.epoch_value))
        #res_best_acc_target = np.zeros(len(config.statistics.epoch_value))
        #res_best_acc_shadows = np.zeros(len(config.statistics.epoch_value))
        res_precision_per_class = np.zeros((len(config.statistics.epoch_value), 10))
        res_recall_per_class = np.zeros((len(config.statistics.epoch_value), 10))
        res_accuracy_per_class = np.zeros((len(config.statistics.epoch_value), 10))
        for index_value_to_test, value_to_test in enumerate(config.statistics.epoch_value):
            config.set_subkey("learning", "epochs", value_to_test)
            precision_general, recall_general, accuracy_general, precision_per_class, recall_per_class, accuracy_per_class = experience_mnist(config, path, index_value_to_test)
            res_precision[index_value_to_test] = precision_general
            res_recall[index_value_to_test] = recall_general
            res_accuracy[index_value_to_test] = accuracy_general
            #res_best_acc_target[index_value_to_test] = best_acc_target
            #res_best_acc_shadows[index_value_to_test] = best_acc_shadows
            res_precision_per_class[index_value_to_test] = precision_per_class
            res_recall_per_class[index_value_to_test] = recall_per_class
            res_accuracy_per_class[index_value_to_test] = accuracy_per_class
            np.save(path + "/res_precision.npy", res_precision)
            np.save(path + "/res_recall.npy", res_recall)
            np.save(path + "/res_accuracy.npy", res_accuracy)
            #np.save(path + "/res_best_acc_target.npy", res_best_acc_target)
            np.save(path + "/res_recall_per_class.npy", res_recall_per_class)
            np.save(path + "/res_precision_per_class.npy", res_precision_per_class)
            np.save(path + "/res_accuracy_per_class.npy", res_accuracy_per_class)
            #np.save(path + "/res_best_acc_shadows.npy", res_best_acc_shadows)

if config.statistics.dataset == "CIFAR10":
    if config.statistics.type == "training_size":
        print("START STATS ON TRAINING SIZE WITH CIFAR10 : ", config.statistics.training_size_value)
        res_precision = np.zeros(len(config.statistics.training_size_value))
        res_recall = np.zeros(len(config.statistics.training_size_value))
        res_accuracy = np.zeros(len(config.statistics.training_size_value))
        # res_best_acc_target = np.zeros(len(config.statistics.training_size_value))
        # res_best_acc_shadows = np.zeros(len(config.statistics.training_size_value))
        res_precision_per_class = np.zeros((len(config.statistics.training_size_value), 10))
        res_recall_per_class = np.zeros((len(config.statistics.training_size_value), 10))
        res_accuracy_per_class = np.zeros((len(config.statistics.training_size_value), 10))
        for index_value_to_test, value_to_test in enumerate(config.statistics.training_size_value):
            config.set_subkey("general", "train_target_size", value_to_test)
            precision_general, recall_general, accuracy_general, precision_per_class, recall_per_class, accuracy_per_class = experience_cifar10(config, path, index_value_to_test)
            res_precision[index_value_to_test] = precision_general
            res_recall[index_value_to_test] = recall_general
            res_accuracy[index_value_to_test] = accuracy_general
            #res_best_acc_target[index_value_to_test] = best_acc_target
            #res_best_acc_shadows[index_value_to_test] = best_acc_shadows
            res_precision_per_class[index_value_to_test] = precision_per_class
            res_recall_per_class[index_value_to_test] = recall_per_class
            res_accuracy_per_class[index_value_to_test] = accuracy_per_class
            np.save(path + "/res_precision.npy", res_precision)
            np.save(path + "/res_recall.npy", res_recall)
            np.save(path + "/res_accuracy.npy", res_accuracy)
            #np.save(path + "/res_best_acc_target.npy", res_best_acc_target)
            np.save(path + "/res_recall_per_class.npy", res_recall_per_class)
            np.save(path + "/res_precision_per_class.npy", res_precision_per_class)
            np.save(path + "/res_accuracy_per_class.npy", res_accuracy_per_class)
            #np.save(path + "/res_best_acc_shadows.npy", res_best_acc_shadows)
    if config.statistics.type == "number_shadow":
        print("START STATS ON NUMBER SHADOW  WITH CIFAR10 : ", config.statistics.number_shadow_value)
        res_precision = np.zeros(len(config.statistics.number_shadow_value))
        res_recall = np.zeros(len(config.statistics.number_shadow_value))
        res_accuracy = np.zeros(len(config.statistics.number_shadow_value))
        #res_best_acc_target = np.zeros(len(config.statistics.number_shadow_value))
        #res_best_acc_shadows = np.zeros(len(config.statistics.number_shadow_value))
        res_precision_per_class = np.zeros((len(config.statistics.number_shadow_value), 10))
        res_recall_per_class = np.zeros((len(config.statistics.number_shadow_value), 10))
        res_accuracy_per_class = np.zeros((len(config.statistics.number_shadow_value), 10))
        for index_value_to_test, value_to_test in enumerate(config.statistics.number_shadow_value):
            config.set_subkey("general", "number_shadow_model", value_to_test)
            precision_general, recall_general, accuracy_general, precision_per_class, recall_per_class, accuracy_per_class = experience_cifar10(config, path, index_value_to_test)
            res_precision[index_value_to_test] = precision_general
            res_recall[index_value_to_test] = recall_general
            res_accuracy[index_value_to_test] = accuracy_general
            #res_best_acc_target[index_value_to_test] = best_acc_target
            #res_best_acc_shadows[index_value_to_test] = best_acc_shadows
            res_precision_per_class[index_value_to_test] = precision_per_class
            res_recall_per_class[index_value_to_test] = recall_per_class
            res_accuracy_per_class[index_value_to_test] = accuracy_per_class
            np.save(path + "/res_precision.npy", res_precision)
            np.save(path + "/res_recall.npy", res_recall)
            np.save(path + "/res_accuracy.npy", res_accuracy)
            #np.save(path + "/res_best_acc_target.npy", res_best_acc_target)
            np.save(path + "/res_recall_per_class.npy", res_recall_per_class)
            np.save(path + "/res_precision_per_class.npy", res_precision_per_class)
            np.save(path + "/res_accuracy_per_class.npy", res_accuracy_per_class)
            #np.save(path + "/res_best_acc_shadows.npy", res_best_acc_shadows)
    if config.statistics.type == "overfitting":
        print("START STATS ON OVERFITTING WITH CIFAR10 : ", config.statistics.epoch_value)
        res_precision = np.zeros(len(config.statistics.epoch_value))
        res_recall = np.zeros(len(config.statistics.epoch_value))
        res_accuracy = np.zeros(len(config.statistics.epoch_value))
        #res_best_acc_target = np.zeros(len(config.statistics.epoch_value))
        #res_best_acc_shadows = np.zeros(len(config.statistics.epoch_value))
        res_precision_per_class = np.zeros((len(config.statistics.epoch_value), 10))
        res_recall_per_class = np.zeros((len(config.statistics.epoch_value), 10))
        res_accuracy_per_class = np.zeros((len(config.statistics.epoch_value), 10))
        for index_value_to_test, value_to_test in enumerate(config.statistics.epoch_value):
            config.set_subkey("learning", "epochs", value_to_test)
            precision_general, recall_general, accuracy_general, precision_per_class, recall_per_class, accuracy_per_class = experience_cifar10(config, path, index_value_to_test)
            res_precision[index_value_to_test] = precision_general
            res_recall[index_value_to_test] = recall_general
            res_accuracy[index_value_to_test] = accuracy_general
            #res_best_acc_target[index_value_to_test] = best_acc_target
            #res_best_acc_shadows[index_value_to_test] = best_acc_shadows
            res_precision_per_class[index_value_to_test] = precision_per_class
            res_recall_per_class[index_value_to_test] = recall_per_class
            res_accuracy_per_class[index_value_to_test] = accuracy_per_class
            np.save(path + "/res_precision.npy", res_precision)
            np.save(path + "/res_recall.npy", res_recall)
            np.save(path + "/res_accuracy.npy", res_accuracy)
            #np.save(path + "/res_best_acc_target.npy", res_best_acc_target)
            np.save(path + "/res_recall_per_class.npy", res_recall_per_class)
            np.save(path + "/res_precision_per_class.npy", res_precision_per_class)
            np.save(path + "/res_accuracy_per_class.npy", res_accuracy_per_class)
            #np.save(path + "/res_best_acc_shadows.npy", res_best_acc_shadows)


if config.statistics.dataset == "CIFAR100":
    if config.statistics.type == "training_size":
        print("START STATS ON TRAINING SIZE WITH CIFAR100 : ", config.statistics.training_size_value)
        res_precision = np.zeros(len(config.statistics.training_size_value))
        res_recall = np.zeros(len(config.statistics.training_size_value))
        res_accuracy = np.zeros(len(config.statistics.training_size_value))
        # res_best_acc_target = np.zeros(len(config.statistics.training_size_value))
        # res_best_acc_shadows = np.zeros(len(config.statistics.training_size_value))
        res_precision_per_class = np.zeros((len(config.statistics.training_size_value), 100))
        res_recall_per_class = np.zeros((len(config.statistics.training_size_value), 100))
        res_accuracy_per_class = np.zeros((len(config.statistics.training_size_value), 100))
        for index_value_to_test, value_to_test in enumerate(config.statistics.training_size_value):
            config.set_subkey("general", "train_target_size", value_to_test)
            precision_general, recall_general, accuracy_general, precision_per_class, recall_per_class, accuracy_per_class = experience_cifar100(config, path, index_value_to_test)
            res_precision[index_value_to_test] = precision_general
            res_recall[index_value_to_test] = recall_general
            res_accuracy[index_value_to_test] = accuracy_general
            #res_best_acc_target[index_value_to_test] = best_acc_target
            #res_best_acc_shadows[index_value_to_test] = best_acc_shadows
            res_precision_per_class[index_value_to_test] = precision_per_class
            res_recall_per_class[index_value_to_test] = recall_per_class
            res_accuracy_per_class[index_value_to_test] = accuracy_per_class
            np.save(path + "/res_precision.npy", res_precision)
            np.save(path + "/res_recall.npy", res_recall)
            np.save(path + "/res_accuracy.npy", res_accuracy)
            #np.save(path + "/res_best_acc_target.npy", res_best_acc_target)
            np.save(path + "/res_recall_per_class.npy", res_recall_per_class)
            np.save(path + "/res_precision_per_class.npy", res_precision_per_class)
            np.save(path + "/res_accuracy_per_class.npy", res_accuracy_per_class)
            #np.save(path + "/res_best_acc_shadows.npy", res_best_acc_shadows)
    if config.statistics.type == "number_shadow":
        print("START STATS ON NUMBER SHADOW  WITH CIFAR100 : ", config.statistics.number_shadow_value)
        res_precision = np.zeros(len(config.statistics.number_shadow_value))
        res_recall = np.zeros(len(config.statistics.number_shadow_value))
        res_accuracy = np.zeros(len(config.statistics.number_shadow_value))
        #res_best_acc_target = np.zeros(len(config.statistics.number_shadow_value))
        #res_best_acc_shadows = np.zeros(len(config.statistics.number_shadow_value))
        res_precision_per_class = np.zeros((len(config.statistics.number_shadow_value), 100))
        res_recall_per_class = np.zeros((len(config.statistics.number_shadow_value), 100))
        res_accuracy_per_class = np.zeros((len(config.statistics.number_shadow_value), 100))
        for index_value_to_test, value_to_test in enumerate(config.statistics.number_shadow_value):
            config.set_subkey("general", "number_shadow_model", value_to_test)
            precision_general, recall_general, accuracy_general, precision_per_class, recall_per_class, accuracy_per_class = experience_cifar100(config, path, index_value_to_test)
            res_precision[index_value_to_test] = precision_general
            res_recall[index_value_to_test] = recall_general
            res_accuracy[index_value_to_test] = accuracy_general
            #res_best_acc_target[index_value_to_test] = best_acc_target
            #res_best_acc_shadows[index_value_to_test] = best_acc_shadows
            res_precision_per_class[index_value_to_test] = precision_per_class
            res_recall_per_class[index_value_to_test] = recall_per_class
            res_accuracy_per_class[index_value_to_test] = accuracy_per_class
            np.save(path + "/res_precision.npy", res_precision)
            np.save(path + "/res_recall.npy", res_recall)
            np.save(path + "/res_accuracy.npy", res_accuracy)
            #np.save(path + "/res_best_acc_target.npy", res_best_acc_target)
            np.save(path + "/res_recall_per_class.npy", res_recall_per_class)
            np.save(path + "/res_precision_per_class.npy", res_precision_per_class)
            np.save(path + "/res_accuracy_per_class.npy", res_accuracy_per_class)
            #np.save(path + "/res_best_acc_shadows.npy", res_best_acc_shadows)
    if config.statistics.type == "overfitting":
        print("START STATS ON OVERFITTING WITH CIFAR100 : ", config.statistics.epoch_value)
        res_precision = np.zeros(len(config.statistics.epoch_value))
        res_recall = np.zeros(len(config.statistics.epoch_value))
        res_accuracy = np.zeros(len(config.statistics.epoch_value))
        #res_best_acc_target = np.zeros(len(config.statistics.epoch_value))
        #res_best_acc_shadows = np.zeros(len(config.statistics.epoch_value))
        res_precision_per_class = np.zeros((len(config.statistics.epoch_value), 100))
        res_recall_per_class = np.zeros((len(config.statistics.epoch_value), 100))
        res_accuracy_per_class = np.zeros((len(config.statistics.epoch_value), 100))
        for index_value_to_test, value_to_test in enumerate(config.statistics.epoch_value):
            config.set_subkey("learning", "epochs", value_to_test)
            precision_general, recall_general, accuracy_general, precision_per_class, recall_per_class, accuracy_per_class = experience_cifar100(config, path, index_value_to_test)
            res_precision[index_value_to_test] = precision_general
            res_recall[index_value_to_test] = recall_general
            res_accuracy[index_value_to_test] = accuracy_general
            #res_best_acc_target[index_value_to_test] = best_acc_target
            #res_best_acc_shadows[index_value_to_test] = best_acc_shadows
            res_precision_per_class[index_value_to_test] = precision_per_class
            res_recall_per_class[index_value_to_test] = recall_per_class
            res_accuracy_per_class[index_value_to_test] = accuracy_per_class
            np.save(path + "/res_precision.npy", res_precision)
            np.save(path + "/res_recall.npy", res_recall)
            np.save(path + "/res_accuracy.npy", res_accuracy)
            #np.save(path + "/res_best_acc_target.npy", res_best_acc_target)
            np.save(path + "/res_recall_per_class.npy", res_recall_per_class)
            np.save(path + "/res_precision_per_class.npy", res_precision_per_class)
            np.save(path + "/res_accuracy_per_class.npy", res_accuracy_per_class)
            #np.save(path + "/res_best_acc_shadows.npy", res_best_acc_shadows)
