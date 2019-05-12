import torch
import torch.nn as nn
import torch.optim as optim
from dataloaders import *
from utils import config
import numpy as np
from model import *
from torch.optim import lr_scheduler
from trainer import *
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import lightgbm as lgb




def experience_cifar100(config, path,param):
    print("START CIFAR100")
    use_cuda = config.general.use_cuda and torch.cuda.is_available()
    torch.manual_seed(config.general.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("START TRAINING TARGET MODEL")
    data_train_target = custum_CIFAR100(True, 0, config, '../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ]))
    data_test_target = custum_CIFAR100(True, 0, config, '../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ]))
    criterion = nn.CrossEntropyLoss()
    train_loader_target = torch.utils.data.DataLoader(data_train_target, batch_size=config.learning.batch_size, shuffle=True)
    test_loader_target = torch.utils.data.DataLoader(data_test_target, batch_size=config.learning.batch_size, shuffle=True)
    dataloaders_target = {"train": train_loader_target, "val": test_loader_target}
    dataset_sizes_target = {"train": len(data_train_target), "val": len(data_test_target)}
    model_target = Net_cifar100().to(device)
    optimizer = optim.SGD(model_target.parameters(), lr=config.learning.learning_rate, momentum=config.learning.momentum)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=config.learning.decrease_lr_factor, gamma=config.learning.decrease_lr_every)
    model_target, best_acc_target, data_test_set, label_test_set, class_test_set = train_model(model_target, criterion, optimizer, exp_lr_scheduler,dataloaders_target,dataset_sizes_target,
                       num_epochs=config.learning.epochs)
    np.save(path + "/res_train_target_"+str(param)+".npy", best_acc_target)
    print("START TRAINING SHADOW MODEL")
    all_shadow_models = []
    all_dataloaders_shadow = []
    data_train_set = []
    label_train_set = []
    class_train_set = []
    for num_model_sahdow in range(config.general.number_shadow_model):
        criterion = nn.CrossEntropyLoss()
        data_train_shadow = custum_CIFAR100(False, num_model_sahdow, config, '../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ]))
        data_test_shadow = custum_CIFAR100(False, num_model_sahdow, config, '../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ]))
        train_loader_shadow = torch.utils.data.DataLoader(data_train_shadow, batch_size=config.learning.batch_size, shuffle=True)
        test_loader_shadow = torch.utils.data.DataLoader(data_test_shadow, batch_size=config.learning.batch_size, shuffle=True)
        dataloaders_shadow = {"train": train_loader_shadow, "val": test_loader_shadow}
        dataset_sizes_shadow = {"train": len(data_train_shadow), "val": len(data_test_shadow)}
        model_shadow = Net_cifar100().to(device)
        optimizer = optim.SGD(model_shadow.parameters(), lr=config.learning.learning_rate, momentum=config.learning.momentum)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=config.learning.decrease_lr_factor, gamma=config.learning.decrease_lr_every)
        model_shadow, best_acc_sh, data_train_set_unit, label_train_set_unit, class_train_set_unit = train_model(model_shadow, criterion, optimizer, exp_lr_scheduler,dataloaders_target,dataset_sizes_target,
                           num_epochs=config.learning.epochs)
        data_train_set.append(data_train_set_unit)
        label_train_set.append(label_train_set_unit)
        class_train_set.append(class_train_set_unit)
        np.save(path + "/res_train_shadow_"+str(num_model_sahdow)+"_"+str(param)+".npy", best_acc_sh)
        all_shadow_models.append(model_shadow)
        all_dataloaders_shadow.append(dataloaders_shadow)
    print("START GETTING DATASET ATTACK MODEL")
    data_train_set = np.concatenate(data_train_set)
    label_train_set = np.concatenate(label_train_set)
    class_train_set = np.concatenate(class_train_set)
    #data_test_set, label_test_set, class_test_set = get_data_for_final_eval([model_target], [dataloaders_target], device)
    #data_train_set, label_train_set, class_train_set = get_data_for_final_eval(all_shadow_models, all_dataloaders_shadow, device)
    data_train_set, label_train_set, class_train_set = shuffle(data_train_set, label_train_set, class_train_set, random_state=config.general.seed)
    data_test_set, label_test_set, class_test_set = shuffle(data_test_set, label_test_set, class_test_set, random_state=config.general.seed)
    #print(data_train_set.shape, data_test_set.shape)
    print("Taille dataset train", len(label_train_set))
    print("Taille dataset test", len(label_test_set))
    print("START FITTING ATTACK MODEL")
    model = lgb.LGBMClassifier(objective='binary', reg_lambda=config.learning.ml.reg_lambd, n_estimators=config.learning.ml.n_estimators)
    model.fit(data_train_set, label_train_set)
    y_pred_lgbm = model.predict(data_test_set)
    precision_general, recall_general, _, _ = precision_recall_fscore_support(y_pred=y_pred_lgbm, y_true=label_test_set, average = "macro")
    accuracy_general = accuracy_score(y_true=label_test_set, y_pred=y_pred_lgbm)
    precision_per_class, recall_per_class, accuracy_per_class = [], [], []
    for idx_class, classe in enumerate(data_train_target.classes):
        all_index_class = np.where(class_test_set == idx_class)
        precision, recall, _, _ = precision_recall_fscore_support(y_pred=y_pred_lgbm[all_index_class], y_true=label_test_set[all_index_class], average = "macro")
        accuracy = accuracy_score(y_true=label_test_set[all_index_class], y_pred=y_pred_lgbm[all_index_class])
        precision_per_class.append(precision)
        recall_per_class.append(recall)
        accuracy_per_class.append(accuracy)
    print("END CIFAR100")
    return (precision_general, recall_general, accuracy_general, precision_per_class, recall_per_class, accuracy_per_class)
