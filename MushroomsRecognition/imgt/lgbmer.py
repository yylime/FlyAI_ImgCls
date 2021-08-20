# lightgbm to cnn
# created by yyl
# time: 2021-07-27
import lightgbm as lgb
import numpy as np
import torch
from torch import nn
import os


def load_net(path, device):
    cnn = torch.load(path, map_location=device)
    # use resnet backbone  and avg pool!!!
    cnn = nn.Sequential(*list(cnn.children())[:-1])
    cnn.to(device)
    cnn.eval()
    return cnn


class Lgbmer:

    def __init__(self, net_path, train_dataloader, valid_dataloader, cfg):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.outputs = cfg.outputs
        self.lgbm_params = cfg.lgbm_params
        # load net
        self.net = load_net(os.path.join(self.outputs, net_path), self.device)
        # extra features
        self.train_x, self.train_y = self.extra_features(train_dataloader)
        self.valid_x, self.valid_y = self.extra_features(valid_dataloader)
        print("Net's features has been loaded")

    @torch.no_grad()
    def extra_features(self, data_loader):
        all_features, all_labels = [], []
        for i, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            features = self.net(inputs)
            all_features.append(features.cpu().data.numpy())
            all_labels.append(targets.cpu().data.numpy())
        # fix dim
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_features = np.reshape(all_features, newshape=(all_features.shape[0], -1))
        all_labels = np.reshape(all_labels, newshape=(all_labels.shape[0],))
        return all_features, all_labels

    def train(self, lgbm_path):
        train_set = lgb.Dataset(self.train_x, self.train_y)
        val_set = lgb.Dataset(self.valid_x, self.valid_y)
        model = lgb.train(self.lgbm_params, train_set, valid_sets=[train_set, val_set], verbose_eval=10)
        model.save_model(os.path.join(self.outputs, lgbm_path))

    def predict(self):
        pass


if __name__ == '__main__':
    from torch import nn

    net_path, lgbm_path = '../data/output/model/resnext50_32x4d.pkl', '../data/output/model/wide_resnet50.txt'
    device = torch.device('cpu')
    cnn = torch.load(net_path, map_location=device)
    cnn.to(device)
    cnn.eval()
    print(cnn)
    x = torch.randn((1, 3, 224, 224))
    y = nn.Sequential(*list(cnn.children())[:-1])(x)
    features = y.cpu().data.numpy()
    print(features.shape)
    fff = np.concatenate([features, features], axis=0)
    fff = np.reshape(fff, newshape=(fff.shape[0], -1))
    print(fff.shape)
    model = lgb.Booster(model_file=lgbm_path)
    print(model.predict(fff))

    params = {'boosting_type': 'gbdt',
              'n_estimators': 3000,
              'objective': 'multiclass',
              'num_class': 9,
              'learning_rate': 0.01,
              #           'bagging_fraction':0.8,
              #           'class_weight':'balanced',
              'early_stopping_rounds': 200,
              'metric': {'auc'},
              #           'reg_alpha' : 0.01,
              #         'reg_lambda' :0.1,
              }
