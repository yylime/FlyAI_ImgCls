from PIL import Image
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as T
import collections
from auto_aug import AutoAugment


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


def lin_comb(a, b, epsilon):
    return epsilon * a + b * (1 - epsilon)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon, self.reduction = epsilon, reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return lin_comb(loss / c, nll, self.epsilon)


class AMSoftmax(nn.Module):
    def __init__(self,
                 in_feats,
                 n_classes=10,
                 m=0.35,
                 s=30):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, lb):
        assert x.size()[0] == lb.size()[0]
        assert x.size()[1] == self.in_feats
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        # print(x_norm.shape, w_norm.shape, costh.shape)
        lb_view = lb.view(-1, 1)
        if lb_view.is_cuda: lb_view = lb_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, lb_view, self.m)
        if x.is_cuda: delt_costh = delt_costh.cuda()
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, lb)
        return loss, costh_m_s


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


# refer https://github.com/yatengLG/Focal-Loss-Pytorch/blob/master/Focal_Loss.py
class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds,
                                  dim=1)  # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)
        preds_logsoft = torch.log(preds_softmax)

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


'''--------------'''


def accuracy(output, target, top_k=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        max_k = max(top_k)
        batch_size = target.size(0)
        _, pred = output.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in top_k:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_tta(img, n, tta_trans):
    imgs = []
    for _ in range(n):
        imgs.append(tta_trans(img=img))
    data = torch.stack(imgs)
    return data


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def np_softmax(x):
    assert len(x.shape) == 1
    x = np.exp(x)
    z = np.sum(x)
    return x / z


def load_net(path, device, features_only=False):
    cnn = torch.load(path, map_location=device)
    # use resnet backbone  and avg pool!!!
    if features_only:
        cnn = nn.Sequential(*list(cnn.children())[:-1])
    cnn.to(device)
    cnn.eval()
    return cnn


@torch.no_grad()
def get_merge_result(models, x_data, tta=True, use_lgbm=False):
    preds = None
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    for i, name in enumerate(models):
        if not use_lgbm:
            cnn = load_net(name, device)
            outputs = cnn(x_data)
            # 先cpu,然后转numoy
            outputs = outputs.cpu()
            prediction = outputs.data.numpy()
            # 如果是tta 默认使用TTA 则需要求均值
            if tta:
                prediction = np.mean(prediction, axis=0)
            # 转化为概率
            prediction = np_softmax(prediction)
        if preds is None:
            preds = prediction
        else:
            preds += prediction
    return preds


class whResize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        # padding
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w + w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h + h_padding))

        img = img.resize(self.size, self.interpolation)

        return img


def get_trans(size=224):
    mean, std = [0.38753143, 0.36847523, 0.27735737], [0.25998375, 0.23844026, 0.2313706]
    return \
        {
            'train': T.Compose([
                T.Resize((size + 32, size + 32)),
                T.RandomChoice(
                    [T.RandomCrop(size, padding=1, pad_if_needed=True, padding_mode='edge'),
                     T.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.8, 1.2))]),
                T.RandomHorizontalFlip(),
                T.RandomAffine(degrees=32),
                T.ColorJitter(0.2, 0.2, 0.2),
                T.ToTensor(),
                T.Normalize(mean, std),
                T.RandomErasing(value='random')
            ]),
            'tta': T.Compose([
                T.Resize((size + 32, size + 32)),
                T.RandomChoice(
                    [T.RandomCrop(size, padding=1, pad_if_needed=True, padding_mode='edge'),
                     T.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.8, 1.2))]),
                T.RandomHorizontalFlip(),
                T.RandomAffine(degrees=32),
                T.ColorJitter(0.2, 0.2, 0.2),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]),
            'val': T.Compose([
                T.Resize((size, size)),
                T.ToTensor(),
                T.Normalize(mean, std)
            ]),
            'auto_aug': T.Compose([
                T.Resize((size, size)),
                T.RandomHorizontalFlip(),
                AutoAugment(dataset='CIFAR'),
                T.ToTensor(),
                T.Normalize(mean, std),
                T.RandomErasing(p=0.2, value='random')
            ]),
            'tta_auto_aug': T.Compose([
                T.Resize((size, size)),
                T.RandomHorizontalFlip(),
                AutoAugment(dataset='CIFAR'),
                T.ToTensor(),
                T.Normalize(mean, std),
            ])
        }


def label_shuffling(ps, ls):
    # ps 表示存储的位置
    # ls 表示标签
    # 原理
    # 1. 计算标签中类别最多的数量记为 k_max
    # 2. 对所有的类别，每个类别生成 k_max的排列，然后对该类别所有样本取余得到长度危k_max的过采样列表
    # 3. 对2所有的列表组合然后重排即可得到新的列表
    # 记录类别的数量
    num_class = max(ls) - min(ls) + 1
    # 第一步，分类数据，得到k——max
    counter = collections.defaultdict(list)
    for i, label in enumerate(ls):
        counter[label].append(i)
    k_max = len(max(counter.items(), key=lambda x: len(x[1]))[1])
    # 第二步采样
    new_idx = []
    for cls in range(num_class):
        permutation = np.random.permutation(k_max)
        new_idx.extend([counter[cls][i % len(counter[cls])] for i in permutation])
    # 第三步重排，可以忽略，因为在训练的时候会使用shuffle,
    assert k_max * num_class == len(new_idx)
    return [ps[i] for i in new_idx], [ls[i] for i in new_idx]
