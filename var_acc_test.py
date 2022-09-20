import os
import argparse
import torchvision
import torch.optim as optim
from torchvision import transforms
import datetime
from models import *
import numpy as np
from utils import print_args
import attack_generator as attack
import time
import pandas as pd

parser = argparse.ArgumentParser(description='PyTorch PGD Adversarial Training')
parser.add_argument('--attack_type', default='pgd20', type=str, help='fgsm/pgd20/cw')
parser.add_argument('--num_steps', type=int, default=4, help='for step size')
parser.add_argument('--seed', type=int, default=7, metavar='S', help='random seed')
parser.add_argument('--dataset', type=str, default="cifar10", help="choose from cifar10,svhn")
parser.add_argument('--rand_init', type=bool, default=True, help="whether to initialize adversarial sample with random noise")
parser.add_argument('--omega', type=float, default=0.001, help="random sample parameter for adv data generation")
parser.add_argument('--name', default='', type=str, help='name of run')
parser.add_argument('--test_batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--dir_path', default='./test_models', type=str)

args = parser.parse_args()
print_args(args)
print(args.attack_type + '_' + args.dataset + '_acc_var.csv')

eps = [16/255, 20/255]
name_eps = ['16/255', '20/255']
dir_path = args.dir_path

# training settings
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

filename_list = []
for i in os.listdir(dir_path):
    filename_list.append(i)

filename_list = [i for i in filename_list if i.split('.')[1] == 'pkl']
filename_list.sort()
print(filename_list)

def cifar100_mapping():
    import json
    with open("cifar100_super_map.json") as json_map:
        tem_map = json.load(json_map)
    ans = {}
    for i in range(100):
        ans[i] = tem_map[str(i)]
    return ans

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

num_classes = 10
print('==> Load Test Data')
if args.dataset == "cifar10":
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship','truck']
if args.dataset == "svhn":
    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
    labels = [str(i) for i in range(10)]
if args.dataset == "cifar100":
    num_classes = 100
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
    labels = [str(i) for i in range(100)]
    super_map = cifar100_mapping()

df = pd.DataFrame(columns=['attack_eps', 'model', 'acc_var', 'total_acc', 'loss_var','loss_mean'] + labels)

for pkl_iter in range(len(filename_list)):
    pd_iter = pkl_iter
    pkl_name = filename_list[pkl_iter]

    print('==> Building model: ', pkl_name)
    model = torch.load(os.path.join(dir_path, pkl_name))
    model.cuda()

    total_acc, class_acc, total_loss, class_loss, class_total = attack.var_eval_clean(model, test_loader, num_classes=num_classes)
    for i,j in enumerate(labels):
        df.loc[pd_iter, j] = class_acc[i]
    df.loc[pd_iter,'total_acc'] = np.mean(class_acc)
    df.loc[pd_iter,'attack_eps'] = '0'
    df.loc[pd_iter,'loss_var'] = np.var(class_loss)
    df.loc[pd_iter,'acc_var'] = np.var(class_acc)
    df.loc[pd_iter,'loss_mean'] = total_loss
    df.loc[pd_iter,'model'] = pkl_name.split('.')[0] 
    print(f'clean     acc_var: {np.var(class_acc)}  loss_var: {np.var(class_loss)}')
    df.to_csv(os.path.join(dir_path, args.attack_type + '_' + args.dataset + '_acc_var.csv'))

    for eps_iter in range(len(eps)):
        pd_iter = len(filename_list) * (eps_iter+1) + pkl_iter

        if args.attack_type == 'pgd20':
            total_acc, class_acc, total_loss, class_loss, class_total = attack.var_eval_robust(model, test_loader, perturb_steps=20, epsilon=eps[eps_iter], step_size=eps[eps_iter]/args.num_steps, loss_fn="cent", category="Madry", rand_init=args.rand_init, num_classes=num_classes)
        elif args.attack_type == 'fgsm':
            total_acc, class_acc, total_loss, class_loss, class_total = attack.var_eval_robust(model, test_loader, perturb_steps=1, epsilon=eps[eps_iter], step_size=eps[eps_iter], loss_fn="cent", category="Madry", rand_init=args.rand_init, num_classes=num_classes)
        elif args.attack_type == 'cw':
            total_acc, class_acc, total_loss, class_loss, class_total = attack.var_eval_robust(model, test_loader, perturb_steps=30, epsilon=eps[eps_iter], step_size=eps[eps_iter]/args.num_steps, loss_fn="cw", category="Madry", rand_init=args.rand_init, num_classes=num_classes)

        for i,j in enumerate(labels):
            df.loc[pd_iter, j] = class_acc[i]
        df.loc[pd_iter,'total_acc'] = np.mean(class_acc)
        df.loc[pd_iter,'attack_eps'] = name_eps[eps_iter]
        df.loc[pd_iter,'loss_var'] = np.var(class_loss)
        df.loc[pd_iter,'loss_mean'] = total_loss
        df.loc[pd_iter,'acc_var'] = np.var(class_acc)
        df.loc[pd_iter,'model'] = pkl_name.split('.')[0] 

        print(f'adv {name_eps[eps_iter]}     acc_var: {np.var(class_acc)}  loss_var: {np.var(class_loss)}')
        df.to_csv(os.path.join(dir_path, args.attack_type + '_' + args.dataset + '_acc_var.csv'))

df.sort_index(inplace=True)
df.to_csv(os.path.join(dir_path, args.attack_type + '_' + args.dataset + '_acc_var.csv'))