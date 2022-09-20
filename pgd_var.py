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
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch PGD Adversarial Training')
parser.add_argument('--epochs', type=int, default=120, metavar='N', help='number of epochs to train')
parser.add_argument('--alpha', type=float, default=0.1, help="weight of regularization")
parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--epsilon', type=float, default=0.031, help='perturbation bound')
parser.add_argument('--num_steps', type=int, default=10, help='maximum perturbation step K')
parser.add_argument('--step_size', type=float, default=0.007, help='step size')
parser.add_argument('--seed', type=int, default=7, metavar='S', help='random seed')
parser.add_argument('--net', type=str, default="WRN_madry",
                    help="decide which network to use,choose from smallcnn,resnet18,WRN")
parser.add_argument('--dataset', type=str, default="cifar10", help="choose from cifar10,svhn")
parser.add_argument('--rand_init', type=bool, default=True, help="whether to initialize adversarial sample with random noise")
parser.add_argument('--omega', type=float, default=0.001, help="random sample parameter for adv data generation")
parser.add_argument('--depth', type=int, default=32, help='WRN depth')
parser.add_argument('--width_factor', type=int, default=10, help='WRN width factor')
parser.add_argument('--drop_rate', type=float, default=0.0, help='WRN drop rate')
parser.add_argument('--name', default='', type=str, help='name of run')
parser.add_argument('--resume', type=str, default='', help='whether to resume training, default: None')
parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
parser.add_argument('--test_batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--test_epsilon', type=float, default=0.062)
parser.add_argument('--superclass', dest='superclass', action='store_true', default=False, help="whether to use superclass for regularization on cifar100")
parser.add_argument('--no_regular', dest='no_regular', action='store_true', default=False, help='whether do regularization')


args = parser.parse_args()

# training settings
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
if args.no_regular:
    args.alpha = 0

exp_id = print_args(args)
out_dir = f'./Results/PGD/{args.dataset}/{args.net}/{time.strftime("%m%d")}_{time.strftime("%H%M", time.localtime())}_seed{args.seed}_'
if args.name != '':
    out_dir += (args.name + '_')
if args.dataset == "cifar100":
    out_dir += f'alpha{str(args.alpha)}_eps{args.epsilon}_super_{args.superclass}_lr{args.lr}_batchsize{args.batch_size}_epoch{args.epochs}'
else:
    out_dir += f'alpha{str(args.alpha)}_eps{args.epsilon}_lr{args.lr}_batchsize{args.batch_size}_epoch{args.epochs}'
writer = SummaryWriter(log_dir=out_dir)

def train(model, train_loader, optimizer):
    starttime = datetime.datetime.now()
    loss_sum = 0
    var_sum = 0
    batch_num = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_num += 1
        data, target = data.cuda(), target.cuda()

        # Get adversarial training data
        x_adv = attack.pgd(model, data, target, epsilon=args.epsilon, step_size=args.step_size, num_steps=args.num_steps, loss_fn='cent', category="Madry", rand_init=args.rand_init)

        model.train()
        optimizer.zero_grad()
        output = model(x_adv)

        if not args.no_regular:
            # calculate standard adversarial training loss
            loss = nn.CrossEntropyLoss(reduction='none')(output, target)
            loss_list = torch.zeros(100).cuda()
            
            for i in range(len(labels)):
                mask = target.eq(i * torch.ones(target.shape).int().cuda())
                loss_c = torch.mean(loss[mask])
                if not torch.isnan(loss_c):
                    loss_list[i] += loss_c

            if args.dataset == "cifar100" and args.superclass:
                cifar100_loss = torch.zeros(20).cuda()
                for i in range(100):
                    cifar100_loss[super_map[i]] += loss_list[i]
                cifar100_loss /= 5
                loss_var = torch.var(cifar100_loss)
            else:
                loss_var = torch.var(loss_list)

            var_sum += loss_var.item()
            final_loss = torch.mean(loss) + float(args.alpha) * loss_var
        else:
            final_loss = nn.CrossEntropyLoss(reduction='mean')(output, target)

        loss_sum += final_loss.item()
        final_loss.backward()
        optimizer.step()

    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    return time, loss_sum/batch_num, var_sum/batch_num


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 60:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 110:
        lr = args.lr * 0.005
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(model, state, checkpoint=out_dir, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(checkpoint, filename))
    torch.save(model, os.path.join(checkpoint, 'model.pkl'))


def cifar100_mapping():
    import json
    with open("cifar100_super_map.json") as json_map:
        tem_map = json.load(json_map)
    ans = {}
    for i in range(100):
        ans[i] = tem_map[str(i)]
    return ans


# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

num_classes = 10
print('==> Load Test Data')
if args.dataset == "cifar10":
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship','truck']
if args.dataset == "svhn":
    trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
    labels = [str(i) for i in range(10)]
if args.dataset == "cifar100":
    num_classes = 100
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
    labels = [str(i) for i in range(100)]
    super_map = cifar100_mapping()

print('==> Load Model')
if args.net == "smallcnn":
    model = SmallCNN().cuda()
    net = "smallcnn"
if args.net == "resnet18":
    model = ResNet18(num_classes).cuda()
    net = "resnet18"
if args.net == "resnet50":
    model = ResNet50(num_classes).cuda()
    net = "resnet50"
if args.net == "WRN":
  # e.g., WRN-34-10
    model = Wide_ResNet(depth=args.depth, num_classes=num_classes, widen_factor=args.width_factor, dropRate=args.drop_rate).cuda()
    net = "WRN{}-{}-dropout{}".format(args.depth, args.width_factor, args.drop_rate)
if args.net == 'WRN_madry':
  # e.g., WRN-32-10
    model = Wide_ResNet_Madry(depth=args.depth, num_classes=num_classes, widen_factor=args.width_factor, dropRate=args.drop_rate).cuda()
    net = "WRN_madry{}-{}-dropout{}".format(args.depth, args.width_factor, args.drop_rate)
print(net)

model = torch.nn.DataParallel(model)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

start_epoch = 0
# Resume
if args.resume:
    print('==> PGD Adversarial Training Resuming from checkpoint ..')
    print(args.resume)
    checkpoint_file_path = os.path.join(args.resume, 'checkpoint.pth.tar')
    assert os.path.isfile(checkpoint_file_path)
    out_dir = args.resume
    checkpoint = torch.load(checkpoint_file_path)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    logger_test = pd.read_csv(os.path.join(args.resume, 'log.csv'), index_col=0)
else:
    print('==> PGD Adversarial Training')
    logger_test = pd.DataFrame(columns=['Epoch', 'Natural Test Acc', 'FGSM Acc', 'PGD20 Acc', 'CW Acc', exp_id])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

test_nat_acc = 0
fgsm_acc = 0
test_pgd20_acc = 0
cw_acc = 0
best_epoch = 0


for epoch in range(start_epoch, args.epochs):
    adjust_learning_rate(optimizer, epoch + 1)
    train_time, train_loss, train_loss_var = train(model, train_loader, optimizer)
    writer.add_scalar('train_loss', train_loss, epoch)
    writer.add_scalar('train_loss_var', train_loss_var, epoch)
    writer.add_scalar('train_time', train_time, epoch)

    ## Evalutions
    total_acc, class_acc, total_loss, class_loss, class_total = attack.var_eval_clean(model, test_loader, num_classes=num_classes)
    test_nat_acc = np.mean(class_acc)
    writer.add_scalar('clean_total_loss', total_loss, epoch)
    writer.add_scalar('clean_loss_var', np.var(class_loss), epoch)
    writer.add_scalar('clean_total_acc', test_nat_acc, epoch)
    writer.add_scalar('clean_acc_var', np.var(class_acc), epoch)
    for i,j in enumerate(labels):
        writer.add_scalar(f'clean_loss/{j}', class_loss[i], epoch)
        writer.add_scalar(f'clean_acc/{j}', class_acc[i], epoch)

    # FGSM
    total_acc, class_acc, total_loss, class_loss, class_total = attack.var_eval_robust(model, test_loader, perturb_steps=1, epsilon=float(args.test_epsilon), step_size=float(args.test_epsilon), loss_fn="cent", category="Madry", rand_init=args.rand_init, num_classes=num_classes)
    fgsm_acc = np.mean(class_acc)
    writer.add_scalar('adv_total_loss', total_loss, epoch)
    writer.add_scalar('adv_loss_var', np.var(class_loss), epoch)
    writer.add_scalar('adv_total_acc', fgsm_acc, epoch)
    writer.add_scalar('adv_acc_var', np.var(class_acc), epoch)
    for i,j in enumerate(labels):
        writer.add_scalar(f'adv_loss/{j}', class_loss[i], epoch)
        writer.add_scalar(f'adv_acc/{j}', class_acc[i], epoch)

    print(
        'Epoch: [%d | %d] | Train Time: %.2f s | Natural Test Acc %.2f | FGSM Test Acc %.2f | FGSM Test Acc Var %.2f |\n' % (
        epoch + 1,
        args.epochs,
        train_time,
        test_nat_acc,
        fgsm_acc,
        np.var(class_acc))
        )

    logger_test.loc[len(logger_test)] = [epoch + 1, test_nat_acc, fgsm_acc, test_pgd20_acc, cw_acc, '']
    logger_test.to_csv(os.path.join(out_dir, 'log.csv'))

    save_checkpoint(model, {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'test_nat_acc': test_nat_acc,
        'optimizer': optimizer.state_dict(),
    })
