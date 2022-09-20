import numpy as np
from models import *

def cwloss(output, target,confidence=50, num_classes=10):
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.sum(loss)
    return loss

def pgd(model, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init, num_classes=10):
    model.eval()
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for k in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
            if loss_fn == "cw":
                loss_adv = cwloss(output,target,num_classes=num_classes)
        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv

def eval_clean(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += nn.CrossEntropyLoss(reduction='sum')(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    log = 'Natrual Test Result ==> Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    # print(log)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def eval_robust(model, test_loader, perturb_steps, epsilon, step_size, loss_fn, category, rand_init):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.enable_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            x_adv = pgd(model,data,target,epsilon,step_size,perturb_steps,loss_fn,category,rand_init=rand_init)
            output = model(x_adv)
            test_loss += nn.CrossEntropyLoss(reduction='sum')(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    log = 'Attack Setting ==> Loss_fn:{}, Perturb steps:{}, Epsilon:{}, Step dize:{} \n Test Result ==> Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(loss_fn,perturb_steps,epsilon,step_size,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    # print(log)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def var_eval_clean(model, test_loader, num_classes):
    model.eval()
    class_loss = [0 for _ in range(num_classes)]
    class_acc = [0 for _ in range(num_classes)]
    class_total = [0 for _ in range(num_classes)]

    with torch.enable_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            loss = nn.CrossEntropyLoss(reduction='none')(output, target)
            for i in range(len(target)):
                class_total[target[i].item()] += 1
                class_loss[target[i].item()] += loss[i].item()
                if target[i].item() == pred[i].item():
                    class_acc[target[i].item()] += 1
    
    return sum(class_acc)/len(test_loader.dataset), [class_acc[i]/class_total[i] for i in range(num_classes)], sum(class_loss)/len(test_loader.dataset), [class_loss[i]/class_total[i] for i in range(num_classes)], class_total

def var_eval_robust(model, test_loader, perturb_steps, epsilon, step_size, loss_fn, category, rand_init, num_classes):
    model.eval()
    class_loss = [0 for _ in range(num_classes)]
    class_acc = [0 for _ in range(num_classes)]
    class_total = [0 for _ in range(num_classes)]

    with torch.enable_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            x_adv = pgd(model,data,target,epsilon,step_size,perturb_steps,loss_fn,category,rand_init=rand_init, num_classes=num_classes)
            output = model(x_adv)
            pred = output.max(1, keepdim=True)[1]
            loss = nn.CrossEntropyLoss(reduction='none')(output, target)
            for i in range(len(target)):
                class_total[target[i].item()] += 1
                class_loss[target[i].item()] += loss[i].item()
                if target[i].item() == pred[i].item():
                    class_acc[target[i].item()] += 1
    
    return sum(class_acc)/len(test_loader.dataset), [class_acc[i]/class_total[i] for i in range(num_classes)], sum(class_loss)/len(test_loader.dataset), [class_loss[i]/class_total[i] for i in range(num_classes)], class_total