import torch
import random
import copy
import numpy as np


def get_gradient(model, loss):
    model.zero_grad()

    loss.backward(retain_graph=True)



def set_gradient(grads, optimizer, shapes):
    for group in optimizer.param_groups:
        length = 0
        for i, p in enumerate(group['params']):
            # if p.grad is None: continue
            i_size = np.prod(shapes[i])
            get_grad = grads[length:length + i_size]
            length += i_size
            p.grad = get_grad.view(shapes[i])


def pcgrad_fn(model, losses, optimizer, mode='mean'):
    grad_list = []
    shapes = []
    shares = []
    for i, loss in enumerate(losses):
        get_gradient(model, loss)
        grads = []
        for p in model.parameters():
            if i == 0:
                shapes.append(p.shape)
            if p.grad is not None:
                grads.append(p.grad.view(-1))
            else:
                grads.append(torch.zeros_like(p).view(-1))
        new_grad = torch.cat(grads, dim=0)
        grad_list.append(new_grad)

        if shares == []:
            shares = (new_grad != 0)
        else:
            shares &= (new_grad != 0)
    #clear memory
    loss_all = 0
    for los in losses:
        loss_all += los
    loss_all.backward()
    grad_list2 = copy.deepcopy(grad_list)
    for g_i in grad_list:
        random.shuffle(grad_list2)
        for g_j in grad_list2:
            g_i_g_j = torch.dot(g_i, g_j)
            if g_i_g_j < 0:
                g_i -= (g_i_g_j) * g_j / (g_j.norm() ** 2)

    grads = torch.cat(grad_list, dim=0)
    grads = grads.view(len(losses), -1)
    if mode == 'mean':
        grads_share = grads * shares.float()

        grads_share = grads_share.mean(dim=0)
        grads_no_share = grads * (1 - shares.float())
        grads_no_share = grads_no_share.sum(dim=0)

        grads = grads_share + grads_no_share
    else:
        grads = grads.sum(dim=0)

    set_gradient(grads, optimizer, shapes)













