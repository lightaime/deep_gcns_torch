import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_kd(all_out, teacher_all_out, outputs, labels, teacher_outputs,
            alpha, temperature):
    """
    loss function for Knowledge Distillation (KD)
    """

    T = temperature

    loss_CE = F.cross_entropy(outputs, labels)
    D_KL = nn.KLDivLoss()(F.log_softmax(all_out / T, dim=1),
                          F.softmax(teacher_all_out / T, dim=1)) * (T * T)
    KD_loss =  (1. - alpha) * loss_CE + alpha * D_KL

    return KD_loss

def loss_kd_only(all_out, teacher_all_out, temperature):
    T = temperature

    D_KL = nn.KLDivLoss()(F.log_softmax(all_out / T, dim=1),
                          F.softmax(teacher_all_out / T, dim=1)) * (T * T)

    return D_KL
