import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoder import MetricNN, LinearNN, GraphNN
from .layers import grad_reverse

def mse_loss(input, target):
    return torch.sum((input - target)**2) / input.data.nelement()


def l1_loss(input, target):
    return torch.sum(torch.abs(input - target)) / input.data.nelement()


def cosine_similarity(input, target):
    cosine_sim = input.unsqueeze(1).bmm(target.unsqueeze(2)).squeeze()
    norm_i = input.norm(p=2, dim=1)
    norm_t = target.norm(p=2, dim=1)
    return cosine_sim / (norm_i * norm_t)

def cosine_loss(input, target):
    cosine_sim = cosine_similarity(input, target)
    cosine_dist = (1 - cosine_sim) / 2
    return cosine_dist


class SemanticLoss(nn.Module):
    def __init__(self, input_size=256):
        super(SemanticLoss,self).__init__()
        self.input_size = input_size
        #self.map = nn.Linear(self.input_size, 300)
        self.map = nn.Sequential(
            nn.Linear(self.input_size,300),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(300, 300),
        )

    def forward(self, input, target):
        input = self.map(input)
        return cosine_loss(input, target)


class DomainLoss(nn.Module):
    def __init__(self, input_size=256, hidden_size=64):
        super(DomainLoss, self).__init__()
        self.input_size = input_size
        # self.map = nn.Linear(self.input_size, 1)
        self.map = nn.Sequential(
            nn.Linear(self.input_size,hidden_size),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size,1),
        )

    def forward(self, input, target):
        input = self.map(input)
        input = F.sigmoid(input).squeeze()
        return F.binary_cross_entropy(input, target)


class DetangledJoinDomainLoss(nn.Module):
    def __init__(self, emb_size=256, w_sem=0.25, w_dom=0.25, w_spa=0.25, lambd=0.5):
        super(DetangledJoinDomainLoss, self).__init__()
        
        self.emb_size = emb_size
        
        self.w_sem = w_sem
        self.w_dom = w_dom
        self.w_spa = w_spa

        self.semantic_loss = SemanticLoss(input_size=int(self.emb_size))
        self.lambd=lambd

        self.domain_loss_mu = DomainLoss(input_size=int(self.emb_size))
        
        self.space_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    def forward(self, im_pos_sem, sk_sem, w2v, im_neg_sem, epoch):
            
        # Semantic Loss
        loss_sem = self.semantic_loss(sk_sem, w2v) + self.semantic_loss(im_pos_sem, w2v) + self.semantic_loss(grad_reverse(im_neg_sem, lambd=self.lambd), w2v)
        loss_sem = loss_sem.mean()
       
        # Space Loss
        loss_spa = self.space_loss(sk_sem, im_pos_sem, im_neg_sem)

        # Domain Loss
        bz = sk_sem.size(0)
        targetSK = torch.zeros(bz)
        targetIM = torch.ones(bz)
        if sk_sem.is_cuda:
            targetSK = targetSK.cuda()
            targetIM = targetIM.cuda()

        if epoch > 25:
            lmb = 1.0
        elif epoch < 5:
            lmb = 0
        else:
            lmb = (epoch-5)/20.0
        loss_dom = self.domain_loss_mu(grad_reverse(sk_sem, lambd=lmb), targetSK)
        loss_dom += self.domain_loss_mu(grad_reverse(im_pos_sem, lambd=lmb), targetIM)
        loss_dom += self.domain_loss_mu(grad_reverse(im_neg_sem, lambd=lmb), targetIM)
        loss_dom = loss_dom/3.0
        # Weighted Loss
        loss = self.w_sem*loss_sem + self.w_dom* loss_dom + self.w_spa*loss_spa

        return loss, loss_sem, loss_dom, loss_spa


