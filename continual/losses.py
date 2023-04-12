import torch
from torch.nn import functional as F


def cls_loss(logits, targets, criterion):
    return criterion(logits, targets)

def distillation_loss(logits, old_logits, tasks, task_id, cls_loss, auto_kd: bool, tau, kd_factor=1.):
    mask = tasks < task_id
    if not mask.any():
        return torch.tensor(0., device=logits.device)
    logits_for_distill = logits[:, :old_logits.shape[1]]

    if auto_kd:
        lbd = old_logits.shape[1] / logits.shape[1]
        cls_loss.mul_(1 - lbd)
        kd_factor = lbd

    kd_loss = F.kl_div(
            F.log_softmax(logits_for_distill[mask] / tau, dim=1),
            F.log_softmax(old_logits[mask] / tau, dim=1),
            reduction='mean', # do not change to batchmean, it acts differently
            log_target=True
    ) * (tau ** 2) * kd_factor

    return kd_loss


def divergence_loss(logits, div_logits, targets, criterion, div_factor, lam=None):
    nb_classes = logits.shape[1]
    nb_new_classes = div_logits.shape[1] - 1
    nb_old_classes = nb_classes - nb_new_classes

    if lam is not None:  # 'lam' is the interpolation Lambda of mixup
        # If using mixup / cutmix
        div_targets = torch.zeros_like(div_logits)
        nb_classes = logits.shape[1]
        nb_new_classes = div_logits.shape[1] - 1
        nb_old_classes = nb_classes - nb_new_classes

        div_targets[:, 0] = targets[:, :nb_old_classes].sum(-1)
        div_targets[:, 1:] = targets[:, nb_old_classes:]
    else:
        div_targets = torch.clone(targets)
        mask_old_cls = div_targets < nb_old_classes
        mask_new_cls = ~mask_old_cls

        div_targets[mask_old_cls] = 0
        div_targets[mask_new_cls] -= nb_old_classes - 1

    div_loss = div_factor * criterion(div_logits, div_targets)
    return div_loss

def relation_distill_loss(query_feats, key_feats, old_query_feats, old_key_feats, tasks, key_tasks, task_id, logits, old_logits, factor):
    distill_loss = 0.

    mask = tasks < task_id
    key_mask = key_tasks < task_id
    if not mask.any() or not key_mask.any():
        return torch.tensor(0., device=logits.device)
    
    block_sim = _calc_similarity_graph(query_feats, key_feats, query_mask=mask, key_mask=key_mask)
    old_block_sim = _calc_similarity_graph(old_query_feats, old_key_feats, query_mask=mask, key_mask=key_mask)

    lbd = old_logits.shape[1] / logits.shape[1]
    for i in range(0, len(block_sim)):
        distill_loss += F.kl_div(block_sim[i].log(), old_block_sim[i].detach(), reduction='mean') * lbd
        # distill_loss_array = F.kl_div(block_sim[i].log(), old_block_sim[i].detach(), reduction='none').sum(-1)
        # distill_loss_array[tasks < task_id] *= lbd
        # distill_loss_array[tasks == task_id] *= 1 - lbd # should only distill old sample relations
        # distill_loss += distill_loss_array.mean(0)

    return distill_loss * factor


def back_distill_loss(query_feats, key_feats, tasks, task_id, logits, old_logits, factor):
    # distill with last layer sim
    distill_loss = 0.
    block_sim = _calc_similarity_graph(query_feats, key_feats)
    
    if old_logits is not None:
        lbd = old_logits.shape[1] / logits.shape[1]
        for i in range(1, len(block_sim)):
            distill_loss_array = F.kl_div(block_sim[i].log(), block_sim[i-1].detach(), reduction='none').sum(-1)
            distill_loss_array[tasks < task_id] *= lbd
            distill_loss_array[tasks == task_id] *= 1 - lbd
            distill_loss += distill_loss_array.mean(0)
    else:
        for i in range(1, len(block_sim)):
            distill_loss += F.kl_div(block_sim[i].log(), block_sim[i-1].detach(), reduction='batchmean')
        
    return distill_loss * factor


def back_distill_loss_v2(query_feats, key_feats, tasks, task_id, logits, old_logits, factor):
    # all distill with base sim
    distill_loss = 0.
    block_sim = _calc_similarity_graph(query_feats, key_feats)
    
    base_sim = block_sim[0].detach()
    if old_logits is not None:
        lbd = old_logits.shape[1] / logits.shape[1]
        for i in range(1, len(block_sim)):
            distill_loss_array = F.kl_div(block_sim[i].log(), base_sim, reduction='none').sum(-1)
            distill_loss_array[tasks < task_id] *= lbd
            distill_loss_array[tasks == task_id] *= 1 - lbd
            distill_loss += distill_loss_array.mean(0)
    else:
        for i in range(1, len(block_sim)):
            distill_loss += F.kl_div(block_sim[i].log(), base_sim, reduction='batchmean')
        
    return distill_loss * factor

def forward_distill_loss(query_feats, key_feats, tasks, task_id, logits, old_logits, factor):
    distill_loss = 0.
    block_sim = _calc_similarity_graph(query_feats, key_feats)
    
    if old_logits is not None:
        lbd = old_logits.shape[1] / logits.shape[1]
        for i in range(1, len(block_sim)):
            distill_loss_array = F.kl_div(block_sim[i-1].log(), block_sim[i].detach(), reduction='none').sum(-1)
            distill_loss_array[tasks < task_id] *= lbd
            distill_loss_array[tasks == task_id] *= 1 - lbd
            distill_loss += distill_loss_array.mean(0)
    else:
        for i in range(1, len(block_sim)):
            distill_loss += F.kl_div(block_sim[i-1].log(), block_sim[i].detach(), reduction='batchmean')
        
    return distill_loss * factor


# def _calc_similarity_graph(query_feats, key_feats, query_mask=None, key_mask=None):
#     # calc similarity graph for each block feat
#     # query and key feats are shaped like list of (B_q, N, C) and (B_k, N, C)
#     block_sim = []
#     for feat_q, feat_k in zip(query_feats, key_feats):
#         if query_mask is not None and key_mask is not None:
#             feat_q = feat_q[query_mask]
#             feat_k = feat_k[key_mask]
        
#         Bk, N, C = feat_k.shape
#         Bq, _, _ = feat_q.shape
#         gfeat_q = feat_q.mean(1) # global query feat, avg across patches
#         gfeat_q = gfeat_q - gfeat_q.mean(0) # zero centered global query feat, avg across the batch
#         gfeat_k = feat_k.mean(1) 
#         gfeat_k = gfeat_k - gfeat_k.mean(0)

#         # gfeat = gfeat / gfeat.norm(dim=-1, keepdim=True) # normalization
#         sim = (gfeat_q @ gfeat_k.T / Bq).softmax(-1)
#         block_sim.append(sim)
    
#     return block_sim

def _calc_similarity_graph(query_feats, key_feats, query_mask=None, key_mask=None):
    # calc similarity graph for each block feat
    # query and key feats are shaped like list of (B_q, N, C) and (B_k, N, C)
    block_sim = []
    for feat_q, feat_k in zip(query_feats, key_feats):
        if query_mask is not None and key_mask is not None:
            feat_q = feat_q[query_mask]
            feat_k = feat_k[key_mask]
        
        Bk, N, C = feat_k.shape
        Bq, _, _ = feat_q.shape
        # gfeat_q = feat_q.mean(1) # global query feat, avg across patches
        gfeat_q = feat_q
        gfeat_q = gfeat_q - gfeat_q.mean((0,1)) # zero centered global query feat, avg across the batch
        # gfeat_k = feat_k.mean(1)
        gfeat_k = feat_k
        gfeat_k = gfeat_k - gfeat_k.mean((0,1))

        # gfeat = gfeat / gfeat.norm(dim=-1, keepdim=True) # normalization
        # sim = (gfeat_q @ gfeat_k.T / Bq).softmax(-1)
        sim = (gfeat_q.permute(1, 0, 2) @ gfeat_k.permute(1, 2, 0) / Bq).softmax(-1).mean(0)
        block_sim.append(sim)
    
    return block_sim