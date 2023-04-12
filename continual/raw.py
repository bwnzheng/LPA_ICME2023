import copy

import torch
from timm.models.layers import trunc_normal_
from torch import nn

import modules.utils as cutils
from . import losses

class ContinualClassifier(nn.Module):
    """Your good old classifier to do continual."""
    def __init__(self, embed_dim, nb_classes):
        super().__init__()

        self.embed_dim = embed_dim
        self.nb_classes = nb_classes
        self.head = nn.Linear(embed_dim, nb_classes, bias=True)
        self.norm = nn.LayerNorm(embed_dim)

    def reset_parameters(self):
        self.head.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, x):
        x = self.norm(x)
        return self.head(x)

    def add_new_outputs(self, n):
        head = nn.Linear(self.embed_dim, self.nb_classes + n, bias=True)
        head.weight.data[:-n] = self.head.weight.data

        head.to(self.head.weight.device)
        self.head = head
        self.nb_classes += n

class Raw(nn.Module):
    """
    :param transformer: The base transformer.
    :param nb_classes: Thhe initial number of classes.
    :param individual_classifier: Classifier config, DyTox is in `1-1`.
    """
    def __init__(
        self,
        transformer,
        nb_classes,
        individual_classifier=''
    ):
        super().__init__()

        self.nb_classes = nb_classes
        self.embed_dim = transformer.embed_dim
        self.individual_classifier = individual_classifier
        self.in_finetuning = False

        self.nb_classes_per_task = [nb_classes]

        self.patch_embed = transformer.patch_embed
        self.pos_embed = transformer.pos_embed
        self.pos_drop = transformer.pos_drop
        # self.sabs = transformer.blocks[:transformer.local_up_to_layer]

        self.transformer_blocks = transformer.blocks
        self.sabs = transformer.blocks[:transformer.local_up_to_layer]
        self.tabs = transformer.blocks[transformer.local_up_to_layer:]
        self.cls_token = transformer.cls_token

        if self.individual_classifier != '':
            in_dim, out_dim = self._get_ind_clf_dim()
            self.head = nn.ModuleList([
                ContinualClassifier(in_dim, out_dim).cuda()
            ])
        else:
            self.head = ContinualClassifier(
                self.embed_dim * len(self.task_tokens), sum(self.nb_classes_per_task)
            ).cuda()

    def end_finetuning(self):
        """Start FT mode, usually with backbone freezed and balanced classes."""
        self.in_finetuning = False

    def begin_finetuning(self):
        """End FT mode, usually with backbone freezed and balanced classes."""
        self.in_finetuning = True

    def update_model(self, nb_new_classes):
        """
        :param nb_new_classes: Number of new classes brought by the new task.
        """
        self.nb_classes_per_task.append(nb_new_classes)

        # Classifier -----------------------------------------------------------
        in_dim, out_dim = self._get_ind_clf_dim()
        self.head.append(
            ContinualClassifier(in_dim, out_dim).cuda()
        )

        # ----------------------------------------------------------------------

    def _get_ind_clf_dim(self):
        """What are the input and output dim of classifier depending on its config.

        By default, DyTox is in 1-1.
        """
        in_dim = self.embed_dim
        out_dim = self.nb_classes_per_task[-1]
        return in_dim, out_dim

    def freeze(self, names):
        """Choose what to freeze depending on the name of the module."""
        cutils.set_parameters(self, requires_grad=True)
        self.train()

        for name in names:
            if name == 'all':
                self.eval()
                return cutils.set_parameters(self)
            elif name == 'sab':
                self.sabs.eval()
                cutils.set_parameters(self.patch_embed, requires_grad=False)
                cutils.set_parameters(self.pos_embed, requires_grad=False)
                cutils.set_parameters(self.sabs, requires_grad=False)
            elif name == 'tab':
                self.tabs.eval()
                cutils.set_parameters(self.tabs, requires_grad=False)
            elif name == 'old_heads':
                self.head[:-1].eval()
                cutils.set_parameters(self.head[:-1], requires_grad=False)
            elif name == 'heads':
                self.head.eval()
                cutils.set_parameters(self.head, requires_grad=False)
            else:
                raise NotImplementedError(f'Unknown name={name}.')

    def param_groups(self):
        return {
            'all': self.parameters(),
            'cls_token': self.cls_token,
            'patch': self.patch_embed.parameters(),
            'pos': [self.pos_embed],
            'old_heads': self.head[:-self.nb_classes_per_task[-1]].parameters() \
                              if self.individual_classifier else \
                              self.head.parameters(),
            'new_head': self.head[-1].parameters() if self.individual_classifier else self.head.parameters(),
            'head': self.head.parameters()
        }

    def reset_classifier(self):
        if isinstance(self.head, nn.ModuleList):
            for head in self.head:
                head.reset_parameters()
        else:
            self.head.reset_parameters()
    
    def reset_parameters(self):

        for b in self.transformer_blocks:
            b.reset_parameters()
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.patch_embed.reset_parameters()

        self.reset_classifier()

    def hook_before_update(self):
        pass

    def hook_after_update(self):
        pass

    def hook_after_epoch(self):
        pass

    def epoch_log(self):
        """Write here whatever you want to log on the internal state of the model."""
        log = {}
        return log

    def get_internal_losses(self, clf_loss):
        """If you want to compute some internal loss, like a EWC loss for example.

        :param clf_loss: The main classification loss (if you wanted to use its gradient for example).
        :return: a dictionnary of losses, all values will be summed in the final loss.
        """
        int_losses = {}
        return int_losses

    def forward_features(self, x):
        # Shared part, this is the ENCODER
        B = x.shape[0]

        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        cls_token = self.cls_token.expand(B, -1, -1)

        for blk in self.sabs:
            x = blk(x)
        
        for blk in self.tabs:
            cls_token = blk(torch.cat((cls_token, x), dim=1))

        return cls_token # (B, 1, C)

    def forward_classifier(self, cls_token):
        cls_token = cls_token.squeeze(1)
        logits = [head(cls_token) for head in self.head]
        logits = torch.cat(logits, dim=1)
        
        return logits
        
    def forward(self, x):
        token = self.forward_features(x)
        logits = self.forward_classifier(token)

        ret = {'logits': logits}

        return ret
    
    def train_forward(self, outputs, samples, targets, tasks, task_id, teacher_model, criterion, args, **kwargs):
        logits = outputs['logits']

        if teacher_model is not None:
            with torch.no_grad():
                old_outputs = teacher_model(samples)
            old_logits = old_outputs['logits']

        loss_dict = dict()

        cls_loss = losses.cls_loss(logits, targets, criterion)
        loss_dict['cls'] = cls_loss

        if teacher_model is not None:
            distill_loss = losses.distillation_loss(logits, old_logits, tasks, task_id, cls_loss, args.auto_kd, args.distillation_tau, args.kd_factor)
            loss_dict['kd'] = distill_loss

        return loss_dict

