import copy
from functools import lru_cache

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


class DyTox(nn.Module):
    """"DyTox for the win!

    :param transformer: The base transformer.
    :param nb_classes: Thhe initial number of classes.
    :param individual_classifier: Classifier config, DyTox is in `1-1`.
    :param head_div: Whether to use the divergence head for improved diversity.
    :param head_div_mode: Use the divergence head in TRaining, FineTuning, or both.
    :param joint_tokens: Use a single TAB forward with masked attention (faster but a bit worse).
    """
    def __init__(
        self,
        transformer,
        nb_classes,
        individual_classifier='',
        head_div=False,
        head_div_mode=['tr', 'ft'],
        joint_tokens=False
    ):
        super().__init__()

        self.nb_classes = nb_classes
        self.embed_dim = transformer.embed_dim
        self.individual_classifier = individual_classifier
        self.use_head_div = head_div
        self.head_div_mode = head_div_mode
        self.head_div = None
        self.joint_tokens = joint_tokens
        self.training_phase = 'tr'

        self.nb_classes_per_task = [nb_classes]

        self.patch_embed = transformer.patch_embed
        self.pos_embed = transformer.pos_embed
        self.pos_drop = transformer.pos_drop
        self.sabs = transformer.blocks[:transformer.local_up_to_layer]

        self.tabs = transformer.blocks[transformer.local_up_to_layer:]

        self.task_tokens = nn.ParameterList([transformer.cls_token])

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
        self.training_phase = 'tr'

    def begin_finetuning(self):
        """End FT mode, usually with backbone freezed and balanced classes."""
        self.training_phase = 'ft'

    def update_model(self, nb_new_classes):
        """Expand model as per the DyTox framework given `nb_new_classes`.

        :param nb_new_classes: Number of new classes brought by the new task.
        """
        self.nb_classes_per_task.append(nb_new_classes)

        # Class tokens ---------------------------------------------------------
        new_task_token = copy.deepcopy(self.task_tokens[-1])
        trunc_normal_(new_task_token, std=.02)
        self.task_tokens.append(new_task_token)
        # ----------------------------------------------------------------------

        # Diversity head -------------------------------------------------------
        if self.use_head_div:
            if self.head_div is None:
                self.head_div = ContinualClassifier(
                    self.embed_dim, self.nb_classes_per_task[-1] + 1
                ).cuda()
            else:
                self.head_div.reset_parameters()
        # ----------------------------------------------------------------------

        # Classifier -----------------------------------------------------------
        if self.individual_classifier != '':
            in_dim, out_dim = self._get_ind_clf_dim()
            self.head.append(
                ContinualClassifier(in_dim, out_dim).cuda()
            )
        else:
            self.head = ContinualClassifier(
                self.embed_dim * len(self.task_tokens), sum(self.nb_classes_per_task)
            ).cuda()
        # ----------------------------------------------------------------------

    def _get_ind_clf_dim(self):
        """What are the input and output dim of classifier depending on its config.

        By default, DyTox is in 1-1.
        """
        if self.individual_classifier == '1-1':
            in_dim = self.embed_dim
            out_dim = self.nb_classes_per_task[-1]
        elif self.individual_classifier == '1-n':
            in_dim = self.embed_dim
            out_dim = sum(self.nb_classes_per_task)
        elif self.individual_classifier == 'n-n':
            in_dim = len(self.task_tokens) * self.embed_dim
            out_dim = sum(self.nb_classes_per_task)
        elif self.individual_classifier == 'n-1':
            in_dim = len(self.task_tokens) * self.embed_dim
            out_dim = self.nb_classes_per_task[-1]
        else:
            raise NotImplementedError(f'Unknown ind classifier {self.individual_classifier}')
        return in_dim, out_dim

    def freeze(self, names):
        """Choose what to freeze depending on the name of the module."""
        requires_grad = False
        cutils.set_parameters(self, requires_grad=not requires_grad)
        self.train()

        for name in names:
            if name == 'all':
                self.eval()
                return cutils.set_parameters(self)
            elif name == 'old_task_tokens':
                cutils.set_parameters(self.task_tokens[:-1], requires_grad=requires_grad)
            elif name == 'task_tokens':
                cutils.set_parameters(self.task_tokens, requires_grad=requires_grad)
            elif name == 'sab':
                self.sabs.eval()
                cutils.set_parameters(self.patch_embed, requires_grad=requires_grad)
                cutils.set_parameters(self.pos_embed, requires_grad=requires_grad)
                cutils.set_parameters(self.sabs, requires_grad=requires_grad)
            elif name == 'tab':
                self.tabs.eval()
                cutils.set_parameters(self.tabs, requires_grad=requires_grad)
            elif name == 'old_heads':
                self.head[:-1].eval()
                cutils.set_parameters(self.head[:-1], requires_grad=requires_grad)
            elif name == 'heads':
                self.head.eval()
                cutils.set_parameters(self.head, requires_grad=requires_grad)
            elif name == 'head_div':
                self.head_div.eval()
                cutils.set_parameters(self.head_div, requires_grad=requires_grad)
            else:
                raise NotImplementedError(f'Unknown name={name}.')

    def param_groups(self):
        return {
            'all': self.parameters(),
            'old_task_tokens': self.task_tokens[:-1],
            'task_tokens': self.task_tokens.parameters(),
            'new_task_tokens': [self.task_tokens[-1]],
            'sa': self.sabs.parameters(),
            'patch': self.patch_embed.parameters(),
            'pos': [self.pos_embed],
            'ca': self.tabs.parameters(),
            'old_heads': self.head[:-self.nb_classes_per_task[-1]].parameters() \
                              if self.individual_classifier else \
                              self.head.parameters(),
            'new_head': self.head[-1].parameters() if self.individual_classifier else self.head.parameters(),
            'head': self.head.parameters(),
            'head_div': self.head_div.parameters() if self.head_div is not None else None
        }

    def reset_classifier(self):
        if isinstance(self.head, nn.ModuleList):
            for head in self.head:
                head.reset_parameters()
        else:
            self.head.reset_parameters()

    def hook_before_update(self):
        pass

    def hook_after_update(self):
        pass

    def hook_after_epoch(self):
        pass

    def epoch_log(self):
        """Write here whatever you want to log on the internal state of the model."""
        log = {}

        # Compute mean distance between class tokens
        mean_dist, min_dist, max_dist = [], float('inf'), 0.
        with torch.no_grad():
            for i in range(len(self.task_tokens)):
                for j in range(i + 1, len(self.task_tokens)):
                    dist = torch.norm(self.task_tokens[i] - self.task_tokens[j], p=2).item()
                    mean_dist.append(dist)

                    min_dist = min(dist, min_dist)
                    max_dist = max(dist, max_dist)

        if len(mean_dist) > 0:
            mean_dist = sum(mean_dist) / len(mean_dist)
        else:
            mean_dist = 0.
            min_dist = 0.

        assert min_dist <= mean_dist <= max_dist, (min_dist, mean_dist, max_dist)
        log['token_mean_dist'] = round(mean_dist, 5)
        log['token_min_dist'] = round(min_dist, 5)
        log['token_max_dist'] = round(max_dist, 5)
        return log

    def forward_features(self, x):
        # Shared part, this is the ENCODER
        B = x.shape[0]

        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.sabs:
            x = blk(x)

        # Specific part, this is what we called the "task specific DECODER"
        if self.joint_tokens:
            return self.forward_tabs_jointtokens(x)

        tokens = []

        for task_token in self.task_tokens:
            task_token = task_token.expand(B, -1, -1)

            for blk in self.tabs:
                task_token = blk(torch.cat((task_token, x), dim=1))

            tokens.append(task_token)

        tokens = torch.cat(tokens, 1)
        return tokens # (B, T, C)

    @lru_cache
    def get_joint_mask(self, N, num_cls, device):
        M = torch.ones(num_cls, num_cls) * -torch.inf
        M[torch.eye(num_cls, dtype=int)==1] = 0
        mask = torch.cat((M, torch.zeros((num_cls, N))), dim=-1).to(device)
        return mask

    def forward_tabs_jointtokens(self, x):
        """Method to do a single TAB forward with all task tokens.
        """
        B, N, C = x.shape

        mask = self.get_joint_mask(N, len(self.task_tokens), x.device)

        task_tokens = torch.cat(
            [task_token.expand(B, 1, -1) for task_token in self.task_tokens],
            dim=1
        )

        for blk in self.tabs:
            task_tokens = blk(
                torch.cat((task_tokens, x), dim=1),
                num_ct=len(self.task_tokens),
                attn_mask=mask
            )

        return task_tokens # (B, T, C)

    def forward_classifier(self, tokens):
        logits = [head(tokens[:, i]) for i, head in enumerate(self.head)]
        logits = torch.cat(logits, dim=1)

        return logits
    
    def forward_div(self, token):
        div_logits = self.head_div(token)
        return div_logits
    
    def forward(self, x):
        tokens = self.forward_features(x) # (B, T, C)
        logits = self.forward_classifier(tokens)
        
        ret = {'logits':logits}

        if self.training and self.head_div is not None:
            logits_div = self.forward_div(tokens[:, -1])
            ret['div'] = logits_div
            if self.training_phase not in self.head_div_mode:
                ret['div'] *= 0.

        return ret

    def train_forward(self, outputs, samples, targets, tasks, task_id, teacher_model, criterion, args, **kwargs):
        logits = outputs['logits']
        div_logits = outputs.get('div')

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

        if div_logits is not None:
            div_loss = losses.divergence_loss(logits, div_logits, targets, criterion, args.head_div, kwargs['lam'])
            loss_dict['div'] = div_loss

        return loss_dict
