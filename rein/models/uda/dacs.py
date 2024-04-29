

import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmseg.registry import MODELS

# from mmseg.core import add_prefix
from mmseg.models import build_segmentor
from .uda_decorator import UDADecorator, get_module
from .dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform)
# from mmseg.models.utils.visualization import subplotimg
# from mmseg.utils.utils import downscale_label_ratio
# from mmseg.models.utils.masking_transforms import BlockMaskGenerator,RandomMaskGenerator,get_mask,get_grid_mask
# from ..utils.shuffle_transforms import BlockShuffle

# from ..decode_heads.daformer_head import DAFormerHead

import torch.nn.functional as F

def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


@MODELS.register_module()
class DACS(UDADecorator):

    def __init__(self, **cfg):
        super(DACS, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.debug_img_interval = cfg['debug_img_interval']
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        assert self.mix == 'class'

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}
        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)

        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None
        

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        with torch.no_grad():
            self.get_imnet_model().eval()
            feat_imnet = self.get_imnet_model().extract_feat(img)
            feat_imnet = [f.detach() for f in feat_imnet]
        lay = -1
        if self.fdist_classes is not None:
            fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
            scale_factor = gt.shape[-1] // feat[lay].shape[-1]
            gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                self.fdist_scale_min_ratio,
                                                self.num_classes,
                                                255).long().detach()
            fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                              fdist_mask)
            self.debug_fdist_mask = fdist_mask
            self.debug_gt_rescale = gt_rescaled
        else:
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    # 获取模型预测的label
    def get_pred_label(self,img,img_metas):
        ImgPred = self.get_model().encode_decode(img, img_metas)
        PredSoftmax = torch.softmax(ImgPred.detach(), dim=1)
        _, ImgPred_label = torch.max(PredSoftmax, dim=1)
        return ImgPred_label
    
    # 训练
    def train_loss(self,log_vars, 
              img,img_metas,
              label,weight = None,
              return_feat=True,prefix="",retain_graph=False):
        
        if weight is None:
            losses = self.get_model().forward_train(
                img, img_metas, label, return_feat=return_feat)
        else:
            losses = self.get_model().forward_train(
                img, img_metas, label, weight,return_feat=return_feat)
        src_feat = losses.pop('features')
        if prefix != "":
            losses = add_prefix(losses, prefix)
        loss, log = self._parse_losses(losses)
        # print(loss)
        log_vars.update(log)
        if retain_graph == True:
            loss.backward(retain_graph=True)
        else:
            loss.backward()
        return src_feat

    def forward_train(self, img, img_metas, gt_semantic_seg, target_img,
                      target_img_metas):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        log_vars = {}
        batch_size,_,W,H = img.shape
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

        means, stds = get_mean_std(img_metas, dev)

        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }


        # 训练 source image
        # mask one class
        """with torch.no_grad():
            src_img = []
            for i in range(batch_size):
                src_labels = torch.unique(gt_semantic_seg[i])
                src_labels = src_labels[src_labels != 255]
                choice = random.choice(src_labels)
                src_mask = gt_semantic_seg[i] == choice
                # src_img.append(img[i] * src_mask)
                img_copy = img[i].clone()
                img_copy[ src_mask.expand(3, -1, -1) ] = -2  # 将 mask 掉的部分设置为黑色，之前是设置为灰色
                src_img.append(img_copy)"""
        
        # mask two class
        src_mask_list = []
        with torch.no_grad():
            src_img = []
            for i in range(batch_size):
                src_labels = torch.unique(gt_semantic_seg[i])
                src_labels = src_labels[src_labels != 255]
                if len(src_labels) < 2:
                    choice = [255,255]
                else:
                    choice = random.sample(src_labels.tolist(), 2)
                src_mask =  torch.logical_or(gt_semantic_seg[i] == choice[0], gt_semantic_seg[i] == choice[1])  

                img_copy = img[i].clone()
                img_copy[ src_mask.expand(3, -1, -1) ] = -2  # 将 mask 掉的部分设置为黑色，之前是设置为灰色
                src_img.append(img_copy)
                src_mask_list.append(src_mask)

        src_img = torch.stack(src_img)

        src_feat = self.train_loss(log_vars = log_vars,
                   img = src_img,img_metas = img_metas,
                   label = gt_semantic_seg,
                   retain_graph=self.enable_fdist)

        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,src_feat)
            feat_loss.backward()
            log_vars.update(add_prefix(feat_log, 'src'))
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')

        # Generate pseudo-label
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        
        with torch.no_grad():
            ema_logits,target_feat = self.get_ema_model().encode_decode(
                target_img, target_img_metas,return_feat=True)

            ema_softmax = torch.softmax(ema_logits, dim=1)
            pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
            ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
            ps_size = np.size(np.array(pseudo_label.cpu()))
            ps_large_p_ratio = torch.sum(ps_large_p).item() / ps_size
            pseudo_weight = ps_large_p_ratio * torch.ones(
                pseudo_prob.shape, device=dev)
            gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

        if self.psweight_ignore_top > 0:
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        

        # gt_semantic_seg:   [batchsize,1,512,512]
        # pseudo_label:  [batchsize,512,512]
        # Apply mixing
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mix_masks = get_class_masks(gt_semantic_seg)
        mix_weight = torch.ones_like(pseudo_weight)

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_img[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
            _, mix_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)

        # 训练 source mixed image
        self.train_loss(log_vars = log_vars,
                   img = mixed_img,
                   img_metas = img_metas,
                   label = mixed_lbl,
                   weight = mix_weight,
                   prefix="mix_s2t")  
         
       
        
       
        
        # 绘制用于debug的图片
        with torch.no_grad():
            if self.local_iter % self.debug_img_interval == 0:
                out_dir = os.path.join(self.train_cfg['work_dir'],'class_mix_debug')
                os.makedirs(out_dir, exist_ok=True)
                
                # 可视化图片
                vis_img = torch.clamp(denorm(src_img, means, stds), 0, 1)
                vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
                vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
                # vis_shuffle_img = torch.clamp( denorm(shuffle_img, means, stds), 0, 1)

                # 可视化模型预测的结果
                vis_src_pred = self.get_pred_label(src_img, img_metas)
                vis_mixed_pred = self.get_pred_label(mixed_img, img_metas)
                # vis_shuffle_pred = self.get_pred_label(shuffle_img,img_metas)


                for j in range(batch_size):
                    rows, cols = 4, 5
                    fig, axs = plt.subplots(rows,cols,figsize=(3 * cols, 3 * rows),gridspec_kw={'hspace': 0.1,'wspace': 0,'top': 0.95,'bottom': 0,'right': 1,'left': 0},)
                    
                    subplotimg(axs[0][0], vis_img[j], 'Source Image')
                    subplotimg(axs[0][1],gt_semantic_seg[j],'Source Seg GT',cmap='cityscapes')
                    subplotimg(axs[0][2], src_mask_list[j][0], 'Domain Mask', cmap='gray')

                    # subplotimg(axs[1][0],vis_shuffle_img[j],'Shuffle Img')
                    # subplotimg(axs[1][3],vis_shuffle_pred[j],'Shuffle Img Pred',cmap='cityscapes')
                    # subplotimg(axs[1][1],shuffle_label[j],'Shuffle Pseudolabel',cmap='cityscapes')



                    subplotimg(axs[1][2], mix_masks[j][0], 'Domain Mask', cmap='gray')
                    subplotimg(axs[1][4],vis_src_pred[j],'Source Img Pred',cmap='cityscapes')
                    

                    subplotimg(axs[2][0], vis_trg_img[j], 'Target Image')
                    subplotimg(axs[2][1],pseudo_label[j],'Target Seg (Pseudo) GT',cmap='cityscapes')
                    subplotimg(axs[2][2], vis_mixed_img[j], 'Mixed Image')
                    subplotimg(axs[2][3], mixed_lbl[j], 'Seg Targ', cmap='cityscapes')
                    subplotimg(axs[2][4],vis_mixed_pred[j],'Mixed Img Pred',cmap='cityscapes')
                   



                    # 显示每个pseudo的softmax的概率
                    subplotimg(axs[3][0],pseudo_prob[j] ,'Pselabel SoftmaxPro',cmap='hot', interpolation='nearest')
                    # 显示大于pseudo域值的像素点有哪些
                    subplotimg(axs[3][1],ps_large_p[j].int() ,'Pselabel mask',cmap='gray')
                    # 计算熵
                    log_probabilities = torch.log2(ema_softmax[j])
                    entropy_map = -torch.sum(ema_softmax[j] * log_probabilities, dim=0)
                    subplotimg(axs[3][2],entropy_map ,'Pseudo label entropy',cmap='hot')


                    # 显示伪标签的熵
                    for ax in axs.flat:
                        ax.axis('off')
                    plt.savefig(
                        os.path.join(out_dir,
                                    f'{(self.local_iter + 1):06d}_{j}.png'))
                    plt.close()
        self.local_iter += 1

        return log_vars
