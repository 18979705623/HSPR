import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import BertPreTrainedModel

from .vilmodel import BertLayerNorm, BertOnlyMLMHead, GlocalTextPathCMT
from .ops import pad_tensors_wgrad, gen_seq_masks, try_cuda


class RegionClassification(nn.Module):
    " for MRC(-kl)"

    def __init__(self, hidden_size, label_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, label_dim))

    def forward(self, input_):
        output = self.net(input_)
        return output


class ClsPrediction(nn.Module):
    def __init__(self, hidden_size, input_size=None):
        super().__init__()
        if input_size is None:
            input_size = hidden_size
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)


class AttentionPrediction(nn.Module):
    def __init__(self, hidden_size, input_size=None):
        super().__init__()
        if input_size is None:
            input_size = hidden_size
        self.query = nn.Linear(input_size, 1)
        self.key = nn.Linear(input_size, 1)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        scores = query * key
        attention_weights = F.softmax(scores, dim=0)
        output = attention_weights.mean()
        return output


class TransformerPrediction(nn.Module):
    def __init__(self, config, label_length=31, num_layers=2, num_heads=8, hidden_size=768):
        super(TransformerPrediction, self).__init__()
        self.linear = nn.Linear(config.hidden_size, hidden_size)
        self.transformer_encoder_layer = TransformerEncoderLayer(hidden_size, num_heads)
        self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer, num_layers)
        self.output = nn.Linear(hidden_size, label_length)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        x = self.linear(input)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=0)
        output = self.softmax(self.output(x))

        return output


class GlocalTextPathCMTPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.bert = GlocalTextPathCMT(config)
        self.room_labels = json.load(open('../datasets/labels/house_pano_info.json'))
        self.room_class_relation = torch.nn.Parameter(
            torch.relu(torch.from_numpy(np.load('../datasets/reg_proximity.npy')).float()),
            requires_grad=False)

        if 'mlm' in config.pretrain_tasks:
            self.mlm_head = BertOnlyMLMHead(self.config)
        if 'rc' in config.pretrain_tasks:
            self.region_classifier = TransformerPrediction(self.config)
        if 'mrc' in config.pretrain_tasks:
            self.image_classifier = RegionClassification(self.config.hidden_size, self.config.image_prob_size)
            if self.config.obj_prob_size > 0 and self.config.obj_prob_size != self.config.image_prob_size:
                self.obj_classifier = RegionClassification(self.config.hidden_size, self.config.obj_prob_size)
            else:
                self.obj_classifier = None
        if 'sap' in config.pretrain_tasks:
            self.global_sap_head = ClsPrediction(self.config.hidden_size)
            self.local_sap_head = ClsPrediction(self.config.hidden_size)
            self.activ_thold_liner = AttentionPrediction(self.config.hidden_size)
            self.conn_scale_liner = AttentionPrediction(self.config.hidden_size, input_size=self.config.hidden_size * 2)
            if config.glocal_fuse:
                self.sap_fuse_linear = ClsPrediction(self.config.hidden_size, input_size=self.config.hidden_size * 2)
            else:
                self.sap_fuse_linear = None
        if 'og' in config.pretrain_tasks:
            self.og_head = ClsPrediction(self.config.hidden_size)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        if 'mlm' in self.config.pretrain_tasks:
            self._tie_or_clone_weights(self.mlm_head.predictions.decoder,
                                       self.bert.embeddings.word_embeddings)

    def compute_label_for_point(self, label_set):
        new_set = []
        for i, list_ in enumerate(label_set):
            new_list = set()
            for x in list_:
                new_list.update(x)
            new_list = list(new_list)
            new_list = torch.LongTensor(new_list)
            new_set.append(new_list)
        new_set = nn.utils.rnn.pad_sequence(new_set, batch_first=True, padding_value=0)
        return try_cuda(new_set)

    def compute_label_for_action(self, obs):
        all_list = []
        max_num_a = 0
        for i, ob in enumerate(obs):
            max_num_a = max(max_num_a, len(ob['candidate']))
        for ob in obs:
            list_view = []
            for adj_loc in ob['candidate']:
                index = adj_loc['pointId']
                label_list = ob['obj_labels'][index]
                list_view.append(try_cuda(torch.LongTensor(label_list)))
            while len(list_view) < max_num_a:
                list_view.append(torch.LongTensor([]))
            all_list.extend(list_view)
        all_list = torch.nn.utils.rnn.pad_sequence(all_list, batch_first=True, padding_value=0)
        return try_cuda(all_list)

    def get_room_type_for_end(self, scan, cand_vpids):
        all_list = torch.LongTensor([])
        for i, ViewpointId in enumerate(cand_vpids):
            room_type = torch.LongTensor([self.room_labels[scan[i]][ViewpointId]])  # int——>tensor
            all_list = torch.cat([all_list, room_type], 0)
        return try_cuda(all_list)

    def get_room_type_for_view(self, scan, cand_vpids):
        all_list = torch.LongTensor([])
        max_num_a = 0
        for i, cand in enumerate(cand_vpids):
            max_num_a = max(max_num_a, len(cand))
        for i, cand in enumerate(cand_vpids):
            list_view = torch.LongTensor([])
            for j, ViewpointId in enumerate(cand):
                room_type = torch.LongTensor([self.room_labels[scan[i]][ViewpointId]])  # int——>tensor
                list_view = torch.cat([list_view, room_type], 0)
            while len(list_view) < max_num_a:
                list_view = torch.cat([list_view, torch.tensor([30])], 0)  # fill
            all_list = torch.cat([all_list, list_view], 0)
        return try_cuda(all_list)

    def get_reasoning_conn(self, view_label_class, text_label_class, batch_size, action_count, visited_masks=None):
        '''Calculate the proximity between the currently seen room type and the target room type,
        using the prior knowledge self.room_class_relation.'''
        text_label_class = text_label_class[:, None, :].expand(-1, action_count, -1). \
            reshape(action_count * batch_size, -1)
        gamma = 0.9
        # one-step reasoning
        one_mid = torch.matmul(view_label_class[:, None, :], self.room_class_relation)
        connectivity = torch.bmm(one_mid, text_label_class[:, :, None]).squeeze(2).reshape(batch_size, -1)
        if visited_masks is not None:
            connectivity.masked_fill_(visited_masks, 0)
        threshold, one_room_index = torch.max(connectivity, 1)

        # two-step reasoning
        # if torch.less(threshold, 0.8).any():
        #     two_mid1 = torch.matmul(self.room_class_relation, view_label_class[:, :, None]).squeeze(2)
        #     two_mid1[two_mid1 == 1] = 0
        #     two_mid1[text_label_class == 1] = 0
        #     two_mid2 = torch.matmul(text_label_class[:, None, :], self.room_class_relation).squeeze(1)
        #     two_mid2[two_mid2 == 1] = 0
        #     two_mid2[two_mid1 == 0] = 0
        #     two_mid1[two_mid2 == 0] = 0
        #
        #     two_mid3 = 0.6 * two_mid1 + 0.4 * gamma * two_mid2
        #     connectivity_reason, two_first_room_index = torch.max(two_mid3, 1)
        #     connectivity_reason = connectivity_reason.reshape(batch_size, -1)
        #     connectivity = torch.where(threshold.unsqueeze(1) < 0.8, connectivity_reason, connectivity)

        # three-step reasoning
        if torch.less(threshold, 0.8).any():
            three_mid1 = torch.matmul(self.room_class_relation, view_label_class[:, :, None]).squeeze(2)
            three_mid1[three_mid1 == 1] = 0  # mask self
            three_mid1[text_label_class == 1] = 0  # mask goal
            three_mid1 = three_mid1.unsqueeze(2).repeat(1, 1, 31)
            three_mid2 = (self.room_class_relation - torch.eye(31).cuda()).unsqueeze(0)  # mask self
            three_mid2 = three_mid2.repeat(batch_size * action_count, 1, 1)
            three_mid2[three_mid1 == 0] = 0  # mask visited viewpoint and goal
            three_mid1[three_mid2 == 0] = 0  # any zero,all zero
            three_mid3 = torch.matmul(text_label_class[:, None, :], self.room_class_relation)
            three_mid3 = three_mid3.repeat(1, 31, 1)
            three_mid3[three_mid3 == 1] = 0  # any zero, all zero
            three_mid3[three_mid1 == 0] = 0
            three_mid3[three_mid2 == 0] = 0
            three_mid2[three_mid3 == 0] = 0
            three_mid1[three_mid3 == 0] = 0

            three_mid4 = 0.6 * three_mid1 + 0.3 * gamma * three_mid2 + 0.1 * gamma * gamma * three_mid3
            connectivity_reason_mid, three_first_room_index = torch.max(three_mid4, 2)
            connectivity_reason, three_second_room_index = torch.max(connectivity_reason_mid, 1)
            connectivity_reason = connectivity_reason.reshape(batch_size, -1)
            connectivity = torch.where(threshold.unsqueeze(1) < 0.8, connectivity_reason, connectivity)
        return connectivity

    def activate_function(self, connectivity, threshold, scaling):
        connectivity = torch.where(connectivity == 1, connectivity + 0.5, connectivity)
        conn = torch.where(connectivity < threshold, (scaling / threshold) * connectivity - scaling,
                           (scaling / (1 - threshold)) * connectivity + scaling * threshold / (threshold - 1))
        return conn

    def calculate_acc_room(self, ground, view, mask):
        pass

    def calculate_acc_room_text(self, ground, text):
        pass

    def forward(self, batch, task, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        if task.startswith('mlm'):
            return self.forward_mlm(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'],
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'],
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'],
                batch['traj_vpids'], batch['traj_cand_vpids'],
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'],
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                batch['txt_labels'], compute_loss
            )
        elif task.startswith('rc'):
            return self.forward_rc(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'],
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'],
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'],
                batch['traj_vpids'], batch['traj_cand_vpids'],
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'],
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                batch['vp_view_mrc_masks'], batch['vp_view_probs'],
                batch['vp_obj_mrc_masks'], batch['vp_obj_probs'], batch['obj_labels'], batch['scan'], compute_loss
            )
        elif task.startswith('mrc'):
            return self.forward_mrc(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'],
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'],
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'],
                batch['traj_vpids'], batch['traj_cand_vpids'],
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'],
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                batch['vp_view_mrc_masks'], batch['vp_view_probs'],
                batch['vp_obj_mrc_masks'], batch['vp_obj_probs'], compute_loss
            )
        elif task.startswith('sap'):
            return self.forward_sap(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'],
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'],
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'],
                batch['traj_vpids'], batch['traj_cand_vpids'],
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'],
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                batch['gmap_visited_masks'],
                batch['global_act_labels'], batch['local_act_labels'], batch['end_vpids'], batch['scan'], compute_loss
            )
        elif task.startswith('og'):
            return self.forward_og(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'],
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'],
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'],
                batch['traj_vpids'], batch['traj_cand_vpids'],
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'],
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                batch['obj_labels'], compute_loss
            )
        elif task.startswith('valid_sap_og'):
            return self.forward_sap_og(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'],
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'],
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'],
                batch['traj_vpids'], batch['traj_cand_vpids'],
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'],
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                batch['gmap_visited_masks'], batch['global_act_labels'], batch['local_act_labels'],
                batch['obj_labels']
            )
        else:
            raise ValueError('invalid task')

    def forward_mlm(
            self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
            txt_labels, compute_loss
    ):
        txt_embeds = self.bert.forward_mlm(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        )

        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(txt_embeds, txt_labels != -1)
        prediction_scores = self.mlm_head(masked_output)

        if compute_loss:
            mask_loss = F.cross_entropy(
                prediction_scores, txt_labels[txt_labels != -1], reduction='none'
            )
            return mask_loss
        else:
            return prediction_scores

    def _compute_masked_hidden(self, hidden, mask):
        '''get only the masked region (don't compute unnecessary hiddens)'''
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

    def pad_tensors(self, tensors, shape, lens=3):
        if tensors is None:
            tensors = torch.zeros(shape)
        output = torch.LongTensor([])
        if tensors.shape[1] < lens:
            padding = torch.zeros(shape[0], lens - tensors.shape[1], shape[2]).to(tensors.device)
            output = torch.cat([tensors, padding], 1)
        elif tensors.shape[1] > lens:
            output = tensors[:, :lens, :]
        else:
            return try_cuda(tensors)
        return try_cuda(output)

    def _compute_gt_labels(self, traj_vpids, scan):
        list_view = torch.LongTensor([])
        for i, traj in enumerate(traj_vpids):
            for j, ViewpointId in enumerate(traj):
                room_type = torch.LongTensor([self.room_labels[scan[i]][ViewpointId]])  # int——>tensor
                list_view = torch.cat([list_view, room_type], 0)
        return try_cuda(list_view)

    def forward_rc(
            self, traj_view_img_fts, traj_obj_img_fts, traj_vpids, scan, compute_loss=True
    ):

        traj_view_img_fts = traj_view_img_fts[:, :36, :]
        traj_obj_img_fts = self.pad_tensors(traj_obj_img_fts, traj_view_img_fts.shape)
        region_fts = torch.cat([traj_view_img_fts, traj_obj_img_fts], 1)

        region_soft_labels = self.region_classifier(region_fts)
        region_targets = self._compute_gt_labels(traj_vpids, scan)
        if compute_loss:
            rc_loss = F.cross_entropy(region_soft_labels, region_targets, reduction='none')
            return rc_loss
        else:
            return region_soft_labels, region_targets

    def forward_mrc(
            self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
            vp_view_mrc_masks, vp_view_probs, vp_obj_mrc_masks, vp_obj_probs, compute_loss=True
    ):
        _, vp_embeds = self.bert(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
            return_gmap_embeds=False
        )

        vp_view_lens = [x[-1] for x in torch.split(traj_vp_view_lens, traj_step_lens)]
        vp_view_embeds = pad_tensors_wgrad(
            [x[1:view_len + 1] for x, view_len in zip(vp_embeds, vp_view_lens)]
        )  # [stop] at 0

        # only compute masked regions for better efficient=cy
        view_masked_output = self._compute_masked_hidden(vp_view_embeds, vp_view_mrc_masks)
        view_prediction_soft_labels = self.image_classifier(view_masked_output)
        view_mrc_targets = self._compute_masked_hidden(vp_view_probs, vp_view_mrc_masks)

        if traj_obj_img_fts is not None:
            vp_obj_lens = [x[-1] for x in torch.split(traj_vp_obj_lens, traj_step_lens)]
            vp_obj_embeds = pad_tensors_wgrad(
                [x[view_len + 1:view_len + obj_len + 1] for x, view_len, obj_len in
                 zip(vp_embeds, vp_view_lens, vp_obj_lens)]
            )
            obj_masked_output = self._compute_masked_hidden(vp_obj_embeds, vp_obj_mrc_masks)
            if self.obj_classifier is None:
                obj_prediction_soft_labels = self.image_classifier(obj_masked_output)
            else:
                obj_prediction_soft_labels = self.obj_classifier(obj_masked_output)
            obj_mrc_targets = self._compute_masked_hidden(vp_obj_probs, vp_obj_mrc_masks)
        else:
            obj_prediction_soft_labels, obj_mrc_targets = None, None

        if compute_loss:
            view_prediction_soft_labels = F.log_softmax(view_prediction_soft_labels, dim=-1)
            view_mrc_loss = F.kl_div(view_prediction_soft_labels, view_mrc_targets, reduction='none').sum(dim=1)
            if obj_prediction_soft_labels is None:
                mrc_loss = view_mrc_loss
            else:
                obj_prediction_soft_labels = F.log_softmax(obj_prediction_soft_labels, dim=-1)
                obj_mrc_loss = F.kl_div(obj_prediction_soft_labels, obj_mrc_targets, reduction='none').sum(dim=1)
                mrc_loss = torch.cat([view_mrc_loss, obj_mrc_loss], 0)
            return mrc_loss
        else:
            return view_prediction_soft_labels, view_mrc_targets, obj_prediction_soft_labels, obj_mrc_targets

    def forward_sap(
            self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
            gmap_visited_masks, global_act_labels, local_act_labels, end_vpids, scan, compute_loss,
    ):
        batch_size = txt_ids.size(0)

        gmap_embeds, vp_embeds = self.bert(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        )
        activate_threshold = torch.abs(self.activ_thold_liner(torch.ones_like(vp_embeds[:, 0])))
        activate_threshold[activate_threshold < 0.7] = 0.7
        conn_scaling_weights = torch.abs(self.conn_scale_liner(torch.cat([gmap_embeds[:, 0], vp_embeds[:, 0]], 1)))
        conn_scaling_weights[conn_scaling_weights < 1.25] = 1.25
        if self.sap_fuse_linear is None:
            fuse_weights = 0.5
        else:
            fuse_weights = torch.sigmoid(self.sap_fuse_linear(
                torch.cat([gmap_embeds[:, 0], vp_embeds[:, 0]], 1)
            ))

        global_logits = self.global_sap_head(gmap_embeds).squeeze(2) * fuse_weights
        global_logits.masked_fill_(gmap_visited_masks, -float('inf'))
        global_logits.masked_fill_(gen_seq_masks(gmap_lens).logical_not(), -float('inf'))
        end_room_type = self.get_room_type_for_end(scan, end_vpids)
        gmap_vpid = gmap_vpids.copy()
        for i in range(batch_size):
            del gmap_vpid[i][0]
        global_room_type = self.get_room_type_for_view(scan, gmap_vpid).reshape(batch_size, -1)
        global_view_labels_oh = try_cuda(torch.zeros(global_room_type.shape[0] * global_room_type.shape[1], 31))
        global_view_labels_oh.scatter_(1, global_room_type.reshape(-1)[:, None], 1)
        end_room_type_oh = try_cuda(torch.zeros(batch_size, 31))
        end_room_type_oh.scatter_(1, end_room_type[:, None], 1)
        global_conn = self.get_reasoning_conn(global_view_labels_oh, end_room_type_oh, batch_size,
                                              global_room_type.shape[1],
                                              gmap_visited_masks[:, 1:])
        global_conn = self.activate_function(global_conn, activate_threshold, conn_scaling_weights)
        global_conn = torch.cat([torch.zeros(batch_size, 1).cuda(), global_conn], 1)
        global_conn.masked_fill_(gmap_visited_masks, 0)
        global_conn.masked_fill_(gen_seq_masks(gmap_lens).logical_not(), 0)
        fused_global_logits = global_logits + global_conn

        local_logits = self.local_sap_head(vp_embeds).squeeze(2) * (1 - fuse_weights)
        vp_nav_masks = pad_tensors_wgrad(
            [x[-1] != 1 for x in torch.split(traj_nav_types, traj_step_lens)]
        )[:, :local_logits.size(1) - 1]
        vp_nav_masks = torch.cat(
            [torch.zeros(len(vp_nav_masks), 1).bool().to(vp_nav_masks.device), vp_nav_masks], 1
        )  # add [stop]
        local_logits.masked_fill_(vp_nav_masks, -float('inf'))
        vp_cand_vpids = []
        for i in range(batch_size):
            vp_cand_vpids.append(traj_cand_vpids[i][-1])
        local_room_type = self.get_room_type_for_view(scan, vp_cand_vpids).reshape(batch_size, -1)
        local_view_labels_oh = try_cuda(torch.zeros(local_room_type.shape[0] * local_room_type.shape[1], 31))
        local_view_labels_oh.scatter_(1, local_room_type.reshape(-1)[:, None], 1)
        local_conn = self.get_reasoning_conn(local_view_labels_oh, end_room_type_oh, batch_size,
                                             local_room_type.shape[1])
        local_conn = self.activate_function(local_conn, activate_threshold, conn_scaling_weights)
        local_conn.masked_fill_(vp_nav_masks[:, 1:local_conn.shape[1] + 1], 0)
        fused_local_logits = torch.clone(local_logits)
        fused_local_logits[:, 1:local_conn.shape[1] + 1] += local_conn

        # residual fusion
        fused_logits = torch.clone(global_logits)
        all_fused_logits = torch.clone(fused_global_logits)
        fused_logits[:, 0] += local_logits[:, 0]  # stop
        all_fused_logits[:, 0] += fused_local_logits[:, 0]

        for i in range(batch_size):
            visited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
            tmp_0 = {}
            tmp_1 = {}
            bw_logits0 = 0
            bw_logits1 = 0
            for j, cand_vpid in enumerate(traj_cand_vpids[i][-1]):
                if cand_vpid in visited_nodes:
                    bw_logits0 += local_logits[i, j + 1]
                    bw_logits1 += fused_local_logits[i, j + 1]
                else:
                    tmp_0[cand_vpid] = local_logits[i, j + 1]
                    tmp_1[cand_vpid] = fused_local_logits[i, j + 1]
            for j, vp in enumerate(gmap_vpids[i]):
                if j > 0 and vp not in visited_nodes:
                    if vp in tmp_0:
                        fused_logits[i, j] += tmp_0[vp]
                        all_fused_logits[i, j] += tmp_1[vp]
                    else:
                        fused_logits[i, j] += bw_logits0
                        all_fused_logits[i, j] += bw_logits1

        stop_singals = [torch.argmax(fused_logit) == 0 for fused_logit in fused_logits]
        for i, stop_singal in enumerate(stop_singals):
            # Stop according to visual logits.
            if stop_singal:
                all_fused_logits[i, 0] += 99

        if compute_loss:
            global_losses = F.cross_entropy(global_logits, global_act_labels, reduction='none')
            local_losses = F.cross_entropy(local_logits, local_act_labels, reduction='none')
            fused_losses = F.cross_entropy(fused_logits, global_act_labels, reduction='none')
            losses = global_losses + local_losses + fused_losses
            return losses
        else:
            return global_logits, local_logits, fused_logits, all_fused_logits, global_act_labels, local_act_labels

    def forward_og(
            self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
            obj_labels, compute_loss
    ):
        gmap_embeds, vp_embeds = self.bert.forward(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
            return_gmap_embeds=False
        )

        vp_view_lens = [x[-1] for x in torch.split(traj_vp_view_lens, traj_step_lens, 0)]
        vp_obj_lens = [x[-1] for x in torch.split(traj_vp_obj_lens, traj_step_lens, 0)]
        obj_embeds = pad_tensors_wgrad([
            x[1 + view_len: 1 + view_len + obj_len] for x, view_len, obj_len in
            zip(vp_embeds, vp_view_lens, vp_obj_lens)
        ])
        obj_masks = gen_seq_masks(torch.stack(vp_obj_lens, 0))

        obj_logits = self.og_head(obj_embeds).squeeze(2)
        obj_logits.masked_fill_(obj_masks.logical_not(), -float('inf'))

        if compute_loss:
            losses = F.cross_entropy(obj_logits, obj_labels, reduction='none')
            return losses
        else:
            return obj_logits

    def forward_sap_og(
            self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
            gmap_visited_masks
    ):
        batch_size = txt_ids.size(0)

        gmap_embeds, vp_embeds = self.bert(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        )

        if self.sap_fuse_linear is None:
            fuse_weights = 0.5
        else:
            fuse_weights = torch.sigmoid(self.sap_fuse_linear(
                torch.cat([gmap_embeds[:, 0], vp_embeds[:, 0]], 1)
            ))

        global_logits = self.global_sap_head(gmap_embeds).squeeze(2) * fuse_weights
        global_logits.masked_fill_(gmap_visited_masks, -float('inf'))
        global_logits.masked_fill_(gen_seq_masks(gmap_lens).logical_not(), -float('inf'))

        local_logits = self.local_sap_head(vp_embeds).squeeze(2) * (1 - fuse_weights)
        vp_nav_masks = pad_tensors_wgrad(
            [x[-1] != 1 for x in torch.split(traj_nav_types, traj_step_lens)]
        )[:, :local_logits.size(1) - 1]
        vp_nav_masks = torch.cat(
            [torch.zeros(len(vp_nav_masks), 1).bool().to(vp_nav_masks.device), vp_nav_masks], 1
        )  # add [stop]
        local_logits.masked_fill_(vp_nav_masks, -float('inf'))

        # fusion
        fused_logits = torch.clone(global_logits)
        fused_logits[:, 0] += local_logits[:, 0]  # stop
        for i in range(batch_size):
            visited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
            tmp = {}
            bw_logits = 0
            for j, cand_vpid in enumerate(traj_cand_vpids[i][-1]):
                if cand_vpid in visited_nodes:
                    bw_logits += local_logits[i, j + 1]
                else:
                    tmp[cand_vpid] = local_logits[i, j + 1]
            for j, vp in enumerate(gmap_vpids[i]):
                if j > 0 and vp not in visited_nodes:
                    if vp in tmp:
                        fused_logits[i, j] += tmp[vp]
                    else:
                        fused_logits[i, j] += bw_logits

        vp_view_lens = [x[-1] for x in torch.split(traj_vp_view_lens, traj_step_lens, 0)]
        vp_obj_lens = [x[-1] for x in torch.split(traj_vp_obj_lens, traj_step_lens, 0)]
        obj_embeds = pad_tensors_wgrad([
            x[1 + view_len: 1 + view_len + obj_len] for x, view_len, obj_len in
            zip(vp_embeds, vp_view_lens, vp_obj_lens)
        ])
        obj_masks = gen_seq_masks(torch.stack(vp_obj_lens, 0))

        obj_logits = self.og_head(obj_embeds).squeeze(2)
        obj_logits.masked_fill_(obj_masks.logical_not(), -float('inf'))

        return global_logits, local_logits, fused_logits, obj_logits
