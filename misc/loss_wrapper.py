import torch
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward

class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        if opt.label_smoothing > 0:
            self.crit = utils.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            self.crit = utils.LanguageModelCriterion()
        self.rl_crit = utils.RewardCriterion()
        self.struc_crit = utils.StructureLosses(opt)

    def forward(self, fc_feats, att_feats, labels, masks, att_masks, gts, gt_indices,
                sc_flag, struc_flag, vocab = None, test_masks = None, test_labels=None, batch_index = None):
        opt = self.opt
        
        out = {}
        if struc_flag:
            if opt.structure_loss_weight < 1:
                if self.opt.caption_model == 'itransformer':
                    pred_res, fea_extract = self.model(fc_feats, att_feats, labels, att_masks)
                    lm_loss = self.crit(pred_res, labels[..., 1:], masks[..., 1:])
                else:
                    lm_loss = self.crit(self.model(fc_feats, att_feats, labels, att_masks), labels[..., 1:],
                                     masks[..., 1:])
                # lm_loss = self.crit(self.model(fc_feats, att_feats, labels, att_masks), labels[..., 1:], masks[..., 1:])
            else:
                lm_loss = torch.tensor(0).type_as(fc_feats)
            if opt.structure_loss_weight > 0:
                if self.opt.caption_model == 'ttransformer':
                    gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks,
                     opt={'sample_method': opt.train_sample_method,
                          'beam_size': opt.train_beam_size,
                          'output_logsoftmax': opt.struc_use_logsoftmax or opt.structure_loss_type == 'softmax_margin' \
                                               or not 'margin' in opt.structure_loss_type,
                          'sample_n': opt.train_sample_n},
                     mode='sample',batch_index=batch_index)
                elif self.opt.caption_model == 'itransformer':
                    gen_result, sample_logprobs, fea_extract = self.model(fc_feats, att_feats, att_masks,
                     opt={'sample_method': opt.train_sample_method,
                          'beam_size': opt.train_beam_size,
                          'output_logsoftmax': opt.struc_use_logsoftmax or opt.structure_loss_type == 'softmax_margin' \
                                               or not 'margin' in opt.structure_loss_type,
                          'sample_n': opt.train_sample_n},
                     mode='sample')

                    fea_extract = torch.squeeze(fea_extract.contiguous().view(-1, 1, fea_extract.size(2)))
                    fea_extract_mask = torch.squeeze(gen_result[..., :].contiguous().view(-1, 1)) != 0
                    index_fea = fea_extract_mask == 1
                    fea_extract = fea_extract[index_fea == True]

                    out['fea_ex'] = fea_extract
                else:
                    gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks,
                        opt={'sample_method':opt.train_sample_method,
                            'beam_size':opt.train_beam_size,
                            'output_logsoftmax': opt.struc_use_logsoftmax or opt.structure_loss_type == 'softmax_margin'\
                                or not 'margin' in opt.structure_loss_type,
                            'sample_n': opt.train_sample_n},
                        mode='sample')
                gts = [gts[_] for _ in gt_indices.tolist()]
                struc_loss = self.struc_crit(sample_logprobs, gen_result, gts)
            else:
                struc_loss = {'loss': torch.tensor(0).type_as(fc_feats),
                              'reward': torch.tensor(0).type_as(fc_feats)}
            loss = (1-opt.structure_loss_weight) * lm_loss + opt.structure_loss_weight * struc_loss['loss']
            out['lm_loss'] = lm_loss
            out['struc_loss'] = struc_loss['loss']
            out['reward'] = struc_loss['reward']
            if out['reward']<=0.3:
                aa=0
        elif not sc_flag:
            if self.opt.caption_model == 'ttransformer':
                pred_res = self.model(fc_feats, att_feats, labels, att_masks, seq_mask=masks, batch_index=batch_index)
                loss = self.crit(pred_res[:,1:,:], test_labels[..., 1:], test_masks[..., 1:], vocab=vocab)
            elif self.opt.caption_model == 'itransformer':
                pred_res, fea_extract = self.model(fc_feats, att_feats, labels, att_masks)
                loss = self.crit(pred_res, labels[..., 1:], masks[..., 1:], vocab=vocab)

                fea_extract = torch.squeeze(fea_extract.contiguous().view(-1, 1, fea_extract.size(2)))
                fea_extract_mask = torch.squeeze(labels[..., 1:].contiguous().view(-1, 1) != 0)
                index_fea = fea_extract_mask == 1
                fea_extract = fea_extract[index_fea == True]

                out['fea_ex'] = fea_extract
            else:
                loss = self.crit(self.model(fc_feats, att_feats, labels, att_masks), labels[..., 1:],
                             masks[..., 1:], vocab = vocab)
        else:
            self.model.eval()
            with torch.no_grad():
                greedy_res, _ = self.model(fc_feats, att_feats, att_masks,
                    mode='sample',
                    opt={'sample_method': opt.sc_sample_method,
                         'beam_size': opt.sc_beam_size})
            self.model.train()
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks,
                    opt={'sample_method':opt.train_sample_method,
                        'beam_size':opt.train_beam_size,
                        'sample_n': opt.train_sample_n},
                    mode='sample')
            gts = [gts[_] for _ in gt_indices.tolist()]
            reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
            reward = torch.from_numpy(reward).float().to(gen_result.device)
            loss = self.rl_crit(sample_logprobs, gen_result.data, reward)
            out['reward'] = reward[:,0].mean()
        out['loss'] = loss
        return out
