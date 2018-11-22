import tensorflow as tf
import numpy as np

import torch
from torch import nn
from torch.nn import functional

class Attention:
    def __init__(self, doc_len):
        self.doc_len = doc_len

    def forward(self, jd_data, cv_data, inf_masks, zero_mask):
        """
        :param cv_data: 3d array, batch_size * doc_len * emb_dim
        :param jd_data: 3d array, batch_size * doc_len * emb_dim
        :return:
        """
        attention_weights = torch.bmm(jd_data, cv_data.permute(0, 2, 1))
        attention_weights = attention_weights + inf_masks

        cv_weights = functional.softmax(attention_weights, dim=2) * zero_mask
        jd_context = torch.bmm(cv_weights, cv_data)
        jd_cat = torch.cat([jd_data, jd_context], dim=2)
        jd_cat = torch.max(jd_cat, dim=1)[0]

        jd_weights = functional.softmax(attention_weights, dim=1) * zero_mask
        cv_context = torch.bmm(jd_weights, jd_data)
        cv_cat = torch.cat([cv_data, cv_context], dim=2)
        cv_cat = torch.max(cv_cat, dim=1)[0]

        return jd_cat, cv_cat

    def get_inf_mask(self, jd_len, cv_len):
        inf_mask = np.zeros(shape=[self.doc_len, self.doc_len])
        inf_mask[jd_len:] = -1e10
        inf_mask[:, cv_len:] = -1e10
        return inf_mask

    def get_zero_mask(self, jd_len, cv_len):
        zero_mask = np.ones(shape=[self.doc_len, self.doc_len])
        zero_mask[jd_len:] = 0
        zero_mask[:, cv_len:] = 0
        return zero_mask

    def get_masks(self, jd_data_np, cv_data_np):
        """
        :param jd_data_np: 3d ndarray: batch * doc_len * emb_dim,
        :param cv_data_np: same
        :return: 3d ndarray: batch * doc_len * doc_len
        """
        jd_lens = [sum(jd.any(axis=1)) for jd in jd_data_np]
        cv_lens = [sum(cv.any(axis=1)) for cv in cv_data_np]
        inf_masks = [self.get_inf_mask(cv_len, jd_len)
                     for cv_len, jd_len in zip(cv_lens, jd_lens)]
        inf_masks = np.array(inf_masks)
        zero_masks = [self.get_zero_mask(cv_len, jd_len)
                      for cv_len, jd_len in zip(cv_lens, jd_lens)]
        zero_masks = np.array(zero_masks)
        return inf_masks, zero_masks



