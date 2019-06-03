"""
LINE模型的定义，主要重写nn.Module下的forward方法和实现保存模型参数的方法
"""
import torch as t
import numpy as np

from torch import nn
from torch.functional import F

class LINE(nn.Module):
    def __init__(self,caseid,entityid,case_count,entity_count,embedding_dim):
        """
        建立模型的初始化参数
        :param caseid: 案件的id
        :param entityid: 实体的id
        :param case_count:案件的数量
        :param entity_count:实体的数量
        :param embedding_dim:embedding的维数
        """
        super(LINE,self).__init__()
        self.caseid = caseid
        self.entityid = entityid
        self.embedding_dim = embedding_dim
        self.case_emb = nn.Embedding(case_count,embedding_dim,sparse=True)
        self.entity_emb = nn.Embedding(entity_count,embedding_dim,sparse=True)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform(self.case_emb.weight.data)
        nn.init.xavier_uniform(self.entity_emb.weight.data)

    def forward(self,pos_caseid,pos_entityid,neg_caseid,neg_entity):
        """
        重写forward函数，传入四个Variable
        :param pos_caseid:正例案件的id
        :param pos_entityid:正例的实体id
        :param neg_caseid:负采样得到的负例案件id
        :param neg_entity:负采样得到的负例实体id
        :return:loss的值
        """
        #正例得分
        pos_case = self.case_emb(pos_caseid)
        pos_entity = self.entity_emb(pos_entityid)
        pos_score = t.mul(pos_case, pos_entity)
        pos_score = t.sum(pos_score, dim=1)
        pos_score = F.logsigmoid(pos_score)
        #负例得分
        neg_case = self.entity_embedding(neg_caseid)
        neg_entity = self.app_embedding(neg_entity)
        neg_score = t.mul(neg_case, neg_entity)
        neg_score = t.sum(neg_score, dim=1)
        neg_score = F.logsigmoid(-1 * neg_score)
        #总得分
        pos_score_sum = t.sum(pos_score)
        neg_score_sum = t.sum(neg_score)
        all_score = -1 * (pos_score_sum + neg_score_sum)

        return all_score


    def save_para(self,filename):
        """
        保存模型的参数，主要是caseid的embedding矩阵和entityid的embedding矩阵
        :param filename: 保存的文件路径
        :return: None
        """
        ...
