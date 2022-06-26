# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer

class BERTGAT(nn.Module):
    
    def __init__(self, bert, num_features):
        super(BERTGAT, self).__init__()
        self.bert = bert
        self.dropout = 0.6      # Dropout value
        self.num_hidden = 8     # Number of hidden units
        self.alpha = 0.2        # Alpha for leaky ReLU
        self.att_head = 8       # Number of attention heads
        self.num_class = 2      # Number of classes

        self.attentions = [ GraphAttentionLayer(num_features,
                                                self.num_hidden,
                                                dropout = self.dropout,
                                                alpha = self.alpha, 
                                                concat=True)
                                                for _ in range(self.att_head)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(self.num_hidden * self.att_head, self.num_class, dropout = self.dropout, alpha = self.alpha, concat=False)

    def forward(self, inputs):
        
        text_bert_indices, bert_segments_ids, dependency_graph, sdat_graph = inputs
        x, _ = self.bert(text_bert_indices, token_type_ids = bert_segments_ids, output_all_encoded_layers=False)
        x = F.dropout(x, self.dropout)
        x = torch.cat([att(x, dependency_graph) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout)
        x = F.elu(self.out_att(x, sdat_graph))
        return F.log_softmax(x, dim=1)