# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolutionLayer

class BERTGCN(nn.Module):
    def __init__(self, bert, opt):
        super(BERTGCN, self).__init__()
        self.opt = opt
        self.bert = bert
        self.gc1 = GraphConvolutionLayer(opt.bert_dim, opt.bert_dim)
        self.gc2 = GraphConvolutionLayer(opt.bert_dim, opt.bert_dim)
        self.gc3 = GraphConvolutionLayer(opt.bert_dim, opt.bert_dim)
        self.gc4 = GraphConvolutionLayer(opt.bert_dim, opt.bert_dim)
        #self.gc5 = GraphConvolutionLayer(opt.bert_dim, opt.bert_dim)
        #self.gc6 = GraphConvolutionLayer(opt.bert_dim, opt.bert_dim)
        #self.gc7 = GraphConvolutionLayer(opt.bert_dim, opt.bert_dim)
        #self.gc8 = GraphConvolutionLayer(opt.bert_dim, opt.bert_dim)
        
        self.fc = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)

    def forward(self, inputs):
        
        text_bert_indices, bert_segments_ids, dependency_graph, affective_graph = inputs

        text_out, _ = self.bert(text_bert_indices, token_type_ids = bert_segments_ids, output_all_encoded_layers=False)

        x = F.relu(self.gc1(text_out, dependency_graph))
        x = F.relu(self.gc2(x, affective_graph))
        x = F.relu(self.gc3(text_out, dependency_graph))
        x = F.relu(self.gc4(x, affective_graph))
        #x = F.relu(self.gc5(text_out, dependency_graph))
        #x = F.relu(self.gc6(x, affective_graph))
        #x = F.relu(self.gc7(text_out, dependency_graph))
        #x = F.relu(self.gc8(x, affective_graph))

        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1) # batch_size x 2*hidden_dim
        output = self.fc(x)
        return output
