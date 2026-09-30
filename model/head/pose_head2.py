import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.head.GCN import GCN_Module
from model.transformer import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer2, TransformerDecoder,TransformerDecoderLayer
from model.transformer_cluster import TransformerEncoderLayer_cluster,TransformerEncoder_cluster
from torch.nn.init import normal_

from utils import build_mlp
from model.head.GAT import GraphAttentionV2Layer
from sklearn.cluster import DBSCAN

class PoseAttent(nn.Module):
    def __init__(self,config):
        super(PoseAttent, self).__init__()
        self.cfg = config
        self.num_STTR_layers=config.num_STTR_layers
        self.T = config.num_frames
        self.N = config.num_boxes
        self.J = config.num_joints
        self.L = config.num_tokens
        embed_feat=config.embed_features
        self.actions_num_classes = config.actions_num_classes
        self.activities_num_classes = config.activities_num_classes
        self.embed_features=embed_feat

        self.group_query = nn.Parameter(torch.randn(embed_feat), requires_grad=True)
        group_tr_layer = TransformerDecoderLayer2(embed_feat, 8 , dropout=0.1, normalize_before=True)
        group_tr_norm = nn.LayerNorm(embed_feat)
        self.group_tr = TransformerDecoder(group_tr_layer, num_layers=2, norm=group_tr_norm)

        self.dropout = nn.Dropout(p=0.5)
        out = self.embed_features // 2
        self.trans_fc = nn.Linear(self.embed_features,out)
        self.trans_fc2 = nn.Linear(self.embed_features,self.embed_features*self.J)

        self.fc_theta = nn.Linear(self.embed_features,self.embed_features)
        self.fc_phi = nn.Linear(self.embed_features,self.embed_features)
        self.fc_g = nn.Linear(self.embed_features,self.embed_features)
        self.activities_fc_s = nn.Linear(self.embed_features, self.activities_num_classes)
        self.activities_fc_t = nn.Linear(self.embed_features, self.activities_num_classes)
        self.actions_fc = nn.Linear(self.embed_features, self.actions_num_classes)
        self.num_groups = 2
        # self.attention_fun = nn.Linear(1000, 1)
        fea_size = embed_feat

        self.att_source_weights = nn.Sequential(
            nn.Linear(fea_size, fea_size, bias=True),
        )
        self.att_context_weights = nn.Sequential(
            nn.Linear(fea_size, fea_size, bias=True),
        )
        self.att_extra_weights = nn.Sequential(
            nn.Linear(fea_size, 1, bias=True),
        )

        self.att_source_weights_t = nn.Sequential(
            nn.Linear(self.N*fea_size, self.N*fea_size, bias=True),
        )
        self.att_context_weights_t = nn.Sequential(
            nn.Linear(self.N*fea_size, self.N*fea_size, bias=True),
        )
        self.att_extra_weights_t = nn.Sequential(
            nn.Linear(self.N*fea_size, 1, bias=True),
        )
        self.lin1 = nn.Linear(self.N*fea_size,self.N*fea_size)
        self.lin2 = nn.Linear(self.L*fea_size,self.N*fea_size)
        self.lin3 = nn.Linear(2*fea_size,fea_size)
        # self.Intra_Group_LSTM = nn.LSTM(fea_size, fea_size, num_layers=1, batch_first=True)
        # self.transBlock = TCT_Block(config.embed_features, config.nhead,  config.dim_feedforward,  config.dropout,  config.activation, config.num_boxes, config.num_joints)
        self.invertBlock = Invert_LSTM_Block(embed_feat,embed_feat)
        self.invertBlockTemp = Invert_LSTM_Block(embed_feat*self.N,embed_feat)

        self.layer1 = GraphAttentionV2Layer(fea_size, fea_size, n_heads=8, is_concat=True,share_weights=True)
        self.activation = nn.ELU()
        self.layer2 = GraphAttentionV2Layer(fea_size, fea_size, n_heads=1, is_concat=True,share_weights=True)

        self.gcn_list = torch.nn.ModuleList([GCN_Module(self.cfg) for i in range(self.cfg.gcn_layers)])

    def get_att_weigths(self, x_s, context=None):
        gammas = {}
        num_players = x_s[0].shape[1]
        if context is not None:
            # B*T, 6, NFB
            context = torch.unsqueeze(context, 1).repeat(1, num_players, 1)
            for g in range(self.num_groups):
                gammas[g] = F.softmax(torch.squeeze(self.att_extra_weights(
                    torch.tanh(self.att_source_weights(x_s[g]) + self.att_context_weights(context)))), dim=-1).view(-1,
                                                                                                                    1,
                                                                                                                    num_players)
        else:
            for g in self.num_groups:
                gammas[g] = F.softmax(torch.squeeze(torch.tanh(self.att_source_weights(x_s[g])))).view(-1, 1,
                                                                                                       num_players)
        # print(torch.squeeze(self.att_extra_weights(
        #             torch.tanh(self.att_source_weights(x_s[g]) + self.att_context_weights(context)))).shape)
        # print(gammas[1].shape)
        return gammas
    
    def get_att_temp_weights(self, x, context=None):
        num_frames = x.shape[1]
        if context is not None:
            context = torch.unsqueeze(context,1).repeat(1,num_frames, 1)
            gammas = F.softmax(torch.squeeze(self.att_extra_weights_t(
                torch.tanh(self.att_source_weights_t(x)+self.att_context_weights_t(context)))), dim= -1).view(-1,1,num_frames)
        else:
            gammas = F.softmax(torch.squeeze(torch.tanh(self.att_source_weights_t(x)))).view(-1, 1, num_frames)

        return gammas
    
    def calc_pairwise_distance_3d(self, X, Y):
        """
        computes pairwise distance between each element
        Args: 
            X: [B,N,D]
            Y: [B,M,D]
        Returns:
            dist: [B,N,M] matrix of euclidean distances
        """
        B = X.shape[0]

        rx = X.pow(2).sum(dim=2).reshape((B, -1, 1))
        ry = Y.pow(2).sum(dim=2).reshape((B, -1, 1))

        dist = rx - 2.0 * X.matmul(Y.transpose(1, 2)) + ry.transpose(1, 2)

        return torch.sqrt(dist)
    
    def calc_relation(self ,x ,p ,NFR):
        lam = 0.5
        B, L, _ = x.shape
        graph_boxes_features_theta = self.fc_theta(x)  # B,L,NFR
        graph_boxes_features_phi = self.fc_phi(x)  # B,L,NFR

        similarity_relation_graph = torch.matmul(graph_boxes_features_theta,
                                                graph_boxes_features_phi.transpose(1, 2))  # B,L,L

        similarity_relation_graph = similarity_relation_graph / np.sqrt(NFR)

        dist = torch.exp(lam * self.calc_pairwise_distance_3d(p,p))
        return F.softmax(similarity_relation_graph * dist)
    
    def space_group_relation(self,x,global_token):
        B, T, N, F = x.shape
        x = x.view(B*T, N, -1)
        x_s = torch.chunk(x, self.num_groups, 1)
        context = torch.mean(x, 1)
        global_token = torch.mean(global_token,0)
        context = context + global_token
        gammas = self.get_att_weigths(x_s, context)

        x_n = []

        for g in range(self.num_groups):
            gammas[g] = gammas[g].view(gammas[g].size(0), -1, 1)
            x_n.append(x_s[g] + x_s[g] * gammas[g])
        

        x_n = torch.cat(x_n,1).view(B,T,N,-1)
        
        return x_n,gammas
    
    def temp_group_relation(self,x ,global_token):
        B, T, N, F = x.shape
        x = x.reshape(B, T, -1) # B, T, N*F
        context = torch.mean(x, 1) # B, N*F
        context = self.lin1(context)
        L, _, C = global_token.shape # L ,B*T , C

        global_token = global_token.reshape(L,B,T,C) 
        global_token = global_token.permute(1,2,3,0).contiguous().reshape(B,T,-1)
        global_token = torch.mean(global_token, 1)
        global_token = self.lin2(global_token)
        
        context = context + global_token
        gammas = self.get_att_temp_weights(x, context)
        gammas = gammas.view(gammas.size(0), -1, 1)
        x = x + x * gammas
        x_n = x.view(B,T,N,-1)
        
        return x_n,gammas

    def forward(self,x,global_token,p):
        B, T, N, F = x.shape
        memory = x.reshape(-1,self.embed_features)

        # p = p.reshape(B,T,N,-1).reshape(B,T*N,-1)
        # ST self attention
        x,gamma_space = self.space_group_relation(x,global_token)
        gamma_sp = torch.stack((gamma_space[0],gamma_space[1]),dim =1).reshape(B,T,N)
        x,gamma_temp = self.temp_group_relation(x,global_token)

        # gamma2 = (gamma_sp*gamma_temp).reshape(B,T,N,1)
        # gamma = torch.nn.functional.normalize(gamma2, p=1, dim=-2)
        # x = gamma*x + x


        # GCN graph modeling
        NFG = self.cfg.num_features_gcn
        NFB = self.cfg.num_features_boxes

        graph_boxes_features = x.reshape(B * T, N, NFG)
        
        visual_info = []
        for i in range(len(self.gcn_list)):
            graph_boxes_features, relation_graph = self.gcn_list[i](graph_boxes_features, p)
            # print("relation_graph:", relation_graph.size())
            visual_info.append(relation_graph.reshape(B, T, N, N))

        # fuse graph_boxes_features with boxes_features
        graph_boxes_features = graph_boxes_features.reshape(B, T, N, NFG)
        boxes_features = x.reshape(B, T, N, NFB)

        # boxes_states= torch.cat( [graph_boxes_features,boxes_features],dim=3)  #B, T, N, NFG+NFB
        boxes_states = graph_boxes_features + boxes_features

        boxes_states = self.dropout(boxes_states)


        # Predict actions
        boxes_states_flat = boxes_states.reshape(-1, NFG)  # B*T*N, NFS
        actions_scores = self.actions_fc(boxes_states_flat)  # B*T*N, actn_num

        # Predict activities
        boxes_states_pooled, _ = torch.max(boxes_states, dim=2)
        boxes_states_pooled_flat = boxes_states_pooled.reshape(-1, NFG)
        activities_scores = self.activities_fc_s(boxes_states_pooled_flat)  # B*T, acty_num

        # Temporal fusion
        actions_scores = actions_scores.reshape(B, T, N, -1)
        actions_scores = torch.mean(actions_scores, dim=1).reshape(B * N, -1)
        activities_scores = activities_scores.reshape(B, T, -1)
        activities_scores = torch.mean(activities_scores, dim=1).reshape(B, -1)

        # # change B = 1 to do GAT
        # # x = x_t.reshape(B,T*N, -1)
        # x = x.reshape(B,T*N, -1)
        # adj_mat = torch.ones((T*N,T*N,1)).to("cuda:0") 
        # x = x.reshape(T*N,-1)
        # x = self.dropout(x)
        # # graph attention layer
        # x = self.layer1(x, adj_mat)
        # x = self.activation(x)
        # x = self.dropout(x)
        # x = self.layer2(x, adj_mat)

        # # clustering
        # clustering = DBSCAN(eps=3, min_samples=1).fit(x)
        # print(clustering.labels_)

        # temp = self.invertBlock(x.reshape(B*T,N,-1))
        # tgt = temp[:,-1,:]
        # tgt = tgt.reshape(-1,self.embed_features)
        # tgt = torch.mean(x.reshape(B,T,N,-1),dim = 2)
        # tgt = tgt.reshape(-1,self.embed_features)

        # memory = self.dropout(memory)
        # tgt = self.dropout(tgt)

        # actions_scores=self.actions_fc(memory)
        # actions_scores = actions_scores.reshape(B,T,N, -1)

        # activities_scores=self.activities_fc_s(tgt)

        # if T!=1:
        #     actions_scores = actions_scores.mean(dim=1).reshape(B * N, -1)

        #     activities_scores=activities_scores.reshape(B,T,-1).mean(dim=1)
        
        # x_n = torch.cat((x_s,x_t),3)
        # x_n = self.lin3(x_n)

        # return actions_scores, activities_scores, cluster_loss
        return actions_scores, activities_scores,  boxes_states,gamma_space,gamma_temp,visual_info


class Invert_LSTM_Block(nn.Module):
    def __init__(self,input_dim,embed_dim):
        super(Invert_LSTM_Block,self).__init__()
        self.linear1 = build_mlp(input_dim=input_dim,
                                hidden_dims=[embed_dim], 
                                output_dim=embed_dim)
        self.Intra_Group_LSTM = nn.LSTM(embed_dim, embed_dim, num_layers=1, batch_first=True)
        self.linear2 = build_mlp(input_dim=embed_dim,
                                hidden_dims=[embed_dim], 
                                output_dim=input_dim)
    def forward(self, x):
        x = self.linear1(x)
        x,_ = self.Intra_Group_LSTM(x)
        x = self.linear2(x)
        return x
