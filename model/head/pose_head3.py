import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.transformer import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer2, TransformerDecoder,TransformerDecoderLayer
from model.transformer_cluster import TransformerEncoderLayer_cluster,TransformerEncoder_cluster
from torch.nn.init import normal_

from utils import build_mlp

#input: config,input_feat,embed_feat
#output: memeory--->size(N,B*T,C_e)
#       tgt-------->size(1,B*T,C_e)
class ST_plus_TR_block_actor(nn.Module):
    def __init__(self,config,embed_feat):
        super(ST_plus_TR_block_actor, self).__init__()
        self.embed_features = embed_feat

        #actor tr encoder
        encoder_layer_actor = TransformerEncoderLayer_cluster(self.embed_features, config.Nhead,total_size=config.total_size,window_size=config.window_size, dropout=config.dropout_porb, normalize_before=True)
        encoder_norm_actor = nn.LayerNorm(self.embed_features)
        self.encoder_actor = TransformerEncoder_cluster(encoder_layer_actor, num_layers=config.num_encoder_layers, norm=encoder_norm_actor)

        #temporal tr encoder
        encoder_layer_temp = TransformerEncoderLayer(self.embed_features, config.Nhead, dropout=config.dropout_porb,
                                                      normalize_before=True)
        encoder_norm_temp = nn.LayerNorm(self.embed_features)
        self.encoder_temp = TransformerEncoder(encoder_layer_temp, num_layers=config.num_encoder_layers,
                                                norm=encoder_norm_temp)
        #actor decoder--first
        decoder_actor=TransformerDecoderLayer2(self.embed_features, config.Nhead, dropout=config.dropout_porb, normalize_before=True)
        decoder_norm_actor= nn.LayerNorm(self.embed_features)
        self.decoder_actor = TransformerDecoder(decoder_actor, num_layers=config.num_decoder_layers, norm=decoder_norm_actor)

        #temporal decoder
        decoder_temp=TransformerDecoderLayer2(self.embed_features,config.Nhead,dropout=config.dropout_porb, normalize_before=True)
        decoder_norm_temp=nn.LayerNorm(self.embed_features)
        self.decoder_temp=TransformerDecoder(decoder_temp,num_layers=config.num_decoder_layers, norm=decoder_norm_temp)

    #x--->N,B*T,C
    #query=1,B*T,C
    def forward(self, x,query):
        N,B,T,C=x.shape
        tgt_len,bsz,dim=query.shape
        actor_o = x.reshape(N,B*T,-1) #(N,B*T,-1)
        #N,B*T,-1
        memory_actor,loss = self.encoder_actor(actor_o)

        #T,B*N,-1
        temp_o=x.permute(2,1,0,3).contiguous().reshape(T,B*N,-1)
        #T,B*N,-1
        memory_temp=self.encoder_temp(temp_o)
        #N,B*T,-1
        memory_temp=memory_temp.reshape(T,B,N,-1).permute(2,1,0,3).contiguous().reshape(N,B*T,-1)
        #N,B*T,-1
        memory=self.decoder_actor(memory_actor,memory_temp)
        memory=memory.reshape(N,B,T,-1)
        #1,B*T,C
        tgt=self.decoder_temp(query,memory)
        tgt=tgt.reshape(tgt_len,bsz,dim)
        return memory,tgt,loss

class ST_plus_TR_block_temp(nn.Module):
    def __init__(self,config,embed_feat):
        super(ST_plus_TR_block_temp, self).__init__()
        self.embed_features = embed_feat


        #actor tr encoder
        encoder_layer_actor = TransformerEncoderLayer_cluster(self.embed_features, config.Nhead,total_size=config.total_size,window_size=config.window_size, dropout=config.dropout_porb, normalize_before=True)
        encoder_norm_actor = nn.LayerNorm(self.embed_features)
        self.encoder_actor = TransformerEncoder_cluster(encoder_layer_actor, num_layers=config.num_encoder_layers, norm=encoder_norm_actor)

        #temporal tr encoder
        encoder_layer_temp = TransformerEncoderLayer(self.embed_features, config.Nhead, dropout=config.dropout_porb,
                                                      normalize_before=True)
        encoder_norm_temp = nn.LayerNorm(self.embed_features)
        self.encoder_temp = TransformerEncoder(encoder_layer_temp, num_layers=config.num_encoder_layers,
                                                norm=encoder_norm_temp)
        #actor decoder--first
        decoder_actor=TransformerDecoderLayer2(self.embed_features, config.Nhead, dropout=config.dropout_porb, normalize_before=True)
        decoder_norm_actor= nn.LayerNorm(self.embed_features)
        self.decoder_actor = TransformerDecoder(decoder_actor, num_layers=config.num_decoder_layers, norm=decoder_norm_actor)

        #temporal decoder
        decoder_temp=TransformerDecoderLayer2(self.embed_features,config.Nhead,dropout=config.dropout_porb, normalize_before=True)
        decoder_norm_temp=nn.LayerNorm(self.embed_features)
        self.decoder_temp=TransformerDecoder(decoder_temp,num_layers=config.num_decoder_layers, norm=decoder_norm_temp)

    #x--->N,B*T,C
    #query=1,B*T,C
    def forward(self, x,query):
        N,B,T,C=x.shape
        tgt_len,bsz,dim=query.shape
        actor_o = x.reshape(N,B*T,-1) #(N,B*T,-1)
        #N,B*T,-1
        memory_actor,loss = self.encoder_actor(actor_o)

        #T,B*N,-1
        temp_o=x.permute(2,1,0,3).contiguous().reshape(T,B*N,-1)
        #T,B*N,-1
        memory_temp=self.encoder_temp(temp_o)
        #T,B*N,-1
        memory_actor=memory_actor.reshape(N,B,T,-1).permute(2,1,0,3).contiguous().reshape(T,B*N,-1)
        #T,B*N,-1
        memory=self.decoder_actor(memory_actor,memory_temp)
        #N,B*T,-1
        memory=memory.reshape(T,B,N,-1).permute(2,1,0,3).contiguous().reshape(N,B,T,-1)
        #1,B*T,C
        tgt=self.decoder_temp(query,memory)
        tgt=tgt.reshape(tgt_len,bsz,dim)
        return memory,tgt,loss

class ST_plus_TR_block_cross(nn.Module):
    def __init__(self,config,embed_feat):
        super(ST_plus_TR_block_cross, self).__init__()

        self.embed_features = embed_feat

        #actor tr encoder
        encoder_layer_actor = TransformerEncoderLayer_cluster(self.embed_features, config.Nhead,total_size=config.total_size,window_size=config.window_size, dropout=config.dropout_porb, normalize_before=True)
        encoder_norm_actor = nn.LayerNorm(self.embed_features)
        self.encoder_actor = TransformerEncoder_cluster(encoder_layer_actor, num_layers=config.num_encoder_layers, norm=encoder_norm_actor)

        #temporal tr encoder
        encoder_layer_temp = TransformerEncoderLayer(self.embed_features, config.Nhead, dropout=config.dropout_porb,
                                                      normalize_before=True)
        encoder_norm_temp = nn.LayerNorm(self.embed_features)
        self.encoder_temp = TransformerEncoder(encoder_layer_temp, num_layers=config.num_encoder_layers,
                                                norm=encoder_norm_temp)
        #actor decoder  --first
        #actor cross decoder_1
        decoder_actor1=TransformerDecoderLayer2(self.embed_features, config.Nhead, dropout=config.dropout_porb, normalize_before=True)
        decoder_norm_actor1= nn.LayerNorm(self.embed_features)
        self.decoder_actor1 = TransformerDecoder(decoder_actor1, num_layers=config.num_decoder_layers, norm=decoder_norm_actor1)
        #actor cross decoder_2
        decoder_actor2=TransformerDecoderLayer2(self.embed_features, config.Nhead, dropout=config.dropout_porb, normalize_before=True)
        decoder_norm_actor2= nn.LayerNorm(self.embed_features)
        self.decoder_actor2 = TransformerDecoder(decoder_actor2, num_layers=config.num_decoder_layers, norm=decoder_norm_actor2)

        #temporal decoder
        decoder_temp=TransformerDecoderLayer2(self.embed_features,config.Nhead,dropout=config.dropout_porb, normalize_before=True)
        decoder_norm_temp=nn.LayerNorm(self.embed_features)
        self.decoder_temp=TransformerDecoder(decoder_temp,num_layers=config.num_decoder_layers, norm=decoder_norm_temp)

    #x--->N,B*T,C
    #query=1,B*T,C
    def forward(self, x,query):
        N,B,T,C=x.shape
        tgt_len,bsz,dim=query.shape
        actor_o = x.reshape(N,B*T,-1) #(N,B*T,-1)
        #N,B*T,-1
        memory_actor,loss = self.encoder_actor(actor_o)

        #T,B*N,-1
        temp_o=x.permute(2,1,0,3).contiguous().reshape(T,B*N,-1)
        #T,B*N,-1
        memory_temp=self.encoder_temp(temp_o)
        #N,B*T,-1
        memory_temp=memory_temp.reshape(T,B,N,-1).permute(2,1,0,3).contiguous().reshape(N,B*T,-1)
        #N,B*T,-1
        memory1=self.decoder_actor1(memory_actor,memory_temp)
        #T,B*N,-1
        memory_actor=memory_actor.reshape(N,B,T,-1).permute(2,1,0,3).contiguous().reshape(T,B*N,-1)
        memory_temp=memory_temp.reshape(N,B,T,-1).permute(2,1,0,3).contiguous().reshape(T,B*N,-1)
        memory2=self.decoder_actor2(memory_actor,memory_temp)

        # N,B,T,-1
        memory1=memory1.reshape(N,B,T,-1)
        memory2 = memory2.reshape(T,B,N,-1).permute(2,1,0,3).contiguous().reshape(N,B,T,-1)
        memory = memory1 + memory2
        #1,B*T,C
        tgt=self.decoder_temp(query,memory.reshape(N,B*T,-1))
        tgt=tgt.reshape(tgt_len,bsz,dim)
        return memory,tgt,loss

class PoseAttent2(nn.Module):
    def __init__(self,config):
        super(PoseAttent2, self).__init__()
        self.num_STTR_layers=config.num_STTR_layers
        self.T = config.num_frames
        self.N = config.num_boxes
        self.J = config.num_joints
        self.L = config.num_tokens
        embed_feat=config.embed_features
        self.actions_num_classes = config.actions_num_classes
        self.activities_num_classes = config.activities_num_classes
        self.embed_features=embed_feat
        self.STTR_module=nn.ModuleList()

        self.group_query = nn.Parameter(torch.randn(embed_feat), requires_grad=True)
        group_tr_layer = TransformerDecoderLayer2(embed_feat, 8 , dropout=0.1, normalize_before=True)
        group_tr_norm = nn.LayerNorm(embed_feat)
        self.group_tr = TransformerDecoder(group_tr_layer, num_layers=2, norm=group_tr_norm)

        for i in range(self.num_STTR_layers):
            self.STTR_module.append(ST_plus_TR_block_cross(config,embed_feat))

        self.dropout = nn.Dropout(p=0.5)
        out = self.embed_features // 2
        self.trans_fc = nn.Linear(self.embed_features,out)
        self.trans_fc2 = nn.Linear(self.embed_features,self.embed_features*self.J)

        self.fc_theta = nn.Linear(self.embed_features,self.embed_features)
        self.fc_phi = nn.Linear(self.embed_features,self.embed_features)
        self.fc_g = nn.Linear(self.embed_features,self.embed_features)
        self.activities_fc_s = nn.Linear(self.embed_features*2, self.activities_num_classes)
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
        lam = 1
        B, T, N, F = x.shape
        x = x.reshape(B*T, N, F)
        graph_boxes_features_theta = self.fc_theta(x)  # B,N,NFR
        graph_boxes_features_phi = self.fc_phi(x)  # B,N,NFR

        similarity_relation_graph = torch.matmul(graph_boxes_features_theta,
                                                graph_boxes_features_phi.transpose(1, 2))  # B,N,N

        similarity_relation_graph = similarity_relation_graph / np.sqrt(NFR)

        dist = torch.exp(lam * self.calc_pairwise_distance_3d(p,p))
        return (similarity_relation_graph * dist).matmul(self.fc_g(x))
    
    def space_group_relation(self,x,global_token):
        B, T, N, F = x.shape
        x = x.view(B*T, N, -1)
        x_s = torch.chunk(x, self.num_groups, 1)
        context = torch.mean(x, 1)
        global_token = torch.mean(global_token,0)
        context = context + global_token
        gammas = self.get_att_weigths(x_s, context)

        group_feas = []
        x_n = []

        for g in range(self.num_groups):
            gammas[g] = gammas[g].view(gammas[g].size(0), -1, 1)
            #todo can change this invertBlock
            temp = self.invertBlock(x_s[g] + x_s[g] * gammas[g])
            x_n.append(x_s[g] + x_s[g] * gammas[g])
            group_feas.append(temp[:,-1,:])
        
        tgt = torch.cat(group_feas, 1)
        x_n = torch.cat(x_n,1).view(B,T,N,-1)
        #B*T,-1
        tgt = tgt.reshape(-1,self.embed_features*2)
        
        return tgt, x_n
    
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
        temp = self.invertBlockTemp(x + x * gammas)
        x = x + x * gammas
        tgt = temp[:,-1,:]
        x_n = x.view(B,T,N,-1)
        tgt = tgt.reshape(-1,self.embed_features)
        
        return tgt, x_n

    def forward(self,x,global_token,p):
        B, T, N, F = x.shape
        memory = x.reshape(-1,self.embed_features)

        # x = self.calc_relation(x,p,F).reshape(B,T,N,F)
        tgt_s, x_s = self.space_group_relation(x,global_token)
        tgt_t, x_t = self.temp_group_relation(x,global_token)

        memory = self.dropout(memory)
        tgt_s = self.dropout(tgt_s)
        tgt_t = self.dropout(tgt_t)

        actions_scores=self.actions_fc(memory)
        actions_scores = actions_scores.reshape(B,T,N, -1)

        activities_scores_s=self.activities_fc_s(tgt_s)
        activities_scores_t=self.activities_fc_t(tgt_t)

        if T!=1:
            actions_scores = actions_scores.mean(dim=1).reshape(B * N, -1)

            activities_scores_s=activities_scores_s.reshape(B,T,-1).mean(dim=1)
            activities_scores_t=activities_scores_t.reshape(B,N,-1).mean(dim=1)
        
        x_n = torch.cat((x_s,x_t),3)
        x_n = self.lin3(x_n)

        # return actions_scores, activities_scores, cluster_loss
        return actions_scores, activities_scores_t, activities_scores_s, x_n


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
