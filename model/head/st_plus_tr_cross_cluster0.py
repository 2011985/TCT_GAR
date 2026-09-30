import torch
import torch.nn as nn

from model.transformer import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer2, TransformerDecoder,TransformerDecoderLayer
from model.transformer_cluster import TransformerEncoderLayer_cluster,TransformerEncoder_cluster
from torch.nn.init import normal_
from utils import build_mlp
from einops import rearrange

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





# class ST_plus_TR_block_cross(nn.Module):
#     def __init__(self,config,embed_feat):
#         super(ST_plus_TR_block_cross, self).__init__()

#         self.embed_features = embed_feat

#         #actor tr encoder
#         encoder_layer_actor = TransformerEncoderLayer_cluster(self.embed_features, config.Nhead,total_size=config.total_size,window_size=config.window_size, dropout=config.dropout_porb, normalize_before=True)
#         encoder_norm_actor = nn.LayerNorm(self.embed_features)
#         self.encoder_actor = TransformerEncoder_cluster(encoder_layer_actor, num_layers=config.num_encoder_layers, norm=encoder_norm_actor)
#         self.encoder_actor_j = TransformerEncoder_cluster(encoder_layer_actor, num_layers=config.num_encoder_layers, norm=encoder_norm_actor)
        
#         # # actor tr encoder
#         # encoder_layer_actor = TransformerEncoderLayer(self.embed_features, config.Nhead, dropout=config.dropout_porb,
#         #                                               normalize_before=True)
#         # encoder_norm_actor = nn.LayerNorm(self.embed_features)
#         # self.encoder_actor = TransformerEncoder(encoder_layer_actor, num_layers=config.num_encoder_layers,
#         #                                         norm=encoder_norm_actor)
#         # self.encoder_actor_j = TransformerEncoder(encoder_layer_actor, num_layers=config.num_encoder_layers,
#         #                                         norm=encoder_norm_actor)

#         #temporal tr encoder
#         encoder_layer_temp = TransformerEncoderLayer(self.embed_features, config.Nhead, dropout=config.dropout_porb,
#                                                       normalize_before=True)
#         encoder_norm_temp = nn.LayerNorm(self.embed_features)
#         self.encoder_temp = TransformerEncoder(encoder_layer_temp, num_layers=config.num_encoder_layers,
#                                                 norm=encoder_norm_temp)
#         self.encoder_temp_j = TransformerEncoder(encoder_layer_temp, num_layers=config.num_encoder_layers,
#                                                 norm=encoder_norm_temp)
#         #actor decoder  --first
#         #actor cross decoder_1
#         decoder_actor1=TransformerDecoderLayer2(self.embed_features, config.Nhead, dropout=config.dropout_porb, normalize_before=True)
#         decoder_norm_actor1= nn.LayerNorm(self.embed_features)
#         self.decoder_actor1 = TransformerDecoder(decoder_actor1, num_layers=config.num_decoder_layers, norm=decoder_norm_actor1)
        
#         #actor cross decoder_2
#         decoder_actor2=TransformerDecoderLayer2(self.embed_features, config.Nhead, dropout=config.dropout_porb, normalize_before=True)
#         decoder_norm_actor2= nn.LayerNorm(self.embed_features)
#         self.decoder_actor2 = TransformerDecoder(decoder_actor2, num_layers=config.num_decoder_layers, norm=decoder_norm_actor2)
        
#         #se block
#         self.se_block = ChannelWiseTransformerBlock(self.embed_features)
#         self.se_block2 = ChannelWiseTransformerBlock(self.embed_features)
#         #conv
#         self.conv1 = nn.Conv1d(2*self.embed_features,self.embed_features,3,padding=1)
#         self.conv2 = nn.Conv1d(2*self.embed_features,self.embed_features,3,padding=1)
#         self.conv = nn.Conv2d(2*self.embed_features,self.embed_features,3,padding=1)

#         #temporal decoder
#         decoder_temp=TransformerDecoderLayer2(self.embed_features,config.Nhead,dropout=config.dropout_porb, normalize_before=True)
#         decoder_norm_temp=nn.LayerNorm(self.embed_features)
#         self.decoder_temp=TransformerDecoder(decoder_temp,num_layers=config.num_decoder_layers, norm=decoder_norm_temp)
#         self.tct_block1 = TCT_Block(self.embed_features, config.nhead,  config.dim_feedforward,  config.dropout,  config.activation, config.num_boxes, config.num_joints,config.num_frames)
#         self.tct_block2 = TCT_Block(self.embed_features, config.nhead,  config.dim_feedforward,  config.dropout,  config.activation, config.num_boxes, config.num_joints,config.num_frames)
#     #x--->N,B*T,C
#     #query=1,B*T,C
#     def forward(self, x, query, query2, joints):
#         N,B,T,C=x.shape
#         tgt_len,bsz,dim=query.shape
#         actor_o = x.reshape(N,B*T,-1) #(N,B*T,-1)
#         actor_o_j = joints.reshape(N,B*T,-1)
#         #N,B*T,-1
#         memory_actor, closs1 = self.encoder_actor(actor_o)
#         # memory_actor = self.encoder_actor(actor_o)
#         # closs1 = 0
#         #N,B*T,-1
#         memory_actor_j, closs2 = self.encoder_actor_j(actor_o_j)
#         # memory_actor_j = self.encoder_actor_j(actor_o_j)
#         # closs2 = 0
#         # memory_actor_compose = self.tct_block1(memory_actor_j,memory_actor,query2)
#         memory_actor_compose = torch.cat((memory_actor,memory_actor_j),dim=2)
#         memory_actor_compose = self.conv1(memory_actor_compose.permute(1,2,0)).permute(2,0,1)

#         tmp_actor = memory_actor_compose.reshape(N,B,T,-1)
#         #T,B*N,-1
#         temp_o=x.permute(2,1,0,3).contiguous().reshape(T,B*N,-1)
#         temp_o_j=joints.permute(2,1,0,3).contiguous().reshape(T,B*N,-1)
#         #T,B*N,-1
#         memory_temp=self.encoder_temp(temp_o)
#         memory_temp_j=self.encoder_temp_j(temp_o_j)
#         query2 = query2.reshape(N,B,T,-1).permute(2,0,1,3).contiguous().reshape(T,B*N,-1)
#         #todo: put compose to here
#         # memory_temp_compose = self.tct_block2(memory_temp_j,memory_temp,query2)
#         memory_temp_compose = torch.cat((memory_temp,memory_temp_j),dim=2)
#         memory_temp_compose = self.conv2(memory_temp_compose.permute(1,2,0)).permute(2,0,1)

#         #N,B*T,-1
#         memory_temp_compose=memory_temp_compose.reshape(T,B,N,-1).permute(2,1,0,3).contiguous().reshape(N,B*T,-1)
#         tmp_temp = memory_temp_compose.reshape(N,B,T,-1)
#         #N,B*T,-1
#         memory_actor_compose = memory_actor_compose.squeeze()
#         memory_temp_compose = memory_temp_compose.squeeze()
#         memory1=self.decoder_actor1(memory_actor_compose,memory_temp_compose)
#         #T,B*N,-1
#         memory_actor_compose=memory_actor_compose.reshape(N,B,T,-1).permute(2,1,0,3).contiguous().reshape(T,B*N,-1)
#         memory_temp_compose=memory_temp_compose.reshape(N,B,T,-1).permute(2,1,0,3).contiguous().reshape(T,B*N,-1)
#         #maybe need to change the order
#         memory2=self.decoder_actor2(memory_temp_compose, memory_actor_compose,)         

#         # N,B,T,-1
#         memory1=memory1.reshape(N,B,T,-1)
#         memory1 = memory1 + tmp_actor
#         # trans_mem1 = memory1.permute(1,3,2,0)
#         # trans_mem1 = self.se_block(trans_mem1)
#         # memory1 = trans_mem1.permute(3,0,2,1)
        
#         memory2 = memory2.reshape(T,B,N,-1).permute(2,1,0,3).contiguous().reshape(N,B,T,-1)
#         memory2 = memory2 + tmp_temp
#         # trans_mem2 = memory2.permute(1,3,2,0)
#         # trans_mem2 = self.se_block2(trans_mem2)
#         # memory2 = trans_mem2.permute(3,0,2,1)
        
#         memory = memory1 + memory2
#         # memory = torch.cat((memory1,memory2),dim=3)
#         # memory = self.conv(memory.permute(1,3,0,2)).permute(2,0,3,1)

#         # trans_mem1 = memory.permute(1,3,2,0)
#         # trans_mem1 = self.se_block(trans_mem1)
#         # memory = trans_mem1.permute(3,0,2,1)

#         #1,B*T,C
#         tgt=self.decoder_temp(query,memory.reshape(N,B*T,-1))
#         tgt=tgt.reshape(tgt_len,bsz,dim) 
#         return memory,tgt,closs2+closs1

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
        tmp_actor = memory_actor.reshape(N,B,T,-1)

        #T,B*N,-1
        temp_o=x.permute(2,1,0,3).contiguous().reshape(T,B*N,-1)
        #T,B*N,-1
        memory_temp=self.encoder_temp(temp_o)
        #N,B*T,-1
        memory_temp=memory_temp.reshape(T,B,N,-1).permute(2,1,0,3).contiguous().reshape(N,B*T,-1)
        tmp_temp = memory_temp.reshape(N,B,T,-1)
        #N,B*T,-1
        memory1=self.decoder_actor1(memory_actor,memory_temp)
        #T,B*N,-1
        memory_actor=memory_actor.reshape(N,B,T,-1).permute(2,1,0,3).contiguous().reshape(T,B*N,-1)
        memory_temp=memory_temp.reshape(N,B,T,-1).permute(2,1,0,3).contiguous().reshape(T,B*N,-1)
        memory2=self.decoder_actor2(memory_temp,memory_actor)         

        # N,B,T,-1
        memory1=memory1.reshape(N,B,T,-1)
        memory1 = memory1 + tmp_actor
        memory2 = memory2.reshape(T,B,N,-1).permute(2,1,0,3).contiguous().reshape(N,B,T,-1)
        memory2 = memory2 + tmp_temp
        memory = memory1 + memory2
        #1,B*T,C
        tgt=self.decoder_temp(query,memory.reshape(N,B*T,-1))
        tgt=tgt.reshape(tgt_len,bsz,dim) 
        return memory,tgt,loss


class TCT_Block(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation,num_boxes,num_joints,num_frames):
        super(TCT_Block, self).__init__()
        # self.tblock = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        # self.tblock2 = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.mha = Attention(d_model, heads=nhead, dim_head=dim_feedforward, dropout=dropout)
        self.bn = nn.BatchNorm2d(d_model)
        self.activation = torch.nn.LeakyReLU()
        self.N = num_boxes
        self.J = num_joints
        self.d = d_model
        self.T = num_frames
        self.conv_joint = nn.Conv1d(d_model,d_model,3,padding=1)
        self.conv_boxes = nn.Conv1d(d_model,d_model,3,padding=1)
        # self.conv2 = nn.Conv1d(d_model,d_model,3,padding=1)

        # decoder_actor=TransformerDecoderLayer2(d_model, nhead, dim_feedforward, dropout, normalize_before=True)
        # decoder_norm_actor= nn.LayerNorm(d_model)
        # self.decoder = TransformerDecoder(decoder_actor, 2, norm=decoder_norm_actor)
        # self.se_block = ChannelWiseTransformerBlock(d_model*2)

        self.fc = build_mlp(input_dim=2*d_model,
                                                 hidden_dims=[d_model], 
                                                 output_dim=d_model)
    def forward(self, joint_fea, boxes_fea):
        '''
        joint_fea:(B,T,N,F)
        boxes_fea:(B,T,N,F)
        '''
        B, T, N, _ = joint_fea.shape
        assert joint_fea.shape == boxes_fea.shape, "Joint and box features must have the same shape."

        # 对 joint_fea 和 boxes_fea 分别应用 1D 卷积
        joint_fea_transformed = self.conv_joint(joint_fea.reshape(B*T*N, -1).transpose(0, 1))
        boxes_fea_transformed = self.conv_boxes(boxes_fea.reshape(B*T*N, -1).transpose(0, 1))

        # 将卷积后的特征恢复到原始的 (B, T, N, F) 形状
        joint_fea_transformed = joint_fea_transformed.transpose(0, 1).reshape(B, T, N, -1)
        boxes_fea_transformed = boxes_fea_transformed.transpose(0, 1).reshape(B, T, N, -1)

        # 对 joint_fea 应用多头自注意力机制
        joint_attended = self.mha(
            joint_fea_transformed.reshape(B*T, N, -1)
        )

        # 恢复自注意力后的特征形状
        joint_attended = joint_attended.reshape(B, T, N, -1)

        # 合并两种特征
        concatenated_features = torch.cat([joint_attended, boxes_fea_transformed], dim=-1)

        # 通过全连接层融合特征
        fused_features = self.fc(concatenated_features)

        # 可选地添加激活函数和批标准化
        fused_features = self.activation(fused_features)
        # if hasattr(self, 'bn'):
        #     fused_features = self.bn(fused_features)

        return fused_features 

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class ChannelWiseTransformerBlock(nn.Module):
    def __init__(self, num_patches, patch_dim=1, dim=64, heads=5, dim_head=64, dropout=0.):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(patch_dim)
        self.projection = nn.Linear(patch_dim ** 2, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        self.mha = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        x = self.avg_pool(z)
        x = x.flatten(-2)

        x = self.projection(x)
        x += self.pos_embedding

        x = self.mha(x)
        x = x.mean(-1).unsqueeze(-1).unsqueeze(-1)
        x = self.sigmoid(x)

        return z * x

class ST_plus_TR_cross_cluster(nn.Module):
    def __init__(self,config):
        super(ST_plus_TR_cross_cluster, self).__init__()
        self.num_STTR_layers=config.num_STTR_layers

        embed_feat=config.embed_features
        self.actions_num_classes = config.actions_num_classes
        self.activities_num_classes = config.activities_num_classes
        self.embed_features=embed_feat
        self.STTR_module=nn.ModuleList()

        self.group_query = nn.Parameter(torch.randn(embed_feat), requires_grad=True) 
        group_tr_layer = TransformerDecoderLayer2(embed_feat, 8 , dropout=0.1, normalize_before=True)
        group_tr_norm = nn.LayerNorm(embed_feat)
        self.group_tr = TransformerDecoder(group_tr_layer, num_layers=2, norm=group_tr_norm)
        self.group_tr2 = TransformerDecoder(group_tr_layer, num_layers=2, norm=group_tr_norm)
        for i in range(self.num_STTR_layers):
            self.STTR_module.append(ST_plus_TR_block_cross(config,embed_feat))

        self.dropout = nn.Dropout(p=0.5)
        self.activities_fc = nn.Linear(self.embed_features, self.activities_num_classes)
        self.actions_fc = nn.Linear(self.embed_features, self.actions_num_classes)
        self.conv = nn.Conv1d(embed_feat,config.num_boxes*embed_feat,3,padding=1)
        # self.joint_fc = nn.Linear(256,512)
        self.TCT_block = TCT_Block(self.embed_features, config.nhead,  config.dim_feedforward,  config.dropout,  config.activation, config.num_boxes, config.num_joints,config.num_frames) 
    def forward(self,x,global_token,poses_features):
        B, T, N, F = x.shape
        #N,B,T,C_e
        x = self.TCT_block(poses_features,x)
        # x+=poses_features
        x = x.permute(2,0,1,3)#.reshape(N,B*T,-1)
        # joints = poses_features.permute(2,0,1,3)
        # joints = self.joint_fc(joints)
        #1,B*T,C_e
        #token=self.token.repeat(B*T,1).reshape(1,B*T,-1)
        token=global_token.mean(dim=0,keepdim=True)
        # token=global_token.reshape(1,B*T,-1)
        #B=N,B*T,C_e
        memory=x
        x = x.reshape(N,B*T,-1)
        #1,B*T,C_e
        group_query = self.group_query.reshape(1,1,-1).repeat(1,B*T,1)
        group = self.group_tr(group_query,x).reshape(1,B*T,-1)
        tgt=token + group

        group_query2 = group_query.repeat(N,1,1)
        group2 = self.group_tr2(group_query2,x).reshape(N,B*T,-1)
        token2 = self.conv(token.permute(1,2,0)).permute(2,0,1).reshape(N,B*T,-1)
        tgt2 = token2+group2

        


        cluster_losses = []
        for i in range(self.num_STTR_layers):
            memory, tgt, t_loss= self.STTR_module[i](memory, tgt)#,tgt2,joints)
            cluster_losses.append(t_loss)
        cluster_loss = sum(cluster_losses)

        #N,B,T,-1
        memory=memory.reshape(N,B,T,-1).permute(1,2,0,3).reshape(-1,self.embed_features)
        #B*T,-1
        tgt=tgt.reshape(-1,self.embed_features)

        memory = self.dropout(memory)
        tgt = self.dropout(tgt)        

        actions_scores=self.actions_fc(memory)
        actions_scores = actions_scores.reshape(B, T, N, -1)
        actions_scores = actions_scores.mean(dim=1).reshape(B * N, -1)

        activities_scores=self.activities_fc(tgt)
        activities_scores=activities_scores.reshape(B,T,-1).mean(dim=1)

        return actions_scores, activities_scores, cluster_loss,tgt

