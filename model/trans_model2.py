import numpy as np
import torch.nn as nn
from roi_align.roi_align import RoIAlign
from backbone.backbone import *
from utils import build_mlp
from .head import HeadBuilder
from pos_encoding import spatialencoding2d

class Trans_volleyball(nn.Module):
    def __init__(self, cfg):
        super(Trans_volleyball, self).__init__()
        self.cfg = cfg
        T, N, self.J = self.cfg.num_frames, self.cfg.num_boxes,self.cfg.num_joints

        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        self.NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn
        NG = self.cfg.num_graph

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False
        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, self.NFB)
        # self.fc_emb_2 = nn.Linear(OC, self.NFB)
        self.nl_emb_1 = nn.LayerNorm([self.NFB])
        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)

        self.fc_actions = nn.Linear(self.NFB, self.cfg.num_actions)
        self.fc_activities = nn.Linear(self.NFB, self.cfg.num_activities)
        # self.conv2d = nn.Conv2d(288, D,3,padding=1)

        #todo
        # self.backbone = BackboneBuilder(cfg.structure.backbone)
        #self.backbone2=BackboneBuilder(cfg.structure.flow_backbone)
        #self.roi_align = RoIAlign(cfg.structure.crop_h, cfg.structure.crop_w)
        #self.flow_roi_align = RoIAlign(14, 14)
        self.global_head=HeadBuilder(cfg.acy_head)
        #self.flow_global_head = HeadBuilder(cfg.structure.flow_head)
        #self.early_head=HeadBuilder(cfg.structure.early_head)
        self.head = HeadBuilder(cfg.head)
        self.pose_head = HeadBuilder(cfg.pose_head)
        self.pose_head2 = HeadBuilder(cfg.pose_head_infer)
        #self.flow_head = HeadBuilder(cfg.structure.head)

        # self.bbox_fc = nn.Linear(12544,256)
        self.bbox_fc = nn.Sequential(nn.Linear(12800,1024),nn.Linear(1024,256))
        self.pose_fc = nn.Sequential(nn.Linear(34,1024),nn.Linear(1024,256))
        self.pose_fc2 = nn.Linear(256,512)
        self.trans_fc2 = nn.Linear(self.cfg.num_features_boxes,self.cfg.num_features_boxes*self.J)

        #self.flow_fc = nn.Sequential(nn.Linear(12544,1024),nn.Linear(1024,256))
        #self.flow_fc = nn.Linear(12544,256)
        # self.embed_fc_1 = nn.Linear(512,256)
        self.embed_fc_2 = nn.Linear(1024,256)
        # embedding_dim = cfg.joint_initial_feat_dim # 3
        self.joint_init_fea_dim = cfg.joint_initial_feat_dim # 3
        print(f'NFB:{self.NFB}')
        print(f'D:{D}')
        print(f'J*NFB:{self.J*self.NFB}')
        print(f'T*self.joint_init_fea_dim:{T*self.joint_init_fea_dim}')
        self.joint_track_projection_layer = nn.Linear(T*self.joint_init_fea_dim, self.NFB)
        self.person_track_projection_layer = nn.Linear(self.J * self.NFB, self.NFB)
        self.joint_track_projection_layer = build_mlp(input_dim= self.joint_init_fea_dim, 
                                                hidden_dims=[self.NFB], 
                                                output_dim=self.NFB,
                                                use_batchnorm=cfg.projection_batchnorm,
                                                dropout=cfg.projection_dropout)
        self.person_track_projection_layer = build_mlp(input_dim=self.J * self.NFB, 
                                                hidden_dims=[self.NFB], 
                                                output_dim=self.NFB,
                                                use_batchnorm=cfg.projection_batchnorm,
                                                dropout=cfg.projection_dropout)

        self.spatial_encoding = cfg.spatial_encoding
        assert  (self.spatial_encoding in [True,False])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        # self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        # print('Load model states from: ', filepath)

    def loadpose(self,filepath):
        state = torch.load(filepath)
        self.pose_head.load_state_dict(state['pose_head_state_dict'])
        # self.pose_head2.load_state_dict(state['pose_head2_state_dict'])
        self.pose_fc.load_state_dict(state['pose_fc_state_dict'])

    def savepose(self, filepath):
        state = {
            'pose_fc_state_dict': self.pose_fc.state_dict(),
            'pose_head_state_dict': self.pose_head.state_dict(),
            # 'pose_head2_state_dict': self.pose_head2.state_dict(),
        }
        torch.save(state, filepath)
        print('pose module saved to:', filepath)

    def savemodel(self, filepath):
        state = {
            'backbone_state_dict': self.backbone.state_dict(),
            # 'fc_emb_state_dict': self.fc_emb.state_dict(),
            # 'fc_actions_state_dict': self.fc_actions.state_dict(),
            # 'fc_activities_state_dict': self.fc_activities.state_dict()
        }

        torch.save(state, filepath)
        print('model saved to:', filepath)

    def forward(self, imgs, gt_boxes,poses):
        B, T, C, H ,W =imgs.shape
        N = gt_boxes.shape[2]
        J = self.J
        OH, OW = self.cfg.out_size

        imgs = imgs.reshape(B*T,C,H,W)
        imgs_features = self.backbone(imgs)

        features_multiscale = []
        for features in imgs_features:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)
        imgs_features = torch.cat(features_multiscale, dim=1)

        OC, OH, OW = imgs_features.shape[-3:]

        device = imgs_features.device
        if self.spatial_encoding is True:
            sp = spatialencoding2d(OC, OH, OW, device).detach()
            imgs_features+=sp

        #roi align
        boxes_in_flat = torch.reshape(gt_boxes, (B*T*N, 4))
        boxes=boxes_in_flat.reshape(B,T*N,4)
        boxes.requires_grad=False
        boxes_idx = torch.arange(B*T, device=boxes_in_flat.device, dtype=torch.int).reshape(-1,1).repeat(1,N) # B*T, N
        boxes_idx_flat=torch.reshape(boxes_idx, (B*T*N,))

        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(imgs_features,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N,D,K,K
        boxes_features=boxes_features.reshape(B*T*N,-1)
        # Embedding to hidden state
        boxes_features = self.fc_emb_1(boxes_features)  # B*T*N, NFB
        boxes_features = F.relu(boxes_features)
        boxes_features = self.dropout_global(boxes_features)
        boxes_features = boxes_features.reshape(B, T, N, -1)

        # Prepare position mask
        graph_boxes_positions = boxes_in_flat  # B*T*N, 4
        graph_boxes_positions[:, 0] = (graph_boxes_positions[:, 0] + graph_boxes_positions[:, 2]) / 2
        graph_boxes_positions[:, 1] = (graph_boxes_positions[:, 1] + graph_boxes_positions[:, 3]) / 2
        graph_boxes_positions = graph_boxes_positions[:, :2].reshape(B*T, N, 2)  # B*T, N, 2

        #global features computing
        global_token=self.global_head(imgs_features.reshape(B,T,OC,OH,OW))
        # global_token=self.embed_fc_2(global_token)

        
        poses=poses[:,:,:,:,:-1].reshape(B*T*N,-1)
        poses_features=self.pose_fc(poses).reshape(B,T,N,-1)
        # actions_scores=self.fc_actions(poses_features)
        # actions_scores = actions_scores.reshape(B,T,N, -1)

        # activities_scores=self.fc_activities(torch.mean(poses_features,dim=2))

        # if T!=1:
        #     actions_scores1 = actions_scores.mean(dim=1).reshape(B * N, -1)

        #     activities_scores1=activities_scores.reshape(B,T,-1).mean(dim=1)

        # actions_scores1, activities_scores1_t,activities_scores1_s,poses_features = self.pose_head(poses_features,global_token, graph_boxes_positions)

        actions_scores2, activities_scores2,boxes_features,gas,gat,vis = self.pose_head2(boxes_features, global_token, graph_boxes_positions)
        # actions_scores3, activities_scores3, aux_loss = self.head(boxes_features, gr_box_features, poses_features)
 
        # return {'activities':[activities_scores1_s, activities_scores1_t, activities_scores2_s, activities_scores2_t, activities_scores3],'actions':[actions_scores1, actions_scores2, actions_scores3],'aux':aux_loss}
        # return {'activities':[activities_scores1_t,activities_scores1_s,activities_scores2,activities_scores3],'actions':[actions_scores1,actions_scores2,actions_scores3],'aux':aux_loss}
        # return {'activities':[activities_scores3],'actions':[actions_scores3],'aux':aux_loss}
        # return {'activities':[activities_scores1_s,activities_scores1_t],'actions':[actions_scores1]}
        return {'activities':[activities_scores2],'actions':[actions_scores2], 'graph':[gas,gat,vis]}
        # return actions_scores1,activities_scores1
        #return [actions_scores1,actions_scores2], [activities_scores1,activities_scores2], aux_loss1+aux_loss2
