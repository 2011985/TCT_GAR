from backbone.backbone import *
from utils import *
from roi_align.roi_align import RoIAlign  # RoIAlign module


class GCN_Module(nn.Module):
    def __init__(self, cfg):
        super(GCN_Module, self).__init__()

        self.cfg = cfg

        NFR = cfg.num_features_relation

        NG = cfg.num_graph
        N = cfg.num_boxes
        T = cfg.num_frames

        NFG = cfg.num_features_gcn
        NFG_ONE = NFG

        self.fc_rn_theta_list = torch.nn.ModuleList([nn.Linear(NFG, NFR) for i in range(NG)])
        self.fc_rn_phi_list = torch.nn.ModuleList([nn.Linear(NFG, NFR) for i in range(NG)])

        self.fc_gcn_list = torch.nn.ModuleList([nn.Linear(NFG, NFG_ONE, bias=False) for i in range(NG)])

        if cfg.dataset_name == 'volleyball':
            self.nl_gcn_list = torch.nn.ModuleList([nn.LayerNorm([N, NFG_ONE]) for i in range(NG)])
        else:
            self.nl_gcn_list = torch.nn.ModuleList([nn.LayerNorm([NFG_ONE]) for i in range(NG)])

    def forward(self, graph_boxes_features, graph_boxes_positions):
        """
        graph_boxes_features  [B*T,N,NFG]
        """

        # GCN graph modeling
        # Prepare boxes similarity relation
        B, N, NFG = graph_boxes_features.shape
        # print("graph_boxes_features:",graph_boxes_features.size())
        NFR = self.cfg.num_features_relation
        NG = self.cfg.num_graph
        NFG_ONE = NFG

        OH, OW = self.cfg.out_size
        pos_threshold = self.cfg.pos_threshold

        # 计算boxes之间的距离
        graph_boxes_distances = calc_pairwise_distance_3d(graph_boxes_positions, graph_boxes_positions)  # B, N, N

        position_mask = (graph_boxes_distances > (pos_threshold * OW))

        relation_graph = None
        graph_boxes_features_list = []

        for i in range(NG):
            graph_boxes_features_theta = self.fc_rn_theta_list[i](graph_boxes_features)  # B,N,NFR
            graph_boxes_features_phi = self.fc_rn_phi_list[i](graph_boxes_features)  # B,N,NFR

            # graph_boxes_features_theta=self.nl_rn_theta_list[i](graph_boxes_features_theta)
            # graph_boxes_features_phi=self.nl_rn_phi_list[i](graph_boxes_features_phi)

            similarity_relation_graph = torch.matmul(graph_boxes_features_theta,
                                                     graph_boxes_features_phi.transpose(1, 2))  # B,N,N

            similarity_relation_graph = similarity_relation_graph / np.sqrt(NFR)

            similarity_relation_graph = similarity_relation_graph.reshape(-1, 1)  # B*N*N, 1

            # Build relation graph
            relation_graph = similarity_relation_graph

            relation_graph = relation_graph.reshape(B, N, N)
            # print("**relation_graph:", relation_graph.size())

            relation_graph[position_mask] = -float('inf')

            relation_graph = torch.softmax(relation_graph, dim=2)

            # Graph convolution
            one_graph_boxes_features = self.fc_gcn_list[i](
                torch.matmul(relation_graph, graph_boxes_features))  # B, N, NFG_ONE
            one_graph_boxes_features = self.nl_gcn_list[i](one_graph_boxes_features)
            one_graph_boxes_features = F.relu(one_graph_boxes_features, inplace=True)

            graph_boxes_features_list.append(one_graph_boxes_features)

        graph_boxes_features = torch.sum(torch.stack(graph_boxes_features_list), dim=0)  # B, N, NFG
        # print("*relation_graph:", relation_graph.size())

        return graph_boxes_features, relation_graph
