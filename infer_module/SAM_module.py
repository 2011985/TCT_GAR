import torch
import torch.nn as nn
class SAM(nn.Module):
    def __init__(self, in_dim, k):
        super(SAM, self).__init__()
        latent_dim = in_dim // 2
        self.k = k

        self.theta = nn.Linear(in_dim, latent_dim, bias=False)
        self.phi = nn.Linear(in_dim, latent_dim, bias=False)
        self.sig = nn.Linear(in_dim,in_dim,bias=False)
        self.W = nn.Linear(in_dim, in_dim, bias=False)
        self.bn = nn.BatchNorm2d(in_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        '''
        :param x: shape [B, N, NFB]
        :return:
        '''
        B, N, _ = x.shape
        phi_x = self.phi(x)
        theta_x= self.theta(x)
        rels = nn.functional.softmax(torch.bmm(phi_x,theta_x.reshape(B,-1,N)),dim=-1)
        alpha = rels.sum(dim=1)+rels.sum(dim=2)
        sel_k_id = torch.argsort(alpha,dim=1,descending=True)[:,:self.k]
        sel_k_id.reshape(-1)
        x_id = torch.Tensor([[i]*self.k for i in range(B)]).reshape(-1).long()
        y_id = sel_k_id.reshape(-1)
        new_x = x[x_id,y_id,:].reshape(B,self.k,-1)
        new_rels = rels[x_id,y_id,:].reshape(B,self.k,-1)[x_id,:,y_id].reshape(B,self.k,self.k)
        sig_x = self.sig(new_x)
        z = self.W(torch.bmm(new_rels,sig_x))+new_x # B,k,NFB
        z_pooled,_ = torch.max(z,dim=1) # B,NFB
        return z_pooled