# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender

class FIRA(GeneralRecommender):
    def __init__(self, config, dataset):
        super(FIRA, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.cf_model = config['cf_model']
        self.n_mm_layers = config['n_mm_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.tau = config['tau']
        self.knn_k = config['knn_k']
        self.n_aspects = config['n_aspects']
        self.c_weight = config['c_weight']
        self.reg_weight = config['reg_weight']
        self.uia_weight = config['uia_weight']
        self.mm_fuse_type = config['mm_fuse_type']
        self.t_interval = config['t_interval']
 
        self.n_nodes = self.n_users + self.n_items
        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.adj = self.scipy_matrix_to_sparse_tenser(self.interaction_matrix, torch.Size((self.n_users, self.n_items)))
        self.num_inters, self.norm_adj = self.get_norm_adj_mat()
        self.num_inters = torch.tensor(1.0 / (self.num_inters + 1e-7), dtype=torch.float32).to(self.device)
        self.dim_sqrt = torch.sqrt(torch.tensor(self.embedding_dim, dtype=torch.float32)).to(self.device)
                
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        img_adjs_file = os.path.join(dataset_path, config['dataset']+'-img_adjs0.pt')
        txt_adjs_file = os.path.join(dataset_path, config['dataset']+'-txt_adjs0.pt')

        self.img_adjs = None
        self.txt_adjs = None
        self.t_v_att = None
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=True)
            if os.path.exists(img_adjs_file):
                self.img_adjs = torch.load(img_adjs_file)
            else:
                if self.v_feat.shape[1] <= (self.n_aspects-1)*self.t_interval:
                    raise RuntimeError('self.t_interval is too large for image features')
                self.disen_v = nn.Parameter(nn.init.xavier_uniform_(torch.zeros([self.n_aspects, self.v_feat.shape[1]-(self.n_aspects-1)*self.t_interval, self.embedding_dim])))
            self.item_v_weight = nn.Parameter(nn.init.xavier_normal_(torch.tensor(np.random.randn(self.n_items, self.n_aspects, 1), dtype=torch.float32, requires_grad=True)))
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=True)
            if os.path.exists(txt_adjs_file):
                self.txt_adjs = torch.load(txt_adjs_file)
            else:
                if self.t_feat.shape[1] <= (self.n_aspects-1)*self.t_interval:
                    raise RuntimeError('self.t_interval is too large for text features')
                self.disen_t = nn.Parameter(nn.init.xavier_uniform_(torch.zeros([self.n_aspects, self.t_feat.shape[1]-(self.n_aspects-1)*self.t_interval, self.embedding_dim])))
            self.item_t_weight = nn.Parameter(nn.init.xavier_normal_(torch.tensor(np.random.randn(self.n_items, self.n_aspects, 1), dtype=torch.float32, requires_grad=True)))
        
        if self.mm_fuse_type == 'att':
            self.i_m_weight = nn.Parameter(nn.init.xavier_normal_(torch.tensor(np.random.randn(1,2), dtype=torch.float32, requires_grad=True)))
              
    def pre_epoch_processing(self):
        self.img_adjs = None
        self.txt_adjs = None
        pass
            
    def scipy_matrix_to_sparse_tenser(self, matrix, shape):
        row = matrix.row
        col = matrix.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(matrix.data)
        return torch.sparse.FloatTensor(i, data, shape).to(self.device)
    
    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        return sumArr, self.scipy_matrix_to_sparse_tenser(L, torch.Size((self.n_nodes, self.n_nodes)))
    
    def get_knn_adj_mat(self, embs):
        embs_norm = embs.div(torch.norm(embs, p=2, dim=-1, keepdim=True))
        sim = torch.mm(embs_norm, embs_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        # adj = torch.sparse.FloatTensor(indices, knn_value.flatten(), adj_size)
        # norm
        return self.compute_normalized_laplacian(adj)
    
    def compute_normalized_laplacian(self, adj):
        # adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        indices = adj.coalesce().indices()
        values = adj.coalesce().values()
        row_sum = torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        r_inv_sqrt.masked_fill_(r_inv_sqrt == float('inf'), 0.)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj.shape)
    
    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        return bpr_loss
    
    def ssl_triple_loss(self, emb1, emb2):
        norm_emb1 = F.normalize(emb1, p=2, dim=-1)
        norm_emb2 = F.normalize(emb2, p=2, dim=-1)
        pos_score = torch.exp(torch.mul(norm_emb1, norm_emb2).sum(dim=-1) / self.tau)
        if len(norm_emb2.shape) == 3:
            norm_all_emb = norm_emb2.permute(0, 2, 1) 
        elif len(norm_emb2.shape) == 2:
            norm_all_emb = norm_emb2.T
        all_score = torch.exp(torch.matmul(norm_emb1, norm_all_emb) / self.tau).sum(dim=-1)
        ssl_loss = -torch.log(pos_score / all_score).mean()
        return ssl_loss
    
    def reg_loss(self, *embs):
        reg_loss = 0
        for emb in embs:
            reg_loss += torch.norm(emb, p=2)
        reg_loss /= embs[-1].shape[0]
        return reg_loss
    
    def trans_windows(self, ori_feats):
        new_feats = []
        for f in range(self.n_aspects):
            new_feats.append(ori_feats[:,f*self.t_interval:ori_feats.shape[1]-(self.n_aspects-1-f)*self.t_interval])
        new_feats = torch.stack(new_feats, dim=1)
        return new_feats
    
    def get_collabrative_graph_embs(self):
        u_g_embeddings, i_g_embeddings = None, None
        if self.cf_model == 'mf':
            i_g_embeddings = self.item_id_embedding.weight
            u_g_embeddings = self.user_embedding.weight
        
        elif self.cf_model == 'lightgcn':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for _ in range(self.n_ui_layers):
                ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
                all_embeddings += [ego_embeddings]
            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings
    
    def get_mm_embs(self, mm_adj, i_m_embs):
        for _ in range(self.n_mm_layers):
            i_m_embs= torch.sparse.mm(mm_adj, i_m_embs)
        return i_m_embs
    
    def forward(self):
        # collaborative graph enbedding
        u_g_embeddings, i_g_embeddings = self.get_collabrative_graph_embs()

        # Intra-modal Multi-relation Learning
        if self.t_feat is not None: 
            # local semantic extraction
            if self.txt_adjs is None:
                print('****************construct text adj********************')
                t_feats_chunks = torch.matmul(self.trans_windows(self.text_embedding.weight).permute(1, 0, 2), self.disen_t)
                # t_feats_chunks = self.trans_windows(self.text_embedding.weight).permute(1, 0, 2)
                t_adjs = []
                # multi-relation graph learning
                with torch.no_grad():
                    for i in range(self.n_aspects):
                        c_adj = self.get_knn_adj_mat(t_feats_chunks[i])
                        t_adjs.append(c_adj)
                self.txt_adjs= t_adjs

            t_d_feats = []
            for c_adj in self.txt_adjs:
                i_t_emb = self.get_mm_embs(c_adj, self.item_id_embedding.weight)
                t_d_feats.append(i_t_emb)
                
            t_d_feats = torch.stack(t_d_feats, dim=1)
        
        if self.v_feat is not None:
            # # local semantic extraction
            if self.img_adjs is None:
                print('****************construct image adj********************')
                v_feats_chunks = torch.matmul(self.trans_windows(self.image_embedding.weight).permute(1, 0, 2), self.disen_v)
                # v_feats_chunks = self.trans_windows(self.image_embedding.weight).permute(1, 0, 2)
                v_adjs = []
                # multi-relation graph learning
                with torch.no_grad():
                    for i in range(self.n_aspects):
                        c_adj = self.get_knn_adj_mat(v_feats_chunks[i])
                        v_adjs.append(c_adj)
                self.img_adjs=v_adjs

            v_d_feats = []
            for c_adj in self.img_adjs:
                i_v_emb = self.get_mm_embs(c_adj, self.item_id_embedding.weight)
                v_d_feats.append(i_v_emb)

            v_d_feats = torch.stack(v_d_feats, dim=1)
        
        # Inter-modal Multi-semantic Mapping and lignment
        if self.v_feat is not None and self.t_feat is not None:
            # inter-modal semantic mapping
            tv_w = (t_d_feats @ v_d_feats.permute(0, 2, 1))
            t_v_w = torch.softmax(tv_w, dim=2)
            self.t_v_att = t_v_w
            v_a_feats = t_v_w @ v_d_feats
            v_t_w = torch.softmax(tv_w, dim=1).permute(0, 2, 1)
            t_a_feats  = v_t_w @ t_d_feats

            t_s_feats = t_d_feats + v_a_feats
            t_weight = F.softmax(self.item_t_weight, dim=1)
            t_f_feats = t_weight * t_s_feats
            # t_f_feats = t_s_feats

            v_s_feats = v_d_feats + t_a_feats
            v_weight = F.softmax(self.item_v_weight, dim=1)
            v_f_feats = v_weight * v_s_feats
            # v_f_feats = v_s_feats

            t_f_feats = t_f_feats.mean(dim=1, keepdim=False)
            v_f_feats = v_f_feats.mean(dim=1, keepdim=False)

            if self.mm_fuse_type == 'att':
                i_weight = F.softmax(self.i_m_weight, dim=1)
                i_m_feats = i_weight[0][0] * v_f_feats + i_weight[0][1] * t_f_feats
            if self.mm_fuse_type == 'sum':
                i_m_feats = v_f_feats + t_f_feats
            if self.mm_fuse_type == 'max':
                i_m_feats = torch.maximum(v_f_feats, t_f_feats)

            u_m_feats = torch.sparse.mm(self.adj, i_m_feats) * self.num_inters[:self.n_users]
            
            # Integration with collaborative and multimodal semantics    
            ua_embeddings = u_g_embeddings + F.normalize(u_m_feats)
            ia_embeddings = i_g_embeddings + F.normalize(i_m_feats)
            
        
        return ua_embeddings, ia_embeddings, [v_d_feats, t_d_feats, t_a_feats, v_a_feats, u_g_embeddings, i_g_embeddings, u_m_feats, i_m_feats]

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        u_embs, i_embs, mm_feats = self.forward()
        v_d_feats, t_d_feats, t_a_feats, v_a_feats, u_g_embs, i_g_embs, u_m_feats, i_m_feats = mm_feats

        # bpr loss
        batch_bpr_loss = self.bpr_loss(u_embs[users], i_embs[pos_items], i_embs[neg_items])
        batch_bpr_loss+=((1 - F.cosine_similarity(u_embs[users], i_g_embs[pos_items], dim=-1).mean()) + (1 - F.cosine_similarity(u_embs[users], i_m_feats[pos_items], dim=-1).mean())) * self.uia_weight

        # alignment loss
        c_loss = 0.0
        # self-alignmment loss
        c_loss += self.ssl_triple_loss(t_a_feats[pos_items].permute(1, 0, 2), t_a_feats[pos_items].permute(1, 0, 2))
        c_loss += self.ssl_triple_loss(v_a_feats[pos_items].permute(1, 0, 2), v_a_feats[pos_items].permute(1, 0, 2))
        # cross-modal alignment loss
        c_loss += self.ssl_triple_loss(v_a_feats[pos_items].permute(1, 0, 2), t_d_feats[pos_items].permute(1, 0, 2))
        c_loss += self.ssl_triple_loss(t_a_feats[pos_items].permute(1, 0, 2), v_d_feats[pos_items].permute(1, 0, 2))

        # id-mm self-supervised loss
        c_loss += self.ssl_triple_loss(u_g_embs[users], u_m_feats[users])
        c_loss += self.ssl_triple_loss(i_g_embs[pos_items], i_m_feats[pos_items])

        # reg loss
        batch_reg_loss = self.reg_loss(u_embs[users], i_embs[pos_items], i_embs[neg_items])
        
        # final loss
        loss = batch_bpr_loss + self.c_weight * c_loss + self.reg_weight * batch_reg_loss

        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embs, item_embs, _ = self.forward()
        scores = torch.matmul(user_embs[user], item_embs.T)
        return scores
