
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool, SAGPooling
import torch
import torch.nn as nn
from torch_geometric.nn.inits import reset
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
import numpy as np

class Encoder1(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder1, self).__init__()
        self.hidden_channels = hidden_channels
        self.conv = GCNConv(in_channels, self.hidden_channels)
        self.prelu = nn.PReLU(self.hidden_channels)

    def forward(self, x, edge_index,weight):




        x1 = self.conv(x, edge_index,edge_weight=weight)


        x1 = self.prelu(x1)
        return x1

class Encoder2(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder2, self).__init__()
        self.hidden_channels = hidden_channels
        self.conv = GCNConv(in_channels, self.hidden_channels)
        self.prelu = nn.PReLU(self.hidden_channels)

    def forward(self, x, edge_index):
        x1 = self.conv(x, edge_index)
        x1 = self.prelu(x1)
        return x1


class Pool(nn.Module):
    def __init__(self, in_channels, ratio=1.0):
        super(Pool, self).__init__()
        self.in_channels=in_channels
        self.sag_pool = SAGPooling(in_channels, ratio)
        self.lin1 = torch.nn.Linear(in_channels * 2, in_channels)



    def forward(self, x, batch,fla,y):

        if fla==1:
            return global_mean_pool(x, batch)


        # print(y)

        importance=y

        x_num = x.size()[0]
        importance=importance.t().reshape((x_num,1)).cuda()

        x=x*importance
        summary=torch.zeros(int(x_num/20),self.in_channels).cuda()###########33
        flag=0
        index=0
        for j in range(x_num):
            summary[index,:]+=x[j]
            flag+=1
            if flag==20:###############
                flag=0
                index+=1
        # print(summary)
        # print(summary.size())
        # exit()
        return summary






class Scorer(nn.Module):
    def __init__(self, hidden_size):
        super(Scorer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

    def forward(self, input1, input2):
        output = torch.sigmoid(torch.sum(input1 * torch.matmul(input2, self.weight), dim=-1))
        return output


class Model(nn.Module):

    def __init__(self, hidden_channels, encoder1,encoder2, pool, scorer):
        super(Model, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.hidden_channels = hidden_channels
        self.pool = pool
        self.scorer = scorer
        self.marginloss = nn.MarginRankingLoss(0.5)
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.scorer)
        reset(self.encoder1)
        reset(self.encoder2)
        reset(self.pool)

    def forward(self, x, edge_index,weight,y,batch=None, index=None):
        r""" Return node and subgraph representations of each node before and after being shuffled """

        hidden1 = self.encoder1(x, edge_index, weight)
        if index is None:
            return hidden1

        z1 = hidden1[index]
        summary1 = self.pool(hidden1,batch,0,y)

        hidden2 = self.encoder2(x, edge_index)
        if index is None:
            return hidden2

        z2 = hidden2[index]
        summary2 = self.pool(hidden2,batch,1,y)


        return z1, z2, summary1, summary2








    def loss(self, hidden1,hidden3, summary1,summary3):
        r"""Computes the margin objective."""
        # print(summary1)
        # print(hidden1)
        # print(summary1.size())
        # print(hidden1.size())
        #
        # print(summary3)
        # print(hidden3)
        # print(summary3.size())
        # print(hidden3.size())
        shuf_index = torch.randperm(summary1.size(0))

        hidden2 = hidden1[shuf_index]
        summary2 = summary1[shuf_index]

        logits_aa = torch.sigmoid(torch.sum(hidden1 * summary1, dim=-1))
        # logits_bb = torch.sigmoid(torch.sum(hidden2 * summary2, dim=-1))
        logits_ab = torch.sigmoid(torch.sum(hidden1 * summary2, dim=-1))
        # logits_ba = torch.sigmoid(torch.sum(hidden2 * summary1, dim=-1))

        logits_ac = torch.sigmoid(torch.sum(hidden1 * summary3, dim=-1))

        # print(logits_aa)
        # print(logits_ab)
        # print(logits_ac)

        ones = torch.ones(logits_aa.size(0)).cuda(logits_aa.device)







        Loss1 = self.marginloss(logits_aa, logits_ab, ones)
        # Loss1=self.MarginLoss(logits_aa, logits_ab)
        # TotalLoss += self.marginloss(logits_bb, logits_ba, ones)

        # print(Loss1)

        Loss2 = self.marginloss(logits_aa, logits_ac, ones)



        TotalLoss = Loss1+0.5*Loss2
        return TotalLoss

    def load_edges(self,save_path):
        with open(save_path + 'test_pos_edges', 'rb') as f:
            pos_edges = pickle.load(f)
        with open(save_path + 'test_neg_edges', 'rb') as f:
            neg_edges = pickle.load(f)

        return pos_edges, neg_edges

    def operator(self, u, v, op='mean'):
        if op == 'mean':
            return (u + v) / 2.0
        elif op == 'l1':
            return np.abs(u - v)
        elif op == 'l2':
            return np.abs(u - v) ** 2
        elif op == 'hadamard':
            return np.multiply(u, v)
        else:
            return None

    def get_dataset_from_embedding(self, embeddings, pos_edges, neg_edges, dict, op='mean'):
        '''
        op can take values from 'mean', 'l1', 'l2', 'hadamard'
        '''

        y = []
        X = []
        edges=[]

        # process positive links
        for u, v, prop in pos_edges:
            # get node representation and average them
            if (u not in dict.keys()) or (v not in dict.keys()):
                continue

            u = dict[u]
            v = dict[v]
            edges.append([u,v,0])

            u_enc = embeddings[u].cpu().detach().numpy()
            v_enc = embeddings[v].cpu().detach().numpy()

            datapoint = self.operator(u_enc, v_enc, op=op)  # (u_enc + v_enc)/2.0

            X.append(datapoint)
            y.append(0.0)

        # process negative links
        for u, v, prop in neg_edges:
            # get node representation and average them
            if (u not in dict.keys()) and (v not in dict.keys()):
                continue

            if (u in dict.keys()) and (v not in dict.keys()):
                u = dict[u]
                u_enc = embeddings[u].cpu().detach().numpy()
                v_enc = u_enc

            if (u not in dict.keys()) and (v in dict.keys()):
                v = dict[v]
                v_enc = embeddings[v].cpu().detach().numpy()
                u_enc = v_enc

            if (u in dict.keys()) and (v in dict.keys()):
                u = dict[u]
                v = dict[v]

                u_enc = embeddings[u].cpu().detach().numpy()
                v_enc = embeddings[v].cpu().detach().numpy()
            edges.append([u,v,1])

            datapoint = self.operator(u_enc, v_enc, op=op)  # (u_enc + v_enc) / 2.0

            X.append(datapoint)
            y.append(1.0)

        dataset = np.array(X), np.array(y),np.array(edges)
        return dataset

    def test(self, emb, dict):

        r"""Evaluates latent space quality via a logistic regression downstream task."""
        #edges_save_basepath = '../dataset/sx-mathoverflow-c2q/'
        # edges_save_basepath = '../dataset/Wikipedia/'
        #edges_save_basepath = '../dataset/reddit/'
        #edges_save_basepath = '../dataset/soc-sign-bitcoinalpha/'
        # edges_save_basepath = '../dataset/ia-enron-employees/'
        #edges_save_basepath = '../dataset/soc-wiki-elec/'
        # edges_save_basepath = '../dataset/ia-escorts-dynamic/'
        edges_save_basepath = '../dataset/fb-forum/'
        # edges_save_basepath = '../dataset/email-dnc/'
        # edges_save_basepath = '../dataset/ia-movielens-user2tags-10m/'
        pos_edges, neg_edges = self.load_edges(edges_save_basepath)
        X, y ,edges= self.get_dataset_from_embedding(emb, pos_edges, neg_edges, dict)

        X_train, X_test, y_train, y_test,x_eg,y_eg = train_test_split(X, y,edges, test_size=0.3)

        logReg = LogisticRegression(solver='lbfgs')
        logReg.fit(X_train, y_train)
        y_score = logReg.predict_proba(X_test)

        acc = logReg.score(X_test, y_test)
        auc = metrics.roc_auc_score(y_true=y_test, y_score=y_score[:, 1])
        #return np.mean(rank), np.mean(hits)
        return acc, auc

    def load_valid_edges(self, save_path):
        with open(save_path + 'valid_pos_edges', 'rb') as f:
            pos_edges = pickle.load(f)
        with open(save_path + 'valid_neg_edges', 'rb') as f:
            neg_edges = pickle.load(f)

        return pos_edges, neg_edges

    def validate(self, emb, dict):
        r"""Evaluates latent space quality via a logistic regression downstream task."""
        #edges_save_basepath = '../dataset/sx-mathoverflow-c2q/'
        # edges_save_basepath = '../dataset/Wikipedia/'
        edges_save_basepath = '../dataset/fb-forum/'
        #edges_save_basepath = '../dataset/soc-sign-bitcoinalpha/'
        # edges_save_basepath = '../dataset/ia-enron-employees/'
        #edges_save_basepath = '../dataset/soc-wiki-elec/'
        #edges_save_basepath = '../dataset/reddit/'
        # edges_save_basepath = '../dataset/ia-retweet-pol/'
        # edges_save_basepath = '../dataset/email-dnc/'
        # edges_save_basepath = '../dataset/ia-movielens-user2tags-10m/'
        pos_edges, neg_edges = self.load_valid_edges(edges_save_basepath)
        X, y ,edges= self.get_dataset_from_embedding(emb, pos_edges, neg_edges, dict)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        logReg = LogisticRegression(solver='lbfgs')
        logReg.fit(X_train, y_train)
        y_score = logReg.predict_proba(X_test)
        acc = logReg.score(X_test, y_test)
        auc = metrics.roc_auc_score(y_true=y_test, y_score=y_score[:, 1])
        return acc, auc