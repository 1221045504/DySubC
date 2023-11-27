import os
import torch
import numpy as np


from scipy import sparse as sp

from torch_geometric.data import Data, Batch
import networkx as nx
import random
import queue



class Subgraph:
    # Class for subgraph extraction

    def __init__(self, x, edge_index, path, maxsize=50, n_order=10):
        self.x = x
        self.path = path
        self.edge_index = np.array(edge_index)
        self.edge_num = edge_index[0].size(0)
        self.node_num = x.size(0)
        self.maxsize = maxsize

        self.sp_adj = sp.csc_matrix((np.ones(self.edge_num), (edge_index[0], edge_index[1])),
                                    shape=[self.node_num, self.node_num])


        self.neighbor = {}
        self.adj_list = {}
        self.subgraph = {}


    def process_adj_list(self):
        for i in range(self.node_num):
            self.adj_list[i] = set()
        for i in range(self.edge_num):
            u, v = self.edge_index[0][i], self.edge_index[1][i]
            self.adj_list[u].add(v)
            self.adj_list[v].add(u)

    def adjust_edge(self, idx, original):
        # Generate time edges for subgraphs
        dic = {}
        exchange_dic={}
        for i in range(len(idx)):
            dic[idx[i]] = i
            exchange_dic[i]=idx[i]

        new_index = [[], []]
        nodes = set(idx)
        for i in idx:
            edge = list(self.adj_list[i] & nodes)
            # edge = [dic[_] for _ in edge]
            # edge = [_ for _ in edge if _ > i]
            # new_index[0] += len(edge) * [dic[i]]
            new_index[0] += len(edge) * [i]
            new_index[1] += edge

        new_index = np.array(new_index)
        new_index = new_index.T
        g = nx.MultiGraph()
        g.add_edges_from(original)
        u = []
        v = []
        p = []
        for j in range(len(new_index)):
            edges_list = []
            for a, b, pro in g.edges(new_index[j][0], data=True):
                if a == new_index[j][0] and b == new_index[j][1]:
                    edges_list.append([a, b, pro])
            for a, b, pro in g.edges(new_index[j][1], data=True):
                if a == new_index[j][1] and b == new_index[j][0]:
                    edges_list.append([a, b, pro])
            edges_list = sorted(edges_list, key=lambda x: x[2]['time'])
            u.append(dic[edges_list[-1][0]])
            v.append(dic[edges_list[-1][1]])
            p.append(edges_list[-1][2]['weight'])

        time_edges = [u, v]
        # time_edges=np.array(time_edges)

        return torch.LongTensor(time_edges), torch.FloatTensor(p),exchange_dic


    def adjust_x(self, idx):
        # Generate node features for subgraphs
        return self.x[idx]

    def getsubgraph(self,edges,time_edge):
        # print(len(edges))
        # print(len(time_edge))

        edges=edges.T
        # print(len(edges))
        g = nx.Graph()
        g.add_edges_from(edges)
        pr=nx.pagerank(g,alpha=0.85)
        ec=nx.eigenvector_centrality(g)

        time_g = nx.MultiGraph()
        time_g.add_edges_from(time_edge)
        neighbor={}
        distance={}
        for i in g.nodes():
            print('Processing node {}.'.format(i))
            num_neighbor=0
            nei=[]
            q = queue.Queue()
            q.put(i)
            nei.append(i)
            dist=0
            dis={}
            dis[i]=1
            while num_neighbor<self.maxsize-1:
                neighbours=[]
                while not q.empty():
                    index=q.get()
                    for n in g.neighbors(index):
                        neighbours.append((index,n))
                dist+=1

                if (self.maxsize-1-num_neighbor)>=len(neighbours):
                    for u,v in neighbours:
                        if v not in nei:
                            q.put(v)
                            nei.append(v)
                            num_neighbor += 1
                            dis[v]=dist

                else:
                    neighbours_score=[]
                    abc=[]
                    for u,v in neighbours:
                        edges_list = []

                        for a, b, pro in time_g.edges(u, data=True):
                            if a == u and b == v:
                                edges_list.append([a, b, pro])





                        edges_list = sorted(edges_list, key=lambda x: x[2]['time'])

                        #neighbours_score.append(edges_list[-1][2]['weight'] + 5*time_g.degree(v))###soc-wiki-elec



                        neighbours_score.append(edges_list[-1][2]['weight']+5*time_g.degree(v))##fb-forum 


                        #neighbours_score.append(edges_list[-1][2]['weight'] + 5 * time_g.degree(v))#mooc

                        #neighbours_score.append(edges_list[-1][2]['weight']+10*time_g.degree(v))#soc-sign-bitcoinalpha 


                        # neighbours_score.append(edges_list[-1][2]['weight'] + time_g.degree(v))  ###ia-escorts-dynamic
                        # neighbours_score.append(edges_list[-1][2]['weight'] + time_g.degree(v))  ###ia-movielens-user2tags-10m sx-mathoverflow-c2q
                        #neighbours_score.append(edges_list[-1][2]['weight'] + time_g.degree(v))  ###sx-mathoverflow-c2q
                        #neighbours_score.append(edges_list[-1][2]['weight']+1*time_g.degree(v)) ##Wikipedia
                        abc.append(v)


                    sorted_neigh=[abc,neighbours_score]
                    sorted_neigh=np.array(sorted_neigh).T

                    sorted_neigh=sorted_neigh[np.argsort(sorted_neigh[:,1]),:]

                    for j in range(sorted_neigh.shape[0]):
                        if sorted_neigh[j][0] not in nei and num_neighbor<self.maxsize-1:
                            nei.append(sorted_neigh[j][0])
                            q.put(sorted_neigh[j][0])
                            num_neighbor += 1
                            dis[sorted_neigh[j][0]] = dist


            nei=np.array(nei)
            neighbor[i]=nei
            distance[i]=dis

        return neighbor,distance


    def build(self,original_edges,dic,imp):
        # Extract subgraphs for all nodes
        if os.path.isfile(self.path + '_subgraph2_pool') and os.stat(self.path + '_subgraph2_pool').st_size != 0:
            print("Exists subgraph2_pool file")
            self.subgraph = torch.load(self.path + '_subgraph2_pool')
            return

        original = []
        for i in range(len(original_edges)):
            original.append((dic[original_edges[i][0]], dic[original_edges[i][1]], original_edges[i][2]))

        # self.neighbor=self.search_neighbor(self.edge_index)
        self.neighbor,distance = self.getsubgraph(self.edge_index,original)

        self.process_adj_list()
        for i in range(self.node_num):
            print('Processing node {} subgraph2_pool.'.format(i))
            nodes = self.neighbor[i][:self.maxsize]
            x = self.adjust_x(nodes)
            edge,weight ,exchange_dic= self.adjust_edge(nodes,original)

            ###########
            importance = np.zeros(x.size()[0])
            for j in range(x.size()[0]):
                importance[j]=imp[int(exchange_dic[j])]
                importance[j] += 1/distance[i][int(exchange_dic[j])]



            # print(importance)
            importance=torch.FloatTensor(importance)
            sum=torch.sum(importance)
            importance=torch.div(importance,sum)
            # print(importance)
            ###########
            self.subgraph[i] = Data(x=x,edge_index= edge,edge_attr=weight,y=importance)

        # self.getsubgraph()

            # print(self.subgraph[i].edge_attr)
            # print(self.subgraph[i].edge_index)
        # print(self.subgraph[0].edge_index)
        torch.save(self.subgraph, self.path + '_subgraph2_pool')
        # torch.save(self.neighbor, self.path + '_neighbor')

    def search(self, node_list):
        # Extract subgraphs for nodes in the list

        batch = []
        index = []
        size = 0
        for node in node_list:
            batch.append(self.subgraph[node])
            index.append(size)
            size += self.subgraph[node].x.size(0)
        index = torch.tensor(index)
        # print(batch[0].edge_index)
        batch = Batch().from_data_list(batch)
        # print(batch.edge_index)
        return batch, index

