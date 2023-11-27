import networkx as nx
import numpy as np
import pickle
import os
import math
import random
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from model import Encoder1,Encoder2, Scorer, Pool,Model
from subgraph1 import Subgraph
import math
import sklearn.preprocessing
import time
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,recall_score
import warnings
warnings.filterwarnings('ignore')
def load_data_as_graph():
    edges = []
    times = []
    ct=0
    # with open('../dataset/sx-mathoverflow-c2q/sx-mathoverflow-c2q.edges') as f:
    with open('../dataset/fb-forum/fb-forum.edges') as f:
    #with open('../dataset/mooc/mooc.edges') as f:

    # with open('../dataset/Wikipedia/Wikipedia.txt') as f:
    #with open('../dataset/reddit/reddit.edges') as f:
    #with open('../dataset/sx-mathoverflow-c2q/sx-mathoverflow-c2q.edges') as f:
    # with open('../dataset/soc-wiki-elec/soc-wiki-elec.edges') as f:
    # with open('../dataset/ia-escorts-dynamic/ia-escorts-dynamic.edges') as f:
    #with open('../dataset/ia-movielens-user2tags-10m/ia-movielens-user2tags-10m.edges') as f:
    # with open('../dataset/ia-retweet-pol/ia-retweet-pol.edges') as f:
    #with open('../dataset/soc-sign-bitcoinalpha/soc-sign-bitcoinalpha.edges') as f:
        for line in f:
            #if ct == 18207:
            #    break
            tokens = line.strip().split(',')
            u = int(tokens[0])
            v = int(tokens[1])
            time = int(tokens[2])  ####3
            weight = int(tokens[2])
            # if v == 3388 or v == 1389 or v == 1870 or v == 3271 or v == 6336 or v == 3228 or v == 7465 or v == 5837:
            #     continue
            # if u == 3388 or u == 1389 or u == 1870 or u == 3271 or u == 6336 or u == 3228 or u == 7465 or u == 5837:
            #     continue
            # if v == 116 or u == 116:
            #     continue
            times.append(time)
            edges.append((u, v, {'weight': weight, 'time': time}))
            ct += 1
    times = np.array(times)
    normal_time = (times - times.min()) / (times.max() - times.min()) * 10
    weight_edges = []
    n = len(edges)
    for i in range(n):
        u, v, pro = edges[i]
        w = normal_time[i]
        weight_edges.append((u, v, {'weight': w, 'time': times[i]}))

    g = nx.MultiGraph()
    g.add_edges_from(weight_edges)

    #########
    no_connect = []
    flag = 0
    for i in nx.connected_components(g):
        if flag == 0:
            flag = 1
            continue
        for j in i:
            no_connect.append(j)
    g_fianl = nx.MultiGraph()
    edges_fianl = []

    for u, v, pos in weight_edges:  #################
        if u in no_connect or v in no_connect:
            continue
        edges_fianl.append((u, v, pos))
    g_fianl.add_edges_from(edges_fianl)
    g = g_fianl
    ##########

    return g

def get_negative_edge(g, first_node=None):
    if first_node is None:
        first_node = np.random.choice(g.nodes())  # pick a random node
    possible_nodes = set(g.nodes())
    neighbours = [n for n in g.neighbors(first_node)] + [first_node]
    possible_nodes.difference_update(neighbours)  # remove the first node and all its neighbours from the candidates
    second_node = np.random.choice(list(possible_nodes))  # pick second node
    edge = (first_node, second_node, {'weight':1, 'time': None})
    return edge

def create_embedding_and_training_data(g, train_edges_fraction=0.75):
    '''
    Create partition of edges into
     -- embedding edges used for learning the embedding
     -- pos edges : positive example of edges for link prediction task

    :param g: nx graph
    :param train_edges_fraction: what fraction of edges to use for embedding learning
    '''

    nodes = g.nodes()
    train_edges = []
    pos_edges = []
    neg_edges = []

    for node in nodes:
        edges_of_node = []
        for e in g.edges(node, data=True): # only gets outgoing edges
            edges_of_node.append(e)

        edges_of_node = sorted(edges_of_node, key=lambda x: x[2]['time'])
        num_edges = len(edges_of_node)

        # training edges per node
        num_train_edges = int(train_edges_fraction * num_edges)
        train_edges.extend(edges_of_node[:num_train_edges])

        # link prediction positive edges
        pos_edges.extend(edges_of_node[num_train_edges:])

    for i in range(len(pos_edges)):
        n_edge = get_negative_edge(g)
        neg_edges.append(n_edge)


    return train_edges, pos_edges, neg_edges


def edges_to_Data(edges):
    num_edge = len(edges)

    dict = {}
    index = 0
    e = []
    edge = np.zeros((12291, 2))  ##forum:12291 bitcoinalpha:21174 wiki:149185  movielens:57944 mathoverflow:142225    mooc:28523  reddit:156971
    j = 0
    for i in range(num_edge):
        u, v, prop = edges[i]
        if u not in dict.keys():
            dict[u] = index
            index += 1
        if v not in dict.keys():
            dict[v] = index
            index += 1
        if (dict[u], dict[v]) not in e:
            e.append((dict[u], dict[v]))
            edge[j][0] = dict[u]
            edge[j][1] = dict[v]
            j += 1

    print(j)
    g = nx.MultiGraph()
    g.add_edges_from(edges)

    num_node = len(g.nodes())
    print(num_node)
    # exit()

    x = np.identity(num_node)

    edge = edge.T
    adj = torch.LongTensor(edge)
    features = torch.from_numpy(x)
    data = Data()
    data.x = features
    data.edge_index = adj
    return data, dict

#######
def deal_edge(edges):

    g1 = nx.MultiGraph()
    g1.add_edges_from(edges)
    no_connect = []
    flag = 0
    for i in nx.connected_components(g1):
        if flag == 0:
            flag = 1
            continue
        for j in i:
            no_connect.append(j)
    e=[]
    for i,j,pos in edges:
        if i in no_connect or j in no_connect:
            continue
        e.append((i, j, pos))
    #print(nx.number_connected_components(g1))
    return e
    #print(nx.number_connected_components(g1))
 ##############

def getImp(edges):
    num_edge = len(edges)
    dict = {}
    index = 0
    e = []
    for i in range(num_edge):
        u, v, prop = edges[i]
        if u not in dict.keys():
            dict[u] = index
            index += 1
        if v not in dict.keys():
            dict[v] = index
            index += 1
        e.append((dict[u],dict[v],prop['weight']))
    imp = np.zeros(index)
    for u,v,weight in e:
        if imp[u]<weight:
            imp[u]=weight
        if imp[v]<weight:
            imp[v]=weight


    imp= (imp - imp.min()) / (imp.max() - imp.min())

    return imp


if __name__ == '__main__':
    g = load_data_as_graph()

    train_edges, pos_edges, neg_edges = create_embedding_and_training_data(g, train_edges_fraction=0.75)

    save_path = '../dataset/fb-forum/'
    """
    cc = list(zip(pos_edges, neg_edges))
    random.shuffle(cc)
    pos_edges[:], neg_edges[:] = zip(*cc)

    total_len = len(pos_edges)
    valid_len = int(total_len * 0.4)
    valid_pos_edges = pos_edges[0:valid_len]
    valid_neg_edges = neg_edges[0:valid_len]
    test_pos_edges = pos_edges[valid_len:-1]
    test_neg_edges = neg_edges[valid_len:-1]

    if os.path.isfile(save_path + 'valid_pos_edges') and os.stat(save_path + 'valid_pos_edges').st_size != 0:
        with open(save_path + 'valid_pos_edges', 'rb') as f:
            valid_pos_edges = pickle.load(f)
    else:
        with open(save_path + 'valid_pos_edges', 'wb') as f:
            pickle.dump(valid_pos_edges, f)

    if os.path.isfile(save_path + 'valid_neg_edges') and os.stat(save_path + 'valid_neg_edges').st_size != 0:
        with open(save_path + 'valid_neg_edges', 'rb') as f:
            valid_neg_edges = pickle.load(f)
    else:
        with open(save_path + 'valid_neg_edges', 'wb') as f:
            pickle.dump(valid_neg_edges, f)

    if os.path.isfile(save_path + 'test_pos_edges') and os.stat(save_path + 'test_pos_edges').st_size != 0:
        with open(save_path + 'test_pos_edges', 'rb') as f:
            test_pos_edges = pickle.load(f)
    else:
        with open(save_path + 'test_pos_edges', 'wb') as f:
            pickle.dump(test_pos_edges, f)

    if os.path.isfile(save_path + 'test_neg_edges') and os.stat(save_path + 'test_neg_edges').st_size != 0:
        with open(save_path + 'test_neg_edges', 'rb') as f:
            test_neg_edges = pickle.load(f)
    else:
        with open(save_path + 'test_neg_edges', 'wb') as f:
            pickle.dump(test_neg_edges, f)
    """

    #######
    train_edges=deal_edge(train_edges)
    #######


    data, dict = edges_to_Data(train_edges)

    ##############
    imp=getImp(train_edges)

    ##############

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # data = data.to(device)

    ppr_path = '../subgraph/fb-forum'
    subgraph = Subgraph(data.x, data.edge_index, ppr_path, 20, 10)
    subgraph.build(train_edges, dict,imp)

    num_hidden = 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model(
        hidden_channels=num_hidden,
        encoder1=Encoder1(872, num_hidden),
        encoder2=Encoder2(872, num_hidden),  ## forum:872 bitcoin:3610 wiki:6479 movielens:14292 mathoverflow:15694 mooc:1735  reddit:10000
        pool=Pool(in_channels=num_hidden),
        scorer=Scorer(num_hidden)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    batch_size =872 #872 3610 6479 14292 15694 1735  10000
    num_node = data.x.size(0)
    hidden_size = 128


    def train(epoch):
        # Model training
        model.train()
        optimizer.zero_grad()
        sample_idx = random.sample(range(data.x.size(0)), batch_size)
        batch, index = subgraph.search(sample_idx)
        z1,z2, summary1,summary2 = model(batch.x.cuda(), batch.edge_index.cuda(),batch.edge_attr.cuda(),batch.y.cuda(), batch.batch.cuda(), index.cuda())
        loss = model.loss(z1,z2, summary1,summary2)
        loss.backward()
        optimizer.step()
        return loss.item()

    def get_all_node_emb(model):
        # Obtain central node embs from subgraphs
        node_list = np.arange(0, num_node, 1)
        list_size = node_list.size
        z = torch.Tensor(list_size, num_hidden).cuda()
        group_nb = math.ceil(list_size/batch_size)
        for i in range(group_nb):
            maxx = min(list_size, (i + 1) * batch_size)
            minn = i * batch_size
            batch, index = subgraph.search(node_list[minn:maxx])
            #weight2 = torch.ones(165820)  # 40386 165820
            node,_, _,_ = model(batch.x.cuda(), batch.edge_index.cuda(),batch.edge_attr.cuda(),batch.y.cuda(), batch.batch.cuda(), index.cuda())
            z[minn:maxx] = node
        return z


    def test(model):
        # Model testing
        model.eval()
        with torch.no_grad():
            emb = get_all_node_emb(model)

        acc, auc = model.test(emb, dict)
        print('accuracy = {}'.format(acc))
        print('auc = {}'.format(auc))
        return acc, auc


    def validate(model):
        # Model validating
        model.eval()
        with torch.no_grad():
            emb = get_all_node_emb(model)
        acc, auc = model.validate(emb, dict)
        print('accuracy = {}'.format(acc))
        print('auc = {}'.format(auc))
        return acc, auc

    def test_nc(model):
        model.eval()

        with torch.no_grad():
            emb = get_all_node_emb(model)
        dict={}
        with open("../dataset/reddit/reddit.edges") as f:
            for lines in f.readlines():
                line=lines.split()
                dict[int(line[0])]=int(line[3])

        data=[]
        label=[]
        for key in dict.keys():

            data.append(emb[key].cpu().detach().numpy())
            label.append(dict[key])
        train, test, train_label, test_label = train_test_split(data, label, test_size=0.25)
        train = np.array(train)
        train_label = np.array(train_label)
        test = np.array(test)
        test_label = np.array(test_label)


        mlr = OneVsRestClassifier(LogisticRegression(max_iter=10000), n_jobs=-1)
        mlr.fit(train, train_label)
        predict = mlr.predict(test)
        acc=accuracy_score(test_label,predict)
        precision=precision_score(test_label,predict,average='weighted')
        return acc,precision







    print('Start training !!!')
    best_auc = 0
    count=0
    f=0
    epoch=0
    while f==0:
        loss = train(epoch)
        print('epoch = {}, loss = {}'.format(epoch, loss))

        acc, auc = validate(model)
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), 'model.pth')
            count=0
        else:
            count+=1
        if count==100:
            f=1
        epoch+=1


    print("!!!!!!!!!test!!!!!!!!!!")
    model.load_state_dict(torch.load('model.pth'))
    auc = 0
    acc = 0
    for i in range(10):
        a, b = test(model)
        acc += a
        auc += b
    acc = acc / 10
    auc = auc / 10
    print('test average accuracy = {}'.format(acc))
    print('test average auc = {}'.format(auc))

