import DAGs_Generator as DG
import GCNutils.utils as utils
import GCNutils.models as models
import numpy as np
import networkx as nx
from scipy import sparse
import scipy as sp
import torch
import torch.nn.functional as F
import torch.nn as nn

def convert_to_feature(duration,demand):
    feature = np.array([],dtype=np.float32)
    for line in range(len(duration)):
        feature = np.append(feature,np.array([duration[line],demand[line][0],demand[line][1]],dtype=np.float32))
    feature = sparse.csr_matrix(feature)
    return feature

def admatrix(edges,n):
    '''
    返回一个图的邻接矩阵
    :param edges: 生成图边信息
    :param n: 节点个数，不包括'Start'和 'Exit'
    :return adjacency_matrix: 图的邻接矩阵     稀疏形式  
    '''
    graph = nx.DiGraph(edges)
    ndlist = [i for i in range(1,n)]
    adjacency_matrix = nx.to_scipy_sparse_matrix(G = graph,nodelist = ndlist,dtype = np.float32)
    return adjacency_matrix

def gcn_embedding(edges,duration,demand):
    '''
    使用GCN仅编码DAG图信息，每个节点保留三维信息
    :param mode: DAG按默认参数生成
    :param duration: DAG中工作流信息
    :para demand: DAG中工作流信息
    :return: features 节点信息
    :return: adj    邻接矩阵   
    '''
    feature = convert_to_feature(duration,demand)
    features = utils.normalize(feature)
    features = features.toarray().reshape([-1,3])
    features = torch.FloatTensor(features)
    adj = admatrix(edges,len(duration)+1)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = utils.normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse.csr_matrix(adj)
    adj = utils.sparse_mx_to_torch_sparse_tensor(adj)
    return features,adj


class DecimaGNN(nn.Module):  # 策略网络
    def __init__(self, input_size, output_size):
        super(DecimaGNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear1 = nn.Linear(self.input_size, 16)
        self.linear2 = nn.Linear(16, self.output_size)

    def forward(self, input_feature):
        output = self.linear1(input_feature)
        output = F.leaky_relu(self.linear2(output))
        return output  # 输出动作概率分布


NonLinearNw1 = DecimaGNN(3,3)
NonLinearNw2 = DecimaGNN(3,3)
NonLinearNw3 = DecimaGNN(3,3)

NonLinearNw1.load_state_dict(torch.load('GCN_initialization/NonLinearNw1.pth', map_location=lambda storage, loc: storage))
NonLinearNw2.load_state_dict(torch.load('GCN_initialization/NonLinearNw2.pth', map_location=lambda storage, loc: storage))
NonLinearNw3.load_state_dict(torch.load('GCN_initialization/NonLinearNw3.pth', map_location=lambda storage, loc: storage))

# edges = [(1, 6), (2, 6), (3, 5), (4, 5), (5, 10), (6, 8), (6, 7), ('Start', 1), ('Start', 2), ('Start', 3), ('Start', 4), ('Start', 9), (7, 'Exit'), (8, 'Exit'), (9, 'Exit'), (10, 'Exit')]
# position = {'Start': (0, 10.5), 'Exit': (12, 10.5), 1: (3, 1), 2: (3, 6), 3: (3, 11), 4: (3, 16), 5: (6, 1), 6: (6, 6), 7: (9, 1), 8: (9, 6), 9: (9, 11), 10: (9, 16)}
# duration = [21, 28, 29, 15, 2, 22, 19, 21, 21, 18]
# demand = [(26.920, 3.252), (2.808, 49.927), (28.557, 3.507), (4.845, 37.866), (1.098, 27.089), (46.571, 1.378), (2.068, 35.691), (3.745, 43.280), (1.50, 45.471), (1.799, 35.478)]

edges, duration, demand, _ = DG.workflows_generator('default')

def Decima_encoder(edges,duration,demand):
    '''
    使用Decima编码器编码DAG图信息
    :param duration: 工作流信息
    :param edges: DAG边信息
    :param demand: 工作流信息
    :return: embeddings dag图的节点编码信息，以字典形式储存。
    '''
    raw_embeddings = [] #原始节点feature
    embeddings =  {}  #编码后的feature字典  job_id : embedding

    cpu_demands = [demand[i][0] for i in range(len(demand))]
    memory_demands = [demand[i][1] for i in range(len(demand))]
    for exetime,cpu_demand,memory_demand in zip(duration,cpu_demands,memory_demands):
        raw_embeddings.append([exetime,cpu_demand,memory_demand])
    raw_embeddings = np.array(raw_embeddings,dtype=np.float32)
    raw_embeddings = torch.from_numpy(raw_embeddings)
    embeddings1 = NonLinearNw1(raw_embeddings) #第一层初始编码信息


    pred0 = DG.search_for_predecessor('Exit',edges[:])
    for ele in pred0:
        embeddings[ele] = embeddings1[ele-1].data

    while(len(embeddings.keys())<len(duration)):
        box = []
        for ele in pred0:
            pred = DG.search_for_predecessor(ele,edges[:])
            for i in pred:
                if i in embeddings.keys():
                    continue
                if i == 'Start':
                    continue
                succ = DG.search_for_all_successors(i,edges[:])
                g = torch.tensor([0,0,0],dtype=torch.float32)
                for j in succ:
                    g += NonLinearNw2(embeddings[j])
                embeddings[i] = NonLinearNw3(g) + embeddings1[i]
                box.append(i)
        pred0 = box
    return embeddings
    
informa = Decima_encoder(edges,duration,demand).keys()
print(informa)





    



