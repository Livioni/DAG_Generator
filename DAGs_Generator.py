import random,math,argparse
import numpy as np
from numpy.random.mtrand import sample
from matplotlib import patches, pyplot as plt
import networkx as nx
from scipy import sparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='default', type=str)#parameters setting
parser.add_argument('--n', default=10, type=int)          #number of DAG  nodes
parser.add_argument('--max_out', default=2, type=float)   #max out_degree of one node
parser.add_argument('--alpha',default=1,type=float)       #shape 
parser.add_argument('--beta',default=1.0,type=float)      #regularity
args = parser.parse_args()

set_dag_size = [20,30,40,50,60,70,80,90]             #random number of DAG  nodes       
set_max_out = [1,2,3,4,5]                              #max out_degree of one node
set_alpha = [0.5,1.0,1.5]                            #DAG shape
set_beta = [0.0,0.5,1.0,2.0]                         #DAG regularity

def DAGs_generate(mode = 'default', n = 10, max_out = 2,alpha = 1,beta = 1.0):
    ##############################################initialize############################################
    if mode != 'default':
        args.n = random.sample(set_dag_size,1)[0]
        args.max_out = random.sample(set_max_out,1)[0]
        args.alpha = random.sample(set_alpha,1)[0]
        args.beta = random.sample(set_alpha,1)[0]
    else: 
        args.n = n
        args.max_out = max_out
        args.alpha = alpha
        args.beta = beta
        args.prob = 1

    length = math.floor(math.sqrt(args.n)/args.alpha)
    mean_value = args.n/length
    random_num = np.random.normal(loc = mean_value, scale = args.beta,  size = (length,1))    
    ###############################################division#############################################
    position = {'Start':(0,4),'Exit':(10,4)}
    generate_num = 0
    dag_num = 1
    dag_list = [] 
    for i in range(len(random_num)):
        dag_list.append([]) 
        for j in range(math.ceil(random_num[i])):
            dag_list[i].append(j)
        generate_num += len(dag_list[i])

    if generate_num != args.n:
        if generate_num<args.n:
            for i in range(args.n-generate_num):
                index = random.randrange(0,length,1)
                dag_list[index].append(len(dag_list[index]))
        if generate_num>args.n:
            i = 0
            while i < generate_num-args.n:
                index = random.randrange(0,length,1)
                if len(dag_list[index])<=1:
                    continue
                else:
                    del dag_list[index][-1]
                    i += 1

    dag_list_update = []
    pos = 1
    max_pos = 0
    for i in range(length):
        dag_list_update.append(list(range(dag_num,dag_num+len(dag_list[i]))))
        dag_num += len(dag_list_update[i])
        pos = 1
        for j in dag_list_update[i]:
            position[j] = (3*(i+1),pos)
            pos += 5
        max_pos = pos if pos > max_pos else max_pos
        position['Start']=(0,max_pos/2)
        position['Exit']=(3*(length+1),max_pos/2)

    ############################################link#####################################################
    into_degree = [0]*args.n            
    out_degree = [0]*args.n             
    edges = []                          
    pred = 0

    for i in range(length-1):
        sample_list = list(range(len(dag_list_update[i+1])))
        for j in range(len(dag_list_update[i])):
            od = random.randrange(1,args.max_out+1,1)
            od = len(dag_list_update[i+1]) if len(dag_list_update[i+1])<od else od
            bridge = random.sample(sample_list,od)
            for k in bridge:
                edges.append((dag_list_update[i][j],dag_list_update[i+1][k]))
                into_degree[pred+len(dag_list_update[i])+k]+=1
                out_degree[pred+j]+=1 
        pred += len(dag_list_update[i])


    ######################################create start node and exit node################################
    for node,id in enumerate(into_degree):#给所有没有入边的节点添加入口节点作父亲
        if id ==0:
            edges.append(('Start',node+1))
            into_degree[node]+=1

    for node,od in enumerate(out_degree):#给所有没有出边的节点添加出口节点作儿子
        if od ==0:
            edges.append((node+1,'Exit'))
            out_degree[node]+=1

    #############################################plot###################################################
    return edges,into_degree,out_degree,position

def plot_DAG(edges,postion):
    g1 = nx.DiGraph()
    g1.add_edges_from(edges)
    nx.draw_networkx(g1, arrows=True, pos=postion)
    plt.savefig("DAG.png", format="PNG")
    return plt.clf

def search_for_successors(node, edges):
        '''
        寻找后续节点
        :param node: 需要查找的节点id
        :param edges: DAG边信息(注意最好传列表的值（edges[:]）进去而不是传列表的地址（edges）!!!)
        :return: node的后续节点id列表
        '''
        map = {}
        if node == 'Exit': return print("error, 'Exit' node do not have successors!")
        for i in range(len(edges)):
            if edges[i][0] in map.keys():
                map[edges[i][0]].append(edges[i][1])
            else:
                map[edges[i][0]] = [edges[i][1]]
        pred = map[node]
        return pred

def search_for_all_successors(node, edges):
    save = node
    node = [node]
    for ele in node:
        succ = search_for_successors(ele,edges)
        if(len(succ)==1 and succ[0]=='Exit'):
            break
        for item in succ:
            if item in node:
                continue
            else:
                node.append(item)
    node.remove(save)
    return node


def search_for_predecessor(node, edges):
    '''
    寻找前继节点
    :param node: 需要查找的节点id
    :param edges: DAG边信息
    :return: node的前继节点id列表
    '''
    map = {}
    if node == 'Start': return print("error, 'Start' node do not have predecessor!")
    for i in range(len(edges)):
        if edges[i][1] in map.keys():
            map[edges[i][1]].append(edges[i][0])
        else:
            map[edges[i][1]] = [edges[i][0]]
    succ = map[node]
    return succ
##### for my graduation project


def workflows_generator(mode='default', n=10, max_out=2, alpha=1, beta=1.0, t_unit=10, resource_unit=100):
    '''
    随机生成一个DAG任务并随机分配它的持续时间和（CPU，Memory）的需求
    :param mode: DAG按默认参数生成
    :param n: DAG中任务数
    :para max_out: DAG节点最大子节点数
    :return: edges      DAG边信息
             duration   DAG节点持续时间
             demand     DAG节点资源需求数量
             position   作图中的位置
    '''
    t = t_unit  # s   time unit
    r = resource_unit  # resource unit
    edges, in_degree, out_degree, position = DAGs_generate(mode, n, max_out, alpha, beta)
    plot_DAG(edges,position)
    duration = []
    demand = []
    # 初始化持续时间
    for i in range(len(in_degree)):
        if random.random() < args.prob:
            # duration.append(random.uniform(t,3*t))
            duration.append(random.sample(range(0, 3 * t), 1)[0])
        else:
            # duration.append(random.uniform(5*t,10*t))
            duration.append(random.sample(range(5 * t, 10 * t), 1)[0])
    # 初始化资源需求
    for i in range(len(in_degree)):
        if random.random() < 0.5:
            demand.append((random.uniform(0.25 * r, 0.5 * r), random.uniform(0.05 * r, 0.01 * r)))
        else:
            demand.append((random.uniform(0.05 * r, 0.01 * r), random.uniform(0.25 * r, 0.5 * r)))

    return edges, duration, demand, position


edges, duration, demand, position = workflows_generator()
print(edges)