import random
from numpy import *
from functools import reduce

def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

class Node(object):
    def __init__(self,layer_index,node_index):
        self.layer_index=layer_index
        self.node_index=node_index
        self.downstream=[]
        self.upstream=[]
        self.output=0
        self.delta=0
    
    #设置节点的输出值，如果节点属于输入层会用到这个函数
    def set_output(self,output):
        self.output=output 
    
    #添加一个到下游节点的连接
    def append_downstream_connection(self,conn):
        self.downstream.append(conn)
        
    #添加一个到上游节点的连接
    def append_upstream_connection(self,conn):
        self.upstream.append(conn)
        
    #计算节点的输出
    def calc_output(self):
        output=reduce(lambda ret,conn:ret+conn.upstream_node.output*conn.weight,self.upstream,0)
        self.output=sigmoid(output)
        
    #计算隐藏层delta
    def calc_hidden_layer_delta(self):
        downstream_delta=reduce(lambda ret,conn:ret+conn.downstream_node.delta*conn.weight,self.downstream,0.0)
        self.delta=self.output*(1-self.output)*downstream_delta
        
    #计算输出层delta
    def calc_output_layer_delta(self,label):
        self.delta=self.output*(1-self.output)*(label-self.output)
        
    #打印节点信息
    def __str__(self):
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str 

#实现一个输出恒为1的节点（计算偏置项Wb时需要）
class ConstNode(object):
    def __init__(self,layer_index,node_index):
        self.layer_index=layer_index
        self.node_index=node_index
        self.downstream=[]
        self.output=1
    
    #添加一个到下游节点的连接
    def append_downstream_connection(self,conn):
        self.downstream.append(conn)
        
    #计算隐藏层delta
    def calc_hidden_layer_delta(self):
        downstream_delta=reduce(lambda ret,conn:ret+conn.downstream_node.delta*conn.weight,self.downstream,0.0)
        self.delta=self.output*(1-self.output)*downstream_delta
        
    #打印节点信息
    def __str__(self):
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str

#Layer对象，负责初始化一层。此外，作为Node的集合对象，提供对Node集合的操作
class Layer(object):
    def __init__(self,layer_index,node_count):
        self.layer_index=layer_index
        self.nodes=[]
        for i in range(node_count):
            self.nodes.append(Node(layer_index,i))
        self.nodes.append(ConstNode(layer_index,node_count))
        
    #设置层的输出值，如果层是输入层会用到这个函数
    def set_output(self,data):
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])
            
    #计算层的输出向量
    def calc_output(self):
        for node in self.nodes[:-1]:#不包括最后一个节点,即ConstNode
            node.calc_output()
            
    #打印层的信息
    def dump(self):
        for node in self.nodes:
            print(node)
            
#Connection对象，主要记录连接的权重，以及该连接所关联的上下游节点
class Connection(object):
    def __init__(self,upstream_node,downstream_node):
        self.upstream_node=upstream_node
        self.downstream_node=downstream_node
        self.weight=random.uniform(-0.1,0.1)
        self.gradient=0.0
        
    def calc_gradient(self):
        self.gradient=self.downstream_node.delta*self.upstream_node.output
        
    def get_gradient(self):
        return self.gradient
    
    def update_weight(self,rate):
        self.calc_gradient()
        self.weight+=rate*self.gradient
        
    def __str__(self):
        return '(%u-%u) -> (%u-%u) = %f' % (
            self.upstream_node.layer_index, 
            self.upstream_node.node_index,
            self.downstream_node.layer_index, 
            self.downstream_node.node_index, 
            self.weight)
    
#Connections对象，提供Connection集合操作
class Connections(object):
    def __init__(self):
        self.connections=[]
    
    def add_connection(self,connection):
        self.connections.append(connection)
        
    def dump(self):
        for conn in self.connections:
            print(conn)
          
#Network对象，提供API
class Network(object):
    #初始化一个全连接神经网络，layers：二维数组，描述神经网络每层节点数
    def __init__(self,layers):
        self.connections=Connections()
        self.layers=[]
        layer_count=len(layers)
        node_count=0
        for i in range(layer_count):
            self.layers.append(Layer(i,layers[i]))
        for layer in range(layer_count-1):
            connections=[Connection(upstream_node,downstream_node)
                        for upstream_node in self.layers[layer].nodes
                        for downstream_node in self.layers[layer+1].nodes[:-1]]
            for conn in connections:
                self.connections.add_connection(conn)
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)
    
    #训练神经网络
    def train(self,labels,data_set,rate,iteration):
        for i in range(iteration):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d],data_set[d],rate)
                # print 'sample %d training finished' % d
    
    #用一个样本训练网络
    def train_one_sample(self,label,sample,rate):
        self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rate)
    
    #计算每个节点的delta
    #self.layers[-2::-1]是Python中一种切片操作，用于获取列表或其他可迭代对象的子序列。
    #-2表示从倒数第二个元素开始。在这里，它指的是self.layers列表中倒数第二个元素，即倒数第二层的节点列表。
    #::-1表示反向迭代，即从倒数第二个元素开始逆序向前遍历。这样做的目的是从倒数第二层开始，依次向前遍历神经网络的隐藏层。
    #因此，self.layers[-2::-1]返回了从倒数第二层开始到第一层（不包括第一层）的所有隐藏层的节点列表。
    def calc_delta(self,label):
        output_nodes=self.layers[-1].nodes
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()
                
    #更新每个连接的权重
    def update_weight(self,rate):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)
                
    #计算每个连接的梯度
    def calc_gradient(self):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()
                    
    #获得网络在一个样本下，每个连接上的梯度
    def get_gradient(self,label,sample):
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()
        
    #根据输入的样本预测输出值
    def predict(self,sample):
        self.layers[0].set_output(sample)
        for i in range(1,len(self.layers)):
            self.layers[i].calc_output()
        return list(map(lambda node:node.output,self.layers[-1].nodes[:-1]))
        #获取这一层的节点时去掉最后一个节点。不过，最后一个节点通常是用于偏置项的
    
    #打印网络信息
    def dump(self):
        for layer in self.layers:
            layer.dump()
       
class Normalizer(object):
    def __init__(self):
        self.mask = [
            0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80
        ]

    def norm(self, number):
        return list(map(lambda m: 0.9 if number & m else 0.1, self.mask))

    def denorm(self, vec):
        binary = list(map(lambda i: 1 if i > 0.5 else 0, vec))
        for i in range(len(self.mask)):
            binary[i] = binary[i] * self.mask[i]
        return reduce(lambda x,y: x + y, binary)


def mean_square_error(vec1, vec2):
    return 0.5 * reduce(lambda a, b: a + b, 
                        list(map(lambda v: (v[0] - v[1]) * (v[0] - v[1]),
                            zip(vec1, vec2))
                        )
                 )

def gradient_check(network, sample_feature, sample_label):
    '''
    梯度检查
    network: 神经网络对象
    sample_feature: 样本的特征
    sample_label: 样本的标签
    '''
    # 计算网络误差
    network_error = lambda vec1, vec2: \
            0.5 * reduce(lambda a, b: a + b, 
                      list(map(lambda v: (v[0] - v[1]) * (v[0] - v[1]),
                          zip(vec1, vec2))))

    # 获取网络在当前样本下每个连接的梯度
    network.get_gradient(sample_feature, sample_label)

    # 对每个权重做梯度检查    
    for conn in network.connections.connections: 
        # 获取指定连接的梯度
        actual_gradient = conn.get_gradient()
    
        # 增加一个很小的值，计算网络的误差
        epsilon = 0.0001
        conn.weight += epsilon
        error1 = network_error(network.predict(sample_feature), sample_label)
    
        # 减去一个很小的值，计算网络的误差
        conn.weight -= 2 * epsilon # 刚才加过了一次，因此这里需要减去2倍
        error2 = network_error(network.predict(sample_feature), sample_label)
    
        # 根据式6计算期望的梯度值
        expected_gradient = (error2 - error1) / (2 * epsilon)
    
        # 打印
        print('expected gradient: \t%f\nactual gradient: \t%f' % (
            expected_gradient, actual_gradient))

def gradient_check_test():
    net = Network([2, 2, 2])
    sample_feature = [0.9, 0.1]
    sample_label = [0.9, 0.1]
    gradient_check(net, sample_feature, sample_label)        
        
''''
首先，代码创建了一个 Normalizer 的实例，这很可能是一个用于对数据进行归一化处理的类的实例。
然后，代码初始化了两个空列表 data_set 和 labels，用于存储训练数据和相应的标签。
接着，代码通过一个循环从 0 到 255（不包括 256），每次增加 8 的步长，生成一个随机整数，然后将这个整数进行归一化处理，并将归一化后的值添加到 data_set 和 labels 列表中。
最后，函数返回了生成的标签列表和数据集列表。
需要注意的是，这段代码中使用了 random.uniform(0, 256) 来生成 0 到 256 之间的随机数，然后将其进行归一化处理。'''
def train_data_set():
    normalizer = Normalizer()
    data_set = []
    labels = []
    for i in range(0, 256, 8):
        n = normalizer.norm(int(random.uniform(0, 256)))
        data_set.append(n)
        labels.append(n)
    return labels, data_set


def train(network):
    labels, data_set = train_data_set()
    network.train(labels, data_set, 0.3, 50)


def test(network, data):
    normalizer = Normalizer()
    norm_data = normalizer.norm(data)
    predict_data = network.predict(norm_data)
    print('\ttestdata(%u)\tpredict(%u)' % (
        data, normalizer.denorm(predict_data)))

def correct_ratio(network):
    normalizer = Normalizer()
    correct = 0.0;
    for i in range(256):
        if normalizer.denorm(network.predict(normalizer.norm(i))) == i:
            correct += 1.0
    print('correct_ratio: %.2f%%' % (correct / 256 * 100))

if __name__ == '__main__':
    net = Network([8, 3, 8])
    train(net)
    
    net.dump()
    correct_ratio(net)