from pymnet import *
import matplotlib.pyplot as plt
import numpy as np
#
#
# def read_multilayer_data(file_path):
#     """
#     读取多层网络数据并提取信息。
#     """
#     layers = set()
#     all_nodes = set()
#     edges = []
#
#     with open(file_path, 'r') as file:
#         for line in file:
#             layer, node1, node2, edge = line.strip().split()
#             layers.add(layer)
#             all_nodes.update([node1, node2])
#             if edge != '0':  # 只添加有连边的记录
#                 edges.append((layer, node1, node2))
#
#     return sorted(layers, key=int), sorted(all_nodes), edges
#
#
# def construct_layer_matrices(layers, all_nodes, edges):
#     """
#     构建每一层的邻接矩阵，确保每层的节点集相同。
#     """
#     layer_matrices = {}
#
#     for layer in layers:
#         n = len(all_nodes)
#         matrix = np.zeros((n, n), dtype=int)
#         for edge in edges:
#             if edge[0] == layer:
#                 i, j = all_nodes.index(edge[1]), all_nodes.index(edge[2])
#                 matrix[i, j] = matrix[j, i] = 1  # 无向图
#         layer_matrices[layer] = matrix
#
#     return layer_matrices
#
#
# def construct_multilayer_network(file_path):
#     """
#     构建多层网络对象，确保每层节点相同。
#     """
#     plt.switch_backend('TkAgg')  # 用于 Matplotlib 的兼容性
#     layers, all_nodes, edges = read_multilayer_data(file_path)
#
#     # 创建多层网络对象
#     multilayer_net = MultilayerNetwork(aspects=1, fullyInterconnected=False)
#
#     # 为所有层添加统一节点集
#     for layer in layers:
#         for node in all_nodes:
#             multilayer_net.add_node(node, layer=int(layer))
#
#     # 添加实际存在的连边
#     for layer, node1, node2 in edges:
#         multilayer_net[node1, node2, int(layer), int(layer)] = 1
#
#     # 构建每一层的邻接矩阵
#     layer_matrices = construct_layer_matrices(layers, all_nodes, edges)
#
#     return multilayer_net, layer_matrices, all_nodes, layers
from scipy.sparse import lil_matrix, csr_matrix
import numpy as np
from matplotlib import pyplot as plt
from pymnet import MultilayerNetwork


def read_multilayer_data(file_path):
    """
    读取多层网络数据并提取信息。
    """
    layers = set()
    all_nodes = set()
    edges = []

    with open(file_path, 'r') as file:
        for line in file:
            layer, node1, node2, edge = line.strip().split()
            layers.add(layer)
            all_nodes.update([node1, node2])
            if edge != '0':  # 只添加有连边的记录
                edges.append((layer, node1, node2))

    return sorted(layers, key=int), sorted(all_nodes), edges


def construct_layer_matrices(layers, all_nodes, edges):
    """
    构建每一层的邻接矩阵，使用稀疏矩阵格式。
    """
    layer_matrices = {}
    n = len(all_nodes)

    # 为每一层创建一个字典，存储节点索引到其邻居的映射
    layer_edges = {layer: [] for layer in layers}

    # 预处理边，按层组织
    for layer, node1, node2 in edges:
        i, j = all_nodes.index(node1), all_nodes.index(node2)
        layer_edges[layer].append((i, j))

    # 为每一层构建稀疏矩阵
    for layer in layers:
        # 使用lil_matrix进行构建（更适合增量构建）
        matrix = lil_matrix((n, n), dtype=int)

        # 填充矩阵
        for i, j in layer_edges[layer]:
            matrix[i, j] = matrix[j, i] = 1  # 无向图

        # 转换为csr_matrix格式以提高后续计算效率
        layer_matrices[layer] = matrix.tocsr()

    return layer_matrices


def construct_multilayer_network(file_path):
    """
    构建多层网络对象，确保每层节点相同。
    """
    plt.switch_backend('TkAgg')  # 用于 Matplotlib 的兼容性
    layers, all_nodes, edges = read_multilayer_data(file_path)

    # 创建多层网络对象
    multilayer_net = MultilayerNetwork(aspects=1, fullyInterconnected=False)

    # 为所有层添加统一节点集
    for layer in layers:
        for node in all_nodes:
            multilayer_net.add_node(node, layer=int(layer))

    # 添加实际存在的连边
    for layer, node1, node2 in edges:
        multilayer_net[node1, node2, int(layer), int(layer)] = 1

    # 构建每一层的邻接矩阵（使用稀疏矩阵）
    layer_matrices = construct_layer_matrices(layers, all_nodes, edges)

    return multilayer_net, layer_matrices, all_nodes, layers

def visualize_multilayer_network(multilayer_net, layers, colors=None, layout='spring', layergap=0.7):
    """
    可视化多层网络。
    """
    if colors is None:
        colors = ['lightgreen', 'pink', 'lightblue', 'yellow', 'cyan', 'purple', 'brown']

    layer_color_dict = {int(layer): colors[i % len(colors)] for i, layer in enumerate(layers)}

    fig = plt.figure(figsize=(20, 16))
    ax = fig.add_subplot(111, projection='3d')
    # draw(multilayer_net, ax=ax, layout=layout, layergap=layergap, layerColorDict=layer_color_dict)
    # plt.show()


# 示例用法
# file_path = "D:\\Users\\14775\\Desktop\\资料\\多层\\多层网络的学习\\多层网络数据集\\London_Multiplex_Transport\\Dataset\\london_transport_multiplex.txt"
# file_path = "D:\\Users\\14775\\Desktop\\资料\\多层\\多层网络的学习\\多层网络数据集\\CKM-Physicians-Innovation_Multiplex_Social\\Dataset\\CKM-Physicians-Innovation_multiplex.txt"
# file_path = "D:\\Users\\14775\\Desktop\\资料\\多层\\多层网络的学习\\多层网络数据集\\London_Multiplex_Transport\\Dataset\\london_transport_multiplex.txt"
# file_path = "D:\\Users\\14775\\Desktop\\资料\\多层\\多层网络的学习\\多层网络数据集\\London_Multiplex_Transport\\Dataset\\london_transport_multiplex.txt"
# file_path = "D:\\Users\\14775\\Desktop\\资料\\多层\\多层网络的学习\\多层网络数据集\\London_Multiplex_Transport\\Dataset\\london_transport_multiplex.txt"
# file_path = "D:\\Users\\14775\\Desktop\\资料\\多层\\多层网络的学习\\多层网络数据集\\arXiv-Netscience_Multiplex_Coauthorship\\Dataset\\arxiv_netscience_multiplex.txt"
file_path = "D:\\Users\\14775\\Desktop\\资料\\多层\\多层网络的学习\\多层网络数据集\\Arabidopsis_Multiplex_Genetic\\Dataset\\arabidopsis_genetic_multiplex.txt"
multilayer_net, layer_matrices, all_nodes, layers = construct_multilayer_network(file_path)
# # print(multilayer_net, layer_matrices, all_nodes, layers)
#
# # 可视化
visualize_multilayer_network(multilayer_net, layers=layers, layout='spring')
