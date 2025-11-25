import numpy as np
import networkx as nx
from collections import defaultdict


def calculate_layer_weight_global(layer_matrices, p=2):
    """
    计算每层的权重。
    """
    weights = {}

    # 预先计算每层的图结构和节点属性
    layer_graphs = {layer: nx.from_scipy_sparse_array(matrix) for layer, matrix in layer_matrices.items()}

    # 移除每层图中的自环
    for G in layer_graphs.values():
        G.remove_edges_from(nx.selfloop_edges(G))

    # 计算每层图的 k核中心性
    layer_k_core = {layer: nx.core_number(G) for layer, G in layer_graphs.items()}

    for layer, G in layer_graphs.items():
        total_I_alpha = 0  # 层内信息的总和
        total_E_alpha = 0  # 层间信息的总和

        for node in G.nodes():
            m_i = layer_k_core[layer][node]  # 节点i的质量（k-core中心性）

            # 计算节点i的层内信息 I_αi
            I_alpha_i = 0
            for j, length in nx.single_source_shortest_path_length(G, node).items():
                if j != node:  # 排除自身
                    m_j = layer_k_core[layer][j]  # 节点j的质量
                    I_alpha_i += (m_i * m_j) / (length ** p)
            total_I_alpha += I_alpha_i

            # 计算节点i的层间信息 E_αi
            E_alpha_i = 0
            layer_neighbors = set(G.neighbors(node))
            for other_layer, other_G in layer_graphs.items():
                if other_layer != layer:
                    other_layer_neighbors = set(other_G.neighbors(node))
                    intersection = len(layer_neighbors & other_layer_neighbors)
                    union = len(layer_neighbors | other_layer_neighbors)
                    E_alpha_i += intersection / (union + 1e-5)  # 加1e-5避免除零错误
            total_E_alpha += E_alpha_i

        # 动态调整 gamma 的值
        if total_I_alpha + total_E_alpha == 0:  # 避免除零错误
            gamma_alpha = 0.5  # 默认值
        else:
            gamma_alpha = total_I_alpha / (total_I_alpha + total_E_alpha)

        gamma_alpha = max(0.0, min(gamma_alpha, 1))

        # 计算层的权重 W_α
        W_alpha = gamma_alpha * total_I_alpha + (1 - gamma_alpha) * total_E_alpha
        weights[layer] = W_alpha

    # 归一化处理
    total_weight = sum(weights.values())
    if total_weight == 0:  # 处理所有权重为0的情况
        return [1.0 / len(layer_matrices) for _ in range(len(layer_matrices))]

    return [weights[layer] / total_weight for layer in sorted(layer_matrices.keys(), key=int)]


# def multilayer_IC_Algorithm(Gs, S_set, p, layer_weights, mc):
#     """
#     多层独立级联传播模型。
#     """
#     layer_count = len(Gs)  # 层数
#     total_influence = 0
#     average_layer_influences = [0 for _ in range(layer_count)]  # 每层的影响力
#
#     # 确保从 1 开始的层号
#     for layer in S_set.keys():
#         S_set[layer] = [node for node in S_set[layer] if Gs[layer].degree(node) > 0]  # 删除度为 0 的节点（不参与传播过程）
#
#     # 将 layer_weights 转换为字典形式，键为层编号
#     layer_weights_dict = {layer: weight for layer, weight in zip(sorted(Gs.keys()), layer_weights)}
#
#     # 进行 mc 次蒙特卡洛模拟
#     for _ in range(mc):
#         global_active_set = set() #全局所有层激活的节点
#         layer_influences = [len(S_set[layer]) for layer in S_set.keys()] #每层影响力
#         current_active_set = {layer: set(S_set[layer]) for layer in S_set.keys()} #当前时间步每层的激活节点
#
#         global_active_set.update(*current_active_set.values())
#         global_active_sets = {layer: set(S_set[layer]) for layer in S_set.keys()} #记录每层的全局激活节点
#
#         new_activate = True
#         while new_activate:
#             new_activate = False
#             new_activate_set = {layer: set() for layer in S_set.keys()} #每层新激活的节点集合
#
#             # 层间传播
#             for source_layer in S_set.keys():
#                 for vi in current_active_set[source_layer]: #遍历当前层的所有激活节点
#                     for target_layer in S_set.keys(): #遍历其他层
#                         if target_layer != source_layer:
#                             # 使用字典形式的 layer_weights  #是否在其他层激活节点
#                             if np.random.random() < layer_weights_dict[source_layer]:
#                                 new_activate_set[target_layer].add(vi)
#                                 global_active_set.add(vi)
#                                 global_active_sets[target_layer].add(vi)
#                                 new_activate = True
#
#             # 层内传播
#             for layer_index in S_set.keys():
#                 G = Gs[layer_index]
#                 for vi in current_active_set[layer_index]:
#                     if vi in G:# 获取当前节点的所有邻居，并排除已激活的邻居
#                         inactive_neighbors = set(G.neighbors(vi)) - global_active_sets[layer_index]
#                         for vj in inactive_neighbors:
#                             if np.random.random() < p:
#                                 new_activate_set[layer_index].add(vj)
#                                 global_active_set.add(vj)
#                                 global_active_sets[layer_index].add(vj)
#                                 new_activate = True
#
#                 layer_influences[layer_index - 1] += len(new_activate_set[layer_index]) #更新每层影响力
#
#             current_active_set = {layer: new_activate_set[layer] for layer in S_set.keys()} #更新每层新激活的节点激活
#
#         total_influence += len(global_active_set)
#         average_layer_influences = [x + y for x, y in zip(average_layer_influences, layer_influences)] #累加每层的影响力
#
#     average_layer_influences = [influence / mc for influence in average_layer_influences]#计算每层的平均影响力
#     average_total_influence = total_influence / mc
#     return average_total_influence, average_layer_influences
def multilayer_IC_Algorithm(Gs, S_set, p, layer_weights, mc):
    """
    多层独立级联传播模型。
    """
    layer_count = len(Gs)  # 层数
    total_influence = 0  # 总影响力
    average_layer_influences = [0 for _ in range(layer_count)]  # 每层的影响力初始化为0

    # 确保从 1 开始的层号
    for layer in S_set.keys():
        S_set[layer] = [node for node in S_set[layer] if Gs[layer].degree(node) > 0]  # 删除度为 0 的节点（不参与传播过程）

    # 将 layer_weights 转换为字典形式，键为层编号
    layer_weights_dict = {layer: weight for layer, weight in zip(sorted(Gs.keys()), layer_weights)}

    # 进行 mc 次蒙特卡洛模拟
    for _ in range(mc):
        global_active_set = set()  # 全局所有层激活的节点
        layer_influences = [len(S_set[layer]) for layer in S_set.keys()]  # 每层的影响力初始化为每层激活节点的数量
        current_active_set = {layer: set(S_set[layer]) for layer in S_set.keys()}  # 当前时间步每层的激活节点

        global_active_set.update(*current_active_set.values())  # 将当前激活的节点添加到全局激活集
        global_active_sets = {layer: set(S_set[layer]) for layer in S_set.keys()}  # 记录每层的全局激活节点

        new_activate = True  # 标记是否有新的节点被激活
        while new_activate:
            new_activate = False  # 假设在这一轮没有新的节点激活
            new_activate_set = {layer: set() for layer in S_set.keys()}  # 每层新激活的节点集合
            layer_to_be_activated_next = {layer: set() for layer in S_set.keys()}  # 用于记录下一时刻层间传播的节点

            # 层内传播（当前时刻）
            for layer_index in S_set.keys():
                G = Gs[layer_index]  # 获取当前层的图
                for vi in current_active_set[layer_index]:  # 遍历当前层的所有激活节点
                    if vi in G:  # 确保节点 vi 存在于当前图中
                        # 获取当前节点的所有邻居，并排除已激活的邻居
                        inactive_neighbors = set(G.neighbors(vi)) - global_active_sets[layer_index]
                        for vj in inactive_neighbors:  # 遍历这些未激活的邻居
                            if np.random.random() < p:  # 通过概率 p 来判断是否激活邻居节点
                                new_activate_set[layer_index].add(vj)
                                global_active_set.add(vj)
                                global_active_sets[layer_index].add(vj)
                                new_activate = True

                layer_influences[layer_index - 1] += len(new_activate_set[layer_index])  # 更新每层的影响力

            # 层间传播（下一时刻）
            for source_layer in S_set.keys():
                for vi in current_active_set[source_layer]:  # 遍历当前层的所有激活节点
                    for target_layer in S_set.keys():  # 遍历其他层
                        if target_layer != source_layer:  # 防止同一层内传播
                            # 使用字典形式的 layer_weights，决定是否激活目标层节点
                            if np.random.random() < layer_weights_dict[source_layer]:
                                layer_to_be_activated_next[target_layer].add(vi)  # 将节点记录为下一时刻层间传播的目标
                                global_active_set.add(vi)
                                global_active_sets[target_layer].add(vi)
                                new_activate = True

            # 更新当前激活节点集合，将下一时刻的层间传播节点加入到激活集合
            current_active_set = {layer: new_activate_set[layer] | layer_to_be_activated_next[layer] for layer in S_set.keys()}

        total_influence += len(global_active_set)  # 更新总影响力
        average_layer_influences = [x + y for x, y in zip(average_layer_influences, layer_influences)]  # 累加每层的影响力

    average_layer_influences = [influence / mc for influence in average_layer_influences]  # 计算每层的平均影响力
    average_total_influence = total_influence / mc  # 计算总的平均影响力
    return average_total_influence, average_layer_influences  # 返回结果