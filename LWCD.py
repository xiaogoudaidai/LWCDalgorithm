import numpy as np
import networkx as nx
import time
import pandas as pd
from collections import defaultdict
import infomap
from scipy.sparse import csr_matrix
import common as cm
import MWIC1



# def calculate_layer_weight_global(layer_matrices, p=2):
#     """
#     计算每层的权重。
#     """
#     weights = {}
#
#     # 预先计算每层的图结构和节点属性
#     layer_graphs = {layer: nx.from_scipy_sparse_array(matrix) for layer, matrix in layer_matrices.items()}
#
#     # 移除每层图中的自环
#     for G in layer_graphs.values():
#         G.remove_edges_from(nx.selfloop_edges(G))
#
#     # 计算每层图的 k核中心性
#     layer_k_core = {layer: nx.core_number(G) for layer, G in layer_graphs.items()}
#
#     for layer, G in layer_graphs.items():
#         total_I_alpha = 0  # 层内信息的总和
#         total_E_alpha = 0  # 层间信息的总和
#
#         for node in G.nodes():
#             m_i = layer_k_core[layer][node]  # 节点i的质量（k-core中心性）
#
#             # 计算节点i的层内信息 I_αi
#             I_alpha_i = 0
#             for j, length in nx.single_source_shortest_path_length(G, node).items():
#                 if j != node:  # 排除自身
#                     m_j = layer_k_core[layer][j]  # 节点j的质量
#                     I_alpha_i += (m_i * m_j) / (length ** p)
#             total_I_alpha += I_alpha_i
#
#             # 计算节点i的层间信息 E_αi
#             E_alpha_i = 0
#             layer_neighbors = set(G.neighbors(node))
#             for other_layer, other_G in layer_graphs.items():
#                 if other_layer != layer:
#                     other_layer_neighbors = set(other_G.neighbors(node))
#                     intersection = len(layer_neighbors & other_layer_neighbors)
#                     union = len(layer_neighbors | other_layer_neighbors)
#                     E_alpha_i += intersection / (union + 1e-5)  # 加1e-5避免除零错误
#             total_E_alpha += E_alpha_i
#
#         # 动态调整 gamma 的值
#         if total_I_alpha + total_E_alpha == 0:  # 避免除零错误
#             gamma_alpha = 0.5  # 默认值
#         else:
#             gamma_alpha = total_I_alpha / (total_I_alpha + total_E_alpha)
#
#         gamma_alpha = max(0.0, min(gamma_alpha, 1))
#
#         # 计算层的权重 W_α
#         W_alpha = gamma_alpha * total_I_alpha + (1 - gamma_alpha) * total_E_alpha
#         weights[layer] = W_alpha
#
#     # 归一化处理
#     total_weight = sum(weights.values())
#     if total_weight == 0:  # 处理所有权重为0的情况
#         return [1.0 / len(layer_matrices) for _ in range(len(layer_matrices))]
#
#     return [weights[layer] / total_weight for layer in sorted(layer_matrices.keys(), key=int)]

#求和归一化
def calculate_layer_weight_global(layer_matrices, p=2):
    weights = {}

    # 预先计算每层的图结构和节点属性
    layer_graphs = {layer: nx.from_scipy_sparse_array(matrix) for layer, matrix in layer_matrices.items()}

    # 移除每层图中的自环
    for G in layer_graphs.values():
        G.remove_edges_from(nx.selfloop_edges(G))

    # 计算每层图的 k-core中心性
    layer_k_core = {layer: nx.core_number(G) for layer, G in layer_graphs.items()}

    # 计算每层的层内信息和层间信息
    total_I_alpha_all = 0  # 所有层的层内信息总和
    total_E_alpha_all = 0  # 所有层的层间信息总和

    # 首先计算所有层的 `I_α` 和 `E_α` 的总和
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

        total_I_alpha_all += total_I_alpha
        total_E_alpha_all += total_E_alpha

        # 输出每层归一化之前的层内信息和层间信息
        print(f"层 {layer} 没有归一化之前的层内信息: {total_I_alpha}, 层间信息: {total_E_alpha}")

    # 归一化 I_α 和 E_α 的总和
    I_alpha_prime_all = total_I_alpha_all
    E_alpha_prime_all = total_E_alpha_all

    # 归一化各层的 I_α 和 E_α
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

        # 归一化当前层的 I_α 和 E_α
        I_alpha_prime = total_I_alpha / I_alpha_prime_all if I_alpha_prime_all != 0 else total_I_alpha
        E_alpha_prime = total_E_alpha / E_alpha_prime_all if E_alpha_prime_all != 0 else total_E_alpha
        print(f"层 {layer} 归一化后的层内信息: {I_alpha_prime}, 层间信息: {E_alpha_prime}")

        # 动态调整 gamma 的值
        if I_alpha_prime + E_alpha_prime == 0:  # 避免除零错误
            gamma_alpha = 0.5  # 默认值
        else:
            gamma_alpha = I_alpha_prime / (I_alpha_prime + E_alpha_prime)

        gamma_alpha = max(0.0, min(gamma_alpha, 1))

        # 计算层的权重 W_α
        W_alpha = gamma_alpha * I_alpha_prime + (1 - gamma_alpha) * E_alpha_prime
        weights[layer] = W_alpha

    # 归一化处理所有层的权重
    total_weight = sum(weights.values())
    if total_weight == 0:  # 处理所有权重为0的情况
        return [1.0 / len(layer_matrices) for _ in range(len(layer_matrices))]

    return [weights[layer] / total_weight for layer in sorted(layer_matrices.keys(), key=int)]


def allocate_nodes_based_on_weights(layer_weights, total_nodes):
    """
    根据层权重比例分配每层的节点数量。
    """
    total_weight = sum(layer_weights)
    proportions = [weight / total_weight for weight in layer_weights]
    allocated_nodes = [int(round(total_nodes * p)) for p in proportions]

    total_allocated = sum(allocated_nodes)
    if total_allocated != total_nodes:
        diff = total_nodes - total_allocated
        sorted_indices = np.argsort(layer_weights)[::-1]
        for idx in sorted_indices:
            if diff == 0:
                break
            allocated_nodes[idx] += np.sign(diff)
            diff -= np.sign(diff)

    return allocated_nodes


def detect_communities_infomap(G):
    """
    使用 Infomap 算法进行社区检测。
    """
    im = infomap.Infomap("--two-level")  # 使用两级社区结构
    for edge in G.edges():
        im.addLink(*edge)
    im.run()

    # 构建社区映射
    community_to_nodes = defaultdict(list)
    node_to_community = {}
    for node, module in im.modules:
        community_to_nodes[module].append(node)
        node_to_community[node] = module

    return node_to_community, dict(community_to_nodes)


def select_nodes_by_community_size(G, num_nodes_to_select):
    """
    基于 Infomap 社区检测结果选择节点。
    """
    # 使用 Infomap 进行社区检测
    node_to_community, community_to_nodes = detect_communities_infomap(G)

    # 计算每个社区的大小占比并分配节点
    total_nodes = len(G)
    community_allocations = {}
    exact_allocations = {
        comm: (len(nodes) / total_nodes) * num_nodes_to_select
        for comm, nodes in community_to_nodes.items()
    }

    # 首先分配整数部分
    community_allocations = {
        comm: int(alloc)
        for comm, alloc in exact_allocations.items()
    }

    # 计算还需要分配的剩余节点数
    remaining = num_nodes_to_select - sum(community_allocations.values())

    if remaining > 0:
        # 按社区大小排序
        sorted_communities = sorted(community_to_nodes.items(),
                                   key=lambda x: len(x[1]),
                                   reverse=True)

        # 循环分配剩余节点给最大的社区
        i = 0
        while remaining > 0:
            comm = sorted_communities[i % len(sorted_communities)][0]
            community_allocations[comm] += 1
            remaining -= 1
            i += 1

    # 在每个社区内选择节点
    selected_nodes = []
    for comm_label, num_to_select in community_allocations.items():
        if num_to_select == 0:
            continue

        comm_nodes = community_to_nodes[comm_label]
        num_to_select = min(num_to_select, len(comm_nodes))

        # 计算社区内每个节点的度
        node_scores = {node: G.degree(node) for node in comm_nodes}

        # 选择度最高的节点
        comm_selected = sorted(node_scores.items(),
                               key=lambda x: x[1],
                               reverse=True)[:num_to_select]
        selected_nodes.extend([node for node, _ in comm_selected])

    # 如果选择的节点数量不够，从未分配满的最大社区中选择
    while len(selected_nodes) < num_nodes_to_select:
        for comm_label in sorted(community_to_nodes,
                                 key=lambda x: len(community_to_nodes[x]),
                                 reverse=True):
            comm_nodes = set(community_to_nodes[comm_label]) - set(selected_nodes)
            if not comm_nodes:
                continue

            # 选择一个额外的节点
            node_scores = {node: G.degree(node) for node in comm_nodes}
            extra_node = max(node_scores.items(), key=lambda x: x[1])[0]
            selected_nodes.append(extra_node)

            if len(selected_nodes) == num_nodes_to_select:
                break

    # 确保不会选择超过要求数量的节点
    selected_nodes = selected_nodes[:num_nodes_to_select]

    assert len(selected_nodes) == num_nodes_to_select, \
        f"选择的节点数量 {len(selected_nodes)} 不等于目标数量 {num_nodes_to_select}"

    return selected_nodes


def multilayer_IC_Algorithm(Gs, S_set, p, layer_weights, mc):
    """
    多层独立级联传播模型。
    """
    layer_count = len(Gs)  # 层数
    total_influence = 0
    average_layer_influences = [0 for _ in range(layer_count)]  # 每层的影响力

    # 确保从 1 开始的层号
    for layer in S_set.keys():
        S_set[layer] = [node for node in S_set[layer] if Gs[layer].degree(node) > 0]  # 删除度为 0 的节点

    # 将 layer_weights 转换为字典形式，键为层编号
    layer_weights_dict = {layer: weight for layer, weight in zip(sorted(Gs.keys()), layer_weights)}

    # 进行 mc 次蒙特卡洛模拟
    for _ in range(mc):
        global_active_set = set()
        layer_influences = [len(S_set[layer]) for layer in S_set.keys()]
        current_active_set = {layer: set(S_set[layer]) for layer in S_set.keys()}

        global_active_set.update(*current_active_set.values())
        global_active_sets = {layer: set(S_set[layer]) for layer in S_set.keys()}

        new_activate = True
        while new_activate:
            new_activate = False
            new_activate_set = {layer: set() for layer in S_set.keys()}

            # 层间传播
            for source_layer in S_set.keys():
                for vi in current_active_set[source_layer]:
                    for target_layer in S_set.keys():
                        if target_layer != source_layer:
                            # 使用字典形式的 layer_weights
                            if np.random.random() < layer_weights_dict[source_layer]:
                                new_activate_set[target_layer].add(vi)
                                global_active_set.add(vi)
                                global_active_sets[target_layer].add(vi)
                                new_activate = True

            # 层内传播
            for layer_index in S_set.keys():
                G = Gs[layer_index]
                for vi in current_active_set[layer_index]:
                    if vi in G:
                        inactive_neighbors = set(G.neighbors(vi)) - global_active_sets[layer_index]
                        for vj in inactive_neighbors:
                            if np.random.random() < p:
                                new_activate_set[layer_index].add(vj)
                                global_active_set.add(vj)
                                global_active_sets[layer_index].add(vj)
                                new_activate = True

                layer_influences[layer_index - 1] += len(new_activate_set[layer_index])

            current_active_set = {layer: new_activate_set[layer] for layer in S_set.keys()}

        total_influence += len(global_active_set)
        average_layer_influences = [x + y for x, y in zip(average_layer_influences, layer_influences)]

    average_layer_influences = [influence / mc for influence in average_layer_influences]
    average_total_influence = total_influence / mc
    return average_total_influence, average_layer_influences


if __name__ == '__main__':
    # 参数设置
    # file_path = "D:\\Users\\14775\\Desktop\\资料\\多层\\多层网络的学习\\多层网络数据集\\arXiv-Netscience_Multiplex_Coauthorship\\Dataset\\arxiv_netscience_multiplex.txt"
    # file_path = "D:\\Users\\14775\\Desktop\\资料\\多层\\多层网络的学习\\多层网络数据集\\London_Multiplex_Transport\\Dataset\\london_transport_multiplex.txt"
    # file_path = "D:\\Users\\14775\\Desktop\\资料\\多层\\多层网络的学习\\多层网络数据集\\Celegans_Multiplex_Genetic\\Dataset\\celegans_genetic_multiplex.txt"
    # file_path = "D:\\Users\\14775\\Desktop\\资料\\多层\\多层网络的学习\\多层网络数据集\\合成网络\\BA-ER-WS.txt"
    # file_path = "D:\\Users\\14775\\Desktop\\资料\\多层\\多层网络的学习\\多层网络数据集\\CKM-Physicians-Innovation_Multiplex_Social\\Dataset\\CKM-Physicians-Innovation_multiplex.txt"
    # file_path = "D:\\Users\\14775\\Desktop\\资料\\多层\\多层网络的学习\\多层网络数据集\\Arabidopsis_Multiplex_Genetic\\Dataset\\arabidopsis_genetic_multiplex.txt"
    # file_path = "D:\\Users\\14775\\Desktop\\资料\\多层\\多层网络的学习\\多层网络数据集\\MoscowAthletics2013_Multiplex_Social\\Dataset\\MoscowAthletics2013_multiplex.txt"
    # file_path = "D:\\Users\\14775\\Desktop\\资料\\多层\\多层网络的学习\\多层网络数据集\\Mus_Multiplex_Genetic\\Dataset\\mus_genetic_multiplex.txt"
    # file_path = "D:\\Users\\14775\\Desktop\\资料\\多层\\多层网络的学习\\多层网络数据集\\合成网络\\BA.txt"
    # file_path = "D:\\Users\\14775\\Desktop\\资料\\多层\\多层网络的学习\\多层网络数据集\\合成网络\\ER.txt"
    # file_path = "D:\\Users\\14775\\Desktop\\资料\\多层\\多层网络的学习\\多层网络数据集\\合成网络\\WS.txt"
    # file_path=  "D:\\Users\\14775\\Desktop\\资料\\多层\\多层网络的学习\\多层网络数据集\\FF-TW-YT\\FF-TW-YT.txt"
    file_path =  "D:\\Users\\14775\\Desktop\\资料\\多层\\多层网络的学习\\多层网络数据集\\Multi_lastfm_asia\\lastfm.txt"
    p = 0.01# 传播概率
    mc = 1000  # Monte Carlo模拟次数

    # 构建多层网络
    multilayer_net, layer_matrices, all_nodes, layers = cm.construct_multilayer_network(file_path)
    Gs = {int(layer): nx.Graph(layer_matrices[layer]) for layer in layer_matrices}
    num_layers = len(layer_matrices)
    # start_time = time.time()
    # 计算层权重
    layer_weights = calculate_layer_weight_global(layer_matrices)
    print(f"层权重: {layer_weights}")

    results = []


    for total_budget in range(50, 701, 50):  # 总预算从50到500，步长50
        print(f"\n===== 处理总预算 {total_budget} 个节点 =====")
        start_time = time.time()

        # 分配每层节点数量（基于层权重和总预算）
        allocated_nodes = allocate_nodes_based_on_weights(layer_weights, total_budget)
        print(f"每层分配的节点数量: {allocated_nodes}")

        selected_nodes_per_layer = {}
        # 对每层进行节点选择
        for layer, num_nodes in zip(sorted(layer_matrices.keys(), key=int), allocated_nodes):
            print(f"\n处理层 {layer}，需要选择 {num_nodes} 个节点")
            G = nx.Graph(layer_matrices[layer])

            # 根据社区大小选择节点
            selected_nodes = select_nodes_by_community_size(G, num_nodes)
            selected_nodes_per_layer[int(layer)] = selected_nodes

            print(f"层 {layer} 选取的节点: {selected_nodes}")

        print("\n所有层的种子节点集合:", selected_nodes_per_layer)

        # 运行传播模型

        ave_inf, layer_inf = MWIC1.multilayer_IC_Algorithm(Gs, selected_nodes_per_layer, p, layer_weights, mc)
        duration = time.time() - start_time
        print(f"平均总影响力: {ave_inf}")
        print(f"每层的平均影响力: {layer_inf}")
        print(f"运行时间: {duration:.2f} 秒")

        # 记录结果
        results.append({
            "每层种子集个数": allocated_nodes,
            "总种子集个数": total_budget,
            "选出的种子集": selected_nodes_per_layer,
            "平均总影响力": ave_inf,
            "每层影响力": layer_inf,
            "运行时间": duration
        })

    # 保存结果到Excel
    df = pd.DataFrame(results)
    # df.to_excel("cqsn-w1 london_0.01.xlsx", index=False)
    # df.to_excel("cqsn-w2 ARXIV_0.05-7.16.xlsx", index=False)
    # df.to_excel("cqsn-w2 ARXIV_0.01-6.23.xlsx", index=False)
    # df.to_excel("cqsn CELE-w3_0.05.xlsx", index=False)
    # df.to_excel("cqsn CELE-w3_0.01-6.23.xlsx", index=False)
    # df.to_excel("cqsn CELE-w2_0.05-6.28.xlsx", index=False)
    # df.to_excel("cqsn w2-BA-ER-WS_0.05-6.28.xlsx", index=False)
    # df.to_excel("cqsn-w2 ARABIDOPSIS_0.05-6.28.xlsx", index=False)
    # df.to_excel("cqsn-w1 ARABIDOPSIS_0.01-6.23.xlsx", index=False)
    # df.to_excel("cqsn-w2 mos_0.05-6.28.xlsx", index=False)
    # df.to_excel("cqsn-w2 ckm_0.01-7.16.xlsx", index=False)
    # df.to_excel("cqsn mus-w2_0.05-6.28.xlsx", index=False)
    # df.to_excel("cqsn w2-BA_0.05-6.28.xlsx", index=False)
    # df.to_excel("cqsn w2-ER_0.05-6.28.xlsx", index=False)
    # df.to_excel("cqsn w2-WS_0.05-6.28.xlsx", index=False)
    # df.to_excel("cqsn-w2 FF-TW-YT_0.05-7.16.xlsx", index=False)
    df.to_excel("cqsn-w2 ladtfm_0.01-11.11.xlsx", index=False)