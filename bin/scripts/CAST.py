import os
import pandas as pd
import networkx as nx
import pickle
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
import collections
from typing import List, Tuple
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import altair as alt
# import plotly.express as px
# import kaleido

import argparse
import pickle

import logging
import sys
logging.basicConfig(filename='python_debug.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)
# logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)



def load_transitive_alignment_results(folder_path):
    all_pairs = []
    for filename in os.listdir(folder_path):
        if filename.startswith("chunk_") and filename.endswith("_realignment.pkl"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'rb') as file:
                # Load the node pairs and scores
                pairs = pickle.load(file)
                all_pairs.extend(pairs)
    return all_pairs

def generate_candidates(G, C, S):
    if (len(C)==1):
        return [n for n in G.neighbors(C[0]) if n in S]
    else:
        res = set([n for n in G.neighbors(C[0])if n in S])
        for item in C[1:]:
            res=res & set([n for n in G.neighbors(item)])
    return list(res)

def update_graph_with_alignment_results(G, alignment_results, min_score):
    new_alignment_results = []
    for node1, node2, score in tqdm(alignment_results):
        if score >= min_score:
            G.add_edge(node1, node2, Cosine=score, origin='transitive_alignment')
            new_alignment_results.append((node1, node2, score))
    print(len(new_alignment_results))
    return G

def calculate_average_weight(graph):
    total_weight = sum(graph[u][v]['Cosine'] for u, v in graph.edges())
    average_weight = total_weight / graph.number_of_edges()
    return average_weight
def add_edges_to_mst(original_graph, mst):
    remaining_edges = [(u, v, original_graph[u][v]['Cosine']) for u, v in original_graph.edges() if not mst.has_edge(u, v)]
    remaining_edges.sort(key=lambda x: x[2], reverse=True)

    average_weight = calculate_average_weight(mst)

    for u, v, weight in remaining_edges:
        mst.add_edge(u, v, Cosine=weight)
        new_average_weight = calculate_average_weight(mst)
        if new_average_weight <= average_weight:
            mst.remove_edge(u, v)
            break
        average_weight = new_average_weight

    return mst

def polish_subgraph_Pure_MST(G):
    if G.number_of_edges() == 0:
        return G
    maximum_spanning_tree = nx.maximum_spanning_tree(G, weight='Cosine')
    return maximum_spanning_tree

def polish_subgraph_Geedy_MST(G):
    if G.number_of_edges() == 0:
        return G
    maximum_spanning_tree = nx.maximum_spanning_tree(G, weight='Cosine')
    polished_subgraph = add_edges_to_mst(G, maximum_spanning_tree)
    return polished_subgraph

def polish_subgraph_hybrid_MST(G):
    # Create an MST with only original edges
    original_edges_graph = nx.Graph(
        (u, v, d) for u, v, d in G.edges(data=True) if d.get('origin') != 'transitive_alignment'
    )
    mst = nx.maximum_spanning_tree(original_edges_graph, weight='Cosine')

    # Check if the MST spans all nodes, if not, start adding transitive edges
    if len(mst.nodes()) < len(G.nodes()):
        print("Original edges do not span all nodes, adding transitive edges as needed.")

        # For each node not in the MST, try to connect it with the least number of transitive edges
        for node in G.nodes():
            if node not in mst.nodes():
                # Find all transitive edges connecting this node to any node in the MST
                candidate_edges = [
                    (u, v, d) for u, v, d in G.edges(node, data=True)
                    if d.get('origin') == 'transitive_alignment' and (u in mst.nodes() or v in mst.nodes())
                ]
                # Sort these edges by weight (assuming higher is better)
                candidate_edges.sort(key=lambda x: x[2]['Cosine'], reverse=True)

                # Add the best edge to the MST, if any
                if candidate_edges:
                    best_edge = candidate_edges[0]
                    mst.add_edge(best_edge[0], best_edge[1], **best_edge[2])

    return mst

def CAST_cluster(G, theta):
    sorted_node=list(sorted(G.degree, key=lambda x: x[1], reverse=True))
    S = [n[0] for n in sorted_node]
    P=[]
    while (len(S)):
        v=S[0]
        C=[]
        C.append(v)
        candidates=generate_candidates(G,C,S)
        while(len(candidates)):
            can_dic={}
            nonematch_flag=1
            for candidate in candidates:
                avg_weight=0
                for node in C:
                    avg_weight += G.edges[candidate,node]['Cosine']
                avg_weight=avg_weight/len(C)
                if avg_weight>theta:
                    can_dic[candidate]=avg_weight
                    nonematch_flag=0
            if (nonematch_flag):
                break
            close_node =  max(can_dic,key=can_dic.get)
            C.append(close_node)
            candidates=generate_candidates(G,C,S)
        P.append(C)
        S=[x for x in S if x not in C]
    return P
def generate_candidates_greedy(G, C, S):
    candidates = set()
    for node in C:
        neighbors = set(G.neighbors(node))
        candidates = candidates.union(neighbors)
    candidates = candidates.intersection(S)
    return list(candidates)

def CAST_cluster_greedy(G, theta):
    sorted_node = list(sorted(G.degree, key=lambda x: x[1], reverse=True))
    S = [n[0] for n in sorted_node]
    P = []
    while len(S):
        v = S[0]
        C = [v]
        candidates = generate_candidates_greedy(G, C, S)
        while len(candidates):
            can_dic = {}
            for candidate in candidates:
                avg_weight = 0
                count = 0
                for node in C:
                    if G.has_edge(candidate, node):
                        avg_weight +=  G.edges[candidate,node]['Cosine']
                        count += 1
                if count > 0:
                    avg_weight /= count
                    if avg_weight > theta:
                        can_dic[candidate] = avg_weight
            if not can_dic:
                break
            close_node = max(can_dic, key=can_dic.get)
            C.append(close_node)
            candidates = generate_candidates(G, C, S)
        P.append(C)
        S = [x for x in S if x not in C]
    return P

# def plot_debug(G_all_pairs):
#     degree_sequence = sorted([d for n, d in G_all_pairs.degree()], reverse=True)
#
#     # Count the number of occurrences of each degree value
#     degree_counts = collections.Counter(degree_sequence)
#     degrees, counts = zip(*degree_counts.items())
#     df = pd.DataFrame({
#         'Degree': degrees,
#         'Node Count': counts
#     })
#
#     # Create a scatter plot using Plotly Express
#     fig = px.scatter(df, x='Node Count', y='Degree', title='Node Count vs Degree Distribution',
#                       labels={'x': 'Node Count', 'y': 'Degree'})
#
#     # Save the figure as a HTML file or display it
#     fig.write_image('degree_loglog_plot.png',engine="kaleido")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Using realignment method to reconstruct the network')
    parser.add_argument('-m', type=str, required=True, default="merged_pairs.tsv", help='raw pairs filename')
    parser.add_argument('-t', type=str, required=True, default="/home/user/LabData/XianghuData/Transitive_Alignment_Distributed/nf_output/transitive_alignment", help='transitive_alignment folder path')
    parser.add_argument('-th', type=float, required=False, default=0.8, help='CAST threshold')
    parser.add_argument('-r', type=str, required=False, default="trans_align_result.tsv", help='output filename')
    parser.add_argument('--minimum_score', type=float, required=False, default=0.6, help='Minimum score to keep in output edges')
    parser.add_argument('--mst_filter', type=str, required=False, default="No", help='Trun on the MST filter')

    args = parser.parse_args()
    raw_pairs_filename = args.m
    transitive_alignment_folder = args.t
    min_score = args.minimum_score
    result_file_path = args.r
    MST_filter = args.mst_filter
    # read the raw pairs file
    all_pairs_df = pd.read_csv(raw_pairs_filename, sep='\t')

    # Anoother format
    if not "CLUSTERID1" in all_pairs_df.columns:
        # rename columns
        all_pairs_df.rename(columns={"scan1": "CLUSTERID1", "scan2": "CLUSTERID2", "score": "Cosine"}, inplace=True)

    # constructed network from edge list
    G_all_pairs = nx.from_pandas_edgelist(all_pairs_df, "CLUSTERID1", "CLUSTERID2", "Cosine")
    print(G_all_pairs.number_of_edges())

    # Example debug message
    original_edges = G_all_pairs.number_of_edges()
    original_nodes = G_all_pairs.number_of_nodes()
    network_density = (2*original_edges)/(original_nodes*(original_nodes-1))
    logging.info(f"Original number of edges:{original_edges}",)
    logging.info(f"Original number of nodes:{original_nodes}", )
    logging.info(f"Network density:{network_density}")

    # plot_debug(G_all_pairs)

    alignment_results = load_transitive_alignment_results(transitive_alignment_folder)
    G_all_pairs = update_graph_with_alignment_results(G_all_pairs, alignment_results, min_score)

    cast_cluster = CAST_cluster(G_all_pairs, args.th)
    # Proceed with the rest of the code to handle CAST clustering results
    cast_components = [G_all_pairs.subgraph(c).copy() for c in cast_cluster]
    cast_components_length  = [len(c) for c in cast_cluster]
    print(cast_components_length)
    if (MST_filter=="Pure_MST"):
        cast_components = [polish_subgraph_Pure_MST(c) for c in cast_components]
    elif (MST_filter=="Greedy_MST"):
        cast_components = [polish_subgraph_Geedy_MST(c) for c in cast_components]
    elif (MST_filter=="Hybrid_MST"):
        cast_components = [polish_subgraph_hybrid_MST(c) for c in cast_components]

    output_results=[]

    # get the all the edges
    for component in cast_components:
        if component.edges():
            for edge in component.edges():
                output_results.append((edge[0],edge[1],G_all_pairs[edge[0]][edge[1]]['Cosine']))

    # Writing the results to a TSV file
    output_df = pd.DataFrame(output_results, columns=["CLUSTERID1", "CLUSTERID2", "Cosine"])

    output_df.to_csv(result_file_path, sep='\t', index=False)

    print(f"Data saved to {result_file_path} successfully.")

