import os
import pandas as pd
import networkx as nx
import pickle
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
import collections
from typing import List, Tuple
from infomap import Infomap

import argparse
import pickle


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
    parser.add_argument('-r', type=str, required=False, default="trans_align_result.tsv", help='output filename')
    parser.add_argument('--minimum_score', type=float, required=False, default=0.6, help='Minimum score to keep in output edges')

    args = parser.parse_args()
    raw_pairs_filename = args.m
    transitive_alignment_folder = args.t
    min_score = args.minimum_score
    result_file_path = args.r
    # read the raw pairs file
    all_pairs_df = pd.read_csv(raw_pairs_filename, sep='\t')

    # Anoother format
    if not "CLUSTERID1" in all_pairs_df.columns:
        # rename columns
        all_pairs_df.rename(columns={"scan1": "CLUSTERID1", "scan2": "CLUSTERID2", "score": "Cosine"}, inplace=True)

    # constructed network from edge list
    G_all_pairs = nx.from_pandas_edgelist(all_pairs_df, "CLUSTERID1", "CLUSTERID2", "Cosine")

    # plot_debug(G_all_pairs)

    alignment_results = load_transitive_alignment_results(transitive_alignment_folder)
    G_all_pairs = update_graph_with_alignment_results(G_all_pairs, alignment_results, min_score)

    infomapWrapper = Infomap("--two-level --silent")

    # Add edges to Infomap from the NetworkX graph with the cosine similarity as weights
    for edge in G_all_pairs.edges(data=True):
        node_a, node_b, data = edge
        weight = data.get('Cosine', 0)  # Default to 1.0 if 'Cosine' is not present
        infomapWrapper.addLink(node_a, node_b, weight)

    # Run Infomap to find communities
    infomapWrapper.run()

    # Extract the communities and assign them to nodes in the original NetworkX graph
    communities = {node.node_id: node.module_id for node in infomapWrapper.tree if node.isLeaf()}

    # Create subgraphs for each community
    community_subgraphs = [G_all_pairs.subgraph([n for n in G_all_pairs if communities[n] == c]).copy() for c in
                           set(communities.values())]

    output_results = []

    # Collect all the edges from the community subgraphs
    for component in community_subgraphs:
        if component.edges():
            for edge in component.edges():
                output_results.append((edge[0], edge[1], component[edge[0]][edge[1]]['Cosine']))

    # Writing the results to a TSV file
    output_df = pd.DataFrame(output_results, columns=["CLUSTERID1", "CLUSTERID2", "Cosine"])

    output_df.to_csv(result_file_path, sep='\t', index=False)

    print(f"Data saved to {result_file_path} successfully.")

