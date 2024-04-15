import os
import sys
import argparse
import networkx as nx
import pandas as pd


def main():
    # pass arguments
    parser = argparse.ArgumentParser(description='Using realignment method to reconstruct the network')
    parser.add_argument('-g', type=str, required=True, default="network.graphml", help='graphml filename')
    parser.add_argument('-m', type=str, required=True, default="merged_pairs.tsv", help='pairs filename')
    parser.add_argument('output_graphml', type=str, default="filtered_graphml.graphml", help='output graphml filename')

    args = parser.parse_args()
    input_graphml = args.g
    input_pairs = args.m
    output_graphml = args.output_graphml

    # read the raw pairs file
    all_pairs_df = pd.read_csv(input_pairs, sep='\t')

    # Anoother format
    if not "CLUSTERID1" in all_pairs_df.columns:
        # rename columns
        all_pairs_df.rename(columns={"scan1": "CLUSTERID1", "scan2": "CLUSTERID2", "score": "Cosine"}, inplace=True)

    # constructed network from edge list
    G_all_pairs = nx.from_pandas_edgelist(all_pairs_df, "CLUSTERID1", "CLUSTERID2", "Cosine")

    # read the graphml file
    G = nx.read_graphml(input_graphml)

    # get rid of all edges in graphml G
    G.remove_edges_from(list(G.edges()))

    # Add edges from G_all_pairs to G
    for node1,node2,score in G_all_pairs.edges(data='Cosine'):
        # checking if nodes are in G
        if str(node1) not in G.nodes():
            continue

        if str(node2) not in G.nodes():
            continue

        # Add edges with attributes
        G.add_edge(str(node1), str(node2))
        G[str(node1)][str(node2)]["deltamz"] = abs(G.nodes[str(node1)]['mz'] - G.nodes[str(node2)]['mz'])
        G[str(node1)][str(node2)]["deltamz_int"] = int(abs(G.nodes[str(node1)]['mz'] - G.nodes[str(node2)]['mz']))
        G[str(node1)][str(node2)]["score"] = score
        G[str(node1)][str(node2)]["matched_peaks"] = "0"
        G[str(node1)][str(node2)]["scan1"] = 0
        G[str(node1)][str(node2)]["scan2"] = 0
        G[str(node1)][str(node2)]["component"] = "N/A"

    # Identify all connected components
    components = list(nx.connected_components(G))

    filtered_components = []
    for component in components:
        if len(component) >= 2:
            filtered_components.append(component)

    # Sort components by size from largest to smallest
    sorted_components = sorted(filtered_components, key=len, reverse=True)


    # Relabeling subgraphs and their edges more efficiently, from largest to smallest
    component_label = 1
    for component in sorted_components:
        # Create a subgraph for the current component
        subgraph = G.subgraph(component)

        # Set component label for nodes in the subgraph
        nx.set_node_attributes(subgraph, component_label, "component")

        # Set component label for edges in the subgraph
        nx.set_edge_attributes(subgraph, component_label, "component")

        # Increment component_label for the next component
        component_label += 1
    # write the graphml file
    nx.write_graphml(G, output_graphml)


if __name__ == '__main__':
    main()