import os
import sys
import argparse
import networkx as nx
import pandas as pd

def main():
    #pass arguments
    parser = argparse.ArgumentParser(description='Using realignment method to reconstruct the network')
    parser.add_argument('-g', type=str,required=True,default="network.graphml", help='graphml filename')
    parser.add_argument('-m', type=str,required=True,default="merged_pairs.tsv", help='pairs filename')    
    parser.add_argument('output_graphml', type=str,default="filtered_graphml.graphml", help='output graphml filename')

    args = parser.parse_args()
    input_graphml = args.g
    input_pairs  = args.m
    output_graphml = args.output_graphml
    

    #read the raw pairs file
    all_pairs_df = pd.read_csv(input_pairs, sep='\t')
    #constructed network from edge list
    G_all_pairs = nx.from_pandas_edgelist(all_pairs_df, "CLUSTERID1", "CLUSTERID2", "Cosine")

    #read the graphml file
    G = nx.read_graphml(input_graphml)

    # get rid of all edges in graphml G
    G.remove_edges_from(list(G.edges()))

    # Add edges from G_all_pairs to G
    for edge in G_all_pairs.edges():
        # checking if nodes are in G
        if str(edge[0]) not in G.nodes():
            continue
        
        if str(edge[1]) not in G.nodes():
            continue

        # Add edges with attributes
        G.add_edge(str(edge[0]),str(edge[1]))
        G[str(edge[0])][str(edge[1])]["deltamz"] = 0
        G[str(edge[0])][str(edge[1])]["deltamz_int"] = 0
        G[str(edge[0])][str(edge[1])]["score"] = 0
        G[str(edge[0])][str(edge[1])]["matched_peaks"] = "0"
        G[str(edge[0])][str(edge[1])]["scan1"] = 0
        G[str(edge[0])][str(edge[1])]["scan2"] = 0
        G[str(edge[0])][str(edge[1])]["component"] = "N/A"

    #write the graphml file
    nx.write_graphml(G, output_graphml)


if __name__ == '__main__':
    main()