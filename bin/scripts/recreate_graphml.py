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
        # Add edges with attributes
        G.add_edge(edge[0],edge[1], deltamz=0, score=0)
    
    #write the graphml file
    nx.write_graphml(G, output_graphml)


if __name__ == '__main__':
    main()