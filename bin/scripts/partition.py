import math
import os
import argparse
import pandas as pd
import networkx as nx

def partition(lst, n):
    division = len(lst) / float(n)
    return [lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n)]

def write_chunks_to_files(chunks, base_directory):
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)
    for i, chunk in enumerate(chunks):
        chunk_file_path = os.path.join(base_directory, f"chunk_{i+1}.tsv")
        with open(chunk_file_path, 'w') as f:
            for pair in chunk:
                f.write(f"{pair[0]}\t{pair[1]}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Using realignment method to reconstruct the network')
    parser.add_argument('-m', type=str, required=True, default="merged_pairs.tsv", help='raw pairs filename')
    parser.add_argument('--n_chunks', type=int, required=True, help='Number of chunks to partition the pairs into')
    args = parser.parse_args()
    
    raw_pairs_filename = args.m
    n_chunks = args.n_chunks

    # read the raw pairs file
    all_pairs_df = pd.read_csv(raw_pairs_filename, sep='\t')

    # Anoother format
    if not "CLUSTERID1" in all_pairs_df.columns:
        # rename columns
        all_pairs_df.rename(columns={"scan1": "CLUSTERID1", "scan2": "CLUSTERID2", "score": "Cosine"}, inplace=True)
        
    # constructed network from edge list
    G_all_pairs = nx.from_pandas_edgelist(all_pairs_df, "CLUSTERID1", "CLUSTERID2", "Cosine")
    

    print(G_all_pairs.number_of_nodes())
    components = [G_all_pairs.subgraph(c).copy() for c in nx.connected_components(G_all_pairs)]
    print(len(components))
    component_len = [len(component) for component in components]
    print(max(component_len))
    node_pairs = [[node1, node2] for [node1, node2] in nx.non_edges(G_all_pairs)]
    chunks = partition(node_pairs, n_chunks)
    
    # Write the chunks to files
    chunks_directory = f"./chunks/"
    write_chunks_to_files(chunks, chunks_directory)
    print(f"Finished processing. Node pairs have been partitioned into {n_chunks} chunks.")
