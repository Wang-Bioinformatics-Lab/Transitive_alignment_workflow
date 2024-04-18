import pandas as pd
import networkx as nx
import pickle
import numpy as np
import collections
from typing import List, Tuple

import argparse
SpectrumTuple = collections.namedtuple(
    "SpectrumTuple", ["precursor_mz", "precursor_charge", "mz", "intensity"]
)

def _cosine_fast(
        spec: SpectrumTuple,
        spec_other: SpectrumTuple,
        fragment_mz_tolerance: float,
        allow_shift: bool,
) -> Tuple[float, List[Tuple[int, int]]]:
    precursor_charge = max(spec.precursor_charge, 1)
    precursor_mass_diff = (
                                  spec.precursor_mz - spec_other.precursor_mz
                          ) * precursor_charge
    # Only take peak shifts into account if the mass difference is relevant.
    num_shifts = 1
    if allow_shift and abs(precursor_mass_diff) >= fragment_mz_tolerance:
        num_shifts += precursor_charge
    other_peak_index = np.zeros(num_shifts, np.uint16)
    mass_diff = np.zeros(num_shifts, np.float32)
    for charge in range(1, num_shifts):
        mass_diff[charge] = precursor_mass_diff / charge

    # Find the matching peaks between both spectra.
    peak_match_scores, peak_match_idx = [], []
    for peak_index, (peak_mz, peak_intensity) in enumerate(
            zip(spec.mz, spec.intensity)
    ):
        # Advance while there is an excessive mass difference.
        for cpi in range(num_shifts):
            while other_peak_index[cpi] < len(spec_other.mz) - 1 and (
                    peak_mz - fragment_mz_tolerance
                    > spec_other.mz[other_peak_index[cpi]] + mass_diff[cpi]
            ):
                other_peak_index[cpi] += 1
        # Match the peaks within the fragment mass window if possible.
        for cpi in range(num_shifts):
            index = 0
            other_peak_i = other_peak_index[cpi] + index
            while (
                    other_peak_i < len(spec_other.mz)
                    and abs(
                peak_mz - (spec_other.mz[other_peak_i] + mass_diff[cpi])
            )
                    <= fragment_mz_tolerance
            ):
                peak_match_scores.append(
                    peak_intensity * spec_other.intensity[other_peak_i]
                )
                peak_match_idx.append((peak_index, other_peak_i))
                index += 1
                other_peak_i = other_peak_index[cpi] + index

    score, peak_matches = 0.0, []
    if len(peak_match_scores) > 0:
        # Use the most prominent peak matches to compute the score (sort in
        # descending order).
        peak_match_scores_arr = np.asarray(peak_match_scores)
        peak_match_order = np.argsort(peak_match_scores_arr)[::-1]
        peak_match_scores_arr = peak_match_scores_arr[peak_match_order]
        peak_match_idx_arr = np.asarray(peak_match_idx)[peak_match_order]
        peaks_used, other_peaks_used = set(), set()
        for peak_match_score, peak_i, other_peak_i in zip(
                peak_match_scores_arr,
                peak_match_idx_arr[:, 0],
                peak_match_idx_arr[:, 1],
        ):
            if (
                    peak_i not in peaks_used
                    and other_peak_i not in other_peaks_used
            ):
                score += peak_match_score
                # Save the matched peaks.
                peak_matches.append((peak_i, other_peak_i))
                # Make sure these peaks are not used anymore.
                peaks_used.add(peak_i)
                other_peaks_used.add(other_peak_i)

    return score, peak_matches


def peak_tuple_to_dic(peakmatches):
    dic = {}
    for peakmatch in peakmatches:
        dic[peakmatch[0]] = peakmatch[1]
    return dic


def norm_intensity(intensity):
    return np.copy(intensity) / np.linalg.norm(intensity)
    # return(intensity)


def realign_path(path,spec_dic):
    final_match_list = []
    spec_1 = spec_dic[path[0]]
    spec_2 = spec_dic[path[0 + 1]]
    # get the matched peaks for the first and second node in the path
    score, peak_matches = _cosine_fast(spec_1, spec_2, 0.5, True)
    final_match_list = peak_matches
    idx = 1
    # find the matched peaks in the rest of the path
    while (len(final_match_list) != 0 and idx < len(path) - 1):
        temp_peakmatch_list = []
        spec_1 = spec_dic[path[idx]]
        spec_2 = spec_dic[path[idx + 1]]
        score, peak_matches = _cosine_fast(spec_1, spec_2, 0.5, True)
        peak_dic1 = peak_tuple_to_dic(final_match_list)
        peak_dic2 = peak_tuple_to_dic(peak_matches)
        # calculate in intersection of two matched peak lists
        for key, value in peak_dic1.items():
            if (peak_dic2.get(value)):
                temp_peakmatch_list.append((key, peak_dic2[value]))
        final_match_list = temp_peakmatch_list
        idx = idx + 1
    # get the original aligned peaks of node1 and node2
    spec_start = spec_dic[path[0]]
    spec_end = spec_dic[path[-1]]
    _, matched_peaks = _cosine_fast(spec_start, spec_end, 0.5, True)
    peak_match_scores = []
    intesity1 = spec_dic[path[0]][3]
    intesity2 = spec_dic[path[-1]][3]
    # Add the new aligned peaks to the original peak matched list
    if (len(final_match_list)):
        final_match_list = final_match_list + matched_peaks
        for matched_peak in final_match_list:
            peak_match_scores.append(intesity1[matched_peak[0]] * intesity2[matched_peak[1]])
    score, peak_matches = 0.0, []
    # calculate the new matched score
    if len(peak_match_scores) > 0:
        peak_match_scores_arr = np.asarray(peak_match_scores)
        peak_match_order = np.argsort(peak_match_scores_arr)[::-1]
        peak_match_scores_arr = peak_match_scores_arr[peak_match_order]
        peak_match_idx_arr = np.asarray(final_match_list)[peak_match_order]
        peaks_used, other_peaks_used = set(), set()
        for peak_match_score, peak_i, other_peak_i in zip(
                peak_match_scores_arr,
                peak_match_idx_arr[:, 0],
                peak_match_idx_arr[:, 1],
        ):
            if (
                    peak_i not in peaks_used
                    and other_peak_i not in other_peaks_used
            ):
                score += peak_match_score
                peak_matches.append((peak_i, other_peak_i))
                peaks_used.add(peak_i)
                other_peaks_used.add(other_peak_i)
        return peak_matches, score
    else:
        return "no match", _



def induced_transitive_network_intersection(G, source, spec_dic, score_threshold=0.3, max_hops=3):
    """
    Realigns nodes in the network starting from a source node based on the shortest path with conditions.
    Args:
    G (nx.Graph): The network graph.
    source (int): The source node ID.
    spec_dic (dict): Dictionary of spectral data.
    score_threshold (float): Minimum score to consider for updating the network.
    max_hops (int): Maximum hops to consider for induced subgraph generation.
    Returns:
    nx.Graph: The induced subgraph around the source node with updated edges.
    """
    def get_best_paths(G, source):
        best_paths = {}
        for target in G.nodes():
            if target == source:
                continue
            try:
                # Paths with min hops
                paths = list(nx.all_shortest_paths(G, source, target))
                # Select path with max sum of cosine scores if multiple shortest paths
                best_path = max(paths, key=lambda path: sum(G[u][v]['Cosine'] for u, v in zip(path[:-1], path[1:])))
                best_paths[target] = best_path
            except nx.NetworkXNoPath:
                continue
        return best_paths
    nodes_within_hops = nx.single_source_shortest_path_length(G, source, cutoff=max_hops)
    subgraph = G.subgraph(nodes_within_hops).copy()
    subgraph_list = list(subgraph.nodes())
    tran_align_node_list = []
    tran_align_node_list.append(source)
    # Step 1: Compute best paths from source to all reachable nodes
    best_paths = get_best_paths(G, source)
    # Step 2: Realign spectra along the best paths and update edges
    for target, path in best_paths.items():
        try:
            # Path realignment
            aligned_peaks,realigned_score = realign_path(path,spec_dic)
            # check if two nodes can be aligned
            if aligned_peaks== "no match":
                #skip if no peak aligned
                continue
            # Check if the new score is above the threshold and the edge does not exist
            if realigned_score >= score_threshold and not G.has_edge(source, target):
                G.add_edge(source, target, Cosine=realigned_score, trans_align_score = realigned_score)
                tran_align_node_list.append(target)
        except Exception as e:
            print(f"An error occurred: {e}")
            print(path)
            break
    # Step 3: Generate the induced subgraph within max_hops from the source
    neighbors_list = list(G.neighbors(source))
    uni_trans_set = set(neighbors_list) | set(tran_align_node_list)
    induced_nodes_list = list(uni_trans_set & set(subgraph_list))
    induced_subgraph = G.subgraph(induced_nodes_list).copy()

    return induced_subgraph

def induced_transitive_network(G, source, spec_dic, score_threshold=0.3, max_hops=3):
    """
    Realigns nodes in the network starting from a source node based on the shortest path with conditions.
    Args:
    G (nx.Graph): The network graph.
    source (int): The source node ID.
    spec_dic (dict): Dictionary of spectral data.
    score_threshold (float): Minimum score to consider for updating the network.
    max_hops (int): Maximum hops to consider for induced subgraph generation.
    Returns:
    nx.Graph: The induced subgraph around the source node with updated edges.
    """
    def get_best_paths(G, source):
        best_paths = {}
        for target in G.nodes():
            if target == source:
                continue
            try:
                # Paths with min hops
                paths = list(nx.all_shortest_paths(G, source, target))
                # Select path with max sum of cosine scores if multiple shortest paths
                best_path = max(paths, key=lambda path: sum(G[u][v]['Cosine'] for u, v in zip(path[:-1], path[1:])))
                best_paths[target] = best_path
            except nx.NetworkXNoPath:
                continue
        return best_paths
    # Step 1: Compute best paths from source to all reachable nodes
    best_paths = get_best_paths(G, source)
    # Step 2: Realign spectra along the best paths and update edges
    for target, path in best_paths.items():
        try:
            # Path realignment
            aligned_peaks,realigned_score = realign_path(path,spec_dic)
            # check if two nodes can be aligned
            if aligned_peaks== "no match":
                #skip if no peak aligned
                continue
            # Check if the new score is above the threshold and the edge does not exist
            if realigned_score >= score_threshold and not G.has_edge(source, target):
                G.add_edge(source, target, Cosine=realigned_score, trans_align_score = realigned_score)
        except Exception as e:
            print(f"An error occurred: {e}")
            print(path)
            break
    # Step 3: Generate the induced subgraph within max_hops from the source
    nodes_within_hops = nx.single_source_shortest_path_length(G, source, cutoff=max_hops)
    induced_subgraph = G.subgraph(nodes_within_hops).copy()
    return induced_subgraph

def remove_singletons(G):
    singletons = [node for node, degree in dict(G.degree()).items() if degree == 0]
    G.remove_nodes_from(singletons)
    return G


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Using realignment method to reconstruct the network')
    parser.add_argument('-m', type=str, required=True, default="merged_pairs.tsv", help='raw pairs filename')
    parser.add_argument('-s', type=int, required=True, default=100, help='source node')
    parser.add_argument('-g', type=str, required=True, default="network.graphml", help='graphml filename')
    parser.add_argument('--max_hops', type=int, required=True, default=4, help='Minimum score to keep in output edges')
    parser.add_argument('--spec_dic', type=str, required=True, default="spec_dic.pkl", help='spectra dictionary filename')
    parser.add_argument('--minimum_score', type=float, required=False, default=0.6,
                        help='Minimum score to keep in output edges')
    parser.add_argument('--induced_option', type=str, required=True, default="intersection",
                        help='induced network options: intersection or union')
    parser.add_argument('output_graphml', type=str, default="filtered_graphml.graphml", help='output graphml filename')

    args = parser.parse_args()
    raw_pairs_filename = args.m
    source = args.s
    max_hops = args.max_hops
    output_graphml = args.output_graphml
    input_graphml = args.g
    spec_dic_path = args.spec_dic
    min_score = args.minimum_score
    induced_option = args.induced_option
    # read the raw pairs file
    all_pairs_df = pd.read_csv(raw_pairs_filename, sep='\t')

    # Anoother format
    if not "CLUSTERID1" in all_pairs_df.columns:
        # rename columns
        all_pairs_df.rename(columns={"scan1": "CLUSTERID1", "scan2": "CLUSTERID2", "score": "Cosine"}, inplace=True)

    # constructed network from edge list
    G_all_pairs = nx.from_pandas_edgelist(all_pairs_df, "CLUSTERID1", "CLUSTERID2", "Cosine")

    # reload
    with open(spec_dic_path, 'rb') as input_file:
        spec_dic = pickle.load(input_file)
    if induced_option == "intersection":
        induced_subgraph = induced_transitive_network_intersection(G_all_pairs,source,spec_dic,min_score,max_hops)
    elif induced_option == "union":
        induced_subgraph = induced_transitive_network(G_all_pairs,source,spec_dic,min_score,max_hops)

    G = nx.read_graphml(input_graphml)

    # get rid of all edges in graphml G
    G.remove_edges_from(list(G.edges()))

    # Add edges from G_all_pairs to G
    for node1,node2,score in induced_subgraph.edges(data='Cosine'):
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
        G[str(node1)][str(node2)]["scan1"] = node1
        G[str(node1)][str(node2)]["scan2"] = node2
        G[str(node1)][str(node2)]["component"] = "N/A"
        # Check if edge has 'trans_align_score' attribute in induced_subgraph and update it in G
        if 'trans_align_score' in induced_subgraph[node1][node2]:
            G[str(node1)][str(node2)]["trans_align_score"] = induced_subgraph[node1][node2]['trans_align_score']

    G = remove_singletons(G)
    # write the graphml file
    nx.write_graphml(G, output_graphml)


