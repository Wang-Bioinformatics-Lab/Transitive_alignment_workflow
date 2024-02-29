
from tqdm import tqdm
import numpy as np
import networkx as nx
from multiprocessing import Pool
import collections
from typing import List, Tuple

import argparse
import pandas as pd
import pickle


# define the spectumtuple structure
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


def realign_path(path):
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


def re_alignment_parallel(args):
    node1, node2 = args
    # if there is a path between node1 and node2 then try to align them
    if nx.has_path(G_all_pairs, node1, node2):
        # find all the shortest path
        all_shortest_hops = [p for p in
                             nx.all_shortest_paths(G_all_pairs, node1, node2, weight=None, method='dijkstra')]
        Max_path_weight = 0
        Max_path = []
        # find the path has the maximum sum of cos score
        for hop in all_shortest_hops:
            Path_weight = 0
            for node_i in range(len(hop) - 1):
                Path_weight = Path_weight + G_all_pairs[hop[node_i]][hop[node_i + 1]]['Cosine']
            if (Path_weight > Max_path_weight):
                Max_path_weight = Path_weight
                Max_path = hop
        matched_peaks, score = realign_path(Max_path)
        # if there are new aligned peaks then retrun (node1, node2, new aligned score)
        if matched_peaks != "no match":
            return (node1, node2, score)
        else:
            return
    else:
        return

if __name__ == '__main__':
    # pass arguments
    parser = argparse.ArgumentParser(description='Using realignment method to reconstruct the network')
    parser.add_argument('-s', type=str, required=True, default="spec_dic.pkl", help='spectra dictionary filename')
    parser.add_argument('-i', type=str, required=True, default="chunk1.tsv", help='input chunk filename')
    parser.add_argument('-m', type=str, required=True, default="merged_pairs.tsv", help='raw pairs filename')
    parser.add_argument('-p', type=int, required=False, default=4, help='the number of paralleled processes')
    parser.add_argument('-r', type=str, required=False, default="trans_align_result.tsv", help='output filename')

    args = parser.parse_args()
    spec_dic_path = args.s
    input_chunk_path = args.i
    raw_pairs_filename = args.m
    processes_number = args.p
    result_file_path = args.r

    # read the raw pairs file
    all_pairs_df = pd.read_csv(raw_pairs_filename, sep='\t')
    # constructed network from edge list
    G_all_pairs = nx.from_pandas_edgelist(all_pairs_df, "CLUSTERID1", "CLUSTERID2", "Cosine")
    print(G_all_pairs.number_of_nodes())

    # reload
    with open(spec_dic_path, 'rb') as input_file:
        spec_dic = pickle.load(input_file)
    print("start to align nodes")

    values = []
    with open(args.i, 'r') as chunk_file:
        for line in chunk_file:
            node1, node2 = line.strip().split()
            values.append([int(node1), int(node2)])



    with Pool(processes=args.p, maxtasksperchild=1000) as pool:
        results = list(pool.imap(re_alignment_parallel, values))

    # Fitlering no results
    results = [value for value in results if value is not None]

    output_path = args.r if args.r.endswith('.pkl') else f"{args.r}.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

