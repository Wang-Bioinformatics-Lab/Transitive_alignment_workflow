from pyteomics import mgf
# from Classic import *
from tqdm import tqdm
import numpy as np
import networkx as nx
from multiprocessing import Pool
import collections
from typing import List, Tuple
import csv
import math
import argparse
import pandas as pd

#define the spectumtuple structure
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
    dic={}
    for peakmatch in peakmatches:
        dic[peakmatch[0]]=peakmatch[1]
    return dic

def norm_intensity(intensity):
    return np.copy(intensity)/np.linalg.norm(intensity)
    #return(intensity)

def realign_path(path):
    final_match_list=[]
    spec_1 = spec_dic[path[0]]
    spec_2 = spec_dic[path[0+1]]
    #get the matched peaks for the first and second node in the path
    score,peak_matches = _cosine_fast(spec_1,spec_2,0.5,True)
    final_match_list=peak_matches
    idx=1
    #find the matched peaks in the rest of the path
    while (len(final_match_list)!=0 and idx <len(path)-1):
        temp_peakmatch_list=[]
        spec_1 = spec_dic[path[idx]]
        spec_2 = spec_dic[path[idx+1]]
        score,peak_matches = _cosine_fast(spec_1,spec_2,0.5,True)
        peak_dic1=peak_tuple_to_dic(final_match_list)
        peak_dic2=peak_tuple_to_dic(peak_matches)
        #calculate in intersection of two matched peak lists
        for key, value in peak_dic1.items():
            if (peak_dic2.get(value)):
                temp_peakmatch_list.append((key,peak_dic2[value]))
        final_match_list=temp_peakmatch_list
        idx=idx+1
    #get the original aligned peaks of node1 and node2
    spec_start = spec_dic[path[0]]
    spec_end = spec_dic[path[-1]]
    _, matched_peaks = _cosine_fast(spec_start,spec_end,0.5,True)
    peak_match_scores = []
    intesity1=spec_dic[path[0]][3]
    intesity2=spec_dic[path[-1]][3]
    #Add the new aligned peaks to the original peak matched list
    if (len(final_match_list)):
        final_match_list=final_match_list+matched_peaks
        for matched_peak in final_match_list:
            peak_match_scores.append(intesity1[matched_peak[0]]*intesity2[matched_peak[1]])
    score, peak_matches = 0.0, []
    #calculate the new matched score
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
        return "no match",_

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
        if new_average_weight >= average_weight:
            mst.remove_edge(u, v)
            break
        average_weight = new_average_weight

    return mst

def polish_subgraph(G):
    if G.number_of_edges() == 0:
        return G
    maximum_spanning_tree = nx.maximum_spanning_tree(G, weight='Cosine')
    polished_subgraph = add_edges_to_mst(G, maximum_spanning_tree)
    return polished_subgraph


def re_alignment_parallel(args):
    node1,node2=args
    #if there is a path between node1 and node2 then try to align them
    if nx.has_path(G_all_pairs, node1, node2):
        #find all the shortest path
        all_shortest_hops = [p for p in nx.all_shortest_paths(G_all_pairs, node1, node2, weight=None, method='dijkstra')]
        Max_path_weight = 0
        Max_path = []
        #find the path has the maximum sum of cos score
        for hop in all_shortest_hops:
            Path_weight = 0
            for node_i in range(len(hop) - 1):
                Path_weight = Path_weight + G_all_pairs[hop[node_i]][hop[node_i + 1]]['Cosine']
            if (Path_weight > Max_path_weight):
                Max_path_weight = Path_weight
                Max_path = hop
        matched_peaks,score = realign_path(Max_path)
        #if there are new aligned peaks then retrun (node1, node2, new aligned score)
        if matched_peaks != "no match":
            return (node1,node2,score)
        else:
            return
    else:
        return

def mgf_processing(mgf_filename):
    spec_dic = {}
    print("start to create spectrum dictionary")
    with open(mgf_filename, 'r') as mgf_file:
        # Read the lines of the MGF file
        for line in mgf_file:
            line = line.strip()
            if line.startswith("BEGIN IONS"):
                mz_array = []
                intensity_array = []
                # Reset the variables when encountering the "BEGIN IONS" line
                scans = None
                pepmass = None
                charge = None
                collision_energy = None
                ion_data = []
            elif line.startswith("SCANS="):
                # Extract the value after "SCANS="
                scans = int(line.split("=")[1])
            elif line.startswith("PEPMASS="):
                # Extract the value after "PEPMASS="
                pepmass = float(line.split("=")[1])
            elif line.startswith("CHARGE="):
                # Extract the value after "CHARGE="
                charge = int(line.split("=")[1])
            elif line.startswith("COLLISION_ENERGY="):
                # Extract the value after "COLLISION_ENERGY="
                collision_energy = float(line.split("=")[1])
            elif line == "END IONS":
                #Process the collected data when encountering the "END IONS" line
                filtered_mz = []
                filtered_intensities = []
                for i, mz in enumerate(mz_array):
                    peak_range = [j for j in range(len(mz_array)) if abs(mz_array[j] - mz) <= 25]
                    sorted_range = sorted(peak_range, key=lambda j: intensity_array[j], reverse=True)
                    if i in sorted_range[:6]:
                        if abs(mz - pepmass) > 17:
                            filtered_mz.append(mz)
                            filtered_intensities.append(intensity_array[i])
                intensity_array = [math.sqrt(x) for x in filtered_intensities]
                # intensity_array = [math.sqrt(x) for x in intensity_array]
                spec_dic[scans] = SpectrumTuple(pepmass, charge, filtered_mz,norm_intensity(intensity_array))
            else:
                # Extract the ion data
                ion_values = line.split("\t")
                if (ion_values[0]):
                    mz = float(ion_values[0])
                    intensity = float(ion_values[1])
                    mz_array.append(mz)
                    intensity_array.append(intensity)
    return spec_dic

def generate_candidates(G, C, S):
    if (len(C)==1):
        return [n for n in G.neighbors(C[0]) if n in S]
    else:
        res = set([n for n in G.neighbors(C[0])if n in S])
        for item in C[1:]:
            res=res & set([n for n in G.neighbors(item)])
    return list(res)

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

if __name__ == '__main__':
    #pass arguments
    parser = argparse.ArgumentParser(description='Using realignment method to reconstruct the network')
    parser.add_argument('-c', type=str,required=True,default="specs_ms.mgf", help='mgf filename')
    parser.add_argument('-m', type=str,required=True,default="merged_pairs.tsv", help='raw pairs filename')
    parser.add_argument('-p', type=int,required=False,default=4, help='the number of paralleled processes')
    parser.add_argument('-r', type=str, required=False, default="trans_align_result.tsv", help='output filename')
    parser.add_argument('-th', type=float, required=False, default=0.8, help='CAST threshold')
    parser.add_argument('--minimum_score', type=float, required=False, default=0.7, help='Minimum score to keep in output edges')
    parser.add_argument('--mst_filter', type=str, required=False, default="No", help='Trun on the MST filter')

    args = parser.parse_args()
    mgf_filename = args.c
    raw_pairs_filename  = args.m
    processes_number = args.p
    result_file_path = args.r
    MST_filter = args.mst_filter

    #read the raw pairs file
    all_pairs_df = pd.read_csv(raw_pairs_filename, sep='\t')
    #constructed network from edge list
    G_all_pairs = nx.from_pandas_edgelist(all_pairs_df, "CLUSTERID1", "CLUSTERID2", "Cosine")

    #creat the spectrum dictionary
    spec_dic = mgf_processing(mgf_filename)
    print("start to align nodes")
    with Pool(processes=args.p, maxtasksperchild=1000) as pool:
        values = [[node1, node2] for [node1, node2] in nx.non_edges(G_all_pairs)]
        results = list(tqdm(pool.imap(re_alignment_parallel, values), total=len(values)))

    # Fitlering no results
    results = [value for value in results if value is not None]

    new_aligned_edge = []
    for item in results:
        if item[2]>=args.minimum_score:
            new_aligned_edge.append(item)

    # add new aligned edge to raw pairs
    for edge in new_aligned_edge:
        G_all_pairs.add_edge(edge[0], edge[1], Cosine=edge[2])

    # creat CAST component
    cast_cluster = CAST_cluster(G_all_pairs, args.th)

    cast_components = [G_all_pairs.subgraph(c).copy() for c in cast_cluster]

    if (MST_filter=="Yes"):
        cast_components = [polish_subgraph(c) for c in cast_components]

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

