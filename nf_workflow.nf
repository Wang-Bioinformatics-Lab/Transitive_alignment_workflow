#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// Spectra as input
params.input_spectra = "data/input_spectra"
params.input_pairs ="data/input_pairs"

params.parallelism = 24

// Topology Filtering

params.topology_cliquemincosine = 0.7
params.networking_min_cosine = 0.7

TOOL_FOLDER = "$baseDir/bin"

process filterNetworkTransitive {
    publishDir "./nf_output/", mode: 'copy'

    conda "$TOOL_FOLDER/conda_env.yml"

    cache false

    cpus 2

    input:
    file input_pairs
    file input_spectra

    output:
    file "filtered_pairs.tsv"

    """
    python $TOOL_FOLDER/scripts/transitive_alignment.py \
    -c ${input_spectra} \
    -m ${input_pairs} \
    -p 2 \
    -th $params.topology_cliquemincosine \
    -r filtered_pairs.tsv \
    --minimum_score $params.networking_min_cosine
    """
}

workflow {
    // Preps input spectrum files
    input_spectra_ch = Channel.fromPath(params.input_spectra+'/*.mgf')
    input_pairs_ch = Channel.fromPath(params.input_pairs+'/*.tsv')

    // Filtering the network
    filtered_networking_pairs_ch = filterNetworkTransitive(input_pairs_ch, input_spectra_ch)

}