#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// Spectra as input
params.input_spectra = "data/input_spectra/specs_ms.mgf"
params.input_pairs ="data/input_pairs/merged_pairs.tsv"

params.input_graphml = ""

// Topology Filtering
params.topology_cliquemincosine = 0.7
params.networking_min_cosine = 0.7

TOOL_FOLDER = "$baseDir/bin"

process filterNetworkTransitive {
    publishDir "./nf_output/", mode: 'copy'

    conda "$TOOL_FOLDER/conda_env.yml"

    cache false

    cpus 16

    input:
    file input_pairs
    file input_spectra

    output:
    file "filtered_pairs.tsv"

    """
    python $TOOL_FOLDER/scripts/transitive_alignment.py \
    -c ${input_spectra} \
    -m ${input_pairs} \
    -p 16 \
    -th $params.topology_cliquemincosine \
    -r filtered_pairs.tsv \
    --minimum_score $params.networking_min_cosine
    """
}

process recreateGraphML {
    publishDir "./nf_output/", mode: 'copy'

    conda "$TOOL_FOLDER/conda_env.yml"

    cache false

    input:
    file input_graphml
    file filtered_networking_pairs

    output:
    file "filtered_graphml.graphml"

    """
    python $TOOL_FOLDER/scripts/recreate_graphml.py \
    -g ${input_graphml} \
    -p ${filtered_networking_pairs} \
    filtered_graphml.graphml
    """
}

workflow {
    // Preps input spectrum files
    input_spectra_ch = Channel.fromPath(params.input_spectra)
    input_pairs_ch = Channel.fromPath(params.input_pairs)

    // Filtering the network
    filtered_networking_pairs_ch = filterNetworkTransitive(input_pairs_ch, input_spectra_ch)

    // Creating graphml
    input_graphml_ch = Channel.fromPath(params.input_graphml)
    filtered_graphml_ch = recreateGraphML(input_graphml_ch, filtered_networking_pairs_ch)

}