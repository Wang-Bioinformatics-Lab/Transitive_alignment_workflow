#!/usr/bin/env nextflow
nextflow.enable.dsl=2

TOOL_FOLDER = "$baseDir/bin"

params.input_pairs = './data/test/merged_pairs.tsv'
params.input_spectra = './data/test/specs_ms.mgf'
params.n_chunks = 100
params.input_graphml = "./data/test/network.graphml"
params.workflow_option = "Induced_network"

// Topology Filtering
params.topology_cliquemincosine = 0.65
params.networking_min_cosine = 0.4
params.mst_filter = "Greedy_MST"

// Induced network
params.source_node = 672
params.max_hops = 4
params.induced_networking_min_cosine = 0.3
params.induced_network_option = "intersection"
params.induced_mst_filter = "Pure_MST"

process Partition {
    conda "$baseDir/bin/conda_env.yml"
    input:
    file merged_pairs
    val n_chunks

    output:
    path "chunks/*.tsv"

    script:
    """
    mkdir -p chunks
    python $TOOL_FOLDER/scripts/partition.py -m ${merged_pairs} --n_chunks ${n_chunks}
    """
}

process Preprocessing {
    conda "$baseDir/bin/conda_env.yml"
    input:
    path specs_mgf

    output:
    path "spec_dic.pkl"

    script:
    """
    python $TOOL_FOLDER/scripts/preprocessing.py -c ${specs_mgf}
    """
}

process TransitiveAlignment {
    conda "$baseDir/bin/conda_env.yml"
    cpus 4
    memory '8 GB'

    input:
    tuple path(chunk),path(spec_dic),file(merged_pairs)

    output:
    path "transitive_alignment/*.pkl"

    script:
    """
    mkdir -p transitive_alignment
    python $TOOL_FOLDER/scripts/Transitive_Alignment.py -s ${spec_dic} -i ${chunk} -m ${merged_pairs} -p 4 -r "transitive_alignment/${chunk.baseName}_realignment.pkl"
    """
}

process CAST {
    publishDir "./nf_output", mode: 'copy'
    conda "$baseDir/bin/conda_env.yml"


    input:
    // To avoid naming collisions
    path to_merge, stageAs: './trans_align_dir/*'
    file merged_pairs

    output:
    file "filtered_pairs.tsv"
    path "python_debug.log"
    //file "degree_loglog_plot.png"

    script:
    """
    python $TOOL_FOLDER/scripts/CAST.py \
    -m ${merged_pairs} \
    -t  ./trans_align_dir \
    -th $params.topology_cliquemincosine \
    -r filtered_pairs.tsv \
    --minimum_score $params.networking_min_cosine \
    --mst_filter $params.mst_filter
    """
}

process recreateGraphML {
    publishDir "./nf_output/", mode: 'copy'

    conda "$TOOL_FOLDER/conda_env.yml"

    cache false

    input:
    file input_mgf
    file input_graphml
    file filtered_networking_pairs

    output:
    file "network"
    file "spectra"

    """
    mkdir spectra
    cp ${input_mgf} spectra/specs_ms.mgf

    mkdir network
    python $TOOL_FOLDER/scripts/recreate_graphml.py \
    -g ${input_graphml} \
    -m ${filtered_networking_pairs} \
    network/network.graphml
    """
}

process InducedNetwork {
    publishDir "./nf_output/", mode: 'copy'

    conda "$TOOL_FOLDER/conda_env.yml"

    cache false

    input:
    file merged_pairs
    file input_mgf
    file input_graphml
    file spec_dic


    output:
    file "network"
    file "spectra"

    """
    mkdir spectra
    cp ${input_mgf} spectra/specs_ms.mgf

    mkdir network
    python $TOOL_FOLDER/scripts/induced_network.py \
    -m ${merged_pairs} \
    -s $params.source_node \
    -g ${input_graphml} \
    --max_hops $params.max_hops \
    --spec_dic ${spec_dic} \
    --minimum_score $params.induced_networking_min_cosine \
    --induced_option $params.induced_network_option \
    --mst_filter $params.induced_mst_filter \
    network/network.graphml
    """
}

workflow {

    // Input channels
    merged_pairs_ch = Channel.fromPath(params.input_pairs)
    specs_mgf_ch = Channel.fromPath(params.input_spectra)

    // Partitioning to create chunk files

    // Preprocessing to create the spectral dictionary

    // Transitive alignment using generated chunk files and spectral dictionary
    chunk_files_ch = Partition(merged_pairs_ch, params.n_chunks)
    spec_dic_ch = Preprocessing(specs_mgf_ch)
    if (params.workflow_option == 'Transitive_alignment'){
        trans_align_ch = TransitiveAlignment(chunk_files_ch.flatten().combine(spec_dic_ch).combine(merged_pairs_ch))

        trans_align_dir_ch = trans_align_ch.collect()

        // Filtering the network
        (filtered_networking_pairs_ch,debug_info_ch) = CAST(trans_align_dir_ch, merged_pairs_ch)
        // Creating graphml
        input_graphml_ch = Channel.fromPath(params.input_graphml)
        recreateGraphML(specs_mgf_ch, input_graphml_ch, filtered_networking_pairs_ch)
        // print out the debug info
        debug_info_ch.subscribe { file ->
            println("Debug info from CAST process:")
            file.readLines().each { line ->
                println(line)
            }
        }
    }
    else if (params.workflow_option == 'Induced_network'){
        input_graphml_ch = Channel.fromPath(params.input_graphml)
        InducedNetwork(merged_pairs_ch,specs_mgf_ch,input_graphml_ch,spec_dic_ch)
    }
}
