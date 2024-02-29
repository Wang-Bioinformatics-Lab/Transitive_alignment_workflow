#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.input = "README.md"

TOOL_FOLDER = "$baseDir/bin"

params.merged_pairs = './data/merged_pairs.tsv'
params.specs_mgf = './data/specs_ms.mgf'
params.n_chunks = 500
params.raw_pairs = './data/merged_pairs.tsv'
params.chunk_dir = './chunks/'
params.spec_dic = 'spec_dic.pkl'
params.trans_alignment_dir = './transitive_alignment/'



process Partition {
    publishDir "./nf_output", mode: 'copy'
    conda "$baseDir/bin/conda_env.yml"
    input:
    file merged_pairs
    val n_chunks

    output:
    path "chunks/*.tsv"

    script:
    """
    mkdir -p chunks
    python $TOOL_FOLDER/partition.py -m ${merged_pairs} --n_chunks ${n_chunks}
    """
}

process Preprocessing {
    publishDir "./nf_output", mode: 'copy'
    conda "$baseDir/bin/conda_env.yml"
    input:
    path specs_mgf

    output:
    path "spec_dic.pkl"

    script:
    """
    python $TOOL_FOLDER/preprocessing.py -c ${specs_mgf}
    """
}

process TransitiveAlignment {
    publishDir "./nf_output", mode: 'copy'
    conda "$baseDir/bin/conda_env.yml"
    cpus 15
    memory '12 GB'

    input:
    tuple path(chunk),path(spec_dic),file(merged_pairs)

    output:
    path "transitive_alignment/*.pkl"

    script:
    """
    mkdir -p transitive_alignment
    python $TOOL_FOLDER/Transitive_Alignment.py -s ${spec_dic} -i ${chunk} -m ${merged_pairs} -p 15 -r "transitive_alignment/${chunk.baseName}_realignment.pkl"
    """
}

workflow {

    // Input channels
    merged_pairs_ch = Channel.fromPath(params.merged_pairs)
    specs_mgf_ch = Channel.fromPath(params.specs_mgf)

    // Partitioning to create chunk files

    // Preprocessing to create the spectral dictionary

    // Transitive alignment using generated chunk files and spectral dictionary
    chunk_files_ch = Partition(merged_pairs_ch, params.n_chunks)
    spec_dic_ch = Preprocessing(specs_mgf_ch)
    TransitiveAlignment(chunk_files_ch.flatten().combine(spec_dic_ch).combine(merged_pairs_ch))
}