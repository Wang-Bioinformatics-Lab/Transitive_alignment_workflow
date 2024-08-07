# Transitive Alignment Workflow

To run the workflow to test simply do

```
make run
```

To learn NextFlow checkout this documentation:

https://www.nextflow.io/docs/latest/index.html

## Installation

You will need to have conda, mamba, and nextflow installed to run things locally. 

## GNPS2 Workflow Input information

Check the definition for the workflow input and display parameters:
https://wang-bioinformatics-lab.github.io/GNPS2_Documentation/workflowdev/


## Deployment to GNPS2

In order to deploy, we have a set of deployment tools that will enable deployment to the various gnps2 systems. To run the deployment, you will need the following setup steps completed:

1. Checked out of the deployment submodules
1. Conda environment and dependencies
1. SSH configuration updated

### Checking out the deployment submodules

use the following commands from the deploy_gnps2 folder. 

You might need to checkout the module, do this by running

```
git submodule init
git submodule update
```

You will also need to specify the user on the server that you've been given that your public key has been associated with. If you want to not enter this every time you do a deployment, you can create a Makefile.credentials file in the deploy_gnps2 folder with the following contents

```
USERNAME=<enter the username>
```

### Deployment Dependencies

You will need to install the dependencies in GNPS2_DeploymentTooling/requirements.txt on your own local machine. 

One way to do this is to use conda to create an environment, for example:

```
conda create -n deploy python=3.8
pip install -r GNPS2_DeploymentTooling/requirements.txt
```

### SSH Configuration

Also update your ssh config file to include the following ssh target:

```
Host ucr-gnps2-dev
    Hostname ucr-lemon.duckdns.org
```

### Deploying to Dev Server

To deploy to development, use the following command, if you don't have your ssh public key installed onto the server, you will not be able to deploy.

```
make deploy-dev
```

### Deploying to Production Server

To deploy to production, use the following command, if you don't have your ssh public key installed onto the server, you will not be able to deploy.

```
make deploy-prod
```

# Workflow Script Instructions

The workflow takes three file inputs:
1. **Input pairs file** (e.g., `merged_pairs.tsv`): Contains the pairs that will be analyzed.
2. **Input spectra file** (e.g., `specs_ms.mgf`): Contains the mass spectrometry data.
3. **Input GraphML file** (e.g., `network.graphml`): Contains the network data in GraphML format.

These files can all be obtained using the GNPS Classic molecular network workflow.

## Script Descriptions

### `partition.py`
- **Purpose**: Partition all the edge pairs that need to be realigned into chunks for efficient processing.
- **Default Setting**: The number of chunks is set to 100.

### `preprocessing.py`
- **Purpose**: Preprocess the spectra files to create a spectra dictionary necessary for the following processes.

### `Transitive_Alignment.py`
- **Purpose**: Perform transitive alignment on a chunk of the edge pairs in parallel to enhance alignment accuracy.
- **Default Settings**: Utilizes 4 CPUs and allocates 8 GB of memory for each process.

### `CAST.py`
- **Purpose**: Gather all the realigned edges files, reintegrate the realigned edges into the original graph, and refine the network. It employs the CAST algorithm for network filtering and refines each component based on the Minimum Spanning Tree (MST) algorithm.

### `recreate_graphml.py`
- **Purpose**: Use the new topology produced by the `CAST.py` script to create a GraphML file, showcasing the results of the Transitive Alignment + CAST approach.

## Workflow Parameter Description

### Workflow Options

#### Full Network Transitive Alignment
Do the transitive alignment for the whole network and use the CAST algorithm to filter out the topology. Enabling this will only make the **Network Topology Parameters** section take effect.

#### Induced Network
Do the transitive alignment only for the source node and induce the network from the source node using either the interaction or union method. Enabling this will only make the **Induced Network Parameters** section take effect.

### Network Topology Parameters

#### Transitive Param - Min Clique Cosine
The score threshold to control the edge score within a CAST clique.

#### Transitive Param - Min Transitive Alignment Score
The score threshold for adding a transitive alignment edge back to the network.

#### MST Filter Option
Filter option for each compound after constructing the MN (This is only for better visualization layout). Available options:
- **Pure MST**: Only apply MST on the MN.
- **Greedy MST**: After applying MST, greedily add back high score edges for higher average edge score.
- **Hybrid MST**: Using the original edges first, if unable to create MST, then use the transitive alignment edges to finish the MST.

### Induced Network Parameters

#### Induced Network Param - Option
- **Intersection the Transitive and MAX Hops**: Using this method, if and only if the nodes in the original network are within the max hops of the source node and also the transitive alignment score is above the setting threshold (**Induced Network Param - Min Transitive Alignment Score**) will be in the induced network results.
- **Union the Transitive and Max Hops**: This method will first do all the transitive alignment from the source node to all the other reachable nodes, and then add back the transitive alignment edge above the threshold (**Induced Network Param - Min Transitive Alignment Score**) to the original network. Afterward, select the node within the max hops from the source node or the edge score above the threshold (**Induced Network Param - Min Transitive Alignment Score**) to the results.

#### Induced Network Param - Source Node
The source node to start the induced network.

#### Induced Network Param - Max Hops
Maximum hops from the source node to the target node.

#### Induced Network Param - Min Transitive Alignment Score
The transitive alignment score threshold.

#### Induced Network Param - MST Filter Option
Filter option for the induced network after constructing (This is only for better visualization layout). Available options:
- **Pure MST**: Only apply MST on the MN.
- **Greedy MST**: After applying MST, greedily add back high score edges for higher average edge score.
- **Hybrid MST**: Using the original edges first, if unable to create MST, then use the transitive alignment edges to finish the MST.


