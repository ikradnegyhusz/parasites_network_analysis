import backboning
import projection
import networkx as nx
import pandas as pd

projections ={
    "simple":projection.simple,
    "hyperbolic":projection.hyperbolic,
    "pearson":projection.pearson
}
backbonings={
    "noise_corrected":backboning.noise_corrected,
    "high_salience_skeleton":backboning.high_salience_skeleton,
    "doubly_stochastic":backboning.doubly_stochastic,
    "disparity_filter":backboning.disparity_filter
}
#convex network reduction missing

G=nx.read_edgelist("data/edges.csv",delimiter=",",nodetype=int)
#get sets of nodes (animal, parasite)
nodes = pd.read_csv("data/nodes.csv",delimiter=",")
animal_nodes = list(set(nodes[nodes[" is_host"]==1]["# index"]))
parasite_nodes = list(set(nodes[nodes[" is_host"]==0]["# index"]))


# function is needed to convert graph to df for backboning (with predefined function)
def graph_to_dataframe(graph):
    edges = nx.to_pandas_edgelist(graph)
    edges.rename(columns={'source': 'src', 'target': 'trg', 'weight': 'nij'}, inplace=True)
    return edges


for p_name,p in projections.items():
    for b_name,b in backbonings.items():
        projected_network = p(G,animal_nodes)
        edges_df = graph_to_dataframe(projected_network)
        result = b(edges_df)
        result.to_csv(f"projections_with_backbonings/{p_name}_{b_name}.csv")