import backboning
import projection
import networkx as nx
import pandas as pd

projection_methods = {
    'simple': {'function': projection.simple,
               'backbonings': {'noise_corrected':backboning.noise_corrected, 'disparity_filter': backboning.disparity_filter}},
    'hyperbolic': {'function': projection.hyperbolic,
                   'backbonings': {'high_salience_skeleton': backboning.high_salience_skeleton}},
    'pearson': {'function': projection.pearson,
                'backbonings': {'noise_corrected':backboning.noise_corrected}}
}

selected_projections = ["pearson"]

#convex network reduction missing
G=nx.read_edgelist("../data/edges.csv",delimiter=",",nodetype=int)
#get sets of nodes (animal, parasite)
nodes = pd.read_csv("../data/nodes.csv",delimiter=",")
animal_nodes = list(set(nodes[nodes[" is_host"]==1]["# index"]))
parasite_nodes = list(set(nodes[nodes[" is_host"]==0]["# index"]))

# function is needed to convert graph to df for backboning (with predefined function)
def graph_to_dataframe(graph):
    edges = nx.to_pandas_edgelist(graph)
    edges.rename(columns={'source': 'src', 'target': 'trg', 'weight': 'nij'}, inplace=True)
    return edges

for key,method in projection_methods.items():
    if key in selected_projections:
        projected_network = method['function'](G,parasite_nodes)
    else:
        continue
    edges_df = graph_to_dataframe(projected_network)
    for label, backboning in method['backbonings'].items():
        result = backboning(edges_df)
        result.to_csv(f"projections_with_backbonings/{key}_{label}.csv")
