import backboning
import projection
import networkx as nx
import pandas as pd
import pickle

projection_methods = {
    'simple': {'function': projection.simple,
                'backbonings': {'disparity_filter':backboning.disparity_filter,'noise_corrected':backboning.noise_corrected}},
    'resource_allocation': {'function': projection.resource_allocation,
                            'backbonings': {'naive':backboning.naive}}
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

for key,projection_method in projection_methods.items():
    print(f"projection: {key}")
    projected_network = projection_method['function'](G,parasite_nodes)
    edges_df = graph_to_dataframe(projected_network)
    edges_df.to_csv(f"projections/{key}.csv")
    for label, backboning_method in projection_method['backbonings'].items():
        print(f"backboining: {label}")
        backboned = backboning_method(edges_df)
        result = backboning.test_densities(backboned,0.0,1.0,0.01/4)
        pickle.dump(result,open(f"stats/{key}_{label}_stats.pkl","wb"))
        backboned.to_csv(f"projections_with_backbonings/{key}_{label}.csv")
