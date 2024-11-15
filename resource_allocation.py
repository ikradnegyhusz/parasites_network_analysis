import numpy as np
import networkx as nx
import pandas as pd
from data.localities_grouping import mapping as locality_to_group


# row stochastic adjacency matrix, of going from host to parasite
def average_resource_allocation(badj: np.array) -> np.array:
    A = badj / badj.sum(axis=1)[:, np.newaxis]  # if the row is zero :()
    A = np.nan_to_num(A, nan=0)
    At = badj.T / badj.T.sum(axis=1)[:, np.newaxis]  #
    At = np.nan_to_num(A, nan=0, posinf=0, neginf=0)
    weights = A @ At
    np.fill_diagonal(weights, 0)
    return (weights + weights.T) / 2


def obtain_bipartite_animals_parasites_graph(df: pd.DataFrame) -> tuple[np.array, list]:
    """All neighbouring node or edge features are copied to parasites (bipartite=1)"""
    G = nx.Graph()

    for _, el in df.iterrows():
        if not G.has_node(el["Host"]):
            G.add_node(el["Host"], bipartite=0)

        if not G.has_node(el["Parasite"]):
            locality = set()
            locality.add(locality_to_group[el["locality"]])
            host_group = set()
            host_group.add(el["hostgroup"])

            G.add_node(
                el["Parasite"],
                bipartite=1,
                host_group=host_group,
                para_group=el["group"],
                locality=locality,
            )
        else:  # if
            # If the parasite is already in the graph - then just its contsnts
            G.nodes[el["Parasite"]]["locality"].add(locality_to_group[el["locality"]])
            G.nodes[el["Parasite"]]["host_group"].add(el["hostgroup"])

            # if not G.has_edge(el["Parasite"], el["Host"]):
            G.add_edge(
                el["Parasite"],
                el["Host"],
            )

    bipartite_one_nodes = sorted(
        n for n, d in G.nodes(data=True) if d.get("bipartite") == 1
    )
    badj = nx.bipartite.biadjacency_matrix(G, row_order=bipartite_one_nodes).toarray()
    return (badj, bipartite_one_nodes, G)
