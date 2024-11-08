import networkx as nx

def get_doubly_stochastic(G: nx.Graph()) -> nx.Graph():
    stadj = nx.adjacency_matrix(G).toarray()
    # check if sum is close to one
    for i, iteration in enumerate(range(1000)):
        stadj = stadj / stadj.sum(axis=1)[:, np.newaxis]
        stadj = stadj / stadj.sum(axis=0)[np.newaxis, :]
        print(f"iteration {i}")
        if np.allclose(stadj.sum(axis=1), 1) and np.allclose(stadj.sum(axis=0), 1):
            print("iteration stopped at", iteration)
            break
