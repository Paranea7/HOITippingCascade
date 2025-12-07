import networkx as nx
import matplotlib.pyplot as plt
import random

n = 9
G = nx.complete_graph(n)

colors = [random.choice(["red", "lightgreen"]) for _ in range(n)]

pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=300)
nx.draw_networkx_edges(G, pos, edge_color="gray", width=1.0, alpha=0.3)

plt.show()