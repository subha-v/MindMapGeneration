import networkx as nx
import matplotlib.pyplot as plt

def filter_subtopics(main_topics, subtopics):
    # Filter out subtopics that are already in the list of main topics
    filtered_subtopics = []
    for subtopic_list in subtopics:
        filtered_subtopics.append([subtopic for subtopic in subtopic_list if subtopic not in main_topics])
    return filtered_subtopics

filtered_subtopics = filter_subtopics(['tang', 'china', 'song', 'chinese'], [['wu', 'li', 'dynasty', 'jin', 'emperor'], ['chinese', 'japan', 'li', 'korea', 'vietnam'], ['sung', 'single', 'charts', 'artist', 'poem'], ['china', 'li', 'jiang', 'wu', 'tang']])


def create_graph(main_topics, subtopics):
    # Add main topics as blue nodes
    G = nx.Graph()
    for i, main_topic in enumerate(main_topics):
        G.add_node(main_topic, color='#5BCEFA')

        # Add subtopics as pink nodes and connect them to their corresponding main topics
        for subtopic in subtopics[i]:
            G.add_node(subtopic, color='pink')
            G.add_edge(main_topic, subtopic)
    
    node_colors = [node[1]['color'] for node in G.nodes(data=True)]
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    plt.figure(figsize=(10, 10))
    nx.draw_networkx(G, pos, node_color=node_colors, node_size=2000, font_size=20, with_labels=True)
    plt.axis('off')
    plt.show()
    plt.savefig('graph.png')

create_graph(['tang', 'china', 'song', 'chinese'], filtered_subtopics)