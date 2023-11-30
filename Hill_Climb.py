import networkx as nx
import matplotlib.pyplot as plt
import random
import copy

cities = [
    "Dehradun",
    "Haridwar",
    "Rishikesh",
    "Muzaffarnagar",
    "Meerut",
    "Ghaziabad",
    "Delhi",
    "Faridabad",
    "Noida",
    "Mathura",
    "Agra",
    "Morena",
    "Gwalior",
    "Jhansi",
    "Bhopal",
    "Nagpur",
    "Karimnagar",
    "Vijaywada",
    "Nellore",
    "Chennai"
]

distances = {
    "Dehradun": {
        "Haridwar": 54,
        "Rishikesh": 29,
        "Muzaffarnagar": 139,
        "Meerut": 76,
        "Ghaziabad": 142,
        "Delhi": 158,
        "Faridabad": 174,
        "Noida": 192,
        "Mathura": 264,
        "Agra": 45,
        "Morena": 230,
        "Gwalior": 119,
        "Jhansi": 235,
        "Bhopal": 386,
        "Nagpur": 761,
        "Karimnagar": 1453,
        "Vijaywada": 1110,
        "Nellore": 1067,
        "Chennai": 1237
    },
    "Haridwar": {
        "Dehradun": 54,
        "Rishikesh": 25,
        "Muzaffarnagar": 116,
        "Meerut": 63,
        "Ghaziabad": 119,
        "Delhi": 135,
        "Faridabad": 151,
        "Noida": 169,
        "Mathura": 241,
        "Agra": 99,
        "Morena": 175,
        "Gwalior": 111,
        "Jhansi": 116,
        "Bhopal": 267,
        "Nagpur": 642,
        "Karimnagar": 1434,
        "Vijaywada": 1091,
        "Nellore": 1048,
        "Chennai": 2316,
    },
    "Rishikesh": {
        "Dehradun": 29,
        "Haridwar": 25,
        "Muzaffarnagar": 124,
        "Meerut": 71,
        "Ghaziabad": 127,
        "Delhi": 143,
        "Faridabad": 159,
        "Noida": 177,
        "Mathura": 249,
        "Agra": 85,
        "Morena": 161,
        "Gwalior": 149,
        "Jhansi": 165,
        "Bhopal": 316,
        "Nagpur": 691,
        "Karimnagar": 1383,
        "Vijaywada": 1040,
        "Nellore": 997,
        "Chennai": 2294,
    },
    "Muzaffarnagar": {
        "Dehradun": 139,
        "Haridwar": 116,
        "Rishikesh": 124,
        "Meerut": 47,
        "Ghaziabad": 103,
        "Delhi": 119,
        "Faridabad": 135,
        "Noida": 153,
        "Mathura": 225,
        "Agra": 90,
        "Morena": 166,
        "Gwalior": 154,
        "Jhansi": 170,
        "Bhopal": 321,
        "Nagpur": 696,
        "Karimnagar": 1388,
        "Vijaywada": 1045,
        "Nellore": 1002,
        "Chennai": 2250,
    },
    "Meerut": {
        "Dehradun": 76,
        "Haridwar": 63,
        "Rishikesh": 71,
        "Muzaffarnagar": 47,
        "Ghaziabad": 56,
        "Delhi": 72,
        "Faridabad": 88,
        "Noida": 106,
        "Mathura": 178,
        "Agra": 47,
        "Morena": 123,
        "Gwalior": 111,
        "Jhansi": 127,
        "Bhopal": 278,
        "Nagpur": 653,
        "Karimnagar": 1345,
        "Vijaywada": 1002,
        "Nellore": 959,
        "Chennai": 2125,
    },
    "Ghaziabad": {
        "Dehradun": 142,
        "Haridwar": 119,
        "Rishikesh": 127,
        "Muzaffarnagar": 103,
        "Meerut": 56,
        "Delhi": 16,
        "Faridabad": 32,
        "Noida": 50,
        "Mathura": 122,
        "Agra": 91,
        "Morena": 167,
        "Gwalior": 155,
        "Jhansi": 171,
        "Bhopal": 322,
        "Nagpur": 697,
        "Karimnagar": 1389,
        "Vijaywada": 1046,
        "Nellore": 1003,
        "Chennai": 2050,
    },
    "Delhi": {
        "Dehradun": 158,
        "Haridwar": 135,
        "Rishikesh": 143,
        "Muzaffarnagar": 119,
        "Meerut": 72,
        "Ghaziabad": 16,
        "Faridabad": 18,
        "Noida": 36,
        "Mathura": 108,
        "Agra": 75,
        "Morena": 151,
        "Gwalior": 139,
        "Jhansi": 155,
        "Bhopal": 306,
        "Nagpur": 681,
        "Karimnagar": 1373,
        "Vijaywada": 1030,
        "Nellore": 987,
        "Chennai": 1965,
    },
    "Faridabad": {
        "Dehradun": 174,
        "Haridwar": 151,
        "Rishikesh": 159,
        "Muzaffarnagar": 135,
        "Meerut": 88,
        "Ghaziabad": 32,
        "Delhi": 18,
        "Noida": 18,
        "Mathura": 90,
        "Agra": 59,
        "Morena": 135,
        "Gwalior": 123,
        "Jhansi": 139,
        "Bhopal": 290,
        "Nagpur": 665,
        "Karimnagar": 1357,
        "Vijaywada": 1014,
        "Nellore": 971,
        "Chennai": 1950,
    },
    "Noida": {
        "Dehradun": 192,
        "Haridwar": 169,
        "Rishikesh": 177,
        "Muzaffarnagar": 153,
        "Meerut": 106,
        "Ghaziabad": 50,
        "Delhi": 36,
        "Faridabad": 18,
        "Mathura": 72,
        "Agra": 41,
        "Morena": 117,
        "Gwalior": 105,
        "Jhansi": 121,
        "Bhopal": 272,
        "Nagpur": 647,
        "Karimnagar": 1339,
        "Vijaywada": 995,
        "Nellore": 952,
        "Chennai": 1935,
    },
    "Mathura": {
        "Dehradun": 264,
        "Haridwar": 241,
        "Rishikesh": 249,
        "Muzaffarnagar": 225,
        "Meerut": 178,
        "Ghaziabad": 122,
        "Delhi": 108,
        "Faridabad": 90,
        "Noida": 72,
        "Agra": 56,
        "Morena": 66,
        "Gwalior": 185,
        "Jhansi": 301,
        "Bhopal": 455,
        "Nagpur": 830,
        "Karimnagar": 1622,
        "Vijaywada": 1387,
        "Nellore": 1344,
        "Chennai": 1800,
    },
    "Agra": {
        "Dehradun": 429,
        "Haridwar": 412,
        "Rishikesh": 394,
        "Muzaffarnagar": 261,
        "Meerut": 210,
        "Ghaziabad": 183,
        "Delhi": 204,
        "Faridabad": 240,
        "Noida": 220,
        "Mathura": 56,
        "Morena": 230,
        "Gwalior": 119,
        "Jhansi": 235,
        "Bhopal": 386,
        "Nagpur": 761,
        "Karimnagar": 1453,
        "Vijaywada": 1110,
        "Nellore": 1067,
        "Chennai": 1237
    },
    "Morena": {
        "Dehradun": 384,
        "Haridwar": 366,
        "Rishikesh": 348,
        "Muzaffarnagar": 215,
        "Meerut": 164,
        "Ghaziabad": 137,
        "Delhi": 158,
        "Faridabad": 194,
        "Noida": 174,
        "Mathura": 66,
        "Agra": 230,
        "Gwalior": 111,
        "Jhansi": 227,
        "Bhopal": 378,
        "Nagpur": 753,
        "Karimnagar": 1445,
        "Vijaywada": 1102,
        "Nellore": 1059,
        "Chennai": 1229
    },
    "Gwalior": {
        "Dehradun": 503,
        "Haridwar": 485,
        "Rishikesh": 467,
        "Muzaffarnagar": 334,
        "Meerut": 283,
        "Ghaziabad": 256,
        "Delhi": 277,
        "Faridabad": 313,
        "Noida": 293,
        "Mathura": 185,
        "Agra": 119,
        "Morena": 111,
        "Jhansi": 116,
        "Bhopal": 267,
        "Nagpur": 642,
        "Karimnagar": 1434,
        "Vijaywada": 1091,
        "Nellore": 1048,
        "Chennai": 1218
    },
    "Jhansi": {
        "Dehradun": 619,
        "Haridwar": 601,
        "Rishikesh": 583,
        "Muzaffarnagar": 450,
        "Meerut": 399,
        "Ghaziabad": 372,
        "Delhi": 393,
        "Faridabad": 429,
        "Noida": 409,
        "Mathura": 301,
        "Agra": 235,
        "Morena": 227,
        "Gwalior": 116,
        "Bhopal": 251,
        "Nagpur": 626,
        "Karimnagar": 1418,
        "Vijaywada": 1075,
        "Nellore": 1032,
        "Chennai": 1202
    },
    "Bhopal": {
        "Dehradun": 773,
        "Haridwar": 755,
        "Rishikesh": 737,
        "Muzaffarnagar": 604,
        "Meerut": 553,
        "Ghaziabad": 526,
        "Delhi": 547,
        "Faridabad": 583,
        "Noida": 563,
        "Mathura": 455,
        "Agra": 389,
        "Morena": 381,
        "Gwalior": 270,
        "Jhansi": 251,
        "Nagpur": 375,
        "Karimnagar": 1067,
        "Vijaywada": 724,
        "Nellore": 681,
        "Chennai": 851
    },
    "Nagpur": {
        "Dehradun": 1148,
        "Haridwar": 1130,
        "Rishikesh": 1112,
        "Muzaffarnagar": 979,
        "Meerut": 928,
        "Ghaziabad": 901,
        "Delhi": 922,
        "Faridabad": 958,
        "Noida": 938,
        "Mathura": 830,
        "Agra": 764,
        "Morena": 756,
        "Gwalior": 645,
        "Jhansi": 626,
        "Bhopal": 375,
        "Karimnagar": 692,
        "Vijaywada": 349,
        "Nellore": 306,
        "Chennai": 476
    },
    "Karimnagar": {
        "Dehradun": 1940,
        "Haridwar": 1922,
        "Rishikesh": 1904,
        "Muzaffarnagar": 1771,
        "Meerut": 1720,
        "Ghaziabad": 1693,
        "Delhi": 1714,
        "Faridabad": 1750,
        "Noida": 1730,
        "Mathura": 1622,
        "Agra": 1556,
        "Morena": 1548,
        "Gwalior": 1437,
        "Jhansi": 1418,
        "Bhopal": 1067,
        "Nagpur": 692,
        "Vijaywada": 344,
        "Nellore": 464,
        "Chennai": 635
    },
    "Vijaywada": {
        "Dehradun": 1705,
        "Haridwar": 1687,
        "Rishikesh": 1669,
        "Muzaffarnagar": 1536,
        "Meerut": 1485,
        "Ghaziabad": 1458,
        "Delhi": 1479,
        "Faridabad": 1515,
        "Noida": 1495,
        "Mathura": 1387,
        "Agra": 1321,
        "Morena": 1313,
        "Gwalior": 1202,
        "Jhansi": 1183,
        "Bhopal": 724,
        "Nagpur": 349,
        "Karimnagar": 344,
        "Nellore": 43,
        "Chennai": 214
    },
    "Nellore": {
        "Dehradun": 1707,
        "Haridwar": 1689,
        "Rishikesh": 1671,
        "Muzaffarnagar": 1538,
        "Meerut": 1487,
        "Ghaziabad": 1460,
        "Delhi": 1481,
        "Faridabad": 1517,
        "Noida": 1497,
        "Mathura": 1389,
        "Agra": 1323,
        "Morena": 1315,
        "Gwalior": 1204,
        "Jhansi": 1185,
        "Bhopal": 681,
        "Nagpur": 306,
        "Karimnagar": 464,
        "Vijaywada": 43,
        "Chennai": 171
    },
    "Chennai": {
        "Dehradun": 1392,
        "Haridwar": 1374,
        "Rishikesh": 1356,
        "Muzaffarnagar": 1223,
        "Meerut": 1172,
        "Ghaziabad": 1145,
        "Delhi": 1166,
        "Faridabad": 1202,
        "Noida": 1182,
        "Mathura": 1074,
        "Agra": 1008,
        "Morena": 1000,
        "Gwalior": 889,
        "Jhansi": 870,
        "Bhopal": 851,
        "Nagpur": 476,
        "Karimnagar": 635,
        "Vijaywada": 214,
        "Nellore": 171
    }
}


# Create a graph using NetworkX
G = nx.Graph()

for city in cities:
    G.add_node(city)

for city1 in cities:
    for city2, distance in distances[city1].items():
        G.add_edge(city1, city2, weight=distance)

# calculate the total distance of a route
def calculate_total_distance(route, distances):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distances[route[i]][route[i + 1]]
    return total_distance

# Initialize the route from Dehradun to Chennai
initial_route = [
    "Dehradun", "Haridwar", "Rishikesh", "Muzaffarnagar", "Meerut", "Ghaziabad", "Delhi",
    "Faridabad", "Noida", "Mathura", "Agra", "Morena", "Gwalior", "Jhansi", "Bhopal", "Nagpur",
    "Karimnagar", "Vijaywada", "Nellore","Chennai"
]

# Define the number of iterations
num_iterations = 1000

# Hill Climbing to find the best route
current_route = copy.deepcopy(initial_route)
best_distance = calculate_total_distance(current_route, distances)

for i in range(num_iterations):
    # Generate a neighboring route by swapping two cities
    neighbor_route = copy.deepcopy(current_route)
    city1, city2 = random.sample(range(1, len(neighbor_route) - 1), 2)  # Avoid swapping Dehradun and Chennai
    neighbor_route[city1], neighbor_route[city2] = neighbor_route[city2], neighbor_route[city1]

    # Calculate distances for current and neighbor routes
    neighbor_distance = calculate_total_distance(neighbor_route, distances)

    # If the neighbor route is better, accept it
    if neighbor_distance < best_distance:
        current_route = neighbor_route
        best_distance = neighbor_distance

# Create a graph for the best route
best_route_graph = nx.Graph()

for i in range(len(current_route) - 1):
    city1, city2 = current_route[i], current_route[i + 1]
    best_route_graph.add_edge(city1, city2, weight=distances[city1][city2])

print("Best Route:", current_route)
print("Best Distance:", best_distance)

# Show the graph with all cities
pos = nx.spring_layout(G)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw(G, pos, with_labels=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

plt.title("Graph with All Cities and Routes")
plt.show()

# Draw the graph for the best route
pos = nx.spring_layout(best_route_graph)
labels = nx.get_edge_attributes(best_route_graph, 'weight')
nx.draw(best_route_graph, pos, with_labels=True)
nx.draw_networkx_edge_labels(best_route_graph, pos, edge_labels=labels)

plt.title("Best Route from Dehradun to Chennai")
plt.show()
