
import streamlit as st
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import requests

# Fonction de géocodage via l’API OpenStreetMap Nominatim (sans geopy)
def geocode_address(address):
    url = f"https://nominatim.openstreetmap.org/search"
    params = {
        'q': address,
        'format': 'json',
        'limit': 1
    }
    headers = {'User-Agent': 'VRP-App'}
    response = requests.get(url, params=params, headers=headers)
    data = response.json()
    if data:
        return float(data[0]['lat']), float(data[0]['lon'])
    return None, None

# Distance approximative
def distance_approx(lat1, lon1, lat2, lon2):
    lat_moy = math.radians((lat1 + lat2) / 2)
    dx = (lon2 - lon1) * 111320 * math.cos(lat_moy)
    dy = (lat2 - lat1) * 111320
    return math.sqrt(dx**2 + dy**2)

def cout_solution(solution, localisations, omega=5):
    total = omega * len(solution)
    for route in solution:
        for i in range(len(route) - 1):
            total += distance_approx(*localisations[route[i]], *localisations[route[i+1]])
    return total

def initialiser_solution(n_clients, n_vehicules):
    clients = list(range(1, n_clients + 1))  # exclut dépôt
    random.shuffle(clients)
    solution = []
    for i in range(n_vehicules):
        route_clients = clients[i::n_vehicules]
        if route_clients:
            solution.append([0] + route_clients + [0])
        else:
            solution.append([0, 0])
    return solution

def crossover(parent1, parent2):
    enfant = []
    for r1, r2 in zip(parent1, parent2):
        inter_r1 = r1[1:-1]
        inter_r2 = r2[1:-1]
        point = random.randint(1, len(inter_r1)-1) if len(inter_r1) > 1 else 1
        merged = inter_r1[:point] + [c for c in inter_r2 if c not in inter_r1[:point]]
        enfant.append([0] + merged + [0])
    return enfant

def mutation(solution):
    voisin = [route[:] for route in solution]
    route = random.choice([r for r in voisin if len(r) > 3])
    i, j = sorted(random.sample(range(1, len(route) - 1), 2))
    route[i], route[j] = route[j], route[i]
    return voisin

def algo_genetique(localisations, n_clients, n_vehicules, pop_size=10, generations=30):
    population = [initialiser_solution(n_clients, n_vehicules) for _ in range(pop_size)]
    for _ in range(generations):
        population = sorted(population, key=lambda s: cout_solution(s, localisations))
        parents = population[:2]
        new_pop = parents[:]
        while len(new_pop) < pop_size:
            enfant = crossover(*parents)
            enfant = mutation(enfant)
            new_pop.append(enfant)
        population = new_pop
    best = min(population, key=lambda s: cout_solution(s, localisations))
    return best, cout_solution(best, localisations)

def plot_solution(solution, localisations):
    depot = localisations[0]
    for i, route in enumerate(solution):
        lat = [localisations[node][0] for node in route]
        lon = [localisations[node][1] for node in route]
        plt.plot(lon, lat, marker='o', label=f"Véhicule {i+1}")
    plt.scatter([depot[1]], [depot[0]], c='red', label='Dépôt', s=100)
    plt.title("Tournées optimisées (Génétique + Dépôt)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()

# Interface Streamlit
st.title("Optimisation de tournées VRP (version interactive sans geopy)")
st.markdown("Utilisation de l’algorithme génétique et géocodage manuel (OpenStreetMap)")

n_clients = st.number_input("Nombre de clients aléatoires à générer", min_value=1, max_value=100, value=5)
n_vehicules = st.slider("Nombre de véhicules", 1, 10, 3)

lat_min = st.number_input("Latitude min", value=50.60)
lat_max = st.number_input("Latitude max", value=50.70)
lon_min = st.number_input("Longitude min", value=3.00)
lon_max = st.number_input("Longitude max", value=3.15)

if lat_max <= lat_min or lon_max <= lon_min:
    st.error("Les bornes géographiques doivent être cohérentes (max > min).")
else:
    depot = ((lat_min + lat_max) / 2, (lon_min + lon_max) / 2)
    clients = {0: depot}

    for i in range(1, n_clients + 1):
        lat = random.uniform(lat_min, lat_max)
        lon = random.uniform(lon_min, lon_max)
        clients[i] = (lat, lon)

    with st.expander("Ajouter manuellement un client (coordonnées)"):
        lat = st.number_input("Latitude client", key="lat_manuel")
        lon = st.number_input("Longitude client", key="lon_manuel")
        if st.button("Ajouter ce client"):
            clients[len(clients)] = (lat, lon)
            st.success("Client ajouté.")

    with st.expander("Ajouter un client par adresse"):
        address = st.text_input("Adresse")
        if st.button("Géocoder l'adresse et ajouter"):
            lat, lon = geocode_address(address)
            if lat and lon:
                clients[len(clients)] = (lat, lon)
                st.success(f"Client ajouté à partir de l’adresse : ({lat}, {lon})")
            else:
                st.error("Adresse introuvable.")

    if st.button("Lancer l'optimisation"):
        n_clients_effectifs = len(clients) - 1
        solution, cout = algo_genetique(clients, n_clients_effectifs, n_vehicules)
        st.success(f"Coût total : {round(cout, 2)}")
        for i, route in enumerate(solution):
            st.write(f"Véhicule {i+1} : {route}")
        plot_solution(solution, clients)
