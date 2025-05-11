
import streamlit as st
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import requests

# Fonction de géocodage via OpenStreetMap (sans geopy)
def geocode_address(address):
    url = "https://nominatim.openstreetmap.org/search"
    params = {'q': address, 'format': 'json', 'limit': 1}
    headers = {'User-Agent': 'VRP-App'}
    response = requests.get(url, params=params, headers=headers)
    data = response.json()
    if data:
        return float(data[0]['lat']), float(data[0]['lon'])
    return None, None

# Calcul de distances
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
    clients = list(range(1, n_clients + 1))
    random.shuffle(clients)
    solution = []
    for i in range(n_vehicules):
        route_clients = clients[i::n_vehicules]
        solution.append([0] + route_clients + [0] if route_clients else [0, 0])
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
    plt.title("🗺️ Tournées optimisées (Génétique)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()

# ----------- Interface améliorée Streamlit -----------
st.set_page_config(page_title="Optimisation VRP", layout="wide")
st.markdown("# 🚚 Optimisation de Tournées de Véhicules")
st.markdown("Optimisez vos trajets à partir d'une zone, d'adresses ou de coordonnées. Application 100% en ligne.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📍 Zone géographique de génération")
    lat_min = st.number_input("Latitude min", value=50.610)
    lat_max = st.number_input("Latitude max", value=50.655)
    lon_min = st.number_input("Longitude min", value=3.01)
    lon_max = st.number_input("Longitude max", value=3.09)
    n_clients = st.slider("Nombre de clients aléatoires", 1, 100, 10)
    n_vehicules = st.slider("Nombre de véhicules", 1, 10, 2)

with col2:
    st.markdown("### ➕ Ajouter manuellement un client")
    lat_manual = st.number_input("Latitude client", key="lat_manuel")
    lon_manual = st.number_input("Longitude client", key="lon_manuel")
    add_manual = st.button("✅ Ajouter ce client")

    st.markdown("### 🏠 Ajouter un client par adresse")
    adresse = st.text_input("Entrez une adresse complète")
    geo_btn = st.button("🔎 Géocoder et ajouter")

# Initialisation des clients
clients = {}
if lat_max <= lat_min or lon_max <= lon_min:
    st.error("La zone géographique n'est pas valide (max doit être > min)")
else:
    depot = ((lat_min + lat_max) / 2, (lon_min + lon_max) / 2)
    clients[0] = depot
    for i in range(1, n_clients + 1):
        clients[i] = (
            random.uniform(lat_min, lat_max),
            random.uniform(lon_min, lon_max)
        )

# Gestion des ajouts manuels
if add_manual:
    clients[len(clients)] = (lat_manual, lon_manual)
    st.success("Client ajouté manuellement !")

if geo_btn:
    lat, lon = geocode_address(adresse)
    if lat and lon:
        clients[len(clients)] = (lat, lon)
        st.success(f"Client ajouté : ({lat:.5f}, {lon:.5f})")
    else:
        st.error("Adresse introuvable")

# Lancement de l'optimisation
if st.button("🚀 Lancer l'optimisation"):
    nb = len(clients) - 1
    sol, c = algo_genetique(clients, nb, n_vehicules)
    st.success(f"✅ Coût total de la solution : {round(c, 2)}")
    for i, route in enumerate(sol):
        st.markdown(f"**Véhicule {i+1}** : {route}")
    plot_solution(sol, clients)
