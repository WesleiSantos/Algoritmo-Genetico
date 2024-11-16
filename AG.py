import numpy as np
import matplotlib.pyplot as plt

# Parâmetros do algoritmo genético
L = 22  # Número de bits para representar x com precisão de 6 casas decimais
min_val = -1
max_val = 2
pop_size = 100  # Tamanho da população
num_generations = 100  # Número de gerações
elitism_rate = 0.1
mutation_rates = [0.01, 0.05, 0.10]
crossover_rates = [0.7, 0.8, 0.9]

# Função objetivo a ser maximizada
def objective_function(x):
    return x * np.sin(10 * np.pi * x) + 1

# Converter um cromossomo binário para um valor real em [-1, 2]
def binary_to_real(binary):
    b10 = int("".join(str(int(bit)) for bit in binary), 2)
    x = min_val + (max_val - min_val) * b10 / (2 ** L - 1)
    return x

# Inicializar a população de forma aleatória
def initialize_population():
    return np.random.randint(2, size=(pop_size, L))

# Avaliar a população com a função de fitness
def evaluate_population(population):
    fitness = np.array([objective_function(binary_to_real(individual)) for individual in population])
    return fitness

# Seleção por Roleta
def roulette_selection(population, fitness):
    fitness = fitness - fitness.min() + 0.1
    probabilities = fitness / fitness.sum()
    indices = np.random.choice(np.arange(pop_size), size=pop_size, replace=True, p=probabilities)
    return population[indices]

# Aplicar elitismo para manter os melhores indivíduos
def apply_elitism(population, fitness):
    num_elites = int(elitism_rate * pop_size)
    elite_indices = np.argsort(fitness)[-num_elites:]
    return population[elite_indices]

# Cruzamento de um ponto
def one_point_crossover(parent1, parent2):
    point = np.random.randint(1, L)
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2

# Operador de mutação simples
def mutate(individual, mutation_rate):
    for i in range(L):
        if np.random.rand() < mutation_rate:
            individual[i] = 1 - individual[i]  # Flip bit
    return individual

# Executar uma simulação com uma taxa de cruzamento e mutação específicas
def run_simulation(crossover_rate, mutation_rate):
    population = initialize_population()
    best_fitness_per_generation = []

    for gen in range(num_generations):
        fitness = evaluate_population(population)
        best_fitness_per_generation.append(np.max(fitness))

        # Seleção e elitismo
        selected_population = roulette_selection(population, fitness)
        elites = apply_elitism(population, fitness)

        # Gerar nova população com cruzamento e mutação
        new_population = []
        while len(new_population) < pop_size - len(elites):
            parent1, parent2 = selected_population[np.random.randint(pop_size)], selected_population[np.random.randint(pop_size)]
            if np.random.rand() < crossover_rate:
                child1, child2 = one_point_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            new_population.extend([mutate(child1, mutation_rate), mutate(child2, mutation_rate)])

        # Adicionar elites e ajustar tamanho da população
        new_population = np.array(new_population[:pop_size - len(elites)])
        population = np.vstack((new_population, elites))

    return best_fitness_per_generation

# Realizar experimentos com diferentes taxas de cruzamento
plt.figure(figsize=(12, 6))
for crossover_rate in crossover_rates:
    best_fitness_crossover = run_simulation(crossover_rate, mutation_rates[0])
    plt.plot(best_fitness_crossover, label=f'Cruzamento {crossover_rate*100}%')

plt.title("Desempenho do Melhor Indivíduo para Diferentes Taxas de Cruzamento")
plt.xlabel("Geração")
plt.ylabel("Fitness do Melhor Indivíduo")
plt.legend()
plt.show()

# Realizar experimentos com diferentes taxas de mutação usando a melhor taxa de cruzamento
plt.figure(figsize=(12, 6))
for mutation_rate in mutation_rates:
    best_fitness_mutation = run_simulation(crossover_rates[2], mutation_rate)
    plt.plot(best_fitness_mutation, label=f'Mutação {mutation_rate*100}%')

plt.title("Desempenho do Melhor Indivíduo para Diferentes Taxas de Mutação")
plt.xlabel("Geração")
plt.ylabel("Fitness do Melhor Indivíduo")
plt.legend()
plt.show()
