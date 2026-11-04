from pathlib import Path

import os
import time
import joblib
import numpy as np

from scipy.sparse import load_npz, save_npz
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    start_time = time.time()

    # ==========================================
    # 1. CREATE OUTPUT FOLDER
    # ==========================================

    os.makedirs((PROJECT_ROOT / 'GA_Artifacts'), exist_ok=True)

    # ==========================================
    # 2. LOAD TF-IDF FEATURES AND LABELS
    # ==========================================

    print("\nLoading TF-IDF feature matrices...")

    X_train = load_npz((PROJECT_ROOT / 'Feature_Artifacts/X_train.npz'))
    X_val = load_npz((PROJECT_ROOT / 'Feature_Artifacts/X_val.npz'))
    X_test = load_npz((PROJECT_ROOT / 'Feature_Artifacts/X_test.npz'))

    y_train = np.load((PROJECT_ROOT / 'Feature_Artifacts/y_train.npy'), allow_pickle=True)
    y_val = np.load((PROJECT_ROOT / 'Feature_Artifacts/y_val.npy'), allow_pickle=True)
    y_test = np.load((PROJECT_ROOT / 'Feature_Artifacts/y_test.npy'), allow_pickle=True)

    vectorizer = joblib.load(
        (PROJECT_ROOT / 'Feature_Artifacts/tfidf_vectorizer.pkl')
    )

    total_features = X_train.shape[1]

    print("Training matrix:", X_train.shape)
    print("Validation matrix:", X_val.shape)
    print("Testing matrix:", X_test.shape)
    print("Total available features:", total_features)

    # ==========================================
    # 3. CREATE SMALL DATA SAMPLES FOR GA
    # ==========================================

    print("\nPreparing samples for genetic algorithm...")

    train_indices, _ = train_test_split(
        np.arange(len(y_train)),
        train_size=12000,
        random_state=42,
        stratify=y_train
    )

    val_indices, _ = train_test_split(
        np.arange(len(y_val)),
        train_size=3000,
        random_state=42,
        stratify=y_val
    )

    X_ga_train = X_train[train_indices]
    y_ga_train = y_train[train_indices]

    X_ga_val = X_val[val_indices]
    y_ga_val = y_val[val_indices]

    print("GA training sample:", X_ga_train.shape)
    print("GA validation sample:", X_ga_val.shape)

    # ==========================================
    # 4. GENETIC ALGORITHM SETTINGS
    # ==========================================

    POPULATION_SIZE = 8
    GENERATIONS = 4
    MUTATION_RATE = 0.002
    ELITE_COUNT = 2

    rng = np.random.default_rng(42)

    # Cache previously evaluated chromosomes
    fitness_cache = {}

    # ==========================================
    # 5. CREATE INITIAL POPULATION
    # ==========================================

    def create_chromosome():

        chromosome = np.zeros(total_features, dtype=bool)

        selected_count = rng.integers(300, 1001)

        selected_indices = rng.choice(
            total_features,
            size=selected_count,
            replace=False
        )

        chromosome[selected_indices] = True

        return chromosome


    population = [
        create_chromosome()
        for _ in range(POPULATION_SIZE)
    ]

    # ==========================================
    # 6. FITNESS FUNCTION
    # ==========================================

    def calculate_fitness(chromosome):

        key = chromosome.tobytes()

        if key in fitness_cache:
            return fitness_cache[key]

        selected_indices = np.flatnonzero(chromosome)

        if len(selected_indices) == 0:
            return 0.0

        # Select only chosen features
        train_data = X_ga_train[:, selected_indices]
        val_data = X_ga_val[:, selected_indices]

        # Lightweight classifier for GA evaluation
        model = SGDClassifier(
            loss="log_loss",
            alpha=0.0001,
            max_iter=5,
            tol=None,
            random_state=42
        )

        model.fit(train_data, y_ga_train)

        predictions = model.predict(val_data)

        score = f1_score(
            y_ga_val,
            predictions,
            average="macro",
            zero_division=0
        )

        fitness_cache[key] = score

        return score


    # ==========================================
    # 7. SELECTION
    # ==========================================

    def select_parent(population, scores):

        candidates = rng.choice(
            len(population),
            size=3,
            replace=False
        )

        winner = max(
            candidates,
            key=lambda i: scores[i]
        )

        return population[winner]


    # ==========================================
    # 8. CROSSOVER
    # ==========================================

    def crossover(parent1, parent2):

        point = rng.integers(1, total_features)

        child = np.concatenate([
            parent1[:point],
            parent2[point:]
        ])

        return child


    # ==========================================
    # 9. MUTATION
    # ==========================================

    def mutate(chromosome):

        mutation_mask = (
            rng.random(total_features) < MUTATION_RATE
        )

        chromosome = chromosome.copy()

        chromosome[mutation_mask] = (
            ~chromosome[mutation_mask]
        )

        # Ensure at least one feature is selected
        if not chromosome.any():
            chromosome[rng.integers(total_features)] = True

        return chromosome


    # ==========================================
    # 10. RUN GENETIC ALGORITHM
    # ==========================================

    best_chromosome = None
    best_fitness = -1

    print("\nStarting Genetic Algorithm...")

    for generation in range(GENERATIONS):

        print(
            f"\nGeneration {generation + 1}/{GENERATIONS}"
        )

        scores = [
            calculate_fitness(chromosome)
            for chromosome in population
        ]

        generation_best = int(np.argmax(scores))
        generation_score = scores[generation_best]

        print("Best validation F1:", round(generation_score, 4))
        print(
            "Selected features:",
            int(population[generation_best].sum())
        )

        if generation_score > best_fitness:
            best_fitness = generation_score
            best_chromosome = population[generation_best].copy()

        # Preserve the best individuals
        elite_indices = np.argsort(scores)[-ELITE_COUNT:]

        new_population = [
            population[i].copy()
            for i in elite_indices
        ]

        # Create the remaining population
        while len(new_population) < POPULATION_SIZE:

            parent1 = select_parent(population, scores)
            parent2 = select_parent(population, scores)

            child = crossover(parent1, parent2)
            child = mutate(child)

            new_population.append(child)

        population = new_population


    # ==========================================
    # 11. SAVE BEST FEATURE MASK
    # ==========================================

    print("\nGenetic Algorithm completed!")

    selected_indices = np.flatnonzero(best_chromosome)

    np.save(
        (PROJECT_ROOT / 'GA_Artifacts/best_feature_mask.npy'),
        best_chromosome
    )

    np.save(
        (PROJECT_ROOT / 'GA_Artifacts/selected_feature_indices.npy'),
        selected_indices
    )

    # Save selected feature names
    feature_names = vectorizer.get_feature_names_out()

    selected_feature_names = feature_names[selected_indices]

    np.save(
        (PROJECT_ROOT / 'GA_Artifacts/selected_feature_names.npy'),
        selected_feature_names
    )

    # ==========================================
    # 12. CREATE SELECTED FEATURE MATRICES
    # ==========================================

    print("\nCreating selected feature matrices...")

    X_train_selected = X_train[:, selected_indices]
    X_val_selected = X_val[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]

    save_npz(
        (PROJECT_ROOT / 'GA_Artifacts/X_train_selected.npz'),
        X_train_selected
    )

    save_npz(
        (PROJECT_ROOT / 'GA_Artifacts/X_val_selected.npz'),
        X_val_selected
    )

    save_npz(
        (PROJECT_ROOT / 'GA_Artifacts/X_test_selected.npz'),
        X_test_selected
    )

    # ==========================================
    # 13. SAVE FINAL REPORT
    # ==========================================

    elapsed_time = (time.time() - start_time) / 60

    print("\n========== GENETIC ALGORITHM REPORT ==========")

    print("Original features:", total_features)

    print("Selected features:", len(selected_indices))

    print("Best validation F1:", round(best_fitness, 4))

    print("Training matrix:", X_train_selected.shape)
    print("Validation matrix:", X_val_selected.shape)
    print("Testing matrix:", X_test_selected.shape)

    print("Time taken:", round(elapsed_time, 2), "minutes")

    print("\nAll GA artifacts saved successfully!")
    print("GENETIC ALGORITHM COMPLETED!")




if __name__ == "__main__":
    main()
