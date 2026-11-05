import numpy as np


class GreyWolfFeatureSelector:
    def __init__(
        self,
        fitness_function,
        population_size=20,
        max_iterations=50,
        alpha_weight=0.5,
        beta_weight=0.3,
        delta_weight=0.2,
        position_min=0.0,
        position_max=1.0,
        random_state=42,
    ):
        if not callable(fitness_function):
            raise TypeError("fitness_function must be callable.")
        if population_size < 3:
            raise ValueError("population_size must be at least 3.")
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")
        if not np.isclose(alpha_weight + beta_weight + delta_weight, 1.0):
            raise ValueError("GWO fitness weights must sum to 1.")
        if position_min >= position_max:
            raise ValueError("position_min must be smaller than position_max.")

        self.fitness_function = fitness_function
        self.population_size = int(population_size)
        self.max_iterations = int(max_iterations)
        self.alpha_weight = float(alpha_weight)
        self.beta_weight = float(beta_weight)
        self.delta_weight = float(delta_weight)
        self.position_min = float(position_min)
        self.position_max = float(position_max)
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.best_position_ = None
        self.best_mask_ = None
        self.best_fitness_ = np.inf
        self.fitness_history_ = []

    @staticmethod
    def sigmoid(positions):
        positions = np.clip(positions, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-positions))

    def binary_mask(self, positions):
        probabilities = self.sigmoid(positions)
        random_values = self.rng.random(probabilities.shape)
        return (random_values < probabilities).astype(np.int8)

    def evaluate_population(self, population):
        scores = np.empty(population.shape[0], dtype=float)

        for index, position in enumerate(population):
            mask = self.binary_mask(position)
            if not np.any(mask):
                scores[index] = np.inf
                continue

            score = self.fitness_function(mask.copy())
            score = float(score)
            scores[index] = score if np.isfinite(score) else np.inf

        return scores

    def _initialize_leaders(self, population, scores):
        order = np.argsort(scores)
        alpha = population[order[0]].copy()
        beta = population[order[1]].copy()
        delta = population[order[2]].copy()
        return alpha, beta, delta

    def fit(self, feature_count):
        if not isinstance(feature_count, (int, np.integer)) or feature_count < 1:
            raise ValueError("feature_count must be a positive integer.")

        self.rng = np.random.default_rng(self.random_state)
        population = self.rng.uniform(
            self.position_min,
            self.position_max,
            size=(self.population_size, feature_count),
        )

        scores = self.evaluate_population(population)
        alpha, beta, delta = self._initialize_leaders(population, scores)

        for iteration in range(self.max_iterations):
            a = 2.0 * (1.0 - iteration / self.max_iterations)

            for wolf_index in range(self.population_size):
                current_position = population[wolf_index].copy()
                candidate_positions = []

                for leader in (alpha, beta, delta):
                    r1 = self.rng.random(feature_count)
                    r2 = self.rng.random(feature_count)
                    coefficient_a = 2.0 * a * r1 - a
                    coefficient_c = 2.0 * r2
                    distance = np.abs(
                        coefficient_c * leader - current_position
                    )
                    candidate = leader - coefficient_a * distance
                    candidate_positions.append(candidate)

                updated_position = np.mean(candidate_positions, axis=0)
                population[wolf_index] = np.clip(
                    updated_position,
                    self.position_min,
                    self.position_max,
                )

            scores = self.evaluate_population(population)
            alpha, beta, delta = self._initialize_leaders(population, scores)

            iteration_best = float(np.min(scores))
            self.fitness_history_.append(iteration_best)

        final_scores = self.evaluate_population(population)
        best_index = int(np.argmin(final_scores))

        self.best_position_ = population[best_index].copy()
        self.best_mask_ = self.binary_mask(self.best_position_)
        if not np.any(self.best_mask_):
            probabilities = self.sigmoid(self.best_position_)
            self.best_mask_[int(np.argmax(probabilities))] = 1

        self.best_fitness_ = float(self.fitness_function(self.best_mask_.copy()))
        return self

    def get_support(self):
        if self.best_mask_ is None:
            raise RuntimeError("Call fit() before requesting the selected features.")
        return self.best_mask_.astype(bool)

    def transform(self, features):
        if self.best_mask_ is None:
            raise RuntimeError("Call fit() before transforming features.")

        features = np.asarray(features)
        if features.ndim != 2:
            raise ValueError("features must be a two-dimensional array.")
        if features.shape[1] != self.best_mask_.size:
            raise ValueError(
                f"Expected {self.best_mask_.size} features, received {features.shape[1]}."
            )

        return features[:, self.get_support()]

    def fit_transform(self, features):
        features = np.asarray(features)
        if features.ndim != 2:
            raise ValueError("features must be a two-dimensional array.")
        self.fit(features.shape[1])
        return self.transform(features)
