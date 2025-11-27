import random
import pandas as pd
from typing import Dict, Tuple, Any
from feature_engineer import FeatureEngineer
from strategy_engine import StrategyEngine

class GeneticOptimizer:
    def __init__(self, data: pd.DataFrame, population_size=50, mutation_rate=0.2, crossover_rate=0.85):
        self.data = data
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        self.parameter_bounds = {
            'sma_n': [5, 50],
            'span_n': [5, 50],
        }
        self.parameter_names = list(self.parameter_bounds.keys())
        self.feature_engine = FeatureEngineer()
        self.strategy_engine = StrategyEngine()

    def _create_individual(self) -> Dict[str, int]:
        return {
            k: random.randint(v[0], v[1]) for k, v in self.parameter_bounds.items()
        }

    def _fitness_function(self, individual: Dict[str, int]) -> float:
        try:
            df_temp = self.feature_engine.add_features(self.data, individual['sma_n'], individual['span_n'])
            df_temp = self.strategy_engine.create_signals(df_temp)
            res = self.strategy_engine.execute_backtest(df_temp)
            
            final_equity = res['equity_curve'].iloc[-1]
            
            drawdowns = res["drawdown"]
            non_zero_drawdowns = drawdowns[drawdowns > 0]
            mean_drawdown = non_zero_drawdowns.mean() if len(non_zero_drawdowns) > 0 else 0
            
            return final_equity - mean_drawdown
        except:
            return -1000.0

    def _crossover(self, p1: Dict, p2: Dict) -> Tuple[Dict, Dict]:
        if random.random() > self.crossover_rate:
            return p1.copy(), p2.copy()
        
        off1, off2 = p1.copy(), p2.copy()
        off1['span_n'] = p2['span_n']
        off2['span_n'] = p1['span_n']
        return off1, off2

    def _mutate(self, ind: Dict) -> Dict:
        mutated = ind.copy()
        for param in self.parameter_names:
            if random.random() < self.mutation_rate:
                bounds = self.parameter_bounds[param]
                change = random.randint(-3, 3)
                new_val = mutated[param] + change
                mutated[param] = max(bounds[0], min(bounds[1], new_val))
        return mutated

    def run(self, generations=10) -> Dict[str, Any]:
        print(f"Starting genetic optimization ({generations} generations)...")
        population = [self._create_individual() for _ in range(self.population_size)]
        
        best_global_ind = None
        best_global_fit = float('-inf')
        
        for gen in range(generations):
            fitnesses = [self._fitness_function(ind) for ind in population]
            
            max_fit = max(fitnesses)
            best_idx = fitnesses.index(max_fit)
            
            if max_fit > best_global_fit:
                best_global_fit = max_fit
                best_global_ind = population[best_idx].copy()
            
            print(f"Generation {gen+1}/{generations} | Best Fitness: {max_fit:.2f} | Best Parameters: {population[best_idx]}")
            
            sorted_pop = [x for _, x in sorted(zip(fitnesses, population), key=lambda pair: pair[0], reverse=True)]
            
            new_pop = []
            new_pop.append(sorted_pop[0]) # Elitism
            new_pop.append(sorted_pop[1])
            
            while len(new_pop) < self.population_size:
                p1 = random.choice(sorted_pop[:int(self.population_size/2)])
                p2 = random.choice(sorted_pop[:int(self.population_size/2)])
                
                o1, o2 = self._crossover(p1, p2)
                new_pop.append(self._mutate(o1))
                if len(new_pop) < self.population_size:
                    new_pop.append(self._mutate(o2))
            
            population = new_pop

        print(f"\nOptimization completed. Best parameters: {best_global_ind}")
        return best_global_ind
