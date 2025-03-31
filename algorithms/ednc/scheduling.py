# algorithms/ednc/scheduling.py
"""
Hybrid Evolutionary-Reinforcement Learning (HE-RL) algorithm for 
packet scheduling in the E-EDNC framework.

This module implements the HE-RL approach to optimize packet scheduling
based on entropy, network conditions, and feedback metrics.
"""

import random
import numpy as np
from collections import defaultdict, deque

class HEPolicy:
    """Helper class to represent policies in the HE-RL algorithm."""
    
    def __init__(self, entropy_threshold=0.5, energy_threshold=0.3, feedback_weights=None):
        self.entropy_threshold = entropy_threshold
        self.energy_threshold = energy_threshold
        self.feedback_weights = feedback_weights or [0.5, 0.3, 0.2]
        self.fitness = 0
    
    def mutate(self, mutation_rate=0.1):
        """Apply mutation to policy parameters."""
        if random.random() < mutation_rate:
            self.entropy_threshold += random.gauss(0, 0.1)
            self.entropy_threshold = max(0, min(1, self.entropy_threshold))
            
        if random.random() < mutation_rate:
            self.energy_threshold += random.gauss(0, 0.05)
            self.energy_threshold = max(0, min(0.5, self.energy_threshold))
            
        if random.random() < mutation_rate and self.feedback_weights:
            idx = random.randint(0, len(self.feedback_weights)-1)
            self.feedback_weights[idx] += random.gauss(0, 0.1)
            self.feedback_weights[idx] = max(0, self.feedback_weights[idx])
            # Normalize weights
            total = sum(self.feedback_weights)
            if total > 0:
                self.feedback_weights = [w/total for w in self.feedback_weights]
    
    def crossover(self, other):
        """Create a new policy by crossing over with another policy."""
        child = HEPolicy()
        # Uniform crossover
        child.entropy_threshold = self.entropy_threshold if random.random() < 0.5 else other.entropy_threshold
        child.energy_threshold = self.energy_threshold if random.random() < 0.5 else other.energy_threshold
        
        # Blend crossover for weights
        alpha = random.random()
        child.feedback_weights = [alpha * self.feedback_weights[i] + (1-alpha) * other.feedback_weights[i] 
                                 for i in range(len(self.feedback_weights))]
        
        return child

def initialize_policies(algorithm, population_size=20):
    """Initialize population of policies for evolutionary algorithm."""
    algorithm.policies = []
    for _ in range(population_size):
        # Create policies with random parameters
        policy = HEPolicy(
            entropy_threshold=random.uniform(0.2, 0.8),
            energy_threshold=random.uniform(0.1, 0.4),
            feedback_weights=[random.random() for _ in range(3)]
        )
        # Normalize feedback weights
        total = sum(policy.feedback_weights)
        if total > 0:
            policy.feedback_weights = [w/total for w in policy.feedback_weights]
        
        algorithm.policies.append(policy)
    
    # Track generations and episodes
    algorithm.current_generation = 0
    algorithm.current_episode = 0
    algorithm.max_generations = 100
    algorithm.episodes_per_generation = 10
    
    # RL parameters
    algorithm.learning_rate = 0.1
    algorithm.discount_factor = 0.9
    algorithm.policy_values = defaultdict(float)  # State-value function
    
    # Tracking for evolutionary algorithm
    algorithm.last_feedback = None
    algorithm.generation_fitness = defaultdict(list)
    
    # Current policy index
    algorithm.current_policy_idx = 0

def he_rl_scheduling(algorithm, packets, constraints):
    """
    Implement the Hybrid Evolutionary-Reinforcement Learning algorithm
    for packet scheduling as described in the paper.
    """
    # Lazy initialization of policy population
    if not hasattr(algorithm, 'policies') or not algorithm.policies:
        initialize_policies(algorithm)
    
    # Check if RL update is needed
    maybe_update_policies_rl(algorithm)
    
    # Select current best policy
    current_policy = None
    if hasattr(algorithm, 'policies') and algorithm.policies:
        # Use the policy with best fitness, or rotate through policies during exploration
        if algorithm.current_episode < algorithm.episodes_per_generation // 2:
            # Exploration phase: rotate through policies
            algorithm.current_policy_idx = (algorithm.current_policy_idx + 1) % len(algorithm.policies)
            current_policy = algorithm.policies[algorithm.current_policy_idx]
        else:
            # Exploitation phase: use best policy
            current_policy = max(algorithm.policies, key=lambda p: p.fitness) 
    
    if not current_policy:
        # Fallback if no policy available
        return default_scheduling(packets, constraints)
    
    # Apply policy to schedule packets
    return apply_policy_for_scheduling(algorithm, current_policy, packets, constraints)

def maybe_update_policies_rl(algorithm):
    """Update policies using RL if enough episodes have passed."""
    if not hasattr(algorithm, 'current_episode'):
        initialize_policies(algorithm)
        return
        
    # Run evolutionary update after enough episodes
    if algorithm.current_episode >= algorithm.episodes_per_generation:
        update_policies_evolutionary(algorithm)
        algorithm.current_episode = 0
        algorithm.current_generation += 1

def update_policies_evolutionary(algorithm):
    """Update policies using evolutionary algorithm."""
    if not algorithm.last_feedback:
        return
        
    # Calculate fitness for each policy based on feedback
    for i, policy in enumerate(algorithm.policies):
        if i in algorithm.generation_fitness:
            policy.fitness = np.mean(algorithm.generation_fitness[i])
    
    # Selection (tournament selection)
    new_population = []
    for _ in range(len(algorithm.policies)):
        # Select 3 random individuals for tournament
        candidates = random.sample(algorithm.policies, min(3, len(algorithm.policies)))
        # Select the best one
        winner = max(candidates, key=lambda p: p.fitness)
        new_population.append(winner)
    
    # Crossover and mutation
    offspring = []
    for _ in range(len(algorithm.policies) // 2):
        parent1, parent2 = random.sample(new_population, 2)
        child1 = parent1.crossover(parent2)
        child2 = parent2.crossover(parent1)
        
        # Mutation
        child1.mutate(mutation_rate=0.2)
        child2.mutate(mutation_rate=0.2)
        
        offspring.extend([child1, child2])
    
    # Replace worst individuals with offspring
    algorithm.policies.sort(key=lambda p: p.fitness)
    replace_count = min(len(offspring), len(algorithm.policies) // 2)
    algorithm.policies[:replace_count] = offspring[:replace_count]
    
    # Reset fitness tracking for next generation
    algorithm.generation_fitness = defaultdict(list)

def apply_policy_for_scheduling(algorithm, policy, packets, constraints):
    """Apply the policy to schedule packets."""
    # Calculate priority for each packet
    priorities = []
    
    for packet in packets:
        # Estimate energy cost based on packet size
        packet_size = packet["packet_size"]
        estimated_energy = packet_size / 1024  # Normalized energy estimate
        
        # Priority function from paper: Ψ(Pi) = γ1H(Pi) - γ2E(Pi)
        entropy = packet["entropy"]
        adjusted_entropy = entropy
        
        # Apply entropy threshold from policy
        entropy_priority = 0
        if adjusted_entropy >= policy.entropy_threshold:
            entropy_priority = adjusted_entropy
        
        # Apply energy threshold from policy
        energy_priority = 0
        if estimated_energy <= policy.energy_threshold:
            energy_priority = -estimated_energy
        
        # Network condition adjustment
        condition_factor = 1.0
        if packet["network_condition"] == "congested":
            condition_factor = 0.7
        elif packet["network_condition"] == "interference":
            condition_factor = 0.8
        
        # Combined priority with policy weights
        priority = (policy.feedback_weights[0] * entropy_priority + 
                   policy.feedback_weights[1] * energy_priority + 
                   policy.feedback_weights[2] * condition_factor)
        
        priorities.append((packet, priority))
    
    # Sort by priority (descending)
    sorted_packets = [p[0] for p in sorted(priorities, key=lambda x: x[1], reverse=True)]
    
    # Apply constraints
    scheduled_packets = []
    total_size = 0
    total_energy = 0
    
    for packet in sorted_packets:
        packet_size = packet["packet_size"]
        estimated_energy = packet_size / 1024
        
        if (total_size + packet_size <= constraints.get("bandwidth", float('inf')) and
            total_energy + estimated_energy <= constraints.get("energy", float('inf'))):
            scheduled_packets.append(packet)
            total_size += packet_size
            total_energy += estimated_energy
    
    # Update RL statistics
    algorithm.current_episode += 1
    
    # Return the scheduled packets
    return scheduled_packets

def reinforcement_learning_update(algorithm, policy_idx, feedback):
    """Update policy values using reinforcement learning."""
    if not algorithm.last_feedback:
        algorithm.last_feedback = feedback
        return
    
    # Calculate reward based on feedback
    reward = (feedback["avg_reliability"] * 0.4 - 
             feedback["avg_latency"]/100 * 0.3 - 
             feedback["avg_energy"]/0.1 * 0.3)
    
    # State representation (simplified)
    current_state = hash((round(feedback["avg_reliability"], 2),
                        round(feedback["avg_latency"], 0),
                        round(feedback["avg_energy"], 2),
                        round(feedback["congestion_level"], 2)))
    
    # Policy value update (TD learning)
    policy = algorithm.policies[policy_idx]
    policy_state = hash((round(policy.entropy_threshold, 2),
                        round(policy.energy_threshold, 2),
                        tuple(round(w, 2) for w in policy.feedback_weights)))
    
    state_key = (current_state, policy_state)
    
    # Q-learning update
    if hasattr(algorithm, 'last_state_key'):
        old_value = algorithm.policy_values[algorithm.last_state_key]
        # TD update
        algorithm.policy_values[algorithm.last_state_key] = old_value + algorithm.learning_rate * (
            reward + algorithm.discount_factor * algorithm.policy_values[state_key] - old_value
        )
    
    algorithm.last_state_key = state_key
    
    # Store fitness for evolutionary algorithm
    algorithm.generation_fitness[policy_idx].append(reward)
    
    # Update last feedback
    algorithm.last_feedback = feedback

def default_scheduling(packets, constraints):
    """Default scheduling strategy when no policy is available."""
    # Sort by entropy (highest first)
    sorted_packets = sorted(packets, key=lambda p: p["entropy"], reverse=True)
    
    # Apply constraints
    scheduled_packets = []
    total_size = 0
    total_energy = 0
    
    for packet in sorted_packets:
        # Estimate resource usage
        packet_size = packet["packet_size"]
        estimated_energy = packet_size / 1024
        
        # Check if packet fits within constraints
        if (total_size + packet_size <= constraints.get("bandwidth", float('inf')) and
            total_energy + estimated_energy <= constraints.get("energy", float('inf'))):
            scheduled_packets.append(packet)
            total_size += packet_size
            total_energy += estimated_energy
    
    return scheduled_packets
