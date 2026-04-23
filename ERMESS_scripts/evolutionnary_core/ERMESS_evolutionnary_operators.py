# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:17:27 2025

@author: JoPHOBEA


Genetic operators module.

This module contains stochastic mutation operators used in the
evolutionary optimization of hybrid energy systems. All operators
are compatible with numba.jit(nopython=True) and modify individuals
in-place.

RESEARCH OPERATORS : active when mode=RESEARCH. They act directly on timeseries without any constraint.
PRO OPERATORS : active when mode=PRO. They act on dispatching and control strategies

"""

import numpy as np
from numba import jit

from ERMESS_scripts.data.indices import *


@jit(nopython=True)
def switch_contract_operator(c,n_contracts):
    """
    Randomly assign a new electricity contract to the individual (RESEARCH operator).
    
    The operator selects uniformly a contract index in [0, n_contracts-1] and updates `c.contract`.
    
    Args:
        c (object): Candidate solution.
        n_contracts (int): Total number of available contracts.
    
    Returns:
        object: Mutated individual with updated contract.
    """
    c.contract=np.random.randint(0,n_contracts,1)[0]
    return(c)

@jit(nopython=True)
def reduce_power_trading_operator(c,selected_storage,hyperparameters_operators):
    """
    Adjust storage power at the time step of maximum grid trade (RESEARCH operator).
    
    The operator increases storage usage for a random storage unit
    at the time index where `c.trades` is maximal, reducing the maximum power taken from grid or Genset.
    
    Args:
        c (object): Candidate solution.
        choices (numpy.ndarray): Operator selection indices.
    
    Returns:
        object: Mutated individual.
    """
    
    arg = np.argmax(c.trades)      
    c.storage_TS[selected_storage][arg] = c.storage_TS[selected_storage][arg]+c.trades[arg]*np.random.uniform(0,hyperparameters_operators[OPER_MAX_AVERAGE,RESEARCH_CONTRACT])        
      
    return(c)

@jit(nopython=True)
def Mutate_production_capacity_operator(c,selected_storage,Bounds_prod,groups,groups_size,unit_productions,n_bits,hyperparameters_operators):    
    """
    Modify production capacity within a randomly selected technology group (RESEARCH operator).
    
    The operator modifies the production capacity within a group and partially
    redistributes the induced production change to storage actions.
    
    Args:
        c (object): Candidate solution.
        choices (numpy.ndarray): Storage selection indices.
        Bounds_prod (numpy.ndarray): Upper production bounds per technology.
        groups (list): List of grouped production indexes.
        prods_U (numpy.ndarray): Unit production time-series.
        n_bits (int): Number of simulation time steps.
        hyperparameters_operators_num (numpy.ndarray): Operator hyperparameter matrix.
    
    Returns:
        object: Mutated individual.
    """
    KILOS_CONVERSION_FACTOR = 1000
    
    group = np.random.randint(len(groups))
    valid_idx = groups[group, :groups_size[group]]
    mask_positive = c.production_set[valid_idx]>0
    candidates = valid_idx[mask_positive]
    if candidates.size > 0:
        selected_producer = candidates[np.random.randint(candidates.size)]
    else:
        selected_producer = valid_idx[np.random.randint(valid_idx.size)]
        
    min_decrease = min(-2,-int(Bounds_prod[selected_producer] / hyperparameters_operators[OPER_INV_LENGTH, RESEARCH_PRODUCTION]))
    max_increase = max(3,int(Bounds_prod[selected_producer] / hyperparameters_operators[OPER_INV_LENGTH, RESEARCH_PRODUCTION]))
    
    modifier=max(0,min(Bounds_prod[selected_producer],c.production_set[selected_producer]+np.random.randint(min_decrease,max_increase)))
    
    # Stochastic modulation factors
    uniform_noise = np.random.uniform(0.0, 1.0) #scaling factor
    gaussian_noise = np.random.normal(1, hyperparameters_operators[OPER_MAX_AVERAGE, RESEARCH_PRODUCTION], n_bits) #per time step
    
    added_prod = (modifier-c.production_set[selected_producer])*unit_productions[selected_producer]*uniform_noise*gaussian_noise/KILOS_CONVERSION_FACTOR    
    c.production_set[selected_producer]=modifier         
    # --- Update storage based on production change ---
    active_bits = np.where(c.trades>0)[0]
    c.storage_TS[selected_storage,active_bits] = c.storage_TS[selected_storage,active_bits]-added_prod[active_bits]
    return(c)

@jit(nopython=True)
def Transfer_production_capacity_operator(c,Bounds_prod):
    """
    Redistribute installed production capacity between active technologies (RESEARCH operator).
    
    This operator performs a discrete capacity transfer between two
    currently active production technologies.
    
    Two producers with strictly positive installed capacity are randomly
    selected. One unit of capacity is subtracted from the first and
    added to the second, while enforcing lower and upper bounds::
    
        P_i' = clip(P_i - 1, 0, Bound_i)
        P_j' = clip(P_j + 1, 0, Bound_j)
    
    This mutation preserves the discrete structural nature of the
    production mix while enabling local reallocation of capacity.
    
    The operator does not create new technologies; it only redistributes
    existing active capacity.
    
    Args:
        c (object): Candidate solution.
        Bounds_prod (numpy.ndarray): Upper capacity bounds for each production technology.
        groups (list): Technology grouping structure (unused here but kept for interface consistency).
    
    Returns:
        object: Mutated individual.
    """
    TRANSFER_STEP = 1
    active_producers  = np.where(c.production_set>0)[0]
    if len(active_producers)>1: # Minimum required to perform a transfer: at least two active units
        producer_from, producer_to = np.random.choice(active_producers,size=2,replace=False)
        c.production_set[producer_from]=max(min(c.production_set[producer_from]-TRANSFER_STEP,Bounds_prod[producer_from]),0)
        c.production_set[producer_to]=max(min(c.production_set[producer_to]+TRANSFER_STEP,Bounds_prod[producer_to]),0)  
    return(c)

@jit(nopython=True)
def Switch_intragroup_productor_operator(c,Bounds_prod,groups,groups_size):
    """
    Perform intra-group technology substitution (RESEARCH operator).
    
    This operator randomly selects a predefined production group
    (composed of mutually exclusive or competing production options)
    and replaces the currently active technology within that group.
    
    Procedure:
        1. A random group is selected.
        2. If at least one technology in the group is active,
           all group capacities are reset to zero.
        3. A new technology within the group is randomly chosen.
        4. Its installed capacity is initialized randomly within bounds.
    
    Mathematically, for group G::
    
        P_k' = 0  for all k in G
        P_j' ~ U(0, Bound_j)  for one j in G
    
    This mutation enables structural topology exploration of the
    production mix, allowing discrete technology replacement rather
    than incremental reallocation.
    
    Args:
        c (object): Candidate solution.
        Bounds_prod (numpy.ndarray): Upper capacity bounds for each production technology.
        groups (list of numpy.ndarray): List of arrays defining mutually exclusive technology groups.
    
    Returns:
        object: Mutated individual.
    """

    candidate_groups = np.where(groups_size>1)[0]
    selected_group = np.random.choice(candidate_groups)
    candidates = groups[selected_group, :groups_size[selected_group]]
    
    if (max(c.production_set[candidates])>0):
        inactive_mask = c.production_set[candidates]==0
        productor_new = np.random.choice(candidates[inactive_mask])
        c.production_set[candidates]=np.repeat(0,len(candidates))
        c.production_set[productor_new]=np.random.randint(0,Bounds_prod[productor_new]) 
    return(c)
                
@jit(nopython=True)
def Mutate_storage_points_operator(c,selected_storage,n_bits,hyperparameters_operators_num):
    """
    Apply stochastic local perturbations to a storage time-series (RESEARCH operator).
    
    The operator modifies randomly selected time steps or time
    subsequences using additive and/or multiplicative Gaussian noise.
    
    Args:
        c (object): Candidate solution.
        choices (numpy.ndarray): Storage selection indices.
        hyperparameters_operators_num (numpy.ndarray): Operator hyperparameter matrix.
    
    Returns:
        object: Mutated individual.
    """
    MIN_MUTATIONS = 1
    EXTRA_MUTATIONS_OFFSET = 4
    RATIO_BETWEEN_NOISES = 50
    
    # --- Determine number of mutation points ---
    max_mutations = int(n_bits / hyperparameters_operators_num[OPER_INV_LENGTH,RESEARCH_STORAGE_PATTERNS]) + EXTRA_MUTATIONS_OFFSET
    num_mutations = np.random.randint(MIN_MUTATIONS, max_mutations)
    
    # Randomly select time indices to mutate (without replacement)
    mutations=np.random.choice(np.arange(len(c.storage_TS[selected_storage])), size=num_mutations, replace=False)
    
    # --- Generate stochastic perturbations ---
    additive_noise = np.random.normal(0,RATIO_BETWEEN_NOISES*hyperparameters_operators_num[OPER_MAX_DEVIATION,RESEARCH_STORAGE_PATTERNS],len(mutations))
    multiplicative_noise = np.random.normal(1,hyperparameters_operators_num[OPER_MAX_DEVIATION,RESEARCH_STORAGE_PATTERNS],len(mutations))
    
    c.storage_TS[selected_storage][mutations]=(c.storage_TS[selected_storage][mutations]+additive_noise)*multiplicative_noise
    return(c)

@jit(nopython=True)
def Mutate_storage_windows_noise_operator(c,selected_storage,n_bits,hyperparameters_operators_num):
    """
    Apply stochastic local perturbations to a storage time-series (RESEARCH operator).
    
    The operator modifies randomly selected time
    sequences (windows) using additive and/or multiplicative Gaussian noise.
    
    Args:
        c (object): Candidate solution.
        choices (numpy.ndarray): Storage selection indices.
        hyperparameters_operators_num (numpy.ndarray): Operator hyperparameter matrix.
    
    Returns:
        object: Mutated individual.
    """
    MIN_REFERENCE_LENGTH = 3
    MAX_SUBSET_LENGTH_CAP = 30
    MAX_SEGMENTS_CAP = 30
    
    subset_length_upper_bound = min(MAX_SUBSET_LENGTH_CAP,max(MIN_REFERENCE_LENGTH,int(n_bits/hyperparameters_operators_num[OPER_INV_LENGTH, RESEARCH_STORAGE_PATTERNS])))

    len_subset = np.random.randint(1,subset_length_upper_bound)   


    max_segments = min(MAX_SEGMENTS_CAP,int(n_bits/(2*hyperparameters_operators_num[OPER_INV_LENGTH, RESEARCH_STORAGE_PATTERNS]))+2)
    num_segments = np.random.randint(1, max_segments)   
    valid_start_range = np.arange(0, n_bits - len_subset)
     
    segment_starts = np.random.choice(valid_start_range,num_segments,replace=False) 
   
    segment_indices = np.empty((len(segment_starts),len_subset),dtype=np.int64)
    for i in range(len(segment_starts)):
        segment_indices[i]=np.arange(segment_starts[i],segment_starts[i]+len_subset)
    matrice_mult = np.random.normal(1,hyperparameters_operators_num[OPER_MAX_DEVIATION,RESEARCH_STORAGE_PATTERNS],segment_indices.size).reshape(segment_indices.shape)
    matrice_add = np.random.normal(0,hyperparameters_operators_num[OPER_MAX_DEVIATION,RESEARCH_STORAGE_PATTERNS],segment_indices.size).reshape(segment_indices.shape)
    for i in range(len(segment_starts)):
        c.storage_TS[selected_storage,segment_indices[i]]=c.storage_TS[selected_storage,segment_indices[i]]*matrice_mult[i]+matrice_add[i]
    return(c)

@jit(nopython=True)
def Mutate_storage_dailypattern_operator(c,selected_storage,time_resolution,n_bits,hyperparameters_operators_num):
    """
    Apply a daily pattern mutation to a storage time series (RESEARCH operator).
    
    This operator modifies the storage dispatch profile by perturbing
    selected hours of the day consistently across all days of the horizon.
    
    Procedure:
        1. A storage unit is selected.
        2. A random subset of intra-day hours is sampled.
        3. For each selected hour, all corresponding time steps
           across every day are identified.
        4. A multiplicative and additive Gaussian perturbation
           is applied consistently across days.
    
    Formally, for selected intra-day hours h and each day d:
    
        S'(d, h) = S(d, h) * α_h + β_h
    
    where:
        - α_h ~ N(1, σ)
        - β_h ~ N(0, kσ)
    
    This operator preserves inter-day structural coherence while
    exploring alternative daily operational patterns.
    
    Args:
        c (object): Candidate solution.
        choices (numpy.ndarray): Vector indicating the storage index to mutate.
        time_resolution (int): Number of time steps per hour.
        n_bits (int): Total number of time steps.
        hyperparameters_operators_num (numpy.ndarray): Hyperparameter matrix controlling mutation amplitude.
    
    Returns:
        object: Mutated individual.
    """
    HOURS_PER_DAY = 24
    RATIO_BETWEEN_NOISES = 5

    steps_per_day = int(time_resolution * HOURS_PER_DAY)
    n_days = int(n_bits / (steps_per_day))
    days_indices = np.arange(n_days)  

    num_mutated_hours = np.random.randint(1, steps_per_day)
                    
    mutated_hours = np.random.choice(np.arange(steps_per_day),num_mutated_hours,replace=False)

    time_indices = np.empty((n_days,num_mutated_hours),dtype=np.int64)
    for day in range(n_days):
        time_indices[day]=mutated_hours+day*int(time_resolution*HOURS_PER_DAY)
    matrice_mult = np.random.normal(1,hyperparameters_operators_num[OPER_MAX_DEVIATION,RESEARCH_STORAGE_PATTERNS],num_mutated_hours)
    matrice_add = np.random.normal(0,RATIO_BETWEEN_NOISES*hyperparameters_operators_num[OPER_MAX_DEVIATION,RESEARCH_STORAGE_PATTERNS],num_mutated_hours)
    for i in range(n_days):
        c.storage_TS[selected_storage,time_indices[day]]=c.storage_TS[selected_storage,time_indices[day]]*matrice_mult+matrice_add
    return(c)

@jit(nopython=True)
def Mutate_storage_global_operator(c,selected_storage,n_store,hyperparameters_operators_num):
    """
    Apply a global scaling mutation to a randomly selected storage profile (RESEARCH operator).
    
    This operator multiplies the entire time series of a storage
    unit by a uniformly sampled scaling factor, modifying power and capacity.
    
    For each storage unit i:
    
        S_i'(t) = S_i(t) * γ_i
    
    where:
    
        γ_i ~ U(1 - ε, 1 + ε)
    
    This operator performs a coarse-grained global adjustment of
    storage utilization intensity while preserving temporal structure.
    
    It enables exploration of overall storage engagement levels
    without modifying dispatch shape.
    
    Args:
        c (object): Candidate solution.
        choices (numpy.ndarray): Mutation selector (unused but kept for interface consistency).
        n_store (int): Number of storage technologies.
        hyperparameters_operators_num (numpy.ndarray): Hyperparameter matrix controlling scaling amplitude.
    
    Returns:
        object: Mutated individual.
    """
    for storage in range(n_store):
        scaling_factor = np.random.uniform(1.0 - 1/hyperparameters_operators_num[OPER_INV_MAGNITUDE,RESEARCH_STORAGE_GLOBAL],1.0 + 1/hyperparameters_operators_num[OPER_INV_MAGNITUDE,RESEARCH_STORAGE_GLOBAL])
        c.storage_TS[storage,] = c.storage_TS[storage,]*scaling_factor
    return(c)

@jit(nopython=True)
def Mutate_storage_windows_scaling_operator(c,selected_storage,n_bits,hyperparameters_operators_num):
    """
    Apply localized stochastic perturbations to a storage time series (RESEARCH operator).
    
    This operator modifies multiple randomly selected temporal
    sub-windows of a given storage dispatch profile.
    
    Procedure:
        1. A storage unit is selected.
        2. Several random time windows are sampled.
        3. For each window, a multiplicative and additive
           random perturbation is applied.
    
    For each selected window W:
    
        S'(t) = S(t) * α + β   for t ∈ W
    
    where:
    
        α ~ U(1 - ε, 1 + ε)
        β ~ U(-δ, δ)
    
    This operator enables fine-grained local exploration of
    dispatch flexibility and short-term behavioral diversity.
    
    Args:
        c (object): Candidate solution.
        choices (numpy.ndarray): Vector indicating the storage index to mutate.
        hyperparameters_operators_num (numpy.ndarray): Hyperparameter matrix controlling window size and mutation amplitude.
    
    Returns:
        object: Mutated individual.
    """
    MAX_SEGMENTS = 40
    RATIO_BETWEEN_NOISES = 100
    
    num_segments = np.random.randint(1,MAX_SEGMENTS)
    max_subset_length = max(3,int(n_bits / hyperparameters_operators_num[OPER_INV_LENGTH, RESEARCH_STORAGE_GLOBAL]))
    segment_length = np.random.randint(1,max_subset_length)     

    valid_start_positions = np.arange(0, n_bits - segment_length)        
    segment_starts = np.random.choice(valid_start_positions,num_segments,replace=True)
    segment_indices = np.empty((len(segment_starts), segment_length), dtype=np.int64)
    for i in range(num_segments):
        segment_indices[i]=np.arange(segment_starts[i],segment_starts[i]+segment_length)
        
    scaling_intensity = hyperparameters_operators_num[OPER_INV_MAGNITUDE,RESEARCH_STORAGE_GLOBAL]
    multiplicative_coeffs = np.random.uniform(1.0 - 1.0 / scaling_intensity,1.0 + 1.0 / scaling_intensity,size=num_segments)
    additive_coeffs = np.random.uniform(-RATIO_BETWEEN_NOISES / scaling_intensity,RATIO_BETWEEN_NOISES / scaling_intensity,size=num_segments)
    
    for i in range(num_segments):
        c.storage_TS[selected_storage,segment_indices[i]]=c.storage_TS[selected_storage,segment_indices[i]]*multiplicative_coeffs[i]+additive_coeffs[i]
    return(c)

@jit(nopython=True)
def Reallocate_storage_mix_points_operator(c,n_store,random_factor,n_bits,hyperparameters_operators_num):
    """
    Transfer energy actions between two storage units (RESEARCH operator) on randomly selected points.
    
    This operator aggressively redistributes charging/discharging actions
    between randomly selected storage systems (preserving global energy balance locally).
    
    Args:
        c (object): Candidate solution.
        n_store (int): Number of storage systems.
        storage_characteristics (numpy.ndarray): Storage technical parameters.
        hyperparameters_operators_num (numpy.ndarray): Operator hyperparameter matrix.
    
    Returns:
        object: Mutated individual.
    """
    MIN_SWITCH_SIZE = 2
    MAX_SWITCH_SIZE = max(3,int(n_bits / hyperparameters_operators_num[OPER_INV_LENGTH, RESEARCH_STORAGE_MIX]))

    (store_in,store_out)=np.random.choice(n_store,2,replace=False)
    
    if (sum(c.storage_sum[np.array([store_in,store_out])])>0):       
             switch_indices = np.random.choice(n_bits,np.random.randint(MIN_SWITCH_SIZE, MAX_SWITCH_SIZE),replace=False)
             # --- Compute redistribution weight ---
             weight = (random_factor+0.5) * (c.storage_sum[store_in] / np.sum(c.storage_sum[np.array([store_in, store_out])]))
             weight = min(1,weight)
             c.storage_TS[store_in,switch_indices]=np.sum(c.storage_TS[:,switch_indices],axis=0)*weight
             c.storage_TS[store_out,switch_indices]=np.sum(c.storage_TS[:,switch_indices],axis=0)*(1-weight)
    return(c)

@jit(nopython=True)
def Transfer_storage_flows_operator(c,n_store,storage_characteristics,n_bits,hyperparameters_operators_num):
    """
    Transfer energy actions between two storage units (RESEARCH operator).
    
    Random contiguous time windows are selected and a fraction of the charging
    or discharging power of one storage unit is transferred to another.
    Charging and discharging actions are handled separately and scaled by
    Gaussian random factors, while maintaining local energy balance between
    the two storage systems.
    
    Args:
        c (object): Candidate solution.
        n_store (int): Number of storage systems.
        storage_characteristics (numpy.ndarray): Storage technical parameters.
        hyperparameters_operators_num (numpy.ndarray): Operator hyperparameter matrix.
    
    Returns:
        object: Mutated individual.
    """

    MAX_SEG_LEN = 30
    MAX_STARTS = 30

    (store_in,store_out)=np.random.choice(n_store,2,replace=False)
    len_subset = np.random.randint(1,min(MAX_SEG_LEN,max(3,int(n_bits/hyperparameters_operators_num[OPER_INV_LENGTH,RESEARCH_STORAGE_MIX]))))                  
    
    max_num_starts = min(MAX_STARTS,int(n_bits/(hyperparameters_operators_num[OPER_INV_LENGTH, RESEARCH_STORAGE_MIX] / 2)) + 2)
    num_segments = np.random.randint(1, max_num_starts)
    segment_starts  = np.random.choice(np.arange(0,n_bits-len_subset),num_segments)              
    segment_indices = np.empty((num_segments,len_subset),dtype=np.int64)
    segment_values = np.empty((num_segments,len_subset),dtype=np.float64)
    
    for i in range(num_segments):
        idx = np.arange(segment_starts[i], segment_starts[i] + len_subset)
        segment_indices[i] = idx
        segment_values[i] = c.storage_TS[store_out,idx]
    
    # Phase 1: transfer of negative components (store_out → store_in)
    NEGATIVE_DIRECTION = -1
    negative_mask = np.where(np.sign(segment_values) == NEGATIVE_DIRECTION, segment_values, 0)
    
    negative_transfer = -np.abs(np.random.normal(hyperparameters_operators_num[OPER_MAX_AVERAGE, RESEARCH_STORAGE_MIX],hyperparameters_operators_num[OPER_MAX_DEVIATION, RESEARCH_STORAGE_MIX],segment_indices.size).reshape(num_segments, len_subset)) * negative_mask
    
    for i in range(num_segments):
        idx = segment_indices[i]
        c.storage_TS[store_in,idx] = c.storage_TS[store_in,idx]-negative_transfer[i]
        c.storage_TS[store_out,idx] = c.storage_TS[store_out,idx]+negative_transfer[i]

    # Phase 2: transfer of positive components (store_in → store_out)
    POSITIVE_DIRECTION = 1    
    positive_mask = np.where(np.sign(segment_values)==POSITIVE_DIRECTION,segment_values,0)
    positive_transfer = np.abs(np.random.normal(hyperparameters_operators_num[OPER_MAX_AVERAGE, RESEARCH_STORAGE_MIX],hyperparameters_operators_num[OPER_MAX_DEVIATION, RESEARCH_STORAGE_MIX],size=segment_indices.size).reshape(num_segments, len_subset)) * positive_mask    
    for i in range(num_segments):
        idx = segment_indices[i]
        scaling_factor = (storage_characteristics[STOR_ROUND_TRIP_EFF, store_in] /storage_characteristics[STOR_ROUND_TRIP_EFF, store_out])
        c.storage_TS[store_in,idx] = c.storage_TS[store_in,idx]+positive_transfer[i]*scaling_factor
        c.storage_TS[store_out,idx] = c.storage_TS[store_out,idx]-positive_transfer[i]
    return(c)

@jit(nopython=True)
def Reduce_storage_volume_operator(c,selected_storage,storage_characteristics,n_bits,hyperparameters_operators_num):
    """
    Redistribute storage energy volume between critical SOC extrema (RESEARCH operator).
    
    This operator modifies the storage dispatch trajectory by
    identifying the minimum and maximum cumulative energy states
    (State of Charge extrema) and redistributing energy between
    time segments separated by these extrema.
    
    Procedure:
        1. Compute cumulative effective storage evolution.
        2. Identify global minimum and maximum SOC points.
        3. Split the time horizon into two complementary regions.
        4. Randomly transfer a fraction of energy between these regions.
    
    The mutation magnitude is proportional to the peak-to-peak
    amplitude of cumulative storage:
    
        shift ~ U(0, ptp(SOC)/κ)
    
    Energy is then redistributed with Gaussian-weighted adjustments.
    
    This operator reshapes long-term storage volume utilization
    while preserving overall feasibility.
    
    Args:
        c (object): Candidate solution.
        choices (numpy.ndarray): Vector selecting the storage unit to mutate.
        storage_characteristics (numpy.ndarray): Matrix containing storage parameters (including efficiency).
        hyperparameters_operators_num (numpy.ndarray): Hyperparameter matrix controlling mutation intensity.
    
    Returns:
        object: Mutated individual.
    """
    storage_series = c.storage_TS[selected_storage]
    normalized_series = storage_series/storage_characteristics[STOR_ROUND_TRIP_EFF,selected_storage]
    losses = np.where(normalized_series-storage_series>0,normalized_series-storage_series,0)
    
    sum_diff_storages = np.cumsum(storage_series+losses) 
    min_point, max_point = np.argmin(sum_diff_storages),np.argmax(sum_diff_storages)
    
    if (min_point!=max_point):
        start_idx = min(min_point, max_point)
        end_idx = max(min_point, max_point)
        
        region_outside = np.concatenate((np.arange(start_idx), np.arange(end_idx, n_bits)))
        region_inside = np.arange(start_idx, end_idx)
        
        sub_storage = [region_outside,region_inside]
        
        changes_outside = np.random.choice(region_outside,np.random.randint(max(2, int(len(region_outside) / hyperparameters_operators_num[OPER_INV_LENGTH, RESEARCH_STORAGE_VOLUME]))),replace=False)
        changes_inside = np.random.choice(region_inside,np.random.randint(1,max(2, int(len(region_inside) / hyperparameters_operators_num[OPER_INV_LENGTH, RESEARCH_STORAGE_VOLUME]))),replace=False)
        
        signal_range = np.ptp(sum_diff_storages)

        shift = np.random.uniform(0,signal_range/hyperparameters_operators_num[OPER_INV_MAGNITUDE,RESEARCH_STORAGE_VOLUME])

        noise_out = np.random.normal(hyperparameters_operators_num[OPER_MAX_AVERAGE, RESEARCH_STORAGE_VOLUME],hyperparameters_operators_num[OPER_MAX_DEVIATION, RESEARCH_STORAGE_VOLUME],len(changes_outside)) / len(changes_outside)
        noise_in = np.random.normal(hyperparameters_operators_num[OPER_MAX_AVERAGE, RESEARCH_STORAGE_VOLUME],hyperparameters_operators_num[OPER_MAX_DEVIATION, RESEARCH_STORAGE_VOLUME],len(changes_inside)) / len(changes_inside)

        if min_point ==min(min_point, max_point):
            c.storage_TS[selected_storage][changes_outside] += shift * noise_out
            c.storage_TS[selected_storage][changes_inside] -= shift * noise_in
        else:
            c.storage_TS[selected_storage][changes_outside] -= shift * noise_out
            c.storage_TS[selected_storage][changes_inside] += shift * noise_in
    return(c)

@jit(nopython=True)
def Reduce_storage_power_operator(c,selected_storage,n_bits,hyperparameters_operators_num):
    """
    Modify storage power extrema (RESEARCH operator).
    
    This operator iteratively perturbs the maximum charging and
    discharging power levels of a storage unit.
    
    At each iteration:
        - The current maximum dispatch value is scaled.
        - The current minimum dispatch value is scaled.
    
    Formally:
    
        S_max' = S_max * α
        S_min' = S_min * β
    
    where α, β are sampled from bounded uniform distributions.
    
    This operator directly explores peak power capabilities
    without altering the global temporal structure.
    
    Args:
        c (object): Candidate solution.
        choices (numpy.ndarray): Vector selecting the storage unit to mutate.
        hyperparameters_operators_num (numpy.ndarray): Hyperparameter matrix controlling mutation amplitude and repetition count.
    
    Returns:
        object: Mutated individual.
    """
    coeffs=np.random.uniform(1-1/hyperparameters_operators_num[OPER_INV_MAGNITUDE,RESEARCH_STORAGE_POWER],1,2)
    max_steps = max(1,int(n_bits/hyperparameters_operators_num[OPER_INV_LENGTH,RESEARCH_STORAGE_POWER]))
    for _ in range(np.random.randint(max_steps)):
        argmax = np.argmax(c.storage_TS[selected_storage])
        argmin = np.argmin(c.storage_TS[selected_storage])
        c.storage_TS[selected_storage][argmax] = c.storage_TS[selected_storage][argmax]*coeffs[0]
        c.storage_TS[selected_storage][argmin] = c.storage_TS[selected_storage][argmin]*coeffs[1]
    return(c)

@jit(nopython=True)
def Merge_opposite_storage_flows_operator(c,n_store,hyperparameters_operators_num):
    """
    Merge opposite storage flows between two storage units (RESEARCH operator).
    
    This operator identifies time steps where two storage systems
    operate in opposite directions (one charging, one discharging).
    
    For a subset of such time steps:
        - The total combined action is assigned to one storage.
        - The second storage dispatch is set to zero.
    
    Formally, for selected times t:
    
        S_i'(t) = S_i(t) + S_j(t)
        S_j'(t) = 0
    
    This mutation promotes coordinated storage behavior
    and reduces inefficient simultaneous opposing actions.
    
    Args:
        c (object): Candidate solution.
        n_store (int): Number of storage technologies.
        hyperparameters_operators_num (numpy.ndarray): Hyperparameter matrix controlling sampling intensity.
    
    Returns:
        object: Mutated individual.
    """
    store_a, store_b = np.random.choice (n_store,2,replace=False)
    series_a = c.storage_TS[store_a]
    series_b = c.storage_TS[store_b]
    opposite_sign_indices = np.where(series_a * series_b < 0)[0]
    if (len(opposite_sign_indices)>1):
        changes = np.random.choice(opposite_sign_indices,max(2,int(len(opposite_sign_indices) / hyperparameters_operators_num[OPER_INV_LENGTH,RESEARCH_INTER_STORAGES])),replace=False)
        aggregated_signal = np.sum(c.storage_TS[:, changes], axis=0)
        c.storage_TS[stores[0],changes]=aggregated_signal
        c.storage_TS[stores[1],changes]=0
    return(c)

@jit(nopython=True)
def Force_storage_trade_consistency_operator(c,selected_storage,n_bits,hyperparameters_operators_num):
    """
    Enforce local scheduling consistency with trade imbalance (RESEARCH operator).
    
    This operator adjusts storage dispatch over a random time window
    to better align with grid trade imbalance.
    
    For a selected window:
        1. Residual mismatches between storage action and trade
           direction are computed.
        2. Opposite-direction energy is partially reallocated.
        3. Compensation is distributed across remaining time steps
           within the window.
    
    Formally:
    
        residual = sign(trade) * min(abs(S), abs(trade))
    
    Energy is redistributed to reduce local infeasibility
    while maintaining total balance within the window.
    
    This operator acts as a semi-guided feasibility correction.
    
    Args:
        c (object): Candidate solution.
        choices (numpy.ndarray): Vector selecting the storage unit to mutate.
        hyperparameters_operators_num (numpy.ndarray): Hyperparameter matrix controlling window length.
    
    Returns:
        object: Mutated individual.
    """
    MAX_LEN = 100
    len_subset = np.random.randint(1,min(MAX_LEN,max(3,int(n_bits / hyperparameters_operators_num[OPER_INV_LENGTH, RESEARCH_STORAGE_TRADES_CONSISTENCY]))))

    start_idx = np.random.choice(np.arange(0,n_bits-len_subset))
    window_indices = np.arange(start_idx, start_idx + len_subset + 1)
    local_storage = c.storage_TS[selected_storage][window_indices]
    local_trades = c.trades[window_indices]
    sign_conflict = local_storage * local_trades < 0
    energy_residuals = np.where(sign_conflict,np.sign(local_trades) * np.minimum(np.abs(local_storage), np.abs(local_trades)),0)
    
    if ((energy_residuals!=0).any() and (energy_residuals==0).any()):
        c.storage_TS[selected_storage,window_indices]=np.where(sign_conflict,local_storage+energy_residuals,local_storage - np.sum(energy_residuals) / np.sum(energy_residuals == 0))
    return(c)

#Smooth the storage actions on a specific daytime for consecutive days
@jit(nopython=True)
def Smooth_interdaily_storage_timeseries_operator(c,selected_storage,random_factor,time_resolution,n_bits,hyperparameters_operators_num):
    """
    Enforce long-term temporal consistency in storage dispatch (RESEARCH operator).
    
    This operator introduces temporal smoothing in selected storage
    time-series segments to promote long-term operational consistency.
    
    Procedure:
        1. A storage unit is selected.
        2. Several multi-day time windows are randomly chosen.
        3. For each window, a moving average is computed over an extended neighborhood.
        4. Original values are partially replaced by a convex combination of:
            - the original time-series values
            - the local moving average
    
    Mathematically, for selected indices t:
    
        x'(t) = α x(t) + (1 - α) MA_window(t)
    
    where:
        - α ∈ [0,1] is a random mixing factor,
        - MA_window is a local moving average over a fixed window width.
    
    This operator preserves feasibility while reducing short-term oscillatory behavior
    and encouraging smoother multi-day storage trajectories. It improves
    inter-temporal realism and mitigates erratic dispatch patterns that may arise
    from purely local mutations.
    
    Args:
        c (object): Candidate solution.
        choices (numpy.ndarray): Vector of mutation choices; index 14 selects
            the storage unit to mutate.
        random_factors (numpy.ndarray): Random vector used to determine mixing intensity.
        time_resolution (int): Number of time steps per hour.
        hyperparameters_operators_num (numpy.ndarray): Hyperparameter matrix controlling mutation frequency.
    
    Returns:
        object: Mutated individual with smoothed storage trajectory.
    """
    HOURS_PER_DAY = 24
    MAX_SEGMENTS_SIZE = 30
    SMOOTHING_WINDOWS_SIZE = 4

    time_span = int(time_resolution*HOURS_PER_DAY)
    len_subset = np.random.randint(1,min(MAX_SEGMENTS_SIZE,max(3,int(n_bits/time_span))))      
    valid_start_range = np.random.choice(np.arange(4*time_span,n_bits-time_span*(len_subset+4)),max(1,int(len(c.storage_TS[selected_storage]/hyperparameters_operators_num[OPER_INV_MAGNITUDE,RESEARCH_INTERDAILY_CONSISTENCY]))))
    num_windows = max(1,int(n_bits /hyperparameters_operators_num[OPER_INV_MAGNITUDE, RESEARCH_INTERDAILY_CONSISTENCY]))
    window_starts = np.random.choice(valid_start_range, num_windows)

    
    daily_indices = np.empty((len(window_starts), len_subset), dtype=np.int64)
    extended_indices = np.empty((len(window_starts), len_subset + 4), dtype=np.int64)
    cumulative_signal = np.empty((len(window_starts), len_subset + 4), dtype=np.float64)
    for i in range(len(window_starts)):
        daily_indices[i]=np.arange(window_starts[i],window_starts[i]+len_subset*time_span,time_span)    
        extended_indices[i]=np.arange(window_starts[i]-2*time_span,window_starts[i]+(len_subset+2)*time_span,time_span)          
        cumulative_signal[i] = np.cumsum(c.storage_TS[selected_storage,extended_indices[i]]) 
    
    
    moving_average = (cumulative_signal[:,SMOOTHING_WINDOWS_SIZE:] - cumulative_signal[:,:-SMOOTHING_WINDOWS_SIZE]) / SMOOTHING_WINDOWS_SIZE 
    replacement = random_factor*c.storage_TS[selected_storage,daily_indices.flatten()]+(1-random_factor)*moving_average.flatten()
    
    c.storage_TS[selected_storage,daily_indices.flatten()] = replacement.flatten()
    return(c)


#Apply the pattern of a random reference day to several days based on their proximity
@jit(nopython=True)
def Copy_interdaily_patterns_operator(c,selected_storage,random_factor,n_bits,time_resolution):
    """
    Promote inter-daily dispatch consistency (RESEARCH operator).
    
    This operator selects a reference day and propagates its
    dispatch pattern to neighboring days with a distance-weighted
    blending factor.
    
    Procedure:
        1. A reference day is selected.
        2. A sequence of neighboring days is identified.
        3. For each day, dispatch is blended with the reference
           pattern using a distance-decaying weight.
    
    Formally, for day d:
    
        S_d' = (1 - α_d) S_d + α_d S_ref
    
    where:
    
        α_d = γ (1 - dist(d)/D)
    
    and γ ∈ [0,1].
    
    This operator encourages realistic inter-daily continuity
    while preserving diversity across distant days.
    
    Args:
        c (object): Candidate solution.
        choices (numpy.ndarray): Vector selecting the storage unit to mutate.
        random_factors (numpy.ndarray): Random vector controlling blending strength.
        n_bits (int): Total number of time steps.
        time_resolution (int): Number of time steps per hour.
    
    Returns:
        object: Mutated individual.
    """
    HOURS_PER_DAY = 24
    n_days = int(n_bits/(HOURS_PER_DAY*time_resolution))
    segment_length = int(time_resolution*HOURS_PER_DAY)
    
    start_day, start_offset, n_steps, direction = np.random.choice(np.arange(n_days-1)),0,np.random.randint(1,n_days),np.random.choice(np.array([-1,1]))
    day_sequence = np.arange(start_day, start_day + n_steps + 1)[:: -direction] % n_days
    
    time_indices = np.empty((len(day_sequence),segment_length),dtype=np.int64)
    
    for i in range(len(day_sequence)):
        base_time = day_sequence[i] * segment_length + start_offset
        time_indices[i]=np.arange(base_time,base_time+segment_length)    
    distances = np.abs(day_sequence-day_sequence[0])
    max_distance  = np.max(distances)
    for i in range(len(day_sequence)):
        similarity = 1 - distances[i] / (2 * max_distance)
        self_factor = 1 - random_factor * similarity
        reference_factor = random_factor * similarity
        c.storage_TS[selected_storage,time_indices[i]] = self_factor*c.storage_TS[selected_storage,time_indices[i]]+reference_factor*c.storage_TS[selected_storage,time_indices[0]]
    return(c)

@jit(nopython=True)
def Smooth_storage_noise_operator(c,selected_storage,random_factor,n_bits,hyperparameters_operators_num):
    """
    Apply stochastic local smoothing to a storage time-series (RESEARCH operator).
    
    This operator selects random subsequences within a storage time-series
    and replaces each value with a weighted combination of its original value
    and a local moving average computed over a surrounding window. 
    
    The effect is to introduce stochastic perturbations while preserving
    local trends and avoiding abrupt discontinuities.
    
    Args:
        c (object): Candidate solution.
        choices (numpy.ndarray): Storage selection indices.
        hyperparameters_operators_num (numpy.ndarray): Operator hyperparameter matrix.
    
    Returns:
        object: Mutated individual.
    """
    MIN_HALFWINDOW_SMOOTHING = 6
    MAX_HALFWINDOW_SMOOTHING_FACTOR = 8
    MAX_LEN = 30
    MIN_SEGMENTS = 1
    MAX_SEGMENTS = 30
    
    window_width = np.int64(min(MIN_HALFWINDOW_SMOOTHING,n_bits/MAX_HALFWINDOW_SMOOTHING_FACTOR))*2
    len_subset = np.random.randint(1,min(MAX_LEN,max(3,int(n_bits/hyperparameters_operators_num[OPER_INV_LENGTH,RESEARCH_CURVE_SMOOTHING]))))              
    n_segments = np.random.randint(MIN_SEGMENTS,min(MAX_SEGMENTS,int(n_bits /hyperparameters_operators_num[OPER_INV_MAGNITUDE, RESEARCH_CURVE_SMOOTHING]) + 2))
    valid_start_positions = np.arange(window_width,n_bits - len_subset - window_width)
    segment_starts = np.random.choice(valid_start_positions, n_segments)

    segment_indices =np.empty((len(segment_starts),len_subset),dtype=np.int64)
    extended_indices =np.empty((len(segment_starts),len_subset+window_width),dtype=np.int64)
    cumulative_signal =np.empty((len(segment_starts),len_subset+window_width),dtype=np.float64)
    for i in range(n_segments):
        segment_indices[i]=np.arange(segment_starts[i],segment_starts[i]+len_subset)
        extended_indices[i] = np.arange(segment_starts[i]-window_width//2,segment_starts[i]+len_subset+window_width//2)
        cumulative_signal[i] = np.cumsum(c.storage_TS[selected_storage,extended_indices[i]]) 
    vec_places = segment_indices.flatten()    

    moving_average = (cumulative_signal[:,window_width:] - cumulative_signal[:,:-window_width]) / window_width 

    updated_values = (random_factor*c.storage_TS[selected_storage,segment_indices.flatten()])+(1-random_factor)*moving_average.flatten()          
    
    c.storage_TS[selected_storage,vec_places] = updated_values   
    return(c)

@jit(nopython=True)
def Distribute_storage_roles_operator(c,random_factor,n_store,storage_characteristics,n_bits,hyperparameters_operators_num):
    """
    Perform inter-storage temporal reallocation (RESEARCH operator).
    
    This operator selects two storage units and redistributes their actions over
    randomly selected time subsets in order to reshape their temporal interaction.
    
    Procedure:
        1. Two storage systems are randomly selected.
        2. Several time sub-windows are sampled.
        3. Effective charging/discharging actions are computed (accounting for efficiency losses).
        4. The aggregated action is decomposed into:
            - a mean temporal trend component
            - a zero-mean fluctuation (noise) component
        5. A convex combination controlled by a random coefficient redistributes:
            - the trend to the first storage
            - the fluctuation to the second storage
    
    Formally, for selected indices t:
    
        S1'(t) = (1 - α) S1(t) + α · trend
        S2'(t) = (1 - α) S2(t) + α · noise(t)
    
    where α ∈ [0,1].
    
    This operator preserves local energy balance while modifying how storage technologies
    share temporal responsibilities. It enhances complementarity exploration between
    storage types (e.g., short-term vs long-term storage).
    
    Args:
        c (object): Candidate solution.
        random_factors (numpy.ndarray): Random vector controlling redistribution intensity.
        n_store (int): Number of storage technologies.
        storage_characteristics (numpy.ndarray): Matrix of storage parameters (including efficiency).
        hyperparameters_operators_num (numpy.ndarray): Hyperparameter matrix controlling mutation window sizes and frequency.
    
    Returns:
        object: Mutated individual.
    """
    store_a, store_b = np.random.choice(n_store,2,replace=False)
    selected_stores = np.array([store_a, store_b])
    
    len_subset = np.random.randint(1,max(3,int(n_bits/hyperparameters_operators_num[OPER_INV_LENGTH,RESEARCH_STORAGE_MIX])))         
    n_segments = np.random.randint(1, min(20,int(n_bits /hyperparameters_operators_num[OPER_INV_MAGNITUDE, RESEARCH_STORAGE_MIX]) + 2))
    
    segment_starts = np.random.choice(np.arange(0,n_bits-len_subset+1),n_segments)  
            
    segment_indices =np.empty((n_segments,len_subset),dtype=np.int64)
    raw_segments = np.empty((2,n_segments,len_subset),dtype=np.float64)
    effective_actions = np.empty((2,n_segments,len_subset),dtype=np.float64)
    for i in range(n_segments):
        indices = np.arange(segment_starts[i],segment_starts[i]+len_subset)
        segment_indices[i] = indices
        raw_segments[:, i] = c.storage_TS[selected_stores][:, indices]
        for j in range(2):
           effective_actions[j,i]=np.where(raw_segments[j,i]<0,raw_segments[j,i]*storage_characteristics[STOR_ROUND_TRIP_EFF,selected_stores[j]],raw_segments[j,i])
        
    combined_signal = effective_actions.sum(axis=0)
    total_action = combined_signal.sum(axis=1)
    segment_trend = total_action/len_subset
    segment_noise = combined_signal - segment_trend[:, None]
    coeff = random_factor
    flat_indices = segment_indices.flatten()

    c.storage_TS[store_a,segment_indices.flatten()] = (1-coeff)*c.storage_TS[store_a,flat_indices]+np.repeat(coeff*segment_trend,len_subset)
    c.storage_TS[store_b,segment_indices.flatten()] = (1-coeff)*c.storage_TS[store_b,flat_indices]+coeff*segment_noise.flatten()
    return(c)

@jit(nopython=True)
def Force_constraint_operator(c,constraint_num,selected_storage,hyperparameters_operators_num):
    """
    Apply constraint-oriented storage adjustment (RESEARCH operator).
    
    This operator modifies storage dispatch in order to improve compliance with a
    system-level constraint (e.g., import reduction or autonomy target).
    
    When constraint_num == 1:
    
        1. Time steps with positive trade imbalance (imports) are identified.
        2. A subset of these time steps is selected.
        3. Storage dispatch is adjusted proportionally to the trade magnitude,
           scaled by a random factor.
        4. A compensatory redistribution is applied to other time steps to
           maintain overall energy balance.
    
    Mathematically:
    
        S'(t₊) = S(t₊) + β · trades(t₊)
        S'(t₋) = S(t₋) − correction
    
    where β is sampled within predefined bounds.
    
    This operator acts as a guided mutation toward feasibility, reducing constraint
    violations without fully overriding evolutionary exploration.
    
    Args:
        c (object): Candidate solution.
        constraint_num (int): Identifier of the active constraint.
        hyperparameters_operators_num (numpy.ndarray): Hyperparameter matrix controlling
            mutation amplitude and sampling density.
    
    Returns:
        object: Mutated individual with improved constraint compliance.
    """
    
    if (constraint_num==CONS_Self_sufficiency) :
      importation_mask = c.trades>0
      importation_indices = np.where(importation_mask)[0]
      len_importation = len(importation_indices)
      if (len_importation>1):
          n_mutations = np.random.randint(1,max(2,int(len_importation /hyperparameters_operators_num[OPER_INV_LENGTH, RESEARCH_CONSTRAINT_FORCING])))
          mutation_indices = np.random.choice(importation_indices,n_mutations,replace=False)
          injected_energy  = c.trades[mutation_indices] * np.random.uniform(hyperparameters_operators_num[OPER_MIN_AVERAGE, RESEARCH_CONSTRAINT_FORCING],hyperparameters_operators_num[OPER_MAX_AVERAGE, RESEARCH_CONSTRAINT_FORCING] )
          
          c.storage_TS[selected_storage,mutation_indices] = c.storage_TS[selected_storage,mutation_indices]+injected_energy
      
          non_importation_indices = np.where(~importation_mask)[0]
          if (len(non_importation_indices)>0):
              n_compensations = np.random.randint(1,1 + min(len(non_importation_indices),max(1,int(len_importation /hyperparameters_operators_num[OPER_INV_LENGTH, RESEARCH_CONSTRAINT_FORCING]))) )
              compensation_indices = np.random.choice(non_importation_indices,n_compensations,replace=False)      
              c.storage_TS[selected_storage,compensation_indices]=c.storage_TS[selected_storage,compensation_indices]-np.sum(injected_energy)/len(compensation_indices)
    elif (constraint_num==CONS_Self_consumption) : 
        exportation_mask = c.trades<0
        exportation_indices = np.where(exportation_mask)[0]
        len_exportation = len(exportation_indices)
        if (len_exportation>1):
            n_mutations = np.random.randint(1,max(2,int(len_exportation /hyperparameters_operators_num[OPER_INV_LENGTH, RESEARCH_CONSTRAINT_FORCING])))
            mutation_indices = np.random.choice(exportation_indices,n_mutations,replace=False)
            absorbed_energy = -c.trades[mutation_indices] * np.random.uniform(hyperparameters_operators_num[OPER_MIN_AVERAGE, RESEARCH_CONSTRAINT_FORCING],hyperparameters_operators_num[OPER_MAX_AVERAGE, RESEARCH_CONSTRAINT_FORCING])
            c.storage_TS[selected_storage,mutation_indices] = c.storage_TS[selected_storage,mutation_indices]-absorbed_energy
            non_exportation_indices = np.where(~exportation_mask)[0]
            if len(non_exportation_indices) > 0:
                n_compensations = np.random.randint(1,1 + min(len(non_exportation_indices),max(1, int(len_exportation /hyperparameters_operators_num[OPER_INV_LENGTH, RESEARCH_CONSTRAINT_FORCING]))))
                compensation_indices = np.random.choice(non_exportation_indices,n_compensations,replace=False)      
                c.storage_TS[selected_storage,compensation_indices]=c.storage_TS[selected_storage,compensation_indices]+np.sum(absorbed_energy)/len(compensation_indices)
    return(c)

@jit(nopython=True)
def Mutate_YDSM_points_operator(c,n_bits,hyperparameters_operators_num):
    """
    Mutate Yearly Demand-Side Management (YDSM) schedules (RESEARCH operator).
    
    Apply stochastic additive and multiplicative perturbations to the yearly DSM vector.
    
    This operator selects random entries in the yearly Demand-Side Management (Y_DSM)
    vector and modifies them using Gaussian additive and multiplicative noise. 
    Negative results are projected to zero to maintain physically meaningful values.
    
    Args:
        c (object): Candidate solution.
        hyperparameters_operators_num (numpy.ndarray): Operator hyperparameter matrix.
    
    Returns:
        object: Mutated individual.
    """
    NOISE_RATIO = 50
    n_mutations = np.random.randint(1,max(2,int(n_bits/hyperparameters_operators_num[OPER_INV_LENGTH,RESEARCH_DSM_NOISE])))
    mutation_indices = np.random.choice(n_bits, n_mutations, replace=False)
    additive_noise = np.random.normal(0,NOISE_RATIO * hyperparameters_operators_num[OPER_MAX_DEVIATION, RESEARCH_DSM_NOISE],n_mutations)
    multiplicative_noise = np.random.normal(1,hyperparameters_operators_num[OPER_MAX_DEVIATION, RESEARCH_DSM_NOISE],n_mutations)

    c.Y_DSM[mutation_indices]=np.maximum(0,(c.Y_DSM[mutation_indices]+additive_noise)*multiplicative_noise)
    return(c)

@jit(nopython=True)
def Mutate_DDSM_points_operator(c,D_DSM_indexes,hyperparameters_operators_num):
    """
    Mutate Daily Demand-Side Management (DDSM) schedules (RESEARCH operator).
    
    Apply stochastic additive and multiplicative perturbations to the yearly DSM vector.
    
    This operator selects random entries in the daily Demand-Side Management (Y_DSM)
    vector and modifies them using Gaussian additive and multiplicative noise. 
    Negative results are projected to zero to maintain physically meaningful values.
    
    Args:
        c (object): Candidate solution.
        D_DSM_indexes (numpy.ndarray): Indexes of mutable DSM days (if applicable).
        random_factors (numpy.ndarray): Pre-generated random coefficients.
        hyperparameters_operators_num (numpy.ndarray): Operator hyperparameter matrix.
    
    Returns:
        object: Mutated individual.
    """
    NOISE_RATIO = 50
    n_time_steps = c.D_DSM.shape[1]
    n_mutations = np.random.randint(1,max(2,int(len(D_DSM_indexes)/hyperparameters_operators_num[OPER_INV_LENGTH,RESEARCH_DSM_NOISE])))
    mutation_rows = np.random.choice(D_DSM_indexes,n_mutations,replace=False)
    additive_noise = np.random.normal(0,NOISE_RATIO * hyperparameters_operators_num[OPER_MAX_DEVIATION, RESEARCH_DSM_NOISE],n_time_steps)    
    multiplicative_noise = np.random.normal(1,hyperparameters_operators_num[OPER_MAX_DEVIATION, RESEARCH_DSM_NOISE],n_time_steps)

    c.D_DSM[mutation_rows]=np.maximum(0,(c.D_DSM[mutation_rows] + additive_noise)*multiplicative_noise)
    return(c)

@jit(nopython=True)
def Smooth_DDSM_windows_operator(c,D_DSM_indexes,random_factor,hyperparameters_operators_num):
    """
    Apply local smoothing to daily DSM schedules (D_DSM).
    
    This operator selects random subsequences of one or several mutable daily DSM vectors
    and replaces them with a weighted average of surrounding values (moving average),
    controlled by a stochastic blending factor. It ensures short-term local smoothness
    in daily DSM schedules.
    
    Args:
        c (object): Candidate solution containing D_DSM.
        D_DSM_indexes (numpy.ndarray): Indexes of mutable DSM days.
        random_factors (numpy.ndarray): Pre-generated random coefficients.
        hyperparameters_operators_num (numpy.ndarray): Operator hyperparameter matrix.
    
    Returns:
        object: Mutated candidate with locally smoothed daily DSM subsequences.
    """
    MIN_HALFWINDOW_SMOOTHING = 3
    MAX_HALFWINDOW_SMOOTHING_FACTOR = 8

    len_day = int(c.D_DSM.shape[1])
    n_mutations = np.random.randint(1,max(2,int(len(D_DSM_indexes)/hyperparameters_operators_num[OPER_INV_LENGTH,RESEARCH_CURVE_SMOOTHING])))
    selected_days = np.random.choice(D_DSM_indexes, n_mutations, replace=False)
    window_width = np.int64(min(MIN_HALFWINDOW_SMOOTHING,len_day/MAX_HALFWINDOW_SMOOTHING_FACTOR))*2
    len_subset = np.random.randint(1,len_day-window_width)  
            
    start_indices = np.random.choice(np.arange(window_width//2,len_day-len_subset-window_width//2))  
    extended_indices = np.arange(start_indices - window_width // 2,start_indices + len_subset + window_width // 2)
    cumulative_signal = np.empty((len(selected_days),len_subset+window_width),dtype=np.float64)
    target_indices = np.arange(start_indices, start_indices + len_subset)
    for i in range(n_mutations):
        cumulative_signal[i] = np.cumsum(c.D_DSM[selected_days[i],extended_indices])             
   
    moving_average = (cumulative_signal[:,window_width:] - cumulative_signal[:,:-window_width]) / window_width 
    for i in range(n_mutations):        
        c.D_DSM[selected_days[i],target_indices] = (random_factor*c.D_DSM[selected_days[i],target_indices])+(1-random_factor)*moving_average[i]  
    return(c)

#Apply the pattern of a random reference day to several days based on their proximity
@jit(nopython=True)
def Copy_DDSM_interdaily_patterns_operator(c,D_DSM_indexes,random_factor,time_resolution):
    """
    Aligns multiple daily DSM schedules to a reference day (inter-daily consistency).
    
    A random reference day is selected, and nearby days are adjusted towards the reference
    according to their distance, using a weighted blend with a stochastic factor. This
    operator preserves inter-daily patterns and consistency while keeping some randomness.
    
    Args:
        c (object): Candidate solution containing D_DSM.
        D_DSM_indexes (numpy.ndarray): Indexes of mutable DSM days.
        random_factors (numpy.ndarray): Pre-generated random coefficients.
        time_resolution (float): Hourly fraction for mapping DSM time steps.
    
    Returns:
        object: Mutated candidate with improved inter-daily consistency in D_DSM.
    """
    n_days = c.D_DSM.shape[0]
    reference_day = np.random.choice(D_DSM_indexes)   
    offset = 0                                       
    span_length = np.random.randint(1, n_days)      
    direction = np.random.choice(np.array([-1, 1]))   

    propagation_days = np.arange(reference_day,reference_day+span_length[2]+1)[::(-1)*direction[3]]%n_days            
    distances = np.abs(propagation_days-propagation_days[0])
    max_distance = np.max((0.1,np.max(distances)))
    reference_profile = c.D_DSM[propagation_days[0]]
    for i in range(len(propagation_days)):
        attenuation = 1 - distances[i] / (2 * max_distance)
        c.D_DSM[propagation_days[i], :] = ((1 - random_factor * attenuation) * c.D_DSM[propagation_days[i], :]+ (random_factor * attenuation) * reference_profile)
    return(c)

#Orient the D_DSM where the trades are >0. Changes partially compensated by a random storage 
@jit(nopython=True)
def Force_DDSM_trades_consistency_operator(c,D_DSM_indexes,time_resolution,selected_storage,random_factors,hyperparameters_operators_num):
    """
    Redistribute daily DSM actions based on trade signals.
    
    This operator modifies randomly selected daily DSM vectors where trades are positive,
    reshaping the profile according to a stochastic coefficient derived from trade differences.
    Changes are partially compensated in a random storage time-series to preserve global balance.
    
    Args:
        c (object): Candidate solution containing D_DSM and storage_TS.
        D_DSM_indexes (numpy.ndarray): Indexes of mutable DSM days.
        time_resolution (float): Hourly fraction for mapping DSM time steps.
        choices (numpy.ndarray): Indexes for storage units.
        random_factors (numpy.ndarray): Pre-generated random coefficients.
        hyperparameters_operators_num (numpy.ndarray): Operator hyperparameter matrix.
    
    Returns:
        object: Mutated candidate with trade-oriented reshaped daily DSM.
    """
    HOURS_PER_DAY = 24
    time_steps_per_day = int(time_resolution * HOURS_PER_DAY)
    n_mutations = np.random.randint(1,max(2,int(len(D_DSM_indexes) /hyperparameters_operators_num[OPER_INV_LENGTH, RESEARCH_DSM_TRADES_CONSISTENCY])))

    selected_days = np.random.choice(D_DSM_indexes,n_mutations,replace=False)

    for day in selected_days:
        start,end = day * time_steps_per_day,(day + 1) * time_steps_per_day
        day_trades = c.trades[start:end]
        original_dsm = c.D_DSM[day].copy()
        max_trade = np.max(day_trades)

        trade_weights = (-(day_trades - max_trade)) ** (2 * random_factors[1])
        normalized_weights = trade_weights / (0.1 + np.sum(trade_weights))

        redistributed_profile = np.sum(original_dsm) * normalized_weights
        
        updated_dsm = (random_factors[2] * original_dsm+ (1 - random_factors[2]) * redistributed_profile)

        c.D_DSM[day] = np.maximum(0, updated_dsm)
        delta_dsm = c.D_DSM[day] - original_dsm

        c.storage_TS[selected_storage, start:end] += (delta_dsm * random_factors[3])
        
    return(c)

@jit(nopython=True)
def Smooth_YDSM_windows_operator(c,random_factor,n_bits,hyperparameters_operators_num):
    """
    Apply local smoothing to yearly DSM schedules (Y_DSM).
    
    This operator selects random subsequences of one or several mutable YDSM vectors
    and replaces them with a weighted average of surrounding values (moving average),
    controlled by a stochastic blending factor. It ensures short-term local smoothness
    in yearly DSM schedules.
    
    Args:
        c (object): Candidate solution containing D_DSM.
        random_factors (numpy.ndarray): Pre-generated random coefficients.
        hyperparameters_operators_num (numpy.ndarray): Operator hyperparameter matrix.
    
    Returns:
        object: Mutated candidate with locally smoothed daily DSM subsequences.
    """
    MIN_HALFWINDOW_SMOOTHING = 6
    MAX_HALFWINDOW_SMOOTHING_FACTOR = 8
    MAX_LEN_SEGMENTS = 30
    MAX_SEGMENTS = 30
    
    window_width = np.int64(min(MIN_HALFWINDOW_SMOOTHING,n_bits/MAX_HALFWINDOW_SMOOTHING_FACTOR))*2
    
    len_subset = np.random.randint(1,min(MAX_LEN_SEGMENTS,max(3,int(n_bits / hyperparameters_operators_num[OPER_INV_LENGTH, RESEARCH_CURVE_SMOOTHING]))))
    n_segments = np.random.randint(1,min(MAX_SEGMENTS,int(n_bits / hyperparameters_operators_num[OPER_INV_MAGNITUDE, RESEARCH_CURVE_SMOOTHING]) + 2)) 
                                   
    segment_starts = np.random.choice(np.arange(window_width, n_bits - len_subset - window_width),n_segments)
    
    segment_indices = np.empty((n_segments, len_subset), dtype=np.int64)
    extended_indices = np.empty((n_segments, len_subset + window_width), dtype=np.int64)
    cumulative_matrix = np.empty((n_segments, len_subset + window_width), dtype=np.float64)
    

    for i in range(n_segments):
        segment_indices[i]=np.arange(segment_starts[i],segment_starts[i]+len_subset)
        extended_indices[i] = np.arange(segment_starts[i]-window_width//2,segment_starts[i]+len_subset+window_width//2)
        cumulative_matrix[i] = np.cumsum(c.Y_DSM[extended_indices[i]])

    

    moving_average  = (cumulative_matrix[:,window_width:] - cumulative_matrix[:,:-window_width]) / window_width 
    flat_indices  = segment_indices.flatten()    

    smoothed_values = moving_average.flatten()

    original_values = c.Y_DSM[flat_indices]

    updated_values = (random_factor * original_values+ (1 - random_factor) * smoothed_values)

    c.Y_DSM[flat_indices] = updated_values

    return(c)

#Apply the pattern of a random reference day to several days based on their proximity
@jit(nopython=True)
def Copy_YDSM_interdaily_patterns_operator(c,random_factor,n_bits,time_resolution):
    """
    Aligns multiple daily DSM schedules to a reference day (inter-daily consistency).
    
    A random reference day is selected, and nearby days are adjusted towards the reference
    according to their distance, using a weighted blend with a stochastic factor. This
    operator preserves inter-daily patterns and consistency while keeping some randomness.
    
    Args:
        c (object): Candidate solution containing D_DSM.
        random_factors (numpy.ndarray): Pre-generated random coefficients.
        time_resolution (float): Hourly fraction for mapping DSM time steps.
    
    Returns:
        object: Mutated candidate with improved inter-daily consistency in D_DSM.
    """
    HOURS_PER_DAY = 24
    
    day_length = int(time_resolution * HOURS_PER_DAY)
    n_days = int(n_bits / day_length)
    
    reference_day = np.random.choice(np.arange(n_days - 1))
    offset_within_day = 0  
    span_length = np.random.randint(1, n_days)
    direction = np.random.choice(np.array([-1, 1]))
    
    propagation_days = (np.arange(reference_day, reference_day + span_length + 1)[:: -direction]% n_days)
    time_indices = np.empty((len(propagation_days), day_length), dtype=np.int64)
    for i in range(len(propagation_days)):
        start = propagation_days[i] * day_length + offset_within_day
        end = start + day_length
        time_indices[i] = np.arange(start, end)

    distances = np.abs(propagation_days - propagation_days[0])
    max_distance = np.max((0.1,np.max(distances)))        
    reference_profile = c.Y_DSM[time_indices[0]]
    for i in range(len(propagation_days)):
        attenuation = 1 - distances[i] / (2 * max_distance)
        c.Y_DSM[time_indices[i]] = ((1 - random_factor * attenuation) * c.Y_DSM[time_indices[i]]+ (random_factor * attenuation) * reference_profile)
    return(c)

#Orient the Y_DSM where the trades are >0. CHanges partially compensated by a random storage 
@jit(nopython=True)
def Force_YDSM_trades_consistency_operator(c,selected_storage,random_factors,n_bits,hyperparameters_operators_num):
    """
    Redistribute yearly DSM actions based on trade signals.
    
    This operator modifies randomly selected yearly DSM vectors where trades are positive,
    reshaping the profile according to a stochastic coefficient derived from trade differences.
    Changes are partially compensated in a random storage time-series to preserve global balance.
    
    Args:
        c (object): Candidate solution containing Y_DSM and storage_TS.
        choices (numpy.ndarray): Indexes for storage units.
        random_factors (numpy.ndarray): Pre-generated random coefficients.
        hyperparameters_operators_num (numpy.ndarray): Operator hyperparameter matrix.
    
    Returns:
        object: Mutated candidate with trade-oriented reshaped daily DSM.
    """
    
    n_mutations = np.random.randint(2,max(3,int(n_bits /hyperparameters_operators_num[OPER_INV_LENGTH, RESEARCH_DSM_TRADES_CONSISTENCY] )))
    mutation_indices = np.random.choice(n_bits,n_mutations,replace=False)    

    original_values = c.Y_DSM[mutation_indices].copy()
    local_trades = c.trades[mutation_indices]
    max_trade = np.max(local_trades)
    trade_weights = (-(local_trades - max_trade)) ** (2 * random_factors[1])
    
    total_dsm = np.sum(original_values)

    normalized_weights = trade_weights / (0.1 + np.sum(trade_weights))

    redistributed_values = total_dsm * normalized_weights

    updated_values = (random_factors[2] * original_values+ (1 - random_factors[2]) * redistributed_values)

    updated_values = np.maximum(0, updated_values)

    c.Y_DSM[mutation_indices] = updated_values
    delta = updated_values - original_values
    
    c.storage_TS[selected_storage][mutation_indices] = c.storage_TS[selected_storage][mutation_indices]+(delta)*random_factors[3]
    return(c)



################################### PRO OPERATORS ##########################################
@jit(nopython=True)
def Switch_dispatching_strategy_operator(c):
    """
    Switch the Power Management Strategy (PMS) (PRO operator).
    
    This operator toggles the PMS strategy between predefined modes
    (e.g., 'LF' and 'CC').
    
    Args:
        c (object): Candidate solution.
    
    Returns:
        object: Mutated individual with updated PMS strategy.
    """
    if (c.DG_strategy == 'LF'):
        c.DG_strategy = 'CC'
    elif (c.DG_strategy == 'CC'):
        c.DG_strategy = 'LF'
    return(c)

@jit(nopython=True)
def Mutate_production_capacity_operator_pro(c,Bounds_prod,groups, groups_size, hyperparameters_operators_num_pro):
    """
    Mutate installed production capacity (PRO operator).
    
    This operator perturbs the installed capacity of a selected
    generation technology (within an exclusive group) using a
    pseudo-normal mutation. The mutation preserves non-negativity
    and allows structural exploration of renewable or dispatchable assets.
    
    Args:
        c (object): Candidate solution.
        hyperparameters_operators_num_pro (np.ndarray): Hyperparameter matrix controlling mutation strength.
        choice (int): Index of the production technology to mutate.
    
    Returns:
        object: Mutated individual with updated production capacity.
    """
    group = np.random.randint(len(groups))
    valid_idx = groups[group, :groups_size[group]]
    mask_positive = c.production_set[valid_idx]>0
    candidates = valid_idx[mask_positive]
    if candidates.size > 0:
        productor = candidates[np.random.randint(candidates.size)]
    else:
        productor = valid_idx[np.random.randint(valid_idx.size)]

    low = min(-2, -int(Bounds_prod[productor] * hyperparameters_operators_num_pro[PRO_PRODUCTION,PRO_OPER_DEVIATION]))
    high = max(3, int(Bounds_prod[productor] * hyperparameters_operators_num_pro[PRO_PRODUCTION,PRO_OPER_DEVIATION]))

    modifier = c.production_set[productor] + np.random.randint(low, high)
    modifier = max(0, min(Bounds_prod[productor], modifier))


    c.production_set[productor]=modifier         
    return(c)

@jit(nopython=True)
def Switch_storages_order_operator(c,random_factors):
    """
    Mutate the storage discharge priority order (PRO operator).
    
    This operator randomly redefines the discharge priority ranking
    of storage units by sorting a vector of random factors. The
    resulting permutation determines the order in which storage
    systems are dispatched during power management.
    
    The mutation is purely combinatorial and modifies the dispatch
    hierarchy without altering physical capacities.
    
    Args:
        c (object): Candidate solution.
        random_factors (np.ndarray): Random values used to generate a new priority permutation.
    
    Returns:
        object: Mutated individual with updated discharge order.
    """
    c.discharge_order = np.argsort(random_factors)
    return(c)

@jit(nopython=True)
def Mutate_DSM_storage_distribution_operator(c,hyperparameters_operators_num_pro):
    """
    Mutate the DSM energy use repartition coefficient (PRO operator).
    
    This operator applies a bounded multiplicative Gaussian mutation
    to the scalar parameter controlling the proportion of energy
    allocated to demand-side management (DSM).
    
    The updated value is clipped to the interval [0, 1]:
    
        x' = clip(x * N(1, sigma), 0, 1)
    
    ensuring physical consistency of the repartition ratio.
    
    Args:
        c (object): Candidate solution.
        hyperparameters_operators_num_pro (np.ndarray): Hyperparameter matrix controlling mutation intensity (sigma).
    
    Returns:
        object: Mutated individual with updated DSM repartition coefficient.
    """
    c.energy_use_coefficient = min(1,max(0,c.energy_use_coefficient*np.random.normal(1,hyperparameters_operators_num_pro[PRO_OPER_DEVIATION,PRO_ENERGY_COEFF])))
    return(c)

@jit(nopython=True)
def Mutate_EMS_Overlap_operator(c,random_factors,hyperparameters_operators_num_pro):
    """
    Mutate EMS Overlap threshold curves (PRO operator).
    
    This operator perturbs the threshold vectors governing
    PMS taking-over behavior (e.g., state-of-charge activation levels).
    
    For each row of `c.PMS_taking_over`, a stochastic
    multiplicative-additive Gaussian perturbation is applied, followed by:
    
    - Clipping to [0, 1]
    - Sorting to preserve monotonic structure
    
    The mutation maintains ordered threshold logic while
    exploring alternative control strategies.
    
    Args:
        c (object): Candidate solution.
        random_factors (np.ndarray): Random selector controlling whether each row is mutated.
        hyperparameters_operators_num_pro (np.ndarray): Hyperparameter matrix controlling mutation intensity.
    
    Returns:
        object: Mutated individual with updated PMS taking-over thresholds.
    """
    if (random_factors[0]<0.5):
        c.overlaps[OVERLAP_INTERN,:] = np.sort(np.minimum(1,np.maximum(0,c.overlaps[OVERLAP_INTERN,:]*np.random.normal(1,hyperparameters_operators_num_pro[PRO_OPER_DEVIATION,PRO_OVERLAP],9)+np.random.normal(0,hyperparameters_operators_num_pro[PRO_OPER_DEVIATION,PRO_OVERLAP]/5,9))))
    if (random_factors[1]<0.5):
        c.overlaps[OVERLAP_EXTERN,:] = np.sort(np.minimum(1,np.maximum(0,c.overlaps[OVERLAP_EXTERN,:]*np.random.normal(1,hyperparameters_operators_num_pro[PRO_OPER_DEVIATION,PRO_OVERLAP],9)+np.random.normal(0,hyperparameters_operators_num_pro[PRO_OPER_DEVIATION,PRO_OVERLAP]/5,9))))
    return(c)

@jit(nopython=True)
def Mutate_DDSM_levels_operator(c,hyperparameters_operators_num_pro):
    """
    Mutate daily DSM minimum activation levels (PRO operator).
    
    This operator applies a stochastic perturbation to the vector
    of minimum daily DSM activation levels.
    
    Each element undergoes a multiplicative-additive Gaussian mutation,
    followed by:
    
    - Clipping to the interval [0, 1]
    - Sorting to preserve monotonic order
    
    This ensures consistent threshold structure while enabling
    exploration of alternative DSM activation policies.
    
    Args:
        c (object): Candidate solution.
        hyperparameters_operators_num_pro (np.ndarray): Hyperparameter matrix controlling mutation intensity.
    
    Returns:
        object: Mutated individual with updated daily DSM minimum activation levels.
    """
    c.D_DSM_minimum_levels = np.sort(np.minimum(1,np.maximum(0,c.D_DSM_minimum_levels*np.random.normal(1,hyperparameters_operators_num_pro[PRO_OPER_DEVIATION,PRO_DSM_LEVELS],23)+np.random.normal(0,hyperparameters_operators_num_pro[PRO_OPER_DEVIATION,PRO_DSM_LEVELS]/5,23))))
    return(c)

@jit(nopython=True)
def Mutate_YDSM_levels_operator(c,hyperparameters_operators_num_pro):
    """
    Mutate yearly DSM minimum activation levels (PRO operator).
    
    This operator perturbs the vector defining minimum yearly
    DSM activation thresholds.
    
    The mutation consists of a multiplicative-additive Gaussian
    perturbation, followed by:
    
    - Clipping to the interval [0, 1]
    - Sorting to maintain monotonic order
    
    This preserves structural feasibility of the yearly
    demand-side control policy.
    
    Args:
        c (object): Candidate solution.
        hyperparameters_operators_num_pro (np.ndarray): Hyperparameter matrix controlling mutation intensity.
    
    Returns:
        object: Mutated individual with updated yearly DSM minimum activation levels.
    """
    c.Y_DSM_minimum_levels = np.sort(np.minimum(1,np.maximum(0,c.Y_DSM_minimum_levels*np.random.normal(1,hyperparameters_operators_num_pro[PRO_OPER_DEVIATION,PRO_DSM_LEVELS],11)+np.random.normal(0,hyperparameters_operators_num_pro[PRO_OPER_DEVIATION,PRO_DSM_LEVELS]/5,11))))
    return(c)

@jit(nopython=True)
def Mutate_DG_min_runtime_operator(c):
    """
    Mutate the minimum diesel generator runtime constraint (PRO operator).
    
    This operator applies a discrete integer mutation to the
    minimum runtime constraint of the diesel generator (DG).
    
    The runtime is incremented or decremented by at most one unit,
    while enforcing a lower bound of 1:
    
        T' = max(1, T + randint(-1, 1))
    
    This mutation enables exploration of DG operational rigidity
    without violating feasibility.
    
    Args:
        c (object): Candidate solution.
    
    Returns:
        object: Mutated individual with updated DG minimum runtime.
    """
    c.DG_min_runtime = max(1,c.DG_min_runtime+np.random.randint(-1,2))
    return(c)

@jit(nopython=True)
def Mutate_DG_min_production_operator(c,hyperparameters_operators_num_pro):
    """
    Mutate the minimum diesel generator production threshold (PRO operator).
    
    This operator applies a log-normal multiplicative mutation
    to the minimum DG production parameter:
    
        P' = exp(log(P + ε) * N(1, sigma))
    
    where ε ensures numerical stability.
    
    This formulation guarantees strict positivity and scale-adaptive
    exploration of the minimum dispatch level.
    
    Args:
        c (object): Candidate solution.
        hyperparameters_operators_num_pro (np.ndarray): Hyperparameter matrix
            controlling mutation strength.
    
    Returns:
        object: Mutated individual with updated DG minimum production.
    """
    c.DG_min_production = max(0.,np.exp(np.log(c.DG_min_production+0.1)*np.random.normal(1,hyperparameters_operators_num_pro[PRO_OPER_DEVIATION,PRO_DG_CONTROL])))
    return(c)

@jit(nopython=True)
def Mutate_storages_capacity_operator(c,hyperparameters_operators_num_pro,choice):
    """
    Mutate installed storage capacity (PRO operator).
    
    This operator applies a log-normal multiplicative perturbation
    to the selected storage unit's capacity, enabling scale-adaptive
    exploration while preserving positivity.
    
    Args:
        c (object): Candidate solution.
        hyperparameters_operators_num_pro (np.ndarray): PRO operator hyperparameters controlling mutation intensity.
        choice (int): Index of the storage unit to mutate.
    
    Returns:
        object: Mutated individual with updated storage capacity.
    """
    c.storages[INDIV_PRO_VOLUME,choice] = max(0.,np.exp(np.log(c.storages[INDIV_PRO_VOLUME,choice]+0.1)*np.random.normal(1,hyperparameters_operators_num_pro[PRO_OPER_DEVIATION,PRO_STORAGE_CAPACITIES])))
    return(c)

@jit(nopython=True)
def Mutate_storages_inpower_operator(c,hyperparameters_operators_num_pro,choice):
    """
    Mutate installed storage power rating (PRO operator).
    
    This operator applies a log-normal multiplicative perturbation to
    the nominal charge/discharge power of a selected storage unit,
    allowing independent exploration of power sizing.
    
    Args:
        c (object): Candidate solution.
        hyperparameters_operators_num_pro (np.ndarray): PRO operator hyperparameters controlling mutation intensity.
        choice (int): Index of the storage unit to mutate.
    
    Returns:
        object: Mutated individual with updated storage power rating.
    """
    c.storages[INDIV_PRO_CHARGE_POWER,choice] = max(0.,np.exp(np.log(c.storages[INDIV_PRO_CHARGE_POWER,choice]+0.1)*np.random.normal(INDIV_PRO_CHARGE_POWER,hyperparameters_operators_num_pro[PRO_OPER_DEVIATION,PRO_STORAGE_POWERS])))
    return(c)

@jit(nopython=True)
def Mutate_storages_outpower_operator(c,hyperparameters_operators_num_pro,choice):
    """
    Mutate installed storage discharge power (PRO operator).
    
    This operator applies a log-normal multiplicative perturbation to
    the nominal discharge power of a selected storage unit, ensuring
    positivity while enabling independent exploration of power sizing.
    
    Mathematically:
        P' = P * exp(sigma * Z)
        
    where Z ~ N(0,1) and sigma is defined in the hyperparameters.
    
    Args:
        c (object): Candidate solution.
        hyperparameters_operators_num_pro (np.ndarray): Hyperparameter matrix controlling mutation intensity.
        choice (int): Index of the storage unit to mutate.
    
    Returns:
        object: Mutated individual with updated discharge power.
    """
    c.storages[INDIV_PRO_DISCHARGE_POWER,choice] = max(0.,np.exp(np.log(c.storages[INDIV_PRO_DISCHARGE_POWER,choice]+0.1)*np.random.normal(1,hyperparameters_operators_num_pro[PRO_OPER_DEVIATION,PRO_STORAGE_POWERS])))
    return(c)

@jit(nopython=True)
def Mutate_initSOC_operator(c,random_factor,choice):
    """
    Initialize or perturb the initial State of Charge (SOC) of a selected storage unit.
    
    This operator scales the SOC of the specified storage unit by a random
    factor and ensures that the resulting value remains within physical bounds
    [0, 1]. It is intended for stochastic exploration of initial conditions.
    
    Args:
        c (object): Candidate solution containing a `storages` array.
        random_factor (float): Random scaling factor used to perturb the SOC.
        choice (int): Index of the storage unit to modify.
    
    Returns:
        object: Candidate solution with updated SOC for the selected storage unit.
    """
    c.storages[INDIV_PRO_SOC_INIT,choice] = max(0.,min(1,c.storages[INDIV_PRO_SOC_INIT,choice]*(0.5+random_factor)))
    return(c)



