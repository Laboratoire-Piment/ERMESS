# -*- coding: utf-8 -*-
"""
Created on Mon May 18 15:47:43 2026

@author: JoPHOBEA
"""

import numpy as np
from ERMESS_scripts.data.data_builder import build_environment


def test_build_environment_runs(fake_structured_data):
    env = build_environment(fake_structured_data)

    assert env is not None
    
class Dummy:
    pass

def fake_structured_data():
    data = Dummy()

    data.storage = Dummy()
    data.storage.characteristics_num = np.array([[1, 2], [3, 4]])
    data.storage.n_store = 2
    data.storage.techs = np.array(["bat1", "bat2"])

    data.time = Dummy()
    data.time.time_resolution = 1
    data.time.duration_years = 1
    data.time.datetime = np.arange(24)
    data.time.n_days = 1

    data.production = Dummy()
    data.production.characteristics_num = np.zeros((2, 2))
    data.production.ids = np.array(["p1", "p2"])
    data.production.capacities = np.array([10, 10])
    data.production.groups = np.array([0, 1])
    data.production.current_prod = np.zeros(2)
    data.production.unit_prods = np.zeros((2, 24))

    data.load = Dummy()
    data.load.non_movable = np.zeros(24)
    data.load.yearly_movable = np.zeros(24)
    data.load.daily_movable = np.zeros(24)

    data.optimization = Dummy()
    data.optimization.constraint_num = 0
    data.optimization.constraint_level = 1.0
    data.optimization.criterion_num = 0
    data.optimization.type_optim = "research"

    data.connection = "on-grid"

    data.hyperparameters = Dummy()
    data.hyperparameters.r_cross = 0.5
    data.hyperparameters.n_iter = 10
    data.hyperparameters.n_pop = 10
    data.hyperparameters.operators_num = np.ones((5, 2))
    data.hyperparameters.cost_constraint = 1.0
    data.hyperparameters.elitism_probability = 0.1

    data.hyperparameterspro = data.hyperparameters

    data.dispatching = Dummy()
    data.dispatching.Defined_items = []
    data.dispatching.energy_use_coefficient = 1.0
    data.dispatching.Y_DSM_minimum_levels = np.zeros(1)
    data.dispatching.D_DSM_minimum_levels = np.zeros(1)
    data.dispatching.DG_strategy = "none"
    data.dispatching.DG_min_runtime = 0
    data.dispatching.DG_min_production = 0
    data.dispatching.Discharge_order = np.zeros(1)
    data.dispatching.Overlaps = np.zeros(1)

    data.tracking = 0
    data.grid = None
    data.genset = None
    data.postProcessConfig = None

    return data

def test_no_nan_in_environment(fake_structured_data):
    env = build_environment(fake_structured_data)

    assert not np.isnan(env.loads.non_movable).any()
    assert not np.isnan(env.production.unit_prods).any()
    
def test_storage_size(fake_structured_data):
    env = build_environment(fake_structured_data)

    assert env.storage.n_store == 2