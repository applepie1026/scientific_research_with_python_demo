import pytest
import numpy as np
import json
import scientific_research_with_python_demo.ambiguity_fuction as af



def test_v2ph():
    with open('template/param.json') as f:
        param_file = json.load(f)
        
    af_obj = af.Periodogram_estimation(param_file)
    desired=np.array([[12/365,24/365,36/365]])*af.m2ph
    assert af_obj.v2ph.shape == (3, 1)
    assert np.allclose(af_obj.v2ph, desired.T)

def test_h2ph():
    with open('template/param.json') as f:
        param_file = json.load(f)
    
    af_obj = af.Periodogram_estimation(param_file)
    af_obj.normal_baseline = np.array([[100,200,300]])*(af.R*np.sin(af.Incidence_angle))/af.m2ph
    desired=np.array([[100,200,300]]).T
    assert af_obj.h2ph.shape == (3, 1)
    assert np.allclose(af_obj.h2ph, desired)

def test_wrap_phase():
    actual=af.Periodogram_estimation.wrap_phase(3*np.pi)
    desired=-np.pi
    assert np.allclose(actual, desired)

def test_simulate_arc_phase():
    with open('template/param.json') as f:
        param_file = json.load(f)
    
    af_obj = af.Periodogram_estimation(param_file)
    af_obj.revisit_cycle=365/af.m2ph
    af_obj.normal_baseline = np.array([[1,2,3]])*(af.R*np.sin(af.Incidence_angle))/af.m2ph
    af_obj.simulate_arc_phase()
    desired=np.array([[2,4,6]]).T
    # noise=af.Periodogram_estimation._add_gaussian_noise(desired, 70)
    # snr=af.Periodogram_estimation._check_snr(desired, noise)
    actual=af_obj._phase_unwrap
    assert np.allclose(actual, desired)
    assert np.allclose(af_obj._snr_check, 70)

def test_construct_searched_space():

    with open('template/param.json') as f:
        param_file = json.load(f)
    
    af_obj = af.Periodogram_estimation(param_file)
    actual=af_obj._construct_parameter_space()
    desired={"height":np.array([-2,-1,0,1,2]), "velocity":np.array([-2,-1,0,1,2])}
    # 
    assert np.allclose(actual["height"], desired["height"])
    assert np.allclose(actual["velocity"], desired["velocity"])

def test_construct_searched_space():
    with open('template/param.json') as f:
        param_file = json.load(f)
    
    af_obj = af.Periodogram_estimation(param_file)
    af_obj.revisit_cycle=365/af.m2ph
    af_obj.normal_baseline = np.array([[1,2,3]])*(af.R*np.sin(af.Incidence_angle))/af.m2ph
    af_obj.simulate_arc_phase()
    af_obj._construct_searched_space()
    desired=np.array([[-4,-3,-2,-1,0,-3,-2,-1,0,1,-2,-1,0,1,2,-1,0,1,2,3,0,1,2,3,4],
                      [-8,-6,-4,-2,0,-6,-4,-2,0,2,-4,-2,0,2,4,-2,0,2,4,6,0,2,4,6,8],
                      [-12,-9,-6,-3,0,-9,-6,-3,0,3,-6,-3,0,3,6,-3,0,3,6,9,0,3,6,9,12]
                      ])
    actual=af_obj._searched_space
    assert np.allclose(actual, desired)

def test_compute_temporal_coherence():
    with open('template/param.json') as f:
        param_file = json.load(f)
    
    af_obj = af.Periodogram_estimation(param_file)
    af_obj.arc_phase=np.array([[2,4,6]]).T
    af_obj._searched_space=np.array([[1,2,3],[1,2,3],[1,2,3]])
    af_obj.compute_temporal_coherence()
    actual=af_obj.coh_t
    desired=np.array([[np.exp(1j)+np.exp(3j)+np.exp(5j),np.exp(0)+np.exp(2j)+np.exp(4j),np.exp(-1j)+np.exp(1j)+np.exp(3j)]])/3
    assert np.allclose(actual, desired)

def test_compute_param_index():
    with open('template/param.json') as f:
        param_file = json.load(f)
    
    af_obj = af.Periodogram_estimation(param_file)
    af_obj.coh_t=np.array(
        [
            [
                np.exp(1) + np.exp(2) + np.exp(1),
                np.exp(3) + np.exp(3) + np.exp(2),
                np.exp(2) + np.exp(4) + np.exp(1),
                np.exp(3) + np.exp(1) + np.exp(2),
            ]
        ]
    )
    af_obj._searched_phase_size=[2,2]
    af_obj.compute_param_index()
    actual=af_obj._param_index
    desired=(0, 1)
    actual_coh=af_obj.coh_t[0][af_obj._best_index]    
    desired_coh=np.exp(2) + np.exp(4) + np.exp(1)
    assert np.allclose(actual, desired)
    assert np.allclose(actual_coh, desired_coh)
    
def test_compute_param():
    with open('template/param1.json') as f:
        param_file = json.load(f)
    af_obj = af.Periodogram_estimation(param_file)
    af_obj._param_index=(0,1)
    af_obj._param={"height":0,"velocity":0}
    af_obj.compute_param()
    actual=af_obj._param
    desired={"height":-1,"velocity":0}
    assert np.allclose(actual["height"], desired["height"])
    assert np.allclose(actual["velocity"], desired["velocity"])

def test_periodogram_estimation():
    with open('template/param2.json') as f:
        param_file = json.load(f)
    af_obj = af.Periodogram_estimation(param_file)
    af_obj.simulate_arc_phase()
    af_obj._periodogram_estimation()
    actual=af_obj._param
    desired={"height":31.0,"velocity":0.05}
    assert np.allclose(actual["height"], desired["height"])
    assert np.allclose(actual["velocity"], desired["velocity"])

def test_serachin_loop():
    with open('template/param2.json') as f:
        param_file = json.load(f)
    af_obj = af.Periodogram_estimation(param_file)
    af_obj.simulate_arc_phase()
    af_obj.searching_loop()
    actual=af_obj._param
    desired={"height":31.0,"velocity":0.05}
    assert abs(actual["height"]- desired["height"])<=0.05
    assert abs(actual["velocity"]-desired["velocity"])<=0.00005

def test_compute_success_rate():
    with open('template/param2.json') as f:
        param_file = json.load(f)
    af_obj = af.Periodogram_estimation(param_file)
    af_obj.simulate_arc_phase()
    af_obj.compute_success_rate()
    actual=af_obj.success_rate
    desired=1
    assert np.allclose(actual, desired)
