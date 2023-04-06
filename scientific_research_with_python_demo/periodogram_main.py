import scientific_research_with_python_demo.utils as utils


def periodogram(par2ph, phase_obs, Num_search, step_orig: float, param_orig):
    """This is a program named "periodogram"
       It is an estimator seraching the solution space to find best (v,h),
       based on (topographic_height+linear_deformation)
       which maximize the temporal coherence γ

    Parameters
    ----------
    v2ph : float
       velocity-to-phase conversion factor
    h2ph : _type_
        height-to-phase conversion factor
    phase_obs : _type_
        simulated obseravation based on 'topographic_height+linear_deformation+niose'
    Num_search : _type_
        size of solution space
    step_orig : dict
        step of searching solution related to parameters[step_h,step_v]
    param_orig : _type_
        original paramters (h,v)

    Returns
    -------
    param : _type_
        The parameters generated after each iteration

    Notes
    -----
    The program consists of a number of modules,
    which enables users to check and upgrade.

    The modules are:
        param_serach:
           creat solution searching space
        coef2phase:
           compute phase based on baselines
        model_phase:
           computer
    """

    # ---------------------------------------------------------
    #  Step 1: construct parameter space ann related phase space
    # ---------------------------------------------------------
    search = dict()  # TODO: HOw to we initialize a dict?
    phase = dict()
    param = dict()
    for key in ("height", "velocity"):
        search[key] = utils._construct_parameter_space(step_orig[key], Num_search[key], param_orig[key])
        phase[key] = utils._coef2phase(par2ph[key], search[key])

    # search_size=[serach_sizeH,serach_sizeV]
    search_size = [Num_search["height"] * 2, Num_search["velocity"] * 2]
    # ---------------------------------------------------------
    # step 2:construct model_phase by using kronecker积 based on (v,h) pairs
    # ---------------------------------------------------------
    phase_model = utils.model_phase(phase["velocity"], phase["height"], search_size)
    # --------------------------------------------------------------------------
    #  Step 3: compute temporal coherence , find max coherence and caculate (v,h)
    # --------------------------------------------------------------------------
    coh_t = utils.simulate_temporal_coherence(phase_obs, phase_model)
    best, index = utils.find_maximum_coherence(coh_t)
    sub = utils.list2dic(["height", "velocity"], utils.index2sub(index, search_size))

    # calculate the best parameters
    for key in ("height", "velocity"):
        param[key] = utils.compute_param(sub[key], step_orig[key], param_orig[key], Num_search[key])
    return param


#