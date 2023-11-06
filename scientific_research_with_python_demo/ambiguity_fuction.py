import numpy as np
from copy import deepcopy
import json

# Constant
WAVELENGTH = 0.056  # [unit:m]
H = 780000  # satellite vertical height[m]
Incidence_angle = 23 * np.pi / 180  # the local incidence angle
R = H / np.cos(Incidence_angle)  # range to the master antenna. test
m2ph = 4 * np.pi / WAVELENGTH


# ------------------------------------------------
class Periodogram_estimation:
    # ------------------------------------------------
    # initial parameters
    # ------------------------------------------------
    def __init__(self, param_file):
        """initialize parameters

        Args:
            param_file (_type_): parameters about acr phase simulation and parameter estimation searching
            Nifg (int): the number of interferograms
            v_orig (float): the deformation rate of the arc
            h_orig (float): the topographic height error of the arc
            SNR (float): the signal phase to noise ratio of the arc based on interferograms
            step_orig (float): the initial step of searching
            std_param (float): the standard deviation of the input parameters related to the arc phase (v,h)
            param_orig (float): the  initial center of the searching parameters space
            param_name (str): the name of the searching parameters
            Num_search_min (int): the minimum number of searching prameters
            Num_search_max (int): the maximum number of searching prameters
            revisit_cycle (int): the revisit cycle of the interferograms
            Bn (float): normal baseline factor
            normal_baseline: the normal baseline of the interferograms based on revisit satellite
        """
        param = deepcopy(param_file)
        self.Nifg = param["Nifg"]
        self.param_sim = param["param_simulation"]
        self.noise_level = param["noise_level"]
        self.step_orig = param["step_orig"]
        self.std_param = param["std_param"]
        self.param_orig = param["param_orig"]
        self.param_name = param["param_name"]
        self.Num_search_min = param["Num_search_min"]
        self.Num_search_max = param["Num_search_max"]
        self.revisit_cycle = param["revisit_cycle"]
        # self.normal_baseline = np.array(param_file['normal_baseline'])
        self.Bn = param["Bn"]
        self.normal_baseline = np.random.randn(1, self.Nifg) * self.Bn
        self._param = param["param_orig"]

    # @property
    def v2ph(self):
        """compute factors of velocity to  arc phase


        time_baseline: the time baseline of the interferograms

        Returns:
            _array_: factors of velocity to  arc phase
        """
        time_baseline = (np.arange(1, self.Nifg + 1, 1)).reshape(1, self.Nifg)
        v2ph = (m2ph * self.revisit_cycle * time_baseline / 365).T
        return v2ph

    # @property
    def h2ph(self):
        """compute factors of topographic height error to  arc phase

        Returns:
            array: factors of topographic height error to  arc phase
        """

        h2ph = (m2ph * self.normal_baseline / (R * np.sin(Incidence_angle))).T
        return h2ph

    def par2ph(self):
        """compute arc phase based on input parameters

        Returns:
            _array_: arc phase based on input parameters
        """
        par2ph = dict()
        par2ph["velocity"] = self.v2ph()
        par2ph["height"] = self.h2ph()
        self._par2ph = par2ph

    @staticmethod
    def wrap_phase(phase):
        """wrap phase to [-pi,pi]

        Args:
            phase (_float_): true phase without  phase ambiguites

        Returns:
            _type_: _description_
        """
        return np.mod(phase + np.pi, 2 * np.pi) - np.pi

    @staticmethod
    def _add_gaussian_noise(Nifg, noise_level_set):
        """construct gaussian noise based on signal and SNR

        Args:
            signal (_array_): arc phase of 'Nifgs' interferograms without phase ambiguity
            SNR (_type_): the signal phase to noise ratio of the arc based on interferograms

        Returns:
            _array_: gaussian noise base on signal size and SNR
        """
        """
        :param signal: 原始信号
        :param SNR: 添加噪声的信噪比
        :return: 生成的噪声
        """
        noise_std = np.zeros((1, Nifg + 1))
        noise_level = np.zeros((Nifg + 1, 1))
        noise_level[0] = np.pi * noise_level_set / 180
        noise_std[0][0] = np.random.randn(1) * noise_level[0]

        for v in range(Nifg):
            noise_level[v + 1] = (np.pi * noise_level_set / 180) + np.random.randn(1) * (np.pi * 5 / 180)
            noise_std[0][v + 1] = np.random.randn(1) * noise_level[v + 1]
        noise_phase = np.zeros((Nifg, 1))
        for i in range(Nifg):
            noise_phase[i] = noise_std[0][i] + noise_std[0][i + 1]

        return noise_phase

    @staticmethod
    def _check_snr(signal, noise):
        """_summary_

        Args:
            signal (_array_): arc phase without phase ambiguities
            noise (_type_): _description_

        Returns:
            _float_: SNR of the arc phase
        """
        signal_power = (1 / signal.shape[0]) * np.sum(np.power(signal, 2))  # 0.5722037
        noise_power = (1 / noise.shape[0]) * np.sum(np.power(noise, 2))  # 0.90688
        SNR = 10 * np.log10(signal_power / noise_power)
        return SNR

    def simulate_arc_phase(self):
        """simulate  observed arc phase based on input parameters
        Parameters:
            v2ph (array): factors of deformation rate to  arc phase
            h2ph (array): factors of topographic height error to  arc phase
            phase_unwrap (array): true arc phase without phase ambiguities
            arc_phase (array): observed arc phase based on interferograms
        """
        # factors of parameters to  arc phase
        # self._h2ph = self.h2ph()
        # self._v2ph = self.v2ph()
        self.par2ph()
        phase_unwrap = np.zeros((self.Nifg, 1))
        for key in self.param_name:
            phase_unwrap += self._par2ph[key] * self.param_sim[key]
        # simulate true arc phase without phase ambiguities
        self._phase_unwrap = phase_unwrap
        # generate gaussian noise of the 'Nifgs' arc phases based on SNR
        noise = Periodogram_estimation._add_gaussian_noise(self.Nifg, self.noise_level)
        phase_true = self._phase_unwrap + noise
        # wrap phase to [-pi,pi] as the observation arc phase
        self.arc_phase = Periodogram_estimation.wrap_phase(phase_true)
        self._snr_check = Periodogram_estimation._check_snr(self._phase_unwrap, noise)

    def _construct_parameter_space(self):
        """construct parameter space based on the searching boundaries and searching step
        Parameters
        ----------
            param_name (str): the name of the searching parameters ("height" or "velocity")
            Num_search_min (int): the minimum number of searching prameters
            Num_search_max (int): the maximum number of searching prameters
            step_orig (float): the searching step of the searching parameters
        Returns
        -------
            _array_: parameters searching space
        """
        param_space = dict()
        for key in self.param_name:
            param_space[key] = np.arange(-self.Num_search_min[key], self.Num_search_max[key] + 1, 1) * self.step_orig[key] + self._param[key]

        return param_space

    def _construct_searched_space(self):
        """construct searched phase space based on the searching boundaries , searching step, and searching parameters space
        Parameters
        ----------
            searched_param (dict): the searching parameters space
            h2ph (array): factors of topographic height error to  arc phase
            v2ph (array): factors of deformation rate to  arc phase
        output
        -------
            _array_: searched phase space
        Notes
        -----
            Since we have a range of parameter v and another range of paramters h every iteration,
            we have got phase_height and phase_v whose dimension
            related to its 'number of search solution'.
            In this case , we have to get a combination of phase based on each v and h
            based on 'The multiplication principle of permutations and combinations'

            For example, we get a range of  parameter v (dimension: 1*num_search_v)
            and a range of parameter  h (dimension: 1*num_search_h)
            In one case , we can have a combination of (v,h) (dimension: num_search_v*num_search_h)

            Since we have 'Number of ifg (Nifg)' interferograms, each parmamters will have Nifg phases.
            Then , we get get a range of phase based parameter's pair (v,h)
            named φ_model (dimension: Nifg*(num_search_v*num_search_v)
            ------------------------------------------------------------------------------------------
            In our case , we can firtsly compute phase
            based on a range of paramters of Nifg interferograms
            φ_height(dimension:Nifg*num_search_h),
            φ_v(dimension:Nifg*num_search_v).

            Then we have to create a combination φ_height and φ_v in the dimension of interferograms
            φ_model (dimension: Nifg*(num_search_v*num_search_v)
            Kronecker product is introduced in our case,
            we use 'kron' to extend dimension of φ_height or φ_v to
            dimension(Nifg*(num_search_v*num_search_v))
            and then get add φ_model by adding extended φ_height and φ_v.
            ------------------------------------------------------------------------------------------
        """
        searched_phase = dict()
        # construct searched parameters space
        self.searched_param = self._construct_parameter_space()

        # construct searched phase space
        for key in self.param_name:
            searched_phase[key] = self.searched_param[key] * self._par2ph[key]

        # shape searched space by kronecker product
        if len(self.param_name) <= 1:
            self._searched_space = searched_phase[self.param_name[0]]
        else:
            self._searched_space = np.kron(searched_phase["velocity"], np.ones((1, self.searched_param["height"].size))) + np.kron(
                np.ones((1, self.searched_param["velocity"].size)), searched_phase["height"]
            )
            # search_space = np.kron(search_phase1, np.ones((1, num_serach[0]))) + np.kron(np.ones((1, num_serach[1])), search_phase2)
            self._searched_phase_size = [
                self.searched_param["height"].size,
                self.searched_param["velocity"].size,
            ]

    def compute_temporal_coherence(self):
        """compute temporal coherence based on searched phase space and observed arc phase per (v,h) pair

        Parameters
        ----------
            arc_phase (array): simulated observation arc phase based on interferograms
            searched_space (array): searched phase space based on (v,h) pair

        Output
        ------
            coh_t (array): temporal coherence of the searched phase space per (v,h) pair

        """

        # resdual_phase = phase_observation - phase_model
        # temporal coherence γ=|(1/Nifgs)Σexp(j*(φ0s_obs-φ0s_modle))|
        coh_phase = self.arc_phase * np.ones((1, self._searched_space.shape[1])) - self._searched_space
        # temporal coherence
        self.coh_t = np.sum(np.exp(1j * coh_phase), axis=0, keepdims=True) / self.Nifg

    def compute_param_index(self):
        """search best coh_t of each paramters (v,h) based on several interferograms
        and get its index in the searched space

        Parameters
        ----------
            coh_t (array): temporal coherence of the searched phase space per (v,h) pair
            searched_phase_size (array): the size of searched phase space

        Output
        ------
            best_index (array): the index of the best coh_t per (v,h) pair in the searched space
        """

        best_index = np.argmax(np.abs(self.coh_t))
        if len(self.param_name) <= 1:
            self._param_index = [best_index]
        else:
            self._param_index = np.unravel_index(best_index, self._searched_phase_size, order="F")

    def compute_param(self):
        """compute the best paramters (v,h) based on the index of the best coh_t per (v,h) pair in the searched space

        Parameters
        ----------
            param_orig (dict): the intial center of searching parameters space
            param_index (array): the index of the best coh_t per (v,h) pair in the searched space
            Num_search_min (array): the minimum index of the searching parameters space
            step_orig (dict): the step of searching parameters space

        Output
        ------
            param (dict): the best paramters (v,h) based on the index of the best coh_t per (v,h) pair in the searched space

        """
        for i, key in enumerate(self.param_name):
            self._param[key] = np.round(self._param[key] + (self._param_index[i] - self.Num_search_min[key]) * self.step_orig[key], 8)

    def _periodogram_estimation(self):
        """parameter estimation based on periodogram method
        this method is based on the following steps:
            step1: construct searched space
            step2: compute temporal coherence
            step3: compute param index
            step4: compute param
        """
        self._construct_searched_space()
        self.compute_temporal_coherence()
        self.compute_param_index()
        self.compute_param()

    def searching_loop(self):
        """searching loop based on periodogram method
        we have set the maximum iteration number as 10 and step bound
        as 1.0e-8 and 1.0e-4 for velocity and height respectively.
        After each iteration, the step of searching space will be reduced by 10 times.
        and the Num_search_min and Num_search_max will be set as 10 from the second iteration to the end.
        The experiments shows that the searching loop could improve the precisionof the estimated parameters
        and somtimes correct the wrong resolution.

        """
        count = 0
        while count <= 10 and self.step_orig["velocity"] >= 1.0e-8 and self.step_orig["height"] >= 1.0e-4:
            self._periodogram_estimation()

            for key in self.param_name:
                self.step_orig[key] = self.step_orig[key] * 0.1
                self.Num_search_min[key] = 10
                self.Num_search_max[key] = 10

            count += 1


def compute_success_rate(param_file):
    """compute the success rate of the parameter estimation based on periodogram method by run the process 100 times
        the acceptable estimation error is set as 0.05 for height and 0.00005 for velocity
    Parameters
    ----------
    param_file : _dict_
        input parameters for the simulation and estimation
    """

    iteration = 0
    success_count = 0
    estimated_param = np.zeros((1000, len(param_file["param_name"])))
    while iteration < 1000:
        est = Periodogram_estimation(param_file)
        # simulate arc phase ,the normal baseline and the noise phase are randomly generated
        est.simulate_arc_phase()
        est.searching_loop()

        # print(est._param)
        if len(param_file["param_name"]) <= 1:
            estimated_param[iteration] = est._param[param_file["param_name"][0]]
            if abs(est._param["height"] - est.param_sim["height"]) < 0.5 or abs(est._param["velocity"] - est.param_sim["velocity"]) < 0.0005:
                success_count += 1
                # print(est._param)
        else:
            if abs(est._param["height"] - est.param_sim["height"]) < 0.5 and abs(est._param["velocity"] - est.param_sim["velocity"]) < 0.0005:
                estimated_param[iteration] = list(est._param.values())
                success_count += 1
                # print(est._param)
        iteration += 1
        del est
    # with open("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/"
    #     + param_file["file_name"]
    #     + "est_param"
    #     + ".csv",
    #      "a") as f:
    # # 按列追加保存
    #     np.savetxt(f, estimated_param, delimiter=",")
    return success_count / 1000


def param_experiment_v(test_param_name, test_param_range, v_range, param_file, save_name):
    for i in range(len(test_param_range)):
        success_rate = np.zeros([1, v_range])
        param_file["test_param_name"] = test_param_range[i]
        print("%d = %s " % test_param_name, test_param_range[i])
        for j in range(len(v_range)):
            param_file["param_simulation"]["velocity"] = v_range[j]
            success_rate[0][j] = compute_success_rate(param_file)
    np.savetxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/" + save_name + "success_rate" + ".csv", success_rate)

    return success_rate
