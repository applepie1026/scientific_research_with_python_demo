import numpy as np
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
        self.Nifg = param_file['Nifg']
        self.v_orig = param_file['v_orig']
        self.h_orig = param_file['h_orig']
        self.SNR = param_file['SNR']
        self.step_orig = param_file['step_orig']
        self.std_param = param_file['std_param']
        self.param_orig = param_file['param_orig']
        self.param_name = param_file['param_name']
        self.Num_search_min = param_file['Num_search_min']
        self.Num_search_max = param_file['Num_search_max']
        self.revisit_cycle = param_file['revisit_cycle']
        self.normal_baseline = np.array(param_file['normal_baseline'])
        self._param=param_file['param_orig']


    @property
    def v2ph(self):
        time_baseline = (np.arange(1, self.Nifg + 1, 1)).reshape(1, self.Nifg)
        self._v2ph=(m2ph * self.revisit_cycle*time_baseline/365).T
        return self._v2ph
    @property
    def h2ph(self):

        # normal_baseline =np.random.randn(1,self.Nifg) *333
        self._h2ph = (m2ph * self.normal_baseline / (R*np.sin(Incidence_angle))).T
        return self._h2ph
    
    @staticmethod
    def wrap_phase(phase):
        return np.mod(phase + np.pi, 2 * np.pi) - np.pi

    @staticmethod
    def _add_gaussian_noise(signal,SNR):
        """
        :param signal: 原始信号
        :param SNR: 添加噪声的信噪比
        :return: 生成的噪声
        """
        noise = np.random.randn(*signal.shape)  # *signal.shape 获取样本序列的尺寸
        noise = noise - np.mean(noise)
        signal_power = (1 / signal.shape[0]) * np.sum(np.power(signal, 2))
        noise_variance = signal_power / np.power(10, (SNR / 10))
        noise = (np.sqrt(noise_variance) / np.std(noise)) * noise

        return noise    
    @staticmethod
    def _check_snr(signal,noise):
            
        signal_power = (1 / signal.shape[0]) * np.sum(np.power(signal, 2))  # 0.5722037
        noise_power = (1 / noise.shape[0]) * np.sum(np.power(noise, 2))  # 0.90688
        SNR = 10 * np.log10(signal_power / noise_power)
        return SNR
    

    def simulate_arc_phase(self):
        self._phase_unwrap=self.v2ph*self.v_orig+self.h2ph*self.h_orig
        noise = Periodogram_estimation._add_gaussian_noise(self._phase_unwrap,self.SNR)
        phase_true = self._phase_unwrap + noise
        self.arc_phase = Periodogram_estimation.wrap_phase(phase_true)
        self._snr_check = Periodogram_estimation._check_snr(self._phase_unwrap,noise)
        
    
    
    def _construct_parameter_space(self):
        param_space=dict()
        for key in self.param_name:
            param_space[key]=np.arange(-self.Num_search_min[key],self.Num_search_max[key]+1,1)*self.step_orig[key]+self.param_orig[key]
        
        return param_space
 

    def _construct_searched_space(self):

        searched_phase=dict()
        # construct searched parameters space
        self.searched_param=self._construct_parameter_space()
        
        # construct searched phase space
        searched_phase["height"]=self.searched_param["height"]*self._h2ph
        searched_phase["velocity"]=self.searched_param["velocity"]*self._v2ph
        # shape searched space by kronecker product

        self._searched_space = np.kron(searched_phase["velocity"], np.ones((1, self.searched_param["height"].size))) + np.kron(np.ones((1, self.searched_param["velocity"].size)), searched_phase["height"])
        # search_space = np.kron(search_phase1, np.ones((1, num_serach[0]))) + np.kron(np.ones((1, num_serach[1])), search_phase2)
        self._searched_phase_size=[self.searched_param["height"].size,self.searched_param["velocity"].size]
        
        
    def compute_temporal_coherence(self):

        # resdual_phase = phase_observation - phase_model
        coh_phase = self.arc_phase * np.ones((1, self._searched_space.shape[1])) - self._searched_space
        # temporal coherence
        self.coh_t = np.sum(np.exp(1j * coh_phase), axis=0, keepdims=True) / self.Nifg

    def compute_param_index(self):

        self._best_index = np.argmax(np.abs(self.coh_t))
        self._param_index=np.unravel_index(self._best_index,self._searched_phase_size,order='F')

    def compute_param(self):
        for i,key in enumerate(self.param_name):
           self._param[key] = np.round(self.param_orig[key] + (self._param_index[i] - self.Num_search_min[key]) * self.step_orig[key], 8)
    

    def _periodogram_estimation(self):
        
        self._construct_searched_space()
        self.compute_temporal_coherence()
        self.compute_param_index()
        self.compute_param()

    def searching_loop(self):
        count = 0
        while count <=10 and self.step_orig["velocity"] >= 1.0e-8 and self.step_orig["height"] >= 1.0e-4:

            self._periodogram_estimation()
            for key in (self.param_name):
                self.step_orig[key] = self.step_orig[key] *0.1
                self.Num_search_min[key] = 10
                self.Num_search_max[key] = 10

            count += 1
    
    def compute_success_rate(self):
        iteration = 0
        success_count = 0
        while iteration<10000:
            self.simulate_arc_phase()
            self.searching_loop()
            if abs(self._param["height"] - self.h_orig) < 0.05 and abs(self._param["velocity"] - self.v_orig) < 0.00005:
                success_count += 1

            iteration+=1
        
        self.success_rate = success_count / iteration

