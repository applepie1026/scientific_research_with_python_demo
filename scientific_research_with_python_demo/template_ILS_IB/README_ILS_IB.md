# ils_demo file
## input data simulation
Float ambiguity solution is used to initilize the ILS estimator. 
To make a demo of ils progress,phase observation simulation is necceesray.
example:
"""
WAVELENGTH = 0.056  # [unit:m]
Nifg = 30
v_orig = 0.01  # [mm/year] 减少v，也可以改善估计结果，相当于减少了重访周期
h_orig = 30  # [m]，整数 30 循环迭代搜索结果有问题
pseudo_obs = np.array([[28, 0.04]]).T
noise_level = 20
param_name = ["height", "velocity"]
normal_baseline = np.random.randn(1, Nifg) * 333
# print(normal_baseline)
time_baseline = np.arange(1, Nifg + 1, 1).reshape(1, Nifg)  # 减小重访周期 dt 能明显改善结果
m2ph = 4 * np.pi / WAVELENGTH
std_param = np.array([40, 0.06])
# calculate the input parameters of phase
v2ph = af.v_coef(time_baseline).T
h2ph = af.h_coef(normal_baseline).T
"""

## searching initial input parameters
### seraching bound
"""
dh = 40
vel = 10
"""
These parameters are used to caculate pseudo-VCM
### float ambiguity solution
Float solutions are set as zero.

### VC matrix guess 
LAMBDA-ILS method is based on VC-matrix of phase_obs.
However,we usually don't know the true VC-matrix.We can use VCE (Variance component estimation)
#### VCE (Variance component estimation)
The whole progress of VCE has three steps.
Its main idea is using ILS to get a guessed phase_unwrap and using phase_unwrap_guess to estimate Q_vc.

Step 1:Guess a VC related pamramters and caculate VC-matrix_guess.
Step 2:Using VC-matrix_guess to etimate phase_unwrap_guess by using LAMBDA-ILS method
Step 3:Estimate more precise VC-matrix by using VCE method based on phase_unwrap_guess

## parameters estimation
Once we have estimated the VC-matrxin，
we can using LAMBDA-method to estimate phase ambiguity.
Exmaple:
[ILS method demo](ILS_demo.py)

