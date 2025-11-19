# diverse_community_example.py
import numpy as np
import pandas as pd
from itertools import product
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ------------------------
# 以下为需要根据 R 的实现替换/扩展的函数接口。
# 我在这里给出简化版本以便示例可运行。
# ------------------------

def BuildPars(in_pars_row):
    # 输入: pandas Series 或 dict，包含参数 S, MuR, SigmaR, MuD, SigmaD, ...
    # 输出: dict，至少包含 S, R, D, A, B（矩阵或数组），以及其他所需项
    S = int(in_pars_row['S'])
    pars = {}
    pars['S'] = S
    # 简化：R、D 为常数向量，A 为 interaction matrix, B 为 zeros matrix（HOI）
    pars['R'] = np.full(S, in_pars_row['MuR'])
    pars['D'] = np.full(S, in_pars_row['MuD'])
    # A: pairwise interactions, 这里用小的负自调项加随机互作（按 MuA, SigmaA）
    muA = in_pars_row['MuA']
    sigmaA = in_pars_row['SigmaA']
    # 对角自调项
    A = np.full((S, S), muA)
    if sigmaA > 0:
        A += np.random.normal(0, sigmaA, size=(S,S))
    np.fill_diagonal(A, -np.abs(np.diag(A)) - 0.1)  # 保证自调为负（稳定）
    pars['A'] = A
    # B: higher-order interaction tensor — 简化为 zero 3D array (S x S x S)
    pars['B'] = np.zeros((S,S,S))
    return pars

def GetTargetAbd(pars, value=0.1):
    # 返回目标稳态丰度向量，长度 S
    S = pars['S']
    return np.full(S, value)

def GetFeasibleB(target_abd, pars):
    # 根据目标丰度和 pars 生成一个满足可行性约束的 B 张量。
    # 这是在 R 中的复杂函数的替代：我们构造一个简单的 B，使得
    # 对每个物种 i, sum_jk B[i,j,k] * target_abd[j] * target_abd[k] = 0（示例）
    S = pars['S']
    B = np.zeros((S,S,S))
    # 简单构造：对角上放小值（示例）
    for i in range(S):
        B[i,i,i] = 0.01  # 你可以用更复杂的生成方式替代
    return B

# Dynamics: 给定当前 abundances x (长度 S) 和参数 dict pars 返回 dx/dt
def Dynamics(x, t, pars):
    # 典型形式（示意）: dx_i/dt = R_i + sum_j A[i,j]*x_j + sum_jk B[i,j,k]*x_j*x_k - D_i * x_i
    R = pars['R']
    D = pars['D']
    A = pars['A']
    B = pars['B']
    S = pars['S']
    dx = np.zeros(S)
    # pairwise term
    pair = A.dot(x)
    # higher-order term: for each i, sum_{j,k} B[i,j,k] x_j x_k
    hoi = np.zeros(S)
    if B is not None:
        # B is SxSxS
        for i in range(S):
            # compute x_j * x_k outer and sum with B[i]
            hoi[i] = np.sum(B[i] * np.outer(x, x))
    dx = R + pair + hoi - D * x
    return dx

def IntegrateDynamics(inistate, pars, endtime=50.0, timestep=0.1, fn=Dynamics):
    times = np.arange(0, endtime + 1e-12, timestep)
    # use odeint (which expects func(x, t, ...))
    sol = odeint(lambda y, t: fn(y, t, pars), inistate, times)
    df = pd.DataFrame(sol, columns=[f"sp{i+1}" for i in range(pars['S'])])
    df.insert(0, 'time', times)
    return df

# ------------------------
# 主脚本（对应你的 R 脚本逻辑）
# ------------------------

def main():
    np.random.seed(1)

    # parameters from your R script
    S = 10
    mu_R = 1
    sigma_R = 0
    mu_D = 1
    sigma_D = 0
    mu_A = 0.05
    sigma_A = 0
    rho_A = 0
    mu_B = 0.0
    sigma_B = 0.0
    intra = "None"
    self_reg = "Quadratic"
    scaling = False
    dist_B = "Normal"

    # build a single-row parameter frame (like crossing with one combination)
    in_pars_row = {
        'S': S,
        'MuR': mu_R,
        'SigmaR': sigma_R,
        'MuD': mu_D,
        'SigmaD': sigma_D,
        'MuA': mu_A,
        'SigmaA': sigma_A,
        'RhoA': rho_A,
        'MuB': mu_B,
        'SigmaB': sigma_B,
        'Intra': intra,
        'SelfReg': self_reg,
        'scaling': scaling,
        'DistB': dist_B
    }

    pars = BuildPars(in_pars_row)
    target_abd = GetTargetAbd(pars, value=0.1)
    ini_state = np.random.uniform(0, 0.05, size=S)
    # ini_state = target_abd + np.random.normal(0, 0.01, size=S)

    end_time = 50.0
    time_step = 0.1

    # integrating dynamics with only pairwise interactions
    # To simulate "No HOIs", set B to zero
    pars_nohoi = dict(pars)
    pars_nohoi['B'] = np.zeros_like(pars['B'])
    out_pw = IntegrateDynamics(inistate=ini_state, pars=pars_nohoi, endtime=end_time, timestep=time_step, fn=Dynamics)

    # melt to long format
    out_pw_long = out_pw.melt(id_vars='time', var_name='variable', value_name='value')
    out_pw_long['Type'] = 'No HOIs'

    # get feasible B and run with HOIs
    const_B = GetFeasibleB(target_abd, pars)
    pars_hoi = dict(pars)
    pars_hoi['B'] = const_B
    out_hoi = IntegrateDynamics(inistate=ini_state, pars=pars_hoi, endtime=end_time, timestep=time_step, fn=Dynamics)
    out_hoi_long = out_hoi.melt(id_vars='time', var_name='variable', value_name='value')
    out_hoi_long['Type'] = 'Constrained HOIs'

    series = pd.concat([out_pw_long, out_hoi_long], ignore_index=True)
    series['Type'] = pd.Categorical(series['Type'], categories=['No HOIs', 'Constrained HOIs'])

    # plotting similar to ggplot facet_wrap
    g = sns.FacetGrid(series, col="Type", sharey=True, height=5, aspect=1.6)
    def lineplot(data, **kwargs):
        ax = plt.gca()
        for var, grp in data.groupby('variable'):
            ax.plot(grp['time'], grp['value'], alpha=0.6, linewidth=1.5)
    g.map_dataframe(lineplot)
    g.set_axis_labels("Time", "Abundance")
    g.fig.suptitle("B", fontsize=16)
    plt.subplots_adjust(top=0.85)
    plt.show()

    # save figure
    g.fig.savefig("Fig1ConstraintExample.png", dpi=300)

if __name__ == "__main__":
    main()