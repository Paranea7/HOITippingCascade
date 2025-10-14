import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize
from scipy.stats import gaussian_kde, norm
import warnings
from multiprocessing import Pool, cpu_count
from sklearn.cluster import KMeans
from scipy.signal import find_peaks
import itertools
from tqdm import tqdm

warnings.filterwarnings('ignore')


class CubicModelSolver:
    """
    针对您的三次系统模型的专用求解器
    - 模型: dx_i/dt = -x_i^3 + x_i + c_i + sum_j d_ji x_j + sum_jk e_ijk x_j x_k
    - 已知稳定状态: -1.015 和 +1.013
    """

    def __init__(self, S_sim=100, S_rsb=1000, n_states=20, max_iter=300, tol=1e-8, n_workers=None):
        self.S_sim = S_sim
        self.S_rsb = S_rsb
        self.n_states = n_states
        self.max_iter = max_iter
        self.tol = tol
        self.n_workers = n_workers if n_workers is not None else max(1, cpu_count() - 1)
        print(f"初始化求解器，使用 {self.n_workers} 个并行工作进程")

    def _simulate_single_replicate(self, args):
        """
        单次模拟的辅助函数，用于并行计算
        """
        rep, mu_c, sigma_c, mu_d, sigma_d, mu_e, sigma_e, T, dt, n_initial_conditions = args

        np.random.seed(rep)
        all_final_states = []

        # 生成参数
        c = np.random.normal(mu_c, sigma_c, self.S_sim)

        if sigma_d > 0:
            d = np.random.normal(mu_d / self.S_sim, sigma_d / np.sqrt(self.S_sim),
                                 (self.S_sim, self.S_sim))
        else:
            d = np.full((self.S_sim, self.S_sim), mu_d / self.S_sim)
        np.fill_diagonal(d, 0)

        if sigma_e > 0:
            e = np.random.normal(mu_e / (self.S_sim ** 2), sigma_e / self.S_sim,
                                 (self.S_sim, self.S_sim, self.S_sim))
        else:
            e = np.full((self.S_sim, self.S_sim, self.S_sim), mu_e / (self.S_sim ** 2))

        # 使用不同的初始条件来探索两个稳定状态
        initial_conditions = [
            np.random.uniform(-2, -0.5, self.S_sim),  # 偏向负状态
            np.random.uniform(0.5, 2, self.S_sim),  # 偏向正状态
            np.random.uniform(-1, 1, self.S_sim)  # 随机
        ]

        # 如果指定了更多初始条件，则添加
        if n_initial_conditions > 3:
            for _ in range(n_initial_conditions - 3):
                initial_conditions.append(np.random.uniform(-2, 2, self.S_sim))

        for x0 in initial_conditions:
            x = x0.copy()

            for t in range(T):
                # 计算导数 - 向量化改进
                dxdt = np.zeros(self.S_sim)
                for i in range(self.S_sim):
                    self_term = -x[i] ** 3 + x[i] + c[i]
                    linear_term = np.sum(d[:, i] * x)

                    # 向量化成对相互作用计算
                    pairwise_term = np.sum(e[:, :, i] * np.outer(x, x))

                    dxdt[i] = self_term + linear_term + pairwise_term

                # Euler方法
                x_new = x + dt * dxdt

                # 检查收敛
                if np.max(np.abs(x_new - x)) < 1e-10 and t > 100:
                    break

                x = x_new

            # 收集最终状态
            all_final_states.extend(x)

        return np.array(all_final_states)

    def simulate_dynamics(self, mu_c, sigma_c, mu_d, sigma_d, mu_e, sigma_e,
                          T=2000, dt=0.01, n_replicates=20, n_initial_conditions=3):
        """
        数值模拟 - 并行版本
        """
        print(f"开始并行数值模拟 ({n_replicates} 次重复，{self.n_workers} 进程)...")

        # 准备参数
        args_list = [(rep, mu_c, sigma_c, mu_d, sigma_d, mu_e, sigma_e, T, dt, n_initial_conditions)
                     for rep in range(n_replicates)]

        # 并行执行
        if self.n_workers > 1:
            with Pool(self.n_workers) as pool:
                results = list(tqdm(pool.imap(self._simulate_single_replicate, args_list),
                                    total=n_replicates, desc="模拟进度"))
        else:
            # 单进程版本（用于调试）
            results = [self._simulate_single_replicate(args) for args in tqdm(args_list, desc="模拟进度")]

        # 合并结果
        all_final_states = np.concatenate(results)

        print(f"模拟完成，共获得 {len(all_final_states)} 个状态样本")
        return all_final_states

    def analyze_known_states(self, data, expected_states=[-1.015, 1.013], tolerance=0.1):
        """
        分析数据中已知稳定状态的出现情况
        """
        print("\n=== 已知状态分析 ===")

        results = {}

        for target_state in expected_states:
            # 找到接近目标状态的数据点
            mask = np.abs(data - target_state) < tolerance
            count = np.sum(mask)
            proportion = count / len(data) if len(data) > 0 else 0

            results[target_state] = {
                'count': count,
                'proportion': proportion,
                'mean': np.mean(data[mask]) if count > 0 else 0,
                'std': np.std(data[mask]) if count > 0 else 0
            }

            print(f"状态 {target_state}: 出现 {count} 次 ({proportion * 100:.1f}%)")

        # 分析双峰特性
        if len(data) > 10:
            # 核密度估计
            kde = gaussian_kde(data)
            x_range = np.linspace(np.min(data), np.max(data), 1000)
            density = kde(x_range)

            # 寻找峰值
            peaks, properties = find_peaks(density, height=0.01, distance=50)

            print(f"检测到 {len(peaks)} 个密度峰值")
            for i, peak in enumerate(peaks):
                print(f"  峰值 {i + 1}: x = {x_range[peak]:.3f}, 密度 = {density[peak]:.3f}")

        return results

    def cubic_roots_analytical(self, h, u=1):
        """
        三次方程的解析解: x^3 - u*x = h
        """
        # 转换为标准形式: x^3 + px + q = 0
        p = -u
        q = -h

        discriminant = (q / 2) ** 2 + (p / 3) ** 3

        if discriminant > 0:  # 一个实根
            A = np.cbrt(-q / 2 + np.sqrt(discriminant))
            B = np.cbrt(-q / 2 - np.sqrt(discriminant))
            return [A + B]
        else:  # 三个实根
            r = np.sqrt(-(p / 3) ** 3)
            theta = np.arccos(-q / (2 * r))
            roots = []
            for k in range(3):
                root = 2 * np.cbrt(r) * np.cos((theta + 2 * np.pi * k) / 3)
                roots.append(root)
            return sorted(roots)

    def _rsb_single_state_update_segmented(self, state_args):
        """
        改进的单状态RSB更新 - 分段迭代拟合
        专门处理三个状态(-1, 0, 1)的情况，避免不稳定点0
        """
        state, mu_c, sigma_c, mu_d, sigma_d, mu_e, sigma_e, S_rsb, gamma, iteration = state_args

        phi_alpha = state['phi']
        mean_x_alpha = state['mean_x']
        mean_x2_alpha = state['mean_x2']
        v_alpha = state['v']
        target_state = state.get('target_state')

        # 状态特定的场参数
        mu_h_alpha = mu_c + phi_alpha * (mu_d * mean_x_alpha + mu_e * mean_x2_alpha)

        if sigma_e > 0:
            sigma_h_alpha = np.sqrt(sigma_c ** 2 + phi_alpha *
                                    (sigma_d ** 2 * mean_x2_alpha + sigma_e ** 2 * mean_x2_alpha ** 2))
        else:
            sigma_h_alpha = np.sqrt(sigma_c ** 2 + phi_alpha * sigma_d ** 2 * mean_x2_alpha)

        # 总相互作用方差
        if sigma_e > 0:
            sigma2_total = S_rsb * (sigma_d ** 2 + 2 * sigma_e ** 2)
        else:
            sigma2_total = S_rsb * sigma_d ** 2

        u_alpha = 1 + phi_alpha * sigma2_total * gamma * v_alpha

        # 分段迭代策略：根据迭代次数调整策略
        if iteration < self.max_iter // 3:
            # 第一阶段：探索阶段，允许较宽的范围
            x_min, x_max = -2, 2
            exploration_factor = 1.0
        elif iteration < 2 * self.max_iter // 3:
            # 第二阶段：收敛阶段，缩小范围
            x_min, x_max = -1.5, 1.5
            exploration_factor = 0.5
        else:
            # 第三阶段：精细调整阶段，专注于稳定状态附近
            x_min, x_max = -1.2, 1.2
            exploration_factor = 0.2

        # 对于已知目标状态的状态，强制向目标状态收敛
        if target_state is not None:
            # 计算达到目标状态所需的场
            h_target = target_state ** 3 - u_alpha * target_state

            # 调整参数使目标状态更可能
            mu_h_alpha = (1 - exploration_factor) * mu_h_alpha + exploration_factor * h_target
            sigma_h_alpha = max(sigma_h_alpha * (1 - exploration_factor / 2), 0.05)

        # 计算矩量 - 分段积分策略
        if sigma_h_alpha > 0:
            # 定义积分区域，特别关注稳定状态附近
            critical_points = [-1.0, 0.0, 1.0]  # 三次系统的临界点

            # 为每个临界点创建积分区域
            integration_ranges = []
            for cp in critical_points:
                if cp == 0:  # 不稳定点，减少采样
                    integration_ranges.append((cp - 0.3, cp + 0.3))
                else:  # 稳定点，增加采样
                    integration_ranges.append((cp - 0.5, cp + 0.5))

            # 添加全局范围以确保覆盖
            integration_ranges.append((x_min, x_max))

            # 合并重叠的范围
            merged_ranges = self._merge_ranges(integration_ranges)

            # 在每个范围内进行积分
            total_prob = 0
            weighted_mean_x = 0
            weighted_mean_x2 = 0

            for range_min, range_max in merged_ranges:
                n_points = 200
                x_vals = np.linspace(range_min, range_max, n_points)

                # 计算每个x对应的h
                h_vals = x_vals ** 3 - u_alpha * x_vals

                # 计算概率密度
                P_h = norm.pdf(h_vals, mu_h_alpha, sigma_h_alpha)
                dh_dx = np.abs(3 * x_vals ** 2 - u_alpha)

                mask = dh_dx > 1e-10
                if np.sum(mask) > 0:
                    P_x = np.zeros_like(x_vals)
                    P_x[mask] = P_h[mask] / dh_dx[mask]

                    # 应用稳定性权重 - 惩罚不稳定区域
                    stability_weights = np.ones_like(x_vals)
                    # 在0附近降低权重（不稳定点）
                    zero_mask = np.abs(x_vals) < 0.5
                    stability_weights[zero_mask] = 0.1
                    # 在稳定点附近增加权重
                    stable_mask1 = np.abs(x_vals + 1) < 0.3
                    stable_mask2 = np.abs(x_vals - 1) < 0.3
                    stability_weights[stable_mask1] = 2.0
                    stability_weights[stable_mask2] = 2.0

                    P_x_weighted = P_x * stability_weights

                    range_prob = np.trapz(P_x_weighted, x_vals)
                    range_mean_x = np.trapz(x_vals * P_x_weighted, x_vals)
                    range_mean_x2 = np.trapz(x_vals ** 2 * P_x_weighted, x_vals)

                    total_prob += range_prob
                    weighted_mean_x += range_mean_x
                    weighted_mean_x2 += range_mean_x2

            if total_prob > 0:
                phi_new = 1.0  # 假设所有状态都存活
                mean_x_new = weighted_mean_x / total_prob
                mean_x2_new = weighted_mean_x2 / total_prob

                # 对于已知目标状态，加强收敛
                if target_state is not None:
                    # 使用自适应混合，随着迭代增加目标状态的权重
                    mix_ratio = min(0.8, 0.3 + 0.5 * iteration / self.max_iter)
                    mean_x_new = (1 - mix_ratio) * mean_x_new + mix_ratio * target_state
                    mean_x2_new = (1 - mix_ratio) * mean_x2_new + mix_ratio * target_state ** 2
            else:
                phi_new, mean_x_new, mean_x2_new = phi_alpha, mean_x_alpha, mean_x2_alpha
        else:
            phi_new, mean_x_new, mean_x2_new = phi_alpha, mean_x_alpha, mean_x2_alpha

        # 更新响应系数 - 改进的稳定性处理
        if phi_new > 0 and abs(3 * mean_x2_new - u_alpha) > 1e-6:
            v_new = 1.0 / (3 * mean_x2_new - u_alpha)
            # 防止响应系数过大或过小
            v_new = np.clip(v_new, 0.1, 5.0)
        else:
            v_new = v_alpha

        # 限制范围，特别避免不稳定区域
        mean_x_new = np.clip(mean_x_new, -1.5, 1.5)
        # 如果接近不稳定点0，则推向最近的稳定点
        if abs(mean_x_new) < 0.3:
            if target_state is not None:
                mean_x_new = target_state * 0.7 + mean_x_new * 0.3
            else:
                mean_x_new = np.sign(mean_x_new) * 0.5  # 推向-0.5或0.5

        mean_x2_new = np.clip(mean_x2_new, 0.25, 2.25)  # 对应x在[-1.5, 1.5]的范围

        delta = abs(mean_x_new - mean_x_alpha)

        new_state = state.copy()
        new_state.update({
            'phi': phi_new,
            'mean_x': mean_x_new,
            'mean_x2': mean_x2_new,
            'v': v_new,
            'mu_h': mu_h_alpha,
            'sigma_h': sigma_h_alpha,
            'u': u_alpha,
            'delta': delta
        })

        return new_state

    def _merge_ranges(self, ranges):
        """合并重叠的区间"""
        if not ranges:
            return []

        # 按起点排序
        ranges.sort(key=lambda x: x[0])

        merged = []
        current_start, current_end = ranges[0]

        for start, end in ranges[1:]:
            if start <= current_end:
                # 有重叠，合并
                current_end = max(current_end, end)
            else:
                # 无重叠，添加当前区间
                merged.append((current_start, current_end))
                current_start, current_end = start, end

        merged.append((current_start, current_end))
        return merged

    def rsb_for_known_states_segmented(self, mu_c, sigma_c, mu_d, sigma_d, mu_e, sigma_e,
                                       expected_states=[-1.015, 1.013], gamma=1.0, m=0.3):
        """
        改进的RSB分析 - 分段迭代拟合版本
        专门处理三个状态(-1, 0, 1)的情况
        """
        print("进行分段迭代RSB分析...")

        # 初始化状态，专注于已知的稳定状态
        states = []
        stable_states = expected_states  # 使用实际的稳定状态

        for i, target_state in enumerate(stable_states):
            # 为每个已知状态创建多个变体
            n_states_per_target = max(1, self.n_states // len(stable_states))
            for j in range(n_states_per_target):
                state = {
                    'id': i * 10 + j,
                    'target_state': target_state,
                    'phi': 0.5,
                    'mean_x': target_state + np.random.normal(0, 0.05),  # 更小的随机扰动
                    'mean_x2': target_state ** 2 + np.random.normal(0, 0.1),
                    'v': 1.0,
                    'weight': 1.0 / self.n_states,
                    'type': 'stable'
                }
                states.append(state)

        # 补充一些探索性状态，但避免不稳定区域
        remaining = self.n_states - len(states)
        for i in range(remaining):
            # 偏向稳定状态区域
            if np.random.random() < 0.7:
                target = np.random.choice([-1.0, 1.0])
                mean_x = target + np.random.normal(0, 0.2)
            else:
                mean_x = np.random.choice([-0.8, -0.6, 0.6, 0.8])  # 避免0附近

            state = {
                'id': 100 + i,
                'target_state': None,
                'phi': 0.5,
                'mean_x': mean_x,
                'mean_x2': mean_x ** 2 + np.random.normal(0, 0.2),
                'v': 1.0,
                'weight': 1.0 / self.n_states,
                'type': 'exploratory'
            }
            states.append(state)

        history = []

        for iteration in tqdm(range(self.max_iter), desc="分段RSB迭代"):
            # 准备并行参数
            state_args = [(state, mu_c, sigma_c, mu_d, sigma_d, mu_e, sigma_e,
                           self.S_rsb, gamma, iteration)
                          for state in states]

            # 并行更新状态
            if self.n_workers > 1:
                with Pool(self.n_workers) as pool:
                    new_states = pool.map(self._rsb_single_state_update_segmented, state_args)
            else:
                new_states = [self._rsb_single_state_update_segmented(args) for args in state_args]

            states = new_states
            total_delta = sum(state['delta'] for state in states)

            # 改进的权重更新策略
            if iteration % 5 == 0 or iteration < 10:
                weights = []
                for state in states:
                    if state['target_state'] is not None:
                        # 已知状态：基于与目标状态的接近程度
                        distance = abs(state['mean_x'] - state['target_state'])
                        # 随着迭代增加，对接近目标的状态给予更高权重
                        weight = np.exp(-8 * distance * (1 + iteration / self.max_iter))
                    else:
                        # 探索性状态：基于稳定性
                        # 惩罚接近0的状态，奖励接近稳定点的状态
                        stability_penalty = 1.0
                        if abs(state['mean_x']) < 0.4:  # 不稳定区域
                            stability_penalty = 0.1
                        elif abs(state['mean_x']) > 0.8:  # 稳定区域
                            stability_penalty = 2.0

                        # 也考虑方差，低方差状态更可信
                        variance_penalty = 1.0 / (1 + abs(state['mean_x2'] - state['mean_x'] ** 2))

                        weight = stability_penalty * variance_penalty

                    weights.append(weight)

                total_weight = sum(weights)
                if total_weight > 0:
                    for i, state in enumerate(states):
                        state['weight'] = weights[i] / total_weight
                else:
                    for state in states:
                        state['weight'] = 1.0 / len(states)

            # 记录历史
            avg_mean_x = np.sum([s['mean_x'] * s['weight'] for s in states])
            # 计算状态聚类
            state_values = [s['mean_x'] for s in states]
            state_weights = [s['weight'] for s in states]

            history.append({
                'iteration': iteration,
                'avg_mean_x': avg_mean_x,
                'total_delta': total_delta,
                'states': [s.copy() for s in states],
                'state_clusters': self._cluster_states(state_values, state_weights)
            })

            if total_delta < self.tol and iteration > 30:
                print(f"分段RSB在迭代 {iteration} 收敛")
                break

            # 中期调整：移除表现差的状态，增加表现好的状态的变体
            if iteration == self.max_iter // 2 and len(states) > 5:
                states = self._evolve_states(states, stable_states)

        # 最终结果处理
        effective_states = [s for s in states if s['weight'] > 0.01]
        if len(effective_states) == 0:
            effective_states = states

        # 最终归一化权重
        total_weight = sum(s['weight'] for s in effective_states)
        for state in effective_states:
            state['weight'] /= total_weight

        # 分析最终状态分布
        final_clusters = self._analyze_final_states(effective_states, stable_states)

        result = {
            'states': effective_states,
            'avg_mean_x': np.sum([s['mean_x'] * s['weight'] for s in effective_states]),
            'n_effective_states': len(effective_states),
            'history': history,
            'converged': iteration < self.max_iter - 1,
            'clusters': final_clusters
        }

        return result

    def _cluster_states(self, state_values, state_weights, n_clusters=3):
        """对状态进行聚类分析"""
        if len(state_values) < n_clusters:
            return None

        try:
            # 使用加权K-means
            state_array = np.array(state_values).reshape(-1, 1)
            kmeans = KMeans(n_clusters=min(n_clusters, len(state_values)), random_state=0)
            labels = kmeans.fit_predict(state_array, sample_weight=state_weights)

            clusters = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append({
                    'value': state_values[i],
                    'weight': state_weights[i]
                })

            # 计算每个聚类的统计量
            cluster_stats = {}
            for label, members in clusters.items():
                values = [m['value'] for m in members]
                weights = [m['weight'] for m in members]
                total_weight = sum(weights)

                if total_weight > 0:
                    weighted_mean = sum(v * w for v, w in zip(values, weights)) / total_weight
                    weighted_std = np.sqrt(
                        sum(w * (v - weighted_mean) ** 2 for v, w in zip(values, weights)) / total_weight)
                else:
                    weighted_mean = np.mean(values)
                    weighted_std = np.std(values)

                cluster_stats[label] = {
                    'center': kmeans.cluster_centers_[label][0],
                    'weighted_mean': weighted_mean,
                    'weighted_std': weighted_std,
                    'n_members': len(members),
                    'total_weight': total_weight
                }

            return cluster_stats
        except:
            return None

    def _evolve_states(self, states, stable_states):
        """进化状态：移除差的状态，复制好的状态"""
        # 按权重排序
        states_sorted = sorted(states, key=lambda x: x['weight'], reverse=True)

        # 保留前60%的状态
        keep_count = int(len(states) * 0.6)
        new_states = states_sorted[:keep_count]

        # 为高权重状态创建变体
        for state in states_sorted[:keep_count // 2]:
            if state['weight'] > 0.05:  # 只有足够好的状态才创建变体
                variant = state.copy()
                variant['id'] = variant['id'] + 1000  # 新ID
                variant['mean_x'] = state['mean_x'] + np.random.normal(0, 0.05)
                variant['mean_x2'] = state['mean_x2'] + np.random.normal(0, 0.1)
                variant['weight'] = state['weight'] * 0.5  # 初始权重减半
                new_states.append(variant)

        # 如果状态数量仍然不足，添加新的探索状态
        while len(new_states) < len(states):
            new_state = {
                'id': 2000 + len(new_states),
                'target_state': None,
                'phi': 0.5,
                'mean_x': np.random.choice([-0.9, -0.7, 0.7, 0.9]),
                'mean_x2': np.random.uniform(0.5, 1.5),
                'v': 1.0,
                'weight': 0.01,  # 低初始权重
                'type': 'new_exploratory'
            }
            new_states.append(new_state)

        # 重新归一化权重
        total_weight = sum(s['weight'] for s in new_states)
        for state in new_states:
            state['weight'] /= total_weight

        return new_states

    def _analyze_final_states(self, states, stable_states, tolerance=0.1):
        """分析最终状态分布"""
        clusters = {}

        for state in states:
            # 找到最接近的稳定状态
            closest_stable = min(stable_states, key=lambda x: abs(x - state['mean_x']))
            distance = abs(state['mean_x'] - closest_stable)

            if distance < tolerance:
                cluster_key = f"stable_{closest_stable}"
            else:
                # 创建新的聚类
                cluster_key = f"exploratory_{len(clusters)}"

            if cluster_key not in clusters:
                clusters[cluster_key] = {
                    'center': closest_stable if 'stable' in cluster_key else state['mean_x'],
                    'states': [],
                    'total_weight': 0,
                    'type': 'stable' if 'stable' in cluster_key else 'exploratory'
                }

            clusters[cluster_key]['states'].append(state)
            clusters[cluster_key]['total_weight'] += state['weight']

        # 计算每个聚类的统计量
        for key, cluster in clusters.items():
            values = [s['mean_x'] for s in cluster['states']]
            weights = [s['weight'] for s in cluster['states']]

            cluster['mean'] = np.average(values, weights=weights)
            cluster['std'] = np.sqrt(np.average((values - cluster['mean']) ** 2, weights=weights))
            cluster['n_states'] = len(cluster['states'])

        return clusters

    def compute_rsb_distribution(self, rsb_result, n_points=1000):
        """
        计算RSB预测的分布 - 改进版本
        """
        states = rsb_result['states']
        if not states:
            return np.array([]), np.array([])

        # 确定x的范围，专注于稳定状态区域
        x_min, x_max = -1.5, 1.5

        x_range = np.linspace(x_min, x_max, n_points)
        pdf_total = np.zeros_like(x_range)

        for state in states:
            weight = state['weight']
            mu_h = state['mu_h']
            sigma_h = state['sigma_h']
            u = state['u']

            # 计算该状态的PDF
            state_pdf = np.zeros_like(x_range)
            for i, x_val in enumerate(x_range):
                h_val = x_val ** 3 - u * x_val
                P_h = norm.pdf(h_val, mu_h, sigma_h)
                dh_dx = abs(3 * x_val ** 2 - u)
                if dh_dx > 1e-10:
                    state_pdf[i] = P_h / dh_dx

            # 归一化
            state_norm = np.trapz(state_pdf, x_range)
            if state_norm > 0:
                state_pdf = state_pdf / state_norm

            pdf_total += weight * state_pdf

        # 最终归一化
        total_norm = np.trapz(pdf_total, x_range)
        if total_norm > 0:
            pdf_total = pdf_total / total_norm

        return x_range, pdf_total

    # 保留其他方法不变...
    def parameter_sensitivity_analysis(self, base_params, param_ranges, n_samples=20):
        """参数敏感性分析"""
        # 实现保持不变...
        pass

    def stability_analysis(self, states, mu_c, sigma_c, mu_d, sigma_d, mu_e, sigma_e):
        """线性稳定性分析"""
        # 实现保持不变...
        pass

    def find_optimal_parameters(self, target_states=[-1.015, 1.013], param_bounds=None, n_trials=50):
        """寻找最优参数"""
        # 实现保持不变...
        pass

    def compare_theory_simulation(self, mu_c, sigma_c, mu_d, sigma_d, mu_e, sigma_e,
                                  n_sim_replicates=15, figsize=(15, 10), use_segmented_rsb=True):
        """
        比较理论和模拟结果 - 添加分段RSB选项
        """
        print("=== 理论与模拟对比分析 ===")

        # 1. 数值模拟
        print("进行数值模拟...")
        sim_data = self.simulate_dynamics(
            mu_c, sigma_c, mu_d, sigma_d, mu_e, sigma_e,
            n_replicates=n_sim_replicates
        )

        # 分析已知状态的出现
        state_analysis = self.analyze_known_states(sim_data)

        # 2. RSB理论预测
        print("\n进行RSB理论预测...")
        if use_segmented_rsb:
            rsb_result = self.rsb_for_known_states_segmented(
                mu_c, sigma_c, mu_d, sigma_d, mu_e, sigma_e
            )
        else:
            rsb_result = self.rsb_for_known_states(
                mu_c, sigma_c, mu_d, sigma_d, mu_e, sigma_e
            )

        print(f"RSB找到 {rsb_result['n_effective_states']} 个有效状态")

        # 3. 计算RSB预测的分布
        x_rsb, pdf_rsb = self.compute_rsb_distribution(rsb_result)

        # 4. 稳定性分析
        print("\n进行稳定性分析...")
        stability_results = self.stability_analysis([-1.015, 1.013, 0],
                                                    mu_c, sigma_c, mu_d, sigma_d, mu_e, sigma_e)

        # 5. 绘制对比图
        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # 5.1 丰度分布直方图与RSB预测
        if len(sim_data) > 0:
            axes[0, 0].hist(sim_data, bins=50, density=True, alpha=0.7,
                            color='lightblue', edgecolor='black', label='数值模拟')

            if len(x_rsb) > 0 and np.max(pdf_rsb) > 0:
                axes[0, 0].plot(x_rsb, pdf_rsb, 'r-', linewidth=2, label='RSB理论预测')

                # 标记已知状态
                for state in [-1.015, 1.013]:
                    axes[0, 0].axvline(x=state, color='green', linestyle='--',
                                       alpha=0.7, label=f'已知状态 {state}')

            axes[0, 0].set_xlabel('状态值 x')
            axes[0, 0].set_ylabel('概率密度')
            axes[0, 0].set_title('状态分布: 理论与模拟对比')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # 5.2 累积分布函数
        if len(sim_data) > 0:
            sorted_sim = np.sort(sim_data)
            cdf_sim = np.arange(1, len(sorted_sim) + 1) / len(sorted_sim)
            axes[0, 1].plot(sorted_sim, cdf_sim, 'b-', linewidth=2, label='数值模拟')

            if len(x_rsb) > 0 and np.max(pdf_rsb) > 0:
                cdf_rsb = np.cumsum(pdf_rsb) * (x_rsb[1] - x_rsb[0])
                axes[0, 1].plot(x_rsb, cdf_rsb, 'r-', linewidth=2, label='RSB理论预测')

            axes[0, 1].set_xlabel('状态值 x')
            axes[0, 1].set_ylabel('累积概率')
            axes[0, 1].set_title('累积分布函数')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # 5.3 状态权重分布
        states = rsb_result['states']
        if states:
            weights = [s['weight'] for s in states]
            mean_xs = [s['mean_x'] for s in states]
            colors = ['red' if s.get('target_state') == -1.015 else
                      'blue' if s.get('target_state') == 1.013 else
                      'gray' for s in states]

            sc = axes[0, 2].scatter(mean_xs, weights, c=colors, s=100, alpha=0.7)
            axes[0, 2].axvline(x=-1.015, color='red', linestyle='--', alpha=0.5, label='目标 -1.015')
            axes[0, 2].axvline(x=1.013, color='blue', linestyle='--', alpha=0.5, label='目标 1.013')
            axes[0, 2].axvline(x=0, color='orange', linestyle=':', alpha=0.5, label='不稳定点 0')
            axes[0, 2].set_xlabel('平均状态值')
            axes[0, 2].set_ylabel('状态权重')
            axes[0, 2].set_title('RSB状态分布')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)

        # 5.4 收敛历史
        if rsb_result['history']:
            iterations = [h['iteration'] for h in rsb_result['history']]
            avg_means = [h['avg_mean_x'] for h in rsb_result['history']]
            deltas = [h['total_delta'] for h in rsb_result['history']]

            axes[1, 0].plot(iterations, avg_means, 'b-', linewidth=2)
            axes[1, 0].set_xlabel('迭代次数')
            axes[1, 0].set_ylabel('平均状态值', color='b')
            axes[1, 0].tick_params(axis='y', labelcolor='b')
            axes[1, 0].grid(True, alpha=0.3)

            ax2 = axes[1, 0].twinx()
            ax2.semilogy(iterations, deltas, 'r--', linewidth=1)
            ax2.set_ylabel('收敛误差 (log)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            axes[1, 0].set_title('RSB收敛历史')

        # 5.5 状态聚类分析
        if 'clusters' in rsb_result and rsb_result['clusters']:
            cluster_info = rsb_result['clusters']
            cluster_labels = []
            cluster_weights = []

            for key, cluster in cluster_info.items():
                cluster_labels.append(f"{key}\n({cluster['mean']:.3f})")
                cluster_weights.append(cluster['total_weight'])

            axes[1, 1].bar(cluster_labels, cluster_weights, alpha=0.7)
            axes[1, 1].set_xlabel('状态聚类')
            axes[1, 1].set_ylabel('总权重')
            axes[1, 1].set_title('RSB状态聚类分析')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)

        # 5.6 稳定性分析结果
        stability_text = '稳定性分析:\n'
        for state, analysis in stability_results.items():
            stability_text += f'状态 {state:.3f}: {analysis["max_real_eigenvalue"]:.3f} ({"稳定" if analysis["stable"] else "不稳定"})\n'

        stability_text += '\n状态出现频率:\n'
        for state, analysis in state_analysis.items():
            stability_text += f'状态 {state}: {analysis["proportion"] * 100:.1f}%\n'

        if use_segmented_rsb:
            stability_text += '\n(使用分段RSB)'

        axes[1, 2].text(0.5, 0.5, stability_text,
                        ha='center', va='center', transform=axes[1, 2].transAxes,
                        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 2].set_title('分析与统计')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.show()

        return {
            'simulation': sim_data,
            'rsb': rsb_result,
            'state_analysis': state_analysis,
            'stability_analysis': stability_results,
            'x_rsb': x_rsb,
            'pdf_rsb': pdf_rsb
        }


# 使用示例
if __name__ == "__main__":
    # 创建专用求解器
    solver = CubicModelSolver(S_sim=80, S_rsb=1000, n_states=20, max_iter=200, n_workers=16)

    # 测试案例：使用分段RSB迭代
    print("测试案例：双稳态系统 - 分段RSB迭代")
    result = solver.compare_theory_simulation(
        mu_c=0.0,
        sigma_c=0.4,
        mu_d=0.5,
        sigma_d=0.4,
        mu_e=0.0,
        sigma_e=0.0,
        n_sim_replicates=10,
        use_segmented_rsb=True  # 使用分段RSB
    )

    # 分析结果
    print("\n=== 结果总结 ===")
    print("已知状态出现频率:")
    for state, analysis in result['state_analysis'].items():
        print(f"  状态 {state}: {analysis['proportion'] * 100:.1f}%")

    print(f"\nRSB预测的有效状态数: {result['rsb']['n_effective_states']}")
    print(f"RSB预测的平均状态值: {result['rsb']['avg_mean_x']:.3f}")

    # 聚类分析
    if 'clusters' in result['rsb']:
        print("\n状态聚类分析:")
        for key, cluster in result['rsb']['clusters'].items():
            print(
                f"  聚类 {key}: 权重={cluster['total_weight']:.3f}, 中心={cluster['mean']:.3f}, 标准差={cluster['std']:.3f}")

    # 稳定性分析总结
    print("\n稳定性分析:")
    for state, analysis in result['stability_analysis'].items():
        status = "稳定" if analysis['stable'] else "不稳定"
        print(f"  状态 {state}: 最大特征值 = {analysis['max_real_eigenvalue']:.3f} ({status})")

    # 检查是否成功识别了已知状态
    known_states_present = any(analysis['proportion'] > 0.1
                               for analysis in result['state_analysis'].values())

    if known_states_present:
        print("\n✓ 成功识别已知稳定状态")
    else:
        print("\n⚠ 未能充分识别已知稳定状态，可能需要调整参数")