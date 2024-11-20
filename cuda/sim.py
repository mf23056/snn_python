import torch
import matplotlib.pyplot as plt
import math
import random
import numpy as np

class LIF:
    def __init__(self, dt=0.01, tau_m=20, V_rest=-65.0, R=10.0, I_back=0.0, V_reset=-65.0, V_th=-50.0, V_ref=3):
        '''
        param dt: 刻み幅 [ms]
        param tau_m: 膜時間定数 [ms]
        param V_rest: 静止電位 [mV]
        param R: 抵抗 [MΩ]
        param I_back: バックグラウンド電流 [nA]
        param V_reset: リセット電位 [mV]
        param V_th: 閾値電位 [mV]
        param V_ref: 不応期 [ms]
        '''
        self.dt = dt
        self.tau_m = tau_m
        self.V_rest = V_rest
        self.R = R
        self.I_back = I_back
        self.V_reset = V_reset
        self.V_th = V_th
        self.V_ref_steps = int(V_ref / dt)

    def __call__(self, I_syn, before_V, ref_time):
        ref_time = torch.maximum(ref_time - 1, torch.zeros_like(ref_time))
        V = torch.where(ref_time > 0, torch.full_like(before_V, self.V_reset),
                        before_V + self.dt * ((1 / self.tau_m) * (-(before_V - self.V_rest) + self.R * (I_syn + self.I_back))))
        spiked = (V >= self.V_th).float()
        ref_time = torch.where(spiked > 0, self.V_ref_steps, ref_time)
        V = torch.where(spiked > 0, torch.full_like(V, self.V_reset), V)
        return spiked, V, ref_time


class StaticSynapse:
    def __init__(self, dt=0.01, tau_syn=25):
        self.dt = dt
        self.tau_syn = tau_syn

    def __call__(self, bin_spike, W, before_I):
        # bin_spike は1か0のバイナリスパイク列
        spikes = bin_spike.unsqueeze(1)  # 各スパイクをテンソルに合わせて繰り返す
        return before_I + self.dt * (-before_I / self.tau_syn) + W * spikes


class Guetig_STDP:
    def __init__(self, dt=0.01, A_plus=0.01, A_minus=0.01, tau_plus=30.0, tau_minus=30.0, alpha=0.999, device='cuda'):
        self.dt = dt
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.alpha = alpha
        self.device = device

    def __call__(self, delta_t):
        delta_w = torch.where(
            delta_t > 0,
            self.A_plus
            * (torch.exp(-delta_t / self.tau_plus) - self.alpha * torch.exp(-delta_t / (self.alpha * self.tau_plus))),
            
            -self.A_minus
            * (torch.exp(delta_t / self.tau_minus) - self.alpha * torch.exp(delta_t / (self.alpha * self.tau_minus))),
        )
        
        # 主対角線はマスキング
        return delta_w.fill_diagonal_(0)


class SNN:
    def __init__(self, n_exc, n_inh, dt=0.01, device='cuda'):
        self.n_exc = n_exc
        self.n_inh = n_inh
        self.n_total = n_exc + n_inh
        self.dt = dt
        self.device = device

        self.neuron = LIF()
        self.synapse = StaticSynapse()
        self.stdp = Guetig_STDP(device=self.device)

        self._initialize_neurons(n_exc, n_inh)
        self._initialize_synapses()

        # 入力ログ用
        self.exc_input_log = []
        self.inh_input_log = []
        self.ei_diff_log = []

    def _initialize_neurons(self, n_exc, n_inh):
        self.positions = torch.rand((self.n_total, 3), device=self.device)
        self.sum_I_syn = torch.zeros(self.n_total, device=self.device)
        self.before_V = torch.full((self.n_total,), -65.0, device=self.device)
        self.ref_time = torch.zeros(self.n_total, device=self.device)
        self.spike_state = torch.ones(self.n_total, device=self.device)
        self.spike_times = -torch.ones(self.n_total, device=self.device)

        neuron_types = ['exc'] * n_exc + ['inh'] * n_inh
        self.neuron_types = neuron_types

    def _initialize_synapses(self):
        C = {"EE": 0.16, "EI": 0.25, "IE": 0.38, "II": 0.1}
        self.before_I = torch.zeros((self.n_total, self.n_total), device=self.device)
        distances = torch.cdist(self.positions, self.positions)
        self.weight_bias = torch.zeros((self.n_total, self.n_total), device=self.device)
        factor = torch.zeros((self.n_total, self.n_total), device=self.device)

        for i, pre_type in enumerate(self.neuron_types):
            for j, post_type in enumerate(self.neuron_types):
                if pre_type == 'exc' and post_type == 'exc':
                    factor[i, j] = C['EE']
                    self.weight_bias[i, j] = 1
                elif pre_type == 'exc' and post_type == 'inh':
                    factor[i, j] = C['EI']
                    self.weight_bias[i, j] = 1
                elif pre_type == 'inh' and post_type == 'exc':
                    factor[i, j] = C['IE']
                    self.weight_bias[i, j] = -1
                elif pre_type == 'inh' and post_type == 'inh':
                    factor[i, j] = C['II']
                    self.weight_bias[i, j] = -1

        synapse_prob = factor * torch.exp(-distances ** 2 / (1 / (factor * math.sqrt(math.pi))) ** 2)
        random_vals = torch.rand_like(synapse_prob)
        self.weights = (synapse_prob > random_vals).float() * self.weight_bias * random_vals
        self.weights.fill_diagonal_(0)

    def run_simulation(self, T=1000):
        before_spike_time = torch.zeros(self.n_total, device=self.device)
        now_spike_time = torch.zeros(self.n_total, device=self.device)
        spike_record = torch.zeros((self.n_total, int(T / self.dt)), device=self.device)
        
        # ログ用テンソルを初期化
        num_steps = int(T / self.dt)
        self.exc_input_log = torch.zeros(num_steps, device=self.device)
        self.inh_input_log = torch.zeros(num_steps, device=self.device)
        self.ei_diff_log = torch.zeros(num_steps, device=self.device)

        for t in range(1, int(T / self.dt)):
            self.before_I = self.synapse(self.spike_state, self.weights, self.before_I)
            self.sum_I_syn = torch.sum(self.before_I, dim=0)
            self.spike_state, self.before_V, self.ref_time = self.neuron(self.sum_I_syn, self.before_V, self.ref_time)

            # STDPの処理（省略なし）
            if torch.any(self.spike_state == 1):
                now_spike_time[self.spike_state == 1] = t * self.dt
                delta_t = now_spike_time.unsqueeze(0) - before_spike_time.unsqueeze(1)
                delta_w = self.stdp(delta_t)
                delta_w.fill_diagonal_(0)
                delta_w = self.weight_bias * delta_w
                delta_w[:, self.spike_state != 1] = 0
                self.weights += delta_w
                self.weights[:self.n_exc, :] = torch.maximum(self.weights[:self.n_exc, :], torch.zeros_like(self.weights[:self.n_exc, :], device=self.device))
                self.weights[self.n_exc:, :] = torch.minimum(self.weights[self.n_exc:, :], torch.zeros_like(self.weights[self.n_exc:, :], device=self.device))
                before_spike_time = now_spike_time

            
            # Excitatory (興奮性) と Inhibitory (抑制性) の入力を分けてログ
            exc_input = torch.sum(self.sum_I_syn[:self.n_exc])
            inh_input = torch.sum(self.sum_I_syn[self.n_exc:])
            ei_diff = exc_input + inh_input  # E - I の差を計算

            # GPU 上でログに保存
            self.exc_input_log[t] = exc_input
            self.inh_input_log[t] = inh_input
            self.ei_diff_log[t] = ei_diff

            spike_record[:, t] = self.spike_state
        return spike_record

    def plot_raster(self, spike_record):
        spike_times = torch.nonzero(spike_record, as_tuple=False)
        plt.figure(figsize=(12, 8))
        plt.scatter(spike_times[:, 1].cpu() * self.dt, spike_times[:, 0].cpu(), marker="|", color="black", s=10)
        plt.xlabel("Time (ms)")
        plt.ylabel("Neuron Index")
        plt.title("Spike Raster Plot")
        plt.savefig("E_I_input.png", dpi=300)
        plt.show()

    def plot_inputs(self):
        # ログを CPU に転送
        exc_input_log = self.exc_input_log.cpu().numpy()
        inh_input_log = self.inh_input_log.cpu().numpy()
        ei_diff_log = self.ei_diff_log.cpu().numpy()

        # プロット
        plt.figure(figsize=(12, 8))
        time = np.arange(len(exc_input_log)) * self.dt
        plt.plot(time, exc_input_log, label='Excitatory Input', color='b')
        plt.plot(time, inh_input_log, label='Inhibitory Input', color='r')
        plt.plot(time, ei_diff_log, label='E-I Input Difference', color='k')
        plt.xlabel('Time (ms)')
        plt.ylabel('Input Sum')
        plt.title('Excitatory and Inhibitory Input Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig("E_I_input.png", dpi=300)
        plt.show()


if __name__ == '__main__':
    torch.manual_seed(42)
    random.seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    network = SNN(n_exc=1000, n_inh=250, device=device)
    T = 2000  # ms
    spike_record = network.run_simulation(T)
    network.plot_raster(spike_record)
    network.plot_inputs()



    
# '''
# parameter
# '''
# N_E = 1000
# N_I = 250
# dt = 0.01
# SIM_TIME = 2000 # ms
# T = int(SIM_TIME/dt) 
    
    
# '''
# simulation
# '''

# nn = NeuralNetwork(N_E, N_I)
# s_bin_log = nn.self_org(T)


# '''
# plot
# '''
# # 結合タイプごとの結合数をカウント
# EE_synapses = [syn for syn in nn.synapses if syn['pre_n'] < N_E and syn['post_n'] < N_E]
# EI_synapses = [syn for syn in nn.synapses if syn['pre_n'] < N_E and syn['post_n'] >= N_E]
# IE_synapses = [syn for syn in nn.synapses if syn['pre_n'] >= N_E and syn['post_n'] < N_E]
# II_synapses = [syn for syn in nn.synapses if syn['pre_n'] >= N_E and syn['post_n'] >= N_E]

# # 可能なシナプス結合数（興奮性、抑制性の総数を使う）
# possible_EE = N_E * (N_E - 1)  # 自己結合を除くために N_E - 1
# possible_EI = N_E * N_I
# possible_IE = N_I * N_E
# possible_II = N_I * (N_I - 1)  # 自己結合を除くために N_I - 1

# # 結合確率の計算
# EE_prob = len(EE_synapses) / possible_EE
# EI_prob = len(EI_synapses) / possible_EI
# IE_prob = len(IE_synapses) / possible_IE
# II_prob = len(II_synapses) / possible_II

# # 結果を出力
# with open("output.txt", "w") as doc:
#     print("シナプス総結合数", len(nn.synapses), file=doc)
#     print('シナプス結合割合', len(nn.synapses)/(N_E+N_I)**2, file=doc)
#     print(f"EE 結合確率: {EE_prob:.4f}", file=doc)
#     print(f"EI 結合確率: {EI_prob:.4f}", file=doc)
#     print(f"IE 結合確率: {IE_prob:.4f}", file=doc)
#     print(f"II 結合確率: {II_prob:.4f}", file=doc)
  
    
# plt.figure(figsize=(12, 10))
# # 興奮性ニューロンと抑制性ニューロンを分けてプロットする
# time_steps, neuron_ids = np.nonzero(s_bin_log)  # スパイクがある位置のインデックスを取得

# # 興奮性ニューロンのスパイクデータをプロット
# exc_neurons = neuron_ids < N_E  # 興奮性ニューロンのインデックス
# plt.scatter(time_steps[exc_neurons] * dt, neuron_ids[exc_neurons], color='b', s=10, label='Excitatory')

# # 抑制性ニューロンのスパイクデータをプロット
# inh_neurons = neuron_ids >= N_E  # 抑制性ニューロンのインデックス
# plt.scatter(time_steps[inh_neurons] * dt, neuron_ids[inh_neurons], color='r', s=10, label='Inhibitory')

# # グラフの設定
# plt.xlabel('Time (ms)')
# plt.ylabel('Neuron IDs')
# plt.title('Spike Raster Plot (Color-Coded by Neuron Type)')
# plt.legend(loc='upper right')
# plt.savefig("spikes_raster.png", dpi=300)

# # Time resolution for blocks (10 ms)
# block_size = 10  # 10msごとに分割
# blocks = int(SIM_TIME / block_size)  # 全シミュレーション時間を10msごとのブロックに分割

# # スパイク頻度を記録する行列を作成
# spike_freq_matrix = np.zeros((len(nn.neurons), blocks))

# # 各ニューロンのスパイク頻度を10msecごとのブロックに集計
# for neuron_id in range(len(nn.neurons)):
#     for block in range(blocks):
#         start_time = int(block * block_size / dt)
#         end_time = int((block + 1) * block_size / dt)
#         spike_freq_matrix[neuron_id, block] = np.sum(s_bin_log[start_time:end_time, neuron_id])

# # スパイク頻度のヒートマップをプロット
# plt.figure(figsize=(12, 6))
# plt.imshow(spike_freq_matrix, aspect='auto', cmap='viridis', interpolation='nearest', origin='lower')
# plt.colorbar(label='Spike Frequency')
# plt.xlabel('Time Blocks (10 ms each)')
# plt.ylabel('Neuron ID')
# plt.title('Spike Frequency per 10ms Block')
# plt.savefig("spike_frequency.png", dpi=300)

# # 1. 全シナプスの重みの平均の変化をプロット
# avg_weights = np.mean(nn.weight_history, axis=1)
# plt.figure(figsize=(8, 4))
# plt.plot(avg_weights)
# plt.title("Average Weight Over Time")
# plt.xlabel("Time Step")
# plt.ylabel("Average Synaptic Weight")
# plt.grid(True)
# plt.savefig("ave_weight.png", dpi=300)

# # 2. 各シナプスの重みの変化をヒートマップで表示
# # plt.figure(figsize=(10, 6))
# # plt.imshow(np.array(nn.weight_history).T, aspect='auto', cmap='coolwarm', origin='lower')
# # plt.colorbar(label='Synaptic Weight')
# # plt.title('Synaptic Weight Changes Over Time')
# # plt.xlabel('Time Step')
# # plt.ylabel('Synapse ID')
# # plt.show()

# # Plot weight changes for weights that became negative
# # negative_weights = np.array(nn.weight_history)
# # negative_weights = np.where(negative_weights < 0, negative_weights, np.nan)  # Set positive weights to NaN

# # plt.figure(figsize=(12, 6))
# # plt.imshow(negative_weights.T, aspect='auto', cmap='coolwarm', interpolation='nearest')
# # plt.colorbar(label='Negative Synaptic Weights')
# # plt.xlabel('Time Steps')
# # plt.ylabel('Synapse Index')
# # plt.title('Negative Weight Changes Over Time')

# # Plot excitatory and inhibitory inputs over time
# plt.figure(figsize=(12, 6))
# plt.plot(np.arange(T) * dt, nn.exc_input_log, label='Excitatory Input', color='b')
# plt.plot(np.arange(T) * dt, nn.inh_input_log, label='Inhibitory Input', color='r')
# plt.plot(np.arange(T) * dt, nn.ei_deffe_log, label='E-I Input Defference', color='k')
# plt.xlabel('Time (ms)')
# plt.ylabel('Input Sum')
# plt.title('Excitatory and Inhibitory Input Over Time')
# plt.legend()
# plt.grid(True)
# plt.savefig("E_I_input.png", dpi=300)

# # Plot E/I ratio over time
# plt.figure(figsize=(12, 6))
# plt.plot(np.arange(T) * dt, nn.ei_ratio_log, label='E/I Ratio', color='g')
# plt.xlabel('Time (ms)')
# plt.ylabel('E/I Ratio')
# plt.title('Excitatory/Inhibitory Ratio Over Time')
# plt.grid(True)
# plt.savefig("E_I_ratio.png", dpi=300)

# # Plot relative weights over time
# ee_relative_weights, ei_relative_weights, ie_relative_weights, ii_relative_weights = zip(*nn.syn_ratio_log)

# plt.figure(figsize=(12, 6))
# time_blocks = np.arange(len(ee_relative_weights)) * block_size
# plt.plot(time_blocks, ee_relative_weights, label='EE Relative Weight', color='b')
# plt.plot(time_blocks, ei_relative_weights, label='EI Relative Weight', color='r')
# plt.plot(time_blocks, ie_relative_weights, label='IE Relative Weight', color='g')
# plt.plot(time_blocks, ii_relative_weights, label='II Relative Weight', color='y')
# plt.xlabel('Time (ms)')
# plt.ylabel('Relative Weight')
# plt.title('Relative Weights of Synapse Types Over Time')
# plt.legend()
# plt.grid(True)
# plt.savefig("w_types.png", dpi=300)




