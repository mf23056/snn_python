import torch
import matplotlib.pyplot as plt
import itertools
import math

class LIF:
    def __init__(self, dt=0.01, tau_m=20, V_rest=-65.0, R=10.0, I_back=0.0, V_reset=-65.0, V_th=-50.0, V_ref=5, bef_V=0, r_timer=0):
        self.dt = dt
        self.tau_m = tau_m
        self.V_rest = V_rest
        self.R = R
        self.I_back = I_back
        self.V_reset = V_reset
        self.V_th = V_th
        self.V_ref = V_ref / dt  # 不応期を刻み幅で割る

    def __call__(self, I_syn, before_V, ref_time):
        V = before_V + self.dt * ((1 / self.tau_m) * (-(before_V - self.V_rest) + self.R * (I_syn + self.I_back)))
        if V >= self.V_th:
            V = self.V_reset
            self.r_timer = self.V_ref
            self.state = 1
        else:
            self.state = 0
        self.bef_V = V
        return self.state

class StaticSynapse:
    def __init__(self, dt=0.01, tau_syn=25, before_I=0):
        self.tau_syn = tau_syn
        self.dt = dt

    def __call__(self, bin_spike, W, before_I):
        I = before_I + self.dt * (-before_I / self.tau_syn) + W * bin_spike
        return I

class Guetig_STDP:
    def __init__(self, dt=0.01, A_plus=0.01, A_minus=0.01, tau_plus=30.0, tau_minus=30.0, alpha=0.999, device='cuda'):
        self.dt = dt
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.alpha = alpha
        self.device = device

    def __call__(self, delta_t):
        delta_w = torch.zeros_like(delta_t, device=self.device)
        delta_w[delta_t > 0] = self.A_plus * (torch.exp(-delta_t[delta_t > 0] / self.tau_plus) - 
                                              self.alpha * torch.exp(-delta_t[delta_t > 0] / (self.alpha * self.tau_plus)))
        delta_w[delta_t <= 0] = -self.A_minus * (torch.exp(delta_t[delta_t <= 0] / self.tau_minus) - 
                                                  self.alpha * torch.exp(delta_t[delta_t <= 0] / (self.alpha * self.tau_minus)))
        return delta_w

class SNN:
    def __init__(self, n_exc, n_inh, dt=0.01, device='cuda'):
        self.n_total = n_exc + n_inh
        self.dt = dt
        self.device = device

        self.neuron = LIF()
        self.synapse = StaticSynapse()
        self.stdp = Guetig_STDP()

        self._initialize_neurons(n_exc, n_inh)
        self._initialize_synapses()

    def _initialize_neurons(self, n_exc, n_inh):
        self.positions = torch.rand((self.n_total, 3), device=self.device)
        self.sum_I_syn = torch.zeros(self.n_total, device=self.device)
        self.before_V = torch.zeros(self.n_total, device=self.device)
        self.ref_time = torch.zeros(self.n_total, device=self.device)
        self.spike_state = torch.zeros(self.n_total, device=self.device)
        self.spike_times = -torch.ones(self.n_total, device=self.device)

        neuron_types = torch.cat((
            torch.full((n_exc,), 1, dtype=torch.int32, device=self.device),
            torch.full((n_inh,), 0, dtype=torch.int32, device=self.device)
        ))
        neuron_types = neuron_types[torch.randperm(neuron_types.size(0))]
        self.neuron_types = neuron_types

    def _initialize_synapses(self):
        self.before_I = torch.zeros((self.n_total, self.n_total), device=self.device)
        self.weights = torch.zeros((self.n_total, self.n_total), device=self.device)
        self.mask_weights = torch.zeros((self.n_total, self.n_total), device=self.device)
        self.synapse_threshold = torch.rand((self.n_total, self.n_total), device=self.device)

        distances = torch.cdist(self.positions, self.positions)
        self.synapse_probabilities = {
            'EE': 0.16,
            'EI': 0.25,
            'IE': 0.38,
            'II': 0.1
        }
        self.distance_decay_factors = self._get_distance_decay_factors()
        self._create_synapses(distances)

    def _get_distance_decay_factors(self):
        return {
            'EE': 1 / (self.synapse_probabilities['EE'] * math.sqrt(math.pi)),
            'EI': 1 / (self.synapse_probabilities['EI'] * math.sqrt(math.pi)),
            'IE': 1 / (self.synapse_probabilities['IE'] * math.sqrt(math.pi)),
            'II': 1 / (self.synapse_probabilities['II'] * math.sqrt(math.pi))
        }

    def _create_synapses(self, distances):
        synapse_prob = torch.zeros((self.n_total, self.n_total), device=self.device)
        for i in range(self.n_total):
            for j in range(self.n_total):
                pre_neuron_type = self.neuron_types[i]
                post_neuron_type = self.neuron_types[j]
                dist = distances[i, j]
                form, weight_bias = self._get_synapse_type_and_weight(pre_neuron_type, post_neuron_type)
                synapse_prob[i, j] = self.synapse_probabilities[form] * torch.exp(-dist**2 / (self.distance_decay_factors[form] ** 2))

        random_values = torch.rand_like(synapse_prob, device=self.device)
        self.weights = (synapse_prob > random_values).float() * weight_bias
        self.mask_weights = self.weights.clone()

    def _get_synapse_type_and_weight(self, pre_neuron_type, post_neuron_type):
        synapse_type = pre_neuron_type * 2 + post_neuron_type
        weights = torch.full_like(synapse_type, 1, device=self.device)
        weights[synapse_type == 2] = -1
        weights[synapse_type == 3] = -1
        return synapse_type, weights

    def run_simulation(self, T=1000):
        spike_record = torch.zeros(self.n_total, int(T / self.dt), device=self.device)

        for t in range(1, int(T / self.dt)):
            self.spike_state = self.neuron(self.sum_I_syn, self.before_V, self.ref_time)
            self.sum_I_syn = self.synapse(self.spike_state, self.weights, self.before_I)
            self.spike_times[self.spike_state == 1] = t * self.dt

            spike_time_defference = self.spike_times.unsqueeze(1) - self.spike_times.unsqueeze(0)
            delta_w = self.stdp(spike_time_defference)

            self.weights += delta_w

            spike_record[:, t] = self.spike_state
            self.before_I = self.sum_I_syn

        return spike_record

    def plot_raster(self, spike_record, T):
        spike_times = torch.nonzero(spike_record == 1, as_tuple=False)
        plt.figure(figsize=(10, 6))
        plt.scatter(spike_times[:, 1].cpu() * self.dt, spike_times[:, 0].cpu(), marker="|", color="black", s=50)
        plt.xlabel("Time (ms)")
        plt.ylabel("Neuron Index")
        plt.title("Spike Raster Plot")
        plt.show()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
network = SNN(n_exc=1000, n_inh=250, device=device)
spike_record = network.run_simulation(T=1000)
network.plot_raster(spike_record, T=1000)



    
'''
parameter
'''
N_E = 1000
N_I = 250
dt = 0.01
SIM_TIME = 2000 # ms
T = int(SIM_TIME/dt) 
    
    
'''
simulation
'''

nn = NeuralNetwork(N_E, N_I)
s_bin_log = nn.self_org(T)


'''
plot
'''
# 結合タイプごとの結合数をカウント
EE_synapses = [syn for syn in nn.synapses if syn['pre_n'] < N_E and syn['post_n'] < N_E]
EI_synapses = [syn for syn in nn.synapses if syn['pre_n'] < N_E and syn['post_n'] >= N_E]
IE_synapses = [syn for syn in nn.synapses if syn['pre_n'] >= N_E and syn['post_n'] < N_E]
II_synapses = [syn for syn in nn.synapses if syn['pre_n'] >= N_E and syn['post_n'] >= N_E]

# 可能なシナプス結合数（興奮性、抑制性の総数を使う）
possible_EE = N_E * (N_E - 1)  # 自己結合を除くために N_E - 1
possible_EI = N_E * N_I
possible_IE = N_I * N_E
possible_II = N_I * (N_I - 1)  # 自己結合を除くために N_I - 1

# 結合確率の計算
EE_prob = len(EE_synapses) / possible_EE
EI_prob = len(EI_synapses) / possible_EI
IE_prob = len(IE_synapses) / possible_IE
II_prob = len(II_synapses) / possible_II

# 結果を出力
with open("output.txt", "w") as doc:
    print("シナプス総結合数", len(nn.synapses), file=doc)
    print('シナプス結合割合', len(nn.synapses)/(N_E+N_I)**2, file=doc)
    print(f"EE 結合確率: {EE_prob:.4f}", file=doc)
    print(f"EI 結合確率: {EI_prob:.4f}", file=doc)
    print(f"IE 結合確率: {IE_prob:.4f}", file=doc)
    print(f"II 結合確率: {II_prob:.4f}", file=doc)
  
    
plt.figure(figsize=(12, 10))
# 興奮性ニューロンと抑制性ニューロンを分けてプロットする
time_steps, neuron_ids = np.nonzero(s_bin_log)  # スパイクがある位置のインデックスを取得

# 興奮性ニューロンのスパイクデータをプロット
exc_neurons = neuron_ids < N_E  # 興奮性ニューロンのインデックス
plt.scatter(time_steps[exc_neurons] * dt, neuron_ids[exc_neurons], color='b', s=10, label='Excitatory')

# 抑制性ニューロンのスパイクデータをプロット
inh_neurons = neuron_ids >= N_E  # 抑制性ニューロンのインデックス
plt.scatter(time_steps[inh_neurons] * dt, neuron_ids[inh_neurons], color='r', s=10, label='Inhibitory')

# グラフの設定
plt.xlabel('Time (ms)')
plt.ylabel('Neuron IDs')
plt.title('Spike Raster Plot (Color-Coded by Neuron Type)')
plt.legend(loc='upper right')
plt.savefig("spikes_raster.png", dpi=300)

# Time resolution for blocks (10 ms)
block_size = 10  # 10msごとに分割
blocks = int(SIM_TIME / block_size)  # 全シミュレーション時間を10msごとのブロックに分割

# スパイク頻度を記録する行列を作成
spike_freq_matrix = np.zeros((len(nn.neurons), blocks))

# 各ニューロンのスパイク頻度を10msecごとのブロックに集計
for neuron_id in range(len(nn.neurons)):
    for block in range(blocks):
        start_time = int(block * block_size / dt)
        end_time = int((block + 1) * block_size / dt)
        spike_freq_matrix[neuron_id, block] = np.sum(s_bin_log[start_time:end_time, neuron_id])

# スパイク頻度のヒートマップをプロット
plt.figure(figsize=(12, 6))
plt.imshow(spike_freq_matrix, aspect='auto', cmap='viridis', interpolation='nearest', origin='lower')
plt.colorbar(label='Spike Frequency')
plt.xlabel('Time Blocks (10 ms each)')
plt.ylabel('Neuron ID')
plt.title('Spike Frequency per 10ms Block')
plt.savefig("spike_frequency.png", dpi=300)

# 1. 全シナプスの重みの平均の変化をプロット
avg_weights = np.mean(nn.weight_history, axis=1)
plt.figure(figsize=(8, 4))
plt.plot(avg_weights)
plt.title("Average Weight Over Time")
plt.xlabel("Time Step")
plt.ylabel("Average Synaptic Weight")
plt.grid(True)
plt.savefig("ave_weight.png", dpi=300)

# 2. 各シナプスの重みの変化をヒートマップで表示
# plt.figure(figsize=(10, 6))
# plt.imshow(np.array(nn.weight_history).T, aspect='auto', cmap='coolwarm', origin='lower')
# plt.colorbar(label='Synaptic Weight')
# plt.title('Synaptic Weight Changes Over Time')
# plt.xlabel('Time Step')
# plt.ylabel('Synapse ID')
# plt.show()

# Plot weight changes for weights that became negative
# negative_weights = np.array(nn.weight_history)
# negative_weights = np.where(negative_weights < 0, negative_weights, np.nan)  # Set positive weights to NaN

# plt.figure(figsize=(12, 6))
# plt.imshow(negative_weights.T, aspect='auto', cmap='coolwarm', interpolation='nearest')
# plt.colorbar(label='Negative Synaptic Weights')
# plt.xlabel('Time Steps')
# plt.ylabel('Synapse Index')
# plt.title('Negative Weight Changes Over Time')

# Plot excitatory and inhibitory inputs over time
plt.figure(figsize=(12, 6))
plt.plot(np.arange(T) * dt, nn.exc_input_log, label='Excitatory Input', color='b')
plt.plot(np.arange(T) * dt, nn.inh_input_log, label='Inhibitory Input', color='r')
plt.plot(np.arange(T) * dt, nn.ei_deffe_log, label='E-I Input Defference', color='k')
plt.xlabel('Time (ms)')
plt.ylabel('Input Sum')
plt.title('Excitatory and Inhibitory Input Over Time')
plt.legend()
plt.grid(True)
plt.savefig("E_I_input.png", dpi=300)

# Plot E/I ratio over time
plt.figure(figsize=(12, 6))
plt.plot(np.arange(T) * dt, nn.ei_ratio_log, label='E/I Ratio', color='g')
plt.xlabel('Time (ms)')
plt.ylabel('E/I Ratio')
plt.title('Excitatory/Inhibitory Ratio Over Time')
plt.grid(True)
plt.savefig("E_I_ratio.png", dpi=300)

# Plot relative weights over time
ee_relative_weights, ei_relative_weights, ie_relative_weights, ii_relative_weights = zip(*nn.syn_ratio_log)

plt.figure(figsize=(12, 6))
time_blocks = np.arange(len(ee_relative_weights)) * block_size
plt.plot(time_blocks, ee_relative_weights, label='EE Relative Weight', color='b')
plt.plot(time_blocks, ei_relative_weights, label='EI Relative Weight', color='r')
plt.plot(time_blocks, ie_relative_weights, label='IE Relative Weight', color='g')
plt.plot(time_blocks, ii_relative_weights, label='II Relative Weight', color='y')
plt.xlabel('Time (ms)')
plt.ylabel('Relative Weight')
plt.title('Relative Weights of Synapse Types Over Time')
plt.legend()
plt.grid(True)
plt.savefig("w_types.png", dpi=300)




