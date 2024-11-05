import matplotlib.pyplot as plt
import numpy as np
from neurons.lif import LIF
from synapses.static_synapse import StaticSynapse
from synapses.stdp_guetig import Guetig_STDP


class NeuralNetwork:
    def __init__(self, n_e, n_i, seed=0):
        np.random.seed(seed)
        self.neurons = self.create_neurons(n_e, n_i)
        self.synapses = self.create_synapse()
        self.weight_history = []  # 重みの変化を記録するリスト
        self.exc_input_log = []  # 興奮性シナプスの入力総和を記録するリスト
        self.inh_input_log = []  # 抑制性シナプスの入力総和を記録するリスト
        self.ei_ratio_log = []  # E/I比を記録するリスト
        self.ei_deffe_log = []
        self.syn_ratio_log = []
        
        
        
        
    def create_neurons(self, exc_n, inh_n):
        n_list = []
        for m in range(exc_n):
            synap_dict = {'id':m,
                          'prop': 'exc', 
                          'pos':np.random.uniform(-1,1,3), 
                          'state': 1,
                          'model':LIF(),
                          'I_sum': 5000,
                          'last_spike': None}

            n_list.append(synap_dict)
            
        for n in range(inh_n):
            synap_dict = {'id':exc_n+n,
                          'prop': 'inh', 
                          'pos':np.random.uniform(-1,1,3), 
                          'state': 1,
                          'model':LIF(),
                          'I_sum': 5000,
                          'last_spike': None}
            n_list.append(synap_dict)
            
        return n_list
        
              
    def create_synapse(self, C={"EE":0.16,"EI":0.25,"IE":0.38,"II":0.1}):
        s_list = []
        self.C = C
        self.Lambda =  {
            'EE':1/(C['EE']*np.sqrt(np.pi)),
            'EI':1/(C['EI']*np.sqrt(np.pi)),
            'IE':1/(C['IE']*np.sqrt(np.pi)),
            'II':1/(C['II']*np.sqrt(np.pi))
        }
            
        
        for pre in self.neurons:
            for post in self.neurons:
                # 距離の計算
                dist = np.sqrt(np.sum((pre['pos'] - post['pos']) ** 2))
                if pre['prop'] == 'exc' and post['prop'] == 'exc': 
                    form = 'EE'
                    weight = np.random.uniform(0,1)
                elif pre['prop'] == 'exc' and post['prop'] == 'inh': 
                    form = 'EI'
                    weight = np.random.uniform(0,1)
                elif pre['prop'] == 'inh' and post['prop'] == 'exc': 
                    form = 'IE'
                    weight = np.random.uniform(-1,0)
                elif pre['prop'] == 'inh' and post['prop'] == 'inh': 
                    form = 'II'
                    weight = np.random.uniform(-1, 0)
                
                if np.random.rand() < self.C[form] * np.exp(-dist / self.Lambda[form] ** 2):
                    syna_dict = {'pre_n': pre['id'],
                                 'post_n': post['id'], 
                                 'weight': weight, 
                                 'model': StaticSynapse(), 
                                 'model_2': Guetig_STDP()}                   
                    s_list.append(syna_dict)          
                    
        return s_list
    
    
    def culc_spikes(self, t):
        spikes = []
        for n in self.neurons:
            spike = n['model'](n['I_sum'])
            if spike:
                n['last_spike'] = t  # スパイクした場合に現在の時刻を記録
            spikes.append(spike)
            
        return spikes
    
    
    def culc_I_post(self):
        # Reset I_sum for all neurons
        for n in self.neurons:
            n['I_sum'] = 0
            
        exc_input_sum = 0
        inh_input_sum = 0
        
        # 各シナプス後電流を計算 => neurons['I_sum']に合算
        for syn in self.synapses:
            pre_n, post_n = syn['pre_n'], syn['post_n']
            I = syn['model'](self.neurons[pre_n]['model'].state,
                           syn['weight'])
            self.neurons[post_n]['I_sum'] += I
            
            # Synapse type (excitatory or inhibitory)
            pre_prop = self.neurons[pre_n]['prop']
            if pre_prop == 'exc':
                exc_input_sum += I
            elif pre_prop == 'inh':
                inh_input_sum += I
        
        # Log the sum of excitatory and inhibitory inputs
        self.exc_input_log.append(exc_input_sum)
        self.inh_input_log.append(inh_input_sum)
        self.ei_deffe_log.append(exc_input_sum + inh_input_sum)
        
        # Calculate and log the E/I ratio
        if inh_input_sum != 0:
            ei_ratio = -(exc_input_sum / inh_input_sum)
        else:
            ei_ratio = np.inf  # Handle division by zero
        
        self.ei_ratio_log.append(ei_ratio)
        
        # 追加: 各シナプスタイプの重みの合計を計算
        weight_sum = sum(syn['weight'] for syn in self.synapses)  # 全シナプスの重みの合計
        ee_weight_sum = sum(syn['weight'] for syn in self.synapses if self.neurons[syn['pre_n']]['prop'] == 'exc' and self.neurons[syn['post_n']]['prop'] == 'exc')
        ei_weight_sum = sum(syn['weight'] for syn in self.synapses if self.neurons[syn['pre_n']]['prop'] == 'exc' and self.neurons[syn['post_n']]['prop'] == 'inh')
        ie_weight_sum = sum(syn['weight'] for syn in self.synapses if self.neurons[syn['pre_n']]['prop'] == 'inh' and self.neurons[syn['post_n']]['prop'] == 'exc')
        ii_weight_sum = sum(syn['weight'] for syn in self.synapses if self.neurons[syn['pre_n']]['prop'] == 'inh' and self.neurons[syn['post_n']]['prop'] == 'inh')
        
        # 各シナプスタイプの重みの相対値を計算
        if weight_sum != 0:
            ee_relative_weight = ee_weight_sum / weight_sum
            ei_relative_weight = ei_weight_sum / weight_sum
            ie_relative_weight = -(ie_weight_sum / weight_sum)
            ii_relative_weight = -(ii_weight_sum / weight_sum)
        else:
            ee_relative_weight = ei_relative_weight = ie_relative_weight = ii_relative_weight = 0  # Handle division by zero

        # 相対値を記録
        self.syn_ratio_log.append((ee_relative_weight, ei_relative_weight, ie_relative_weight, ii_relative_weight))
    
    
    def change_weight(self):
        for syn in self.synapses:
            post_spike = self.neurons[syn['post_n']]['model'].state
            pre_prop = self.neurons[syn['pre_n']]['prop']
            
        # ポストニューロンが発火したときのみ更新
            if post_spike:
                # スパイクタイミング差を計算（事後 - 事前）
                pre_time = self.neurons[syn['pre_n']]['last_spike']
                post_time = self.neurons[syn['post_n']]['last_spike']
                
                if pre_time is not None and post_time is not None:
                    delta_t = pre_time - post_time
                    
                    # Guetig STDPモデルに基づく重み更新
                    weight_update = syn['model_2'](delta_t)
                    if pre_prop == 'exc':
                        syn['weight'] += weight_update
                    elif pre_prop == 'inh':
                        syn['weight'] -= weight_update
                        
                    # 重みの履歴を記録
                    self.weight_history.append([syn['weight'] for syn in self.synapses])
    
            
    
    def self_org(self, T):
        s_bin_log = []
        
        for m in range(T):
            spikes = self.culc_spikes(m * dt)
            self.culc_I_post()
            self.change_weight()
            
            
            s_bin_log.append(spikes)
        
        return np.array(s_bin_log)
    
    
    
'''
parameter
'''
N_E = 1000
N_I = 250
dt = 0.01
SIM_TIME = 1000 # ms
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




