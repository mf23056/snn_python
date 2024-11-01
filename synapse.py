import numpy as np
import matplotlib.pyplot as plt

# permitive
class StaticSynapse:
    def __init__(self, dt=0.01, tau_syn=25, before_I=0):
        '''
        param dt: 刻み幅 (ms)
        param tau_syn: シナプスの時定数 (ms)
        param before_I: 初期のシナプス電流
        '''
        self.tau_syn = tau_syn  # シナプスの時定数
        self.dt = dt            # 時間の刻み幅
        self.before_I = before_I # 初期のシナプス電流

    # postsynaptic currentを計算
    def __call__(self, bin_spike: int, W: float) -> float:
        '''
        param bin_spike: 入力データ (バイナリスパイク)
        param W: シナプスの重み
        return: postsynaptic current (latest state)
        '''
        # postsynaptic currentの計算
        I = self.before_I + self.dt * (-self.before_I / self.tau_syn) + W * bin_spike

        
        # 内部状態の更新
        self.before_I = I
        
        return I



# shot-term synaptic plasticity
class STSP_thodyks:
    def __init__(self, dt=0.01, tau_syn=20, STD_tau_f=100, STD_tau_d=500, STD_B=0.45,
                 STF_tau_f=500, STF_tau_d=50, STF_B=0.15, before_U=0, before_X=1, before_I=0):
        '''
        param dt: 刻み幅
        param tau_syn: シナプスの時定数(ms)
        param STD_tau_f: (ms)
        param STD_tau_d: (ms)
        param STD_B: 
        param U: Utilization (U) - Fraction of available neurotransmitter resources ready for release 
                 (release probability). Increases after a spike and decays with a time constant between spikes.
        param X: Availability (X) - Fraction of neurotransmitter resources remaining (i.e., available for release).
                 Decreases after a spike and recovers with a time constant between spikes.
        param I: Postsynaptic current (I) - The electrical current generated in the postsynaptic neuron.
                 Depends on both utilization (U) and availability (X) and reflects the strength of synaptic transmission.
        '''
        self.dt = dt
        self.tau_syn = tau_syn
        self.STD_tau_f = STD_tau_f
        self.STD_tau_d = STD_tau_d
        self.STD_B = STD_B
        self.STF_tau_f = STF_tau_f
        self.STF_tau_d = STF_tau_d
        self.STF_B = STF_B
        self.before_U = before_U  # Previous utilization (U)
        self.before_X = before_X  # Previous availability (X)
        self.before_I = before_I  # Previous postsynaptic current (I)        

    # culcurate postsynaptic current
    def __call__(self, bin_spike: float, property: str, weight: float) -> float:
        '''
        Calculate the postsynaptic current based on input and synapse properties.

        param bin_spike: Input data (binary spike)
        param property: Synapse type ('inh' for inhibitory, 'exc' for excitatory)
        param weight: Synapse weight
        param before_U: Previous state of U
        param before_X: Previous state of X
        param before_I: Previous postsynaptic current
        return: Current postsynaptic current
        '''
        
        if property not in ['inh', 'exc']:
            raise ValueError("Invalid property. Use 'inh' for inhibitory or 'exc' for excitatory.")
        
        if(property=='inh'):
            U = self.before_U + self.dt * (- self.before_U / self.STD_tau_f) + self.STD_B * (1 - self.before_U) * bin_spike
            X = self.before_X + self.dt * ((1-self.before_X) / self.STD_tau_d) - U * self.before_X * bin_spike
            I = self.before_I + self.dt * (- (self.before_I / self.tau_syn)) + weight * U * self.before_X * bin_spike

        elif(property=='exc'):
            U = self.before_U + self.dt * (-self.before_U / self.STF_tau_f) + self.STF_B * (1 - self.before_U) * bin_spike
            X = self.before_X + self.dt * ((1-self.before_X) / self.STF_tau_d) - U * self.before_X * bin_spike
            I = self.before_I + self.dt * (-self.before_I / self.tau_syn) + weight * U * self.before_X * bin_spike
        
        # Update internal states
        self.before_U = U
        self.before_X = X
        self.before_I = I
        
        return I
    


# Long-term plasticity

class Guetig_STDP:
    def __init__(self, dt=0.01, A_plus=0.01, A_minus=0.01, tau_plus=30.0, tau_minus=30.0, alpha=0.999):
        '''
        Parameters:
        delta_t (float): Time difference between post-synaptic spike and pre-synaptic spike (t_post - t_pre).
        A_plus (float): Amplitude of the weight change for potentiation (when delta_t > 0).
        A_minus (float): Amplitude of the weight change for depression (when delta_t < 0).
        tau_plus (float): Time constant for potentiation.
        tau_minus (float): Time constant for depression.
        '''
        self.dt = dt
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.alpha = alpha
        
    
    
    def __call__(self, delta_t) -> float:
        '''
        Returns:
        float: Change in synaptic weight.
        '''
        
        if delta_t > 0:
            # Long-term potentiation (LTP) with Gütig's modification
            delta_w = self.A_plus * (np.exp(-delta_t / self.tau_plus) - self.alpha * np.exp(-delta_t / (self.alpha * self.tau_plus)))
        else:
            # Long-term depression (LTD) with Gütig's modification
            delta_w = -self.A_minus * (np.exp(delta_t / self.tau_minus) - self.alpha * np.exp(delta_t / (self.alpha * self.tau_minus)))

        return delta_w


if __name__ == "__main__":
    '''
    thodyks short-term plasticity
    '''
    np.random.seed(42)  # 再現性のために乱数シードを設定
    T = 300  # 総シミュレーション時間(ms)
    dt = 0.01  # 刻み幅(ms)
    time = np.arange(0, T, dt)  # 時間軸
    spike_prob = 0.001  # 各時刻でスパイクが発生する確率
    spike_data = np.random.binomial(1, spike_prob, len(time))  # バイナリスパイクデータ

    # テスト用のSTSP_thodyksインスタンスを作成
    synapse = STSP_thodyks(dt=dt)

    # 各時刻における U, X, I を保存するリスト
    U_vals = []
    X_vals = []
    I_vals = []

    # シナプスのタイプ（興奮性シナプス）を使用
    synapse_type = 'exc'
    weight = 1.0  # シナプスの重み

    # シミュレーションループ
    for spike in spike_data:
        I = synapse(spike, synapse_type, weight)
        U_vals.append(synapse.before_U)
        X_vals.append(synapse.before_X)
        I_vals.append(I)

    # プロットの準備
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # スパイクデータのプロット
    axs[0].plot(time, spike_data, label="Spikes", color="black")
    axs[0].set_ylabel("Spike")
    axs[0].legend(loc="upper right")
    axs[0].set_title("Spike Data and Synaptic Variables Over Time")
    axs[0].grid()

    # Uのプロット
    axs[1].plot(time, U_vals, label="U (Utilization)", color="blue")
    axs[1].set_ylabel("U (Utilization)")
    axs[1].legend(loc="upper right")

    # Xのプロット
    axs[1].plot(time, X_vals, label="X (Availability)", color="green")
    axs[1].set_ylabel("X (Availability)")
    axs[1].legend(loc="upper right")
    axs[1].grid()

    # Iのプロット
    axs[2].plot(time, I_vals, label="I (Postsynaptic Current)", color="red")
    axs[2].set_xlabel("Time (ms)")
    axs[2].set_ylabel("I (Postsynaptic Current)")
    axs[2].legend(loc="upper right")
    axs[2].grid()

    plt.tight_layout()
    plt.show()

    '''
    common static synapse model
    '''

    # ランダムスパイクデータを生成
    np.random.seed(42)  # 再現性のために乱数シードを設定
    T = 300  # 総シミュレーション時間(ms)
    dt = 0.01  # 刻み幅(ms)
    time = np.arange(0, T, dt)  # 時間軸
    spike_prob = 0.001  # 各時刻でスパイクが発生する確率
    spike_data = np.random.binomial(1, spike_prob, len(time))  # バイナリスパイクデータ

    # StaticSynapseクラスのインスタンスを作成
    tau_syn = 20  # シナプスの時定数
    weight = 1.0  # シナプスの重み
    synapse = StaticSynapse(dt=dt, tau_syn=tau_syn)

    # ポストシナプス電流の計算
    postsynaptic_current = []
    for spike in spike_data:
        I = synapse(bin_spike=spike, W=weight)
        postsynaptic_current.append(I)

    # プロット
    plt.figure(figsize=(10, 6))

    # スパイクデータのプロット
    plt.subplot(2, 1, 1)
    plt.plot(time, spike_data, drawstyle='steps-post', label='Spike Data')
    plt.ylabel('Spike (0 or 1)')
    plt.title('Random Spike Input and Postsynaptic Current')
    plt.grid(True)

    # ポストシナプス電流のプロット
    plt.subplot(2, 1, 2)
    plt.plot(time, postsynaptic_current, label='Postsynaptic Current')
    plt.xlabel('Time (ms)')
    plt.ylabel('Current (I)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
    '''
    guetig stdp - long term plasticity
    '''
    
    # Guetig STDPのインスタンスを作成
    stdp = Guetig_STDP()

    # delta_tの範囲を設定
    delta_ts = np.linspace(-100, 100, 400)
    changes = [stdp(dt) for dt in delta_ts]

    # プロット
    plt.figure(figsize=(10, 6))
    plt.plot(delta_ts, changes, label='Change in Synaptic Weight', color='blue')
    plt.title('Guetig STDP: Change in Synaptic Weight vs. Delta t')
    plt.xlabel('Delta t (ms)')
    plt.ylabel('Change in Weight (Δw)')
    plt.axhline(0, color='red', linestyle='--')  # y=0のラインを追加
    plt.grid()
    plt.legend()
    plt.show()