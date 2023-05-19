"""
THIS CODE IS FOR:
- Training the agents myself (dont consider overfitting right now)
- Doing a basic hyperparameter optimization

- TODO, plot reward learned etc
- TODO, get other performance indicator such as sortonio or max drawdown


NEED Help:
    We use a deep reinforcement learning agent to make trading actions, which can be either buy, sell, or hold.
    Where are these 3 possibilities defined ? I could not see it (action value can take [-1,0-1] ??)
    
"""


OFF_POLICY_MODELS = ["ddpg", "td3", "sac"]
ON_POLICY_MODELS = ["ppo", "a2c"]

import os,sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

from talib import RSI, MACD, CCI, DX, ROC, ULTOSC, WILLR, OBV, HT_DCPHASE
from drl_agents.elegantrl_models import DRLAgent as DRLAgent_erl
from environment_Alpaca import CryptoEnvAlpaca
from function_finance_metrics import compute_data_points_per_year, compute_eqw, sharpe_iid


#ALPACA_LIMITS = np.array([0.001, 0.01]) # Minimum buy limits for each currency (here we only have BTC so the dimension is 1, it doesnt matter since crypto dont have min buy)
ALPACA_LIMITS = np.array([0.001]) 
no_candles_for_train = 20000 #I dont know what that is, its used to define the average_episode_step_min when sampling hyperparams
no_candles_for_val = 5000  #I dont know what that is, its not used




def extract_TA(price_df):
    #extract standard technical indicators from the data, there are hundreds to chose from
    ta_df = pd.DataFrame(index = price_df.index)
    ta_df['price'] = price_df['close']
    ta_df['dprice'] = (price_df['close'] - price_df['open'])/price_df['open']
    ta_df['rsi'] = RSI(price_df['close'], timeperiod=14)
    ta_df['macd'], _, _ = MACD(price_df['close'], fastperiod=12,slowperiod=26, signalperiod=9)
    ta_df['cci'] = CCI(price_df['high'], price_df['low'], price_df['close'], timeperiod=14)
    ta_df['dx'] = DX(price_df['high'], price_df['low'], price_df['close'], timeperiod=14)
    ta_df['roc'] = ROC(price_df['close'], timeperiod=10)
    ta_df['ultosc'] = ULTOSC(price_df['high'], price_df['low'], price_df['close'])
    ta_df['willr'] = WILLR(price_df['high'],price_df['low'], price_df['close'])
    ta_df['obv'] = OBV(price_df['close'], price_df['volume'])
    ta_df['ht_dcp'] = HT_DCPHASE(price_df['close'])
    ta_df['volume'] = price_df['volume']
    ta_df['volat'] = (price_df['high'] - price_df['low'])/price_df['close']
    
    
    #twitter sentiment -> Anthony
    
    #On-chain data -> Luis
    
    #Order-book data -> Aurelien
    
    return ta_df



def main():
    
    """
    The code below was taken from the function_train_test.py from FinRL-Crypto    
    """
    
    
    price_df1 = pd.read_csv('Data/Binance_OCHL/1h/BTCUSDT-1h-data.csv')
    ta_feature_df1 = extract_TA(price_df1)
    ta_feature_df1 = ta_feature_df1.dropna()
    price_df1 = price_df1.loc[ta_feature_df1.index]
    #price_df1 = price_df1.iloc[:5000]
    
    #price_df2 = pd.read_csv('Data/Binance_OCHL/1h/ETHUSDT-1h-data.csv')
    #ta_feature_df2 = extract_TA(price_df2)
    #ta_feature_df2 = ta_feature_df2.dropna()
    #price_df2 = price_df2.loc[ta_feature_df2.index] 
    #price_df2 = price_df2.iloc[:5000]
    
    #print(price_df1)
    #print(price_df2)
    #print(ta_feature_df1)
    
    #price_array = np.array([price_df1['close'].to_numpy(),price_df2['close'].to_numpy()]).reshape(-1,2)
    price_array = price_df1['close'].to_numpy().reshape(-1,1)
    feature_array = ta_feature_df1.to_numpy()
    feature_names = ta_feature_df1.columns
    
    #plot feature intercorrelation
    nfeatures = feature_array.shape[1]
    cor_matrix = np.ones((nfeatures,nfeatures))
    for i in range(nfeatures):
        for j in range(i+1, nfeatures):
            cor_matrix[i,j] = stats.pearsonr(feature_array[:,i],feature_array[:,j])[0]
            cor_matrix[j,i] = cor_matrix[i,j]
    
    plt.figure(figsize=(7,7))
    plt.imshow(np.abs(cor_matrix), origin='lower', vmin=0, vmax = 1, cmap = 'inferno')
    plt.xticks(np.arange(nfeatures), feature_names, rotation=90)
    plt.yticks(np.arange(nfeatures), feature_names)
    plt.show()
    
    nt = len(price_array)
    ntrain = int(2*nt/3)
    train_indices = np.arange(ntrain)
    test_indices = np.arange(ntrain,nt)
    
    print('price_array', price_array.shape) #price array can contain several currencies
    print('feature_array', feature_array.shape) #contains all the features (technical indicators etc)
    print('train_indices', train_indices.shape)
    print('test_indices', test_indices.shape)
    print()
    
    
    
    #(data_from_processor, price_array, tech_array, time_array)
    #(249990, 8) (24999, 10) (24999, 70) (24999,)
    
    model_name = 'ppo'
    
    #trial = "<optuna.trial._trial.Trial object at 0x000001F277789940>"
    env = CryptoEnvAlpaca

    env_params = {'ALPACA_limits':ALPACA_LIMITS, 'lookback': 1, 'norm_cash': 0.000244140625, 'norm_stocks': 0.00390625, 'norm_tech': 3.0517578125e-05, 'norm_reward': 0.0009765625, 'norm_action': 10000}
    erl_params = {'learning_rate': 0.03, 'batch_size': 512, 'gamma': 0.99, 'net_dimension': 1024, 'target_step': 37500, 'eval_time_gap': 60, 'break_step': 45000.0}
    break_step = erl_params['break_step']
    cwd = './train_results/cwd_tests/model_CPCV_ppo_5m_50H_25k'
    gpu_id = 0
    
    
    sharpe_bot, sharpe_eqw = train_and_test(price_array, feature_array, train_indices,
                                                test_indices, env, model_name, env_params,
                                                erl_params, break_step, cwd, gpu_id)
    
    
    print()
    print('Sharpe ratio RL agent', sharpe_bot)
    print('Sharpe ratio baseline (Equal weight portfolio)',sharpe_eqw)









def train_and_test(price_array, tech_array, train_indices, test_indices, env, 
                   model_name, env_params, erl_params,break_step, cwd, gpu_id):
    
    """
    Train and test a DRL agent with a given set of hyper parameters
    
    INPUT:
        trial:
        price_array:    2D array of the price (cen be more than one price)
        tech_array:     2D array of features
        train_indices:  list of indices to use for train
        test_indices:   list of indices to use for test
        env:
        model_name:     DRL agent to use (ppo, a2c, ddpg, td3, sac)
        env_params:
        erl_params:
        break_step:
        cwd:
        gpu_id:
            
    OUTPUT:
        sharpe_bot:
        sharpe_eqw:
        
    """
    
    print('Training the %s agent...' % model_name)   
    train_agent(price_array,tech_array,train_indices,env, model_name, env_params,erl_params,break_step,cwd,gpu_id)
    
    print('Testing the %s agent...' % model_name)  
    sharpe_bot, sharpe_eqw = test_agent(price_array, tech_array, test_indices, env, 
                                                      env_params, model_name, cwd, gpu_id, erl_params)
    
    return sharpe_bot, sharpe_eqw




def train_agent(price_array, tech_array, train_indices, env, model_name, env_params, erl_params, break_step, cwd, gpu_id):
    print('No. Train Samples:', len(train_indices), '\n')
    price_array_train = price_array[train_indices, :]
    tech_array_train = tech_array[train_indices, :]

    agent = DRLAgent_erl(env=env, price_array=price_array_train, tech_array=tech_array_train, env_params=env_params, if_log=True)
    model = agent.get_model(model_name,gpu_id,model_kwargs=erl_params)
    agent.train_model(model=model,cwd=cwd,total_timesteps=break_step)
    
    
    
    
    
def test_agent(price_array, tech_array, test_indices, env, env_params, model_name, cwd, gpu_id, erl_params):
    print('\nNo. Test Samples:', len(test_indices))
    price_array_test = price_array[test_indices, :]
    tech_array_test = tech_array[test_indices, :]

    data_config = {"price_array": price_array_test,"tech_array": tech_array_test,"if_train": False}
    env_instance = env(config=data_config,env_params=env_params,if_log=True)
    net_dimension = erl_params['net_dimension']

    account_value_erl = DRLAgent_erl.DRL_prediction(model_name=model_name, cwd=cwd,
                                                    net_dimension=net_dimension,
                                                    environment=env_instance, gpu_id=gpu_id)
    
    plt.plot(account_value_erl)
    plt.show()
    
    
    lookback = env_params['lookback']
    indice_start = lookback - 1
    indice_end = len(price_array_test) - lookback

    # EVALUATING the DRL agent performance (need to understand whats going on)
    #data_points_per_year = compute_data_points_per_year(trial.user_attrs["timeframe"])#
    #data_points_per_year = 365*24*60 #if its 1min data
    data_points_per_year = 365*24 #if 1h data
    account_value_eqw, eqw_PnL, eqw_cumrets = compute_eqw(price_array_test, indice_start, indice_end)
    dataset_size = np.shape(eqw_PnL)[0]
    factor = data_points_per_year / dataset_size
    
    account_value_eqw = np.array(account_value_eqw)
    account_value_erl = np.array(account_value_erl)
    eqw_PnL = account_value_eqw[1:] - account_value_eqw[:-1]
    sharpe_eqw, _ = sharpe_iid(eqw_PnL, bench=0, factor=factor, log=False)
    bot_PnL = account_value_erl[1:] - account_value_erl[:-1]
    sharpe_bot, _ = sharpe_iid(bot_PnL, bench=0, factor=factor, log=False)
    
    #Note, you can also get the sortonio ratio with sortino_iid function
    
    
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')
    plt.rcParams.update({'font.size': 20})
    
    nt = len(bot_PnL)
    plt.figure(figsize=(11,4))
    plt.plot(account_value_eqw/account_value_eqw[0], label = 'Baseline (Equal weight)', color = 'green')
    plt.plot(account_value_erl/account_value_erl[0], label = 'PPO agent', color = 'blue')
    plt.yscale('log')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Cum. PnL')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(7,4))
    bot_ROI = np.array([bot_PnL[i]/account_value_erl[i] for i in range(nt)]) 
    plt.hist(bot_ROI*100, bins=75, color = 'blue', alpha = 0.5, density=True)
    plt.axvline(np.mean(bot_ROI*100), color = 'black', linestyle = '--')
    plt.ylabel('Density')
    plt.xlabel('ROI by timeperiod [%]')
    plt.xlim(-7,7)
    plt.show()
    print('Mean Agent ROI is %.5f%% / timepoint' % np.mean(bot_ROI*100))
    print('Cummulated ROI is %.2f%%' % (((account_value_erl[-1] - account_value_erl[0])/account_value_erl[0])*100))

    plt.figure(figsize=(7,4))
    eqw_ROI = np.array([eqw_PnL[i]/account_value_eqw[i] for i in range(nt)]) 
    plt.hist(eqw_ROI*100, bins=75, color = 'green', alpha = 0.5, density=True)
    plt.axvline(np.mean(eqw_ROI*100), color = 'black', linestyle = '--')
    plt.xlim(-7,7)
    plt.ylabel('Density')
    plt.xlabel('ROI by timeperiod [%]')
    plt.show()
    print('Mean Baseline ROI is %.5f%% / timepoint' % np.mean(eqw_ROI*100))
    print('Cummulated ROI is %.2f%%' % (((account_value_eqw[-1] - account_value_eqw[0])/account_value_eqw[0])*100))
    
    
    

    return sharpe_bot, sharpe_eqw



def sample_hyperparams(trial):
    average_episode_step_min = no_candles_for_train + 0.25 * no_candles_for_train
    sampled_erl_params = {
        "learning_rate": trial.suggest_categorical("learning_rate", [3e-2, 2.3e-2, 1.5e-2, 7.5e-3, 5e-6]),
        "batch_size": trial.suggest_categorical("batch_size", [512, 1280, 2048, 3080]),
        "gamma": trial.suggest_categorical("gamma", [0.85, 0.99, 0.999]),
        "net_dimension": trial.suggest_categorical("net_dimension", [2 ** 9, 2 ** 10, 2 ** 11, 2 ** 12]),
        "target_step": trial.suggest_categorical("target_step",
                                                 [average_episode_step_min, round(1.5 * average_episode_step_min),
                                                  2 * average_episode_step_min]),
        "eval_time_gap": trial.suggest_categorical("eval_time_gap", [60]),
        "break_step": trial.suggest_categorical("break_step", [3e4, 4.5e4, 6e4])
    }

    # environment normalization and lookback
    sampled_env_params = {
        "lookback": trial.suggest_categorical("lookback", [1]),
        "norm_cash": trial.suggest_categorical("norm_cash", [2 ** -12]),
        "norm_stocks": trial.suggest_categorical("norm_stocks", [2 ** -8]),
        "norm_tech": trial.suggest_categorical("norm_tech", [2 ** -15]),
        "norm_reward": trial.suggest_categorical("norm_reward", [2 ** -10]),
        "norm_action": trial.suggest_categorical("norm_action", [10000])
    }
    return sampled_erl_params, sampled_env_params



if __name__ == "__main__":
    main()
