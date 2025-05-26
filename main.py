import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import LabelEncoder
from evobagging_methods import EvoBagging 

def check_dataset(dataset, mode='shape'):
    if mode == 'info':
        print(dataset.info())
    elif mode == 'head':
        print(dataset.head())
    elif mode == 'shape':
        print(f"{dataset.shape}")
    else:
        raise ValueError("Invalid mode. Use 'info', 'head' or 'shape'.")
    
def one_hot_encoding(data):
    non_numerial_columns = ['protocol_type', 'service', 'flag']
    data_encoded = pd.get_dummies(data, columns=non_numerial_columns, drop_first=True, dtype=int) # drop first: 忽略資料標題
    
    return data_encoded

def normalization(data):
    scaler = MinMaxScaler()
    
    features = data.drop(columns=['label'])
    
    feature_encoded = pd.DataFrame(scaler.fit_transform(features),
                                   columns=features.columns,
                                   index=features.index)
    
    data = pd.concat([feature_encoded, data['label']], axis=1)
    
    return data

def change_label(df):
    """
        將 label 轉換成 5 類別：
        DoS, R2L, Probe, U2R, Normal
    """
    df = df.copy()
    
    df['label'] = df.label.replace([
        'apache2','back','land','neptune','mailbomb','pod','processtable',
        'smurf','teardrop','udpstorm','worm'
    ], 'DoS')

    df['label'] = df.label.replace([
        'ftp_write','guess_passwd','httptunnel','imap','multihop','named',
        'phf','sendmail','snmpgetattack','snmpguess','spy','warezclient',
        'warezmaster','xlock','xsnoop'
    ], 'R2L')      

    df['label'] = df.label.replace([
        'ipsweep','mscan','nmap','portsweep','saint','satan'
    ], 'Probe')

    df['label'] = df.label.replace([
        'buffer_overflow','loadmodule','perl','ps','rootkit','sqlattack','xterm'
    ], 'U2R')

    # 其餘未列入的視為 'Normal'
    df['label'] = df.label.where(df.label.isin(['DoS', 'R2L', 'Probe', 'U2R']), 'Normal')
    
    return df

def label_distribution(data1, data2):
    """
        繪製資料集的標籤分佈圖
    """
    data1_counts = data1['label'].value_counts()
    data2_counts = data2['label'].value_counts()
    
    combined  = pd.DataFrame({
        'Train': data1_counts,
        'Test': data2_counts
    }).fillna(0) # 若某些 label 在其中一個資料集中缺少，填 0
    
    # 自訂排序
    order = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
    
    combined = combined.reindex(order)
    
    # 繪製圖形
    combined.plot(kind='bar', figsize=(12, 6))
    plt.title('NSL-KDD Train vs Test Label Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(rotation=45) 
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.show()
    
columns = (['duration'
,'protocol_type'
,'service'
,'flag'
,'src_bytes'
,'dst_bytes'
,'land'
,'wrong_fragment'
,'urgent'
,'hot'
,'num_failed_logins'
,'logged_in'
,'num_compromised'
,'root_shell'
,'su_attempted'
,'num_root'
,'num_file_creations'
,'num_shells'
,'num_access_files'
,'num_outbound_cmds'
,'is_host_login'
,'is_guest_login'
,'count'
,'srv_count'
,'serror_rate'
,'srv_serror_rate'
,'rerror_rate'
,'srv_rerror_rate'
,'same_srv_rate'
,'diff_srv_rate'
,'srv_diff_host_rate'
,'dst_host_count'
,'dst_host_srv_count'
,'dst_host_same_srv_rate'
,'dst_host_diff_srv_rate'
,'dst_host_same_src_port_rate'
,'dst_host_srv_diff_host_rate'
,'dst_host_serror_rate'
,'dst_host_srv_serror_rate'
,'dst_host_rerror_rate'
,'dst_host_srv_rerror_rate'
,'label'
,'level'])   

def main():
    # 載入NSL-KDD資料集

    # 訓練資料集讀取
    train_dataset = pd.read_csv('KDDTrain+.csv')
    train_dataset.columns = columns
    train_dataset = train_dataset.drop(columns=['level']) # 去除 level 欄位
    train_dataset_header = list(train_dataset.columns) 
    train_dataset = pd.DataFrame(data=train_dataset, columns=train_dataset_header)

    # 測試資料集讀取
    test_dataset = pd.read_csv('KDDTest+.csv')
    test_dataset.columns = columns
    test_dataset = test_dataset.drop(columns=['level']) # 去除 level 欄位
    test_dataset_header = list(test_dataset.columns)
    test_dataset = pd.DataFrame(data=test_dataset, columns=test_dataset_header)
    
    # one-hot encoding
    train_dataset = one_hot_encoding(train_dataset)
    test_dataset = one_hot_encoding(test_dataset)
    
    # Normalization
    train_dataset = normalization(train_dataset)
    test_dataset = normalization(test_dataset)
    
    # Label Change
    train_dataset = change_label(train_dataset)
    test_dataset = change_label(test_dataset)
    
    # Drop Normal data
    train_dataset = train_dataset[train_dataset['label'] != 'Normal']
    test_dataset = test_dataset[test_dataset['label'] != 'Normal']
    
    # 訓練的題目與答案
    X_train = train_dataset.drop(columns=['label']) 
    Y_train = train_dataset['label'] 

    # 測試的題目與答案
    X_test = test_dataset.drop(columns=['label']) 
    Y_test = test_dataset['label'] 

    oversampler = ADASYN() # sampling_strategy: 欲平衡的類別比例, random_state: 隨機種子, n_neighbors: 近鄰數量
    label_encoder = LabelEncoder()

    Y_train_encoded = label_encoder.fit_transform(Y_train) 
    Y_test_encoded = label_encoder.transform(Y_test)

    X_train, Y_train = oversampler.fit_resample(X_train, Y_train_encoded) # 進行過採樣

    Y_train = pd.DataFrame(Y_train, columns=['label'], index=X_train.index)
    
    # EvoBagging parameter setting
    n_select = 5  # 選擇袋子數量
    n_new_bags = 3  # 新袋子數量
    max_initial_size = 1000  # 初始袋子尺寸
    n_crossover = 4  # 交配袋子數量
    n_mutation = 2  # 突變袋子數量
    mutation_size = 50  # 突變資料量
    size_coef = 1000  # 袋子大小權重
    metric = 'f1'  # 評估指標
    procs = 4  # 並行處理的進程數
    
    num_initial_bags = 10

    
    evoBag = EvoBagging(X_train, Y_train,
                        n_select, n_new_bags,
                        max_initial_size, n_crossover,
                        n_mutation, mutation_size,
                        size_coef, metric, procs)
    
    initial_bags = {}