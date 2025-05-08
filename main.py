import pandas as pd

from sklearn.preprocessing import MinMaxScaler

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
        Dos, R2L, Probe, U2R, Normal
    """
    df = df.copy()
    
    df['label'] = df.label.replace([
        'apache2','back','land','neptune','mailbomb','pod','processtable',
        'smurf','teardrop','udpstorm','worm'
    ], 'Dos')

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
    df['label'] = df.label.where(df.label.isin(['Dos', 'R2L', 'Probe', 'U2R']), 'Normal')
    
    return df

if __name__ == '__main__':
    # 訓練資料集讀取
    train_dataset = pd.read_csv('C:/Users/hansc/OneDrive/School/專題製作/Code/NSL-KDD Dataset/KDDTrain+.csv', encoding='utf-8')
    train_dataset_header = list(train_dataset.columns)
    train_dataset = pd.DataFrame(data=train_dataset, columns=train_dataset_header)

    # 測試資料集讀取
    test_dataset = pd.read_csv('C:/Users/hansc/OneDrive/School/專題製作/Code/NSL-KDD Dataset/KDDTrain+.csv', encoding='utf-8')
    test_dataset_header = list(test_dataset.columns)
    test_dataset = pd.DataFrame(data=test_dataset, columns=test_dataset_header)
    
    # 資料預處理 - one-hot encoding
    train_dataset = one_hot_encoding(train_dataset)
    test_dataset = one_hot_encoding(test_dataset)
    
    # 資料預處理 - MinMaxScaler 轉換
    train_dataset = normalization(train_dataset)
    test_dataset = normalization(test_dataset)
    
    # 資料預處理 - label 轉換
    train_dataset = change_label(train_dataset)