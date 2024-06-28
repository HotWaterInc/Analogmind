from sklearn.preprocessing import MinMaxScaler

def normalize_data_min_max(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)


