from scipy.io import loadmat
from numpy import array,reshape
from tqdm import tqdm


def parse_dataset(dataset_path,window_size,limit):
    #Loading the raw mat file, and extracting the raw (filtered) eeg Data and the associated labels
    raw_mat = loadmat(dataset_path)
    data = list(raw_mat['o'])
    labels = data[0][0][4]
    eeg = array(data[0][0][5])

    X = []
    y = []

    for i in tqdm(range(len(eeg)-window_size)):
        if labels[i][0] == 0:
            continue
        X.append(eeg.T[0:,i-window_size:i+window_size])
        y.append(labels[i][0]-1)

    return X[:limit],y[:limit]

def flattenMatrixDataset(D_M):
    return reshape(D_M,[len(D_M),D_M.shape[1]*D_M.shape[2]])




