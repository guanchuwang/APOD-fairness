import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

import pickle

class PandasDataSet(TensorDataset):

    def __init__(self, *dataframes):
        tensors = (self._df_to_tensor(df) for df in dataframes)
        super(PandasDataSet, self).__init__(*tensors)

    def _df_to_tensor(self, df):
        if isinstance(df, pd.Series):
            df = df.to_frame('dummy')
        return torch.from_numpy(df.values).float()


def load_ICU_data(path):

    input_data = pd.read_csv(path)

    print(type(input_data))
    # sensitive attributes; we identify 'race' and 'sex' as sensitive attributes
    sensitive_attribs = ['gender']
    # Z = (input_data.loc[:, sensitive_attribs]
    #      .assign(Sex_Code_Text=lambda df: (df['gender'] == 'Male').astype(int)))

    # print((Z.loc[0:2000, sensitive_attribs] == 0).sum())
    Z = input_data.loc[:, sensitive_attribs]
    # print(Z)
    # sensitive_attribs = ['race', 'sex']
    # Z = (input_data.loc[:, sensitive_attribs]
    #      .assign(race=lambda df: (df['race'] == 'White').astype(int),
    #              sex=lambda df: (df['sex'] == 'Male').astype(int)))

    # targets; 1 when someone makes over 50k , otherwise 0
    y = (input_data['I_am_working_in_field'] == -1).astype(int)

    # features; note that the 'target' and sentive attribute columns are dropped

    # 'Ethnic_Code_Text', 'DateOfBirth', 'Screening_Date',
    # 'Person_ID', 'AssessmentID', 'Case_ID', 'Agency_Text', 'LastName',
    # 'FirstName', 'MiddleName'

    X = (input_data
         .drop(columns=['gender', 'I_am_working_in_field', 'user_id'])
         .fillna('Unknown')
         .pipe(pd.get_dummies, drop_first=False))

    # print(X)

    print(f"features X: {X.shape[0]} samples, {X.shape[1]} attributes")
    print(f"targets y: {y.shape} samples")
    print(f"sensitives Z: {Z.shape[0]} samples, {Z.shape[1]} attributes")
    return X, y, Z

def main():

    # load ICU data set
    X, y, Z = load_ICU_data('./datasets/pokec-dataset/region_job.csv')

    n_instance = X.shape[0]
    n_features = X.shape[1]
    n_sensitive = Z.shape[1]

    print('n_instance:', n_instance, 'n_features', n_features, 'n_sensitive', n_sensitive)

    # split into train/test set
    (X_train, X_test, y_train, y_test, Z_train, Z_test) = train_test_split(X, y, Z,
                                                                           train_size=30000,
                                                                           test_size=100,
                                                                           stratify=y, random_state=7)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(list(X_train.keys()))


    print('Z train Sex minority:', ((Z_train['gender'] == 0)).sum() * 1. / len(Z_train['gender']))
    print('Z test Sex minority:', ((Z_test['gender'] == 0)).sum()*1./len(Z_test['gender']))
    print('Y train minority:', ((y_train == 1)).sum()*1./len(y_train))
    print('Y test minority:', ((y_test == 1)).sum()*1./len(y_test))

    X_train, y_train, Z_train = X_train.values, y_train.values, Z_train.values.squeeze(axis=1)
    X_test, y_test, Z_test = X_test.values, y_test.values, Z_test.values.squeeze(axis=1)

    shuffle_index = np.random.permutation(X_train.shape[0])
    X_train, y_train, Z_train = X_train[shuffle_index], y_train[shuffle_index], Z_train[shuffle_index]
    shuffle_index = np.random.permutation(X_test.shape[0])
    X_test, y_test, Z_test = X_test[shuffle_index], y_test[shuffle_index], Z_test[shuffle_index]



    # with open('data/pokec/pokec_train.pkl', 'wb') as fileObject:
    #     pickle.dump((X_train, y_train, Z_train), fileObject)  # 保存list到文件
    #     fileObject.close()
    #
    # with open('data/pokec/pokec_test.pkl', 'wb') as fileObject:
    #     pickle.dump((X_test, y_test, Z_test), fileObject)  # 保存list到文件
    #     fileObject.close()
    #
    # with open('data/pokec/pokec_train.pkl', 'rb') as fileObject:
    #     X_train, y_train, Z_train = pickle.load(fileObject)
    #     fileObject.close()
    #
    # with open('data/pokec/pokec_test.pkl', 'rb') as fileObject:
    #     X_test, y_test, Z_test = pickle.load(fileObject)
    #     fileObject.close()

    # print(X_train, y_train, Z_train)
    # print(X_test, y_test, Z_test)

    # print((Z_train == 0).sum())
    # print((Z_test == 0).sum())

    # np.save('data/adult-data/adult_train.pkl', [X_train, y_train, Z_train])
    # np.save('data/adult-data/adult_test.pkl', [X_test, y_test, Z_test])

    

if __name__ == "__main__":

    torch.manual_seed(7)
    np.random.seed(4) # 11
    # random.seed(7)

    main()