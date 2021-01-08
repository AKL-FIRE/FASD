from sklearn.feature_selection import VarianceThreshold
import DataReader
import pandas as pd

control_data = DataReader.DataShow("./data/control_rawdata.npy")
control_data.load_data()

features = ['C', 'I', 'O', 'F', 'M', 'LTXE', 'CIOFM', 'CIOFMLTXE', 'Scorr', 'Var', 'EyeMvt']

data = pd.DataFrame(control_data.data[0][0][17:, 3:14], columns=features)
selector = VarianceThreshold(1)
X = data.to_numpy()
selector.fit(X)

print('Variances is %s'%selector.variances_)
print('After transform is \n%s'%selector.transform(X))
print('The surport is %s'%selector.get_support(True))  # 如果为True那么返回的是被选中的特征的下标
print('The surport is %s'%selector.get_support(False))  # 如果为FALSE那么返回的是布尔类型的列表，反应是否选中这列特征
print('The feature is %s' % [x for (index, x) in enumerate(features) if index in selector.get_support(True)])
print('After reverse transform is \n%s'%selector.inverse_transform(selector.transform(X)))