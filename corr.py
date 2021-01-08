import DataReader
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

control_data = DataReader.DataShow("./data/control_rawdata.npy")
control_data.load_data()

features = ['C', 'I', 'O', 'F', 'M', 'LTXE', 'CIOFM', 'CIOFMLTXE', 'Scorr', 'Var', 'EyeMvt']

data = pd.DataFrame(control_data.data[0][0][:,3:14],
             columns=features)
# 计算特征之间的相似度
corr_values1 = data.corr()

f, ax= plt.subplots(figsize = (11, 11))
sns.heatmap(corr_values1, cmap='RdBu', linewidths = 0.05, ax = ax, annot=True)
# 设置Axes的标题
ax.set_title('Correlation between features')
plt.show()

