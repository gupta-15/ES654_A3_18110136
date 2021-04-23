from sklearn.datasets import load_digits
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

X = pd.DataFrame(load_digits().data)
y = pd.DataFrame(load_digits().target,columns= ['target'])

X = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents, columns = ['Component 1', 'Component 2'])
finalDf = pd.concat([principalDf, y], axis = 1)
#print(finalDf['target'])

fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_title('2-component PCA')
digits = [0,1, 2,3,4,5,6,7,8,9]
colors = ['r', 'g', 'b','c','m','y','k','#fa8174','0.75','#8EBA42']
for digit, color in zip(digits,colors):
    indices = (finalDf['target'] == digit)
    ax.scatter(finalDf.loc[indices, 'Component 1'], finalDf.loc[indices, 'Component 2'], c = color, s = 40)
ax.legend(digits)
ax.grid()
plt.show()