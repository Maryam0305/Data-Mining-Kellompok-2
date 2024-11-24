# %%
import pandas as pd

# Load dataset
data = pd.read_csv('data/pima_diabetes.csv')
data.head()


# %%
data.info()
print(data.isnull().sum())


# %%
# Contoh membersihkan data (sesuaikan dengan kebutuhan dataset Anda)
data = data.dropna()
data.to_csv('data/cleaned_pima_diabetes.csv', index=False)


# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Distribusi data
sns.pairplot(data)
plt.show()


# %%
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()



