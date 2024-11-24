import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_dataset(data):
    # Handle missing values
    for column in data.columns:
        if data[column].dtype in ['int64', 'float64']:
            data[column].fillna(data[column].median(), inplace=True)
        else:
            data[column].fillna(data[column].mode()[0], inplace=True)
    
    # Handle outliers using IQR
    for column in data.select_dtypes(include=['int64', 'float64']).columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[column] = data[column].clip(lower_bound, upper_bound)
    
    # Encode categorical columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    # Split dataset
    if 'Penjualan Bersih' in data.columns:
        X = data.drop('Penjualan Bersih', axis=1)
        y = data['Penjualan Bersih']
    else:
        X = data
        y = None
    
    # Standardize numerical data
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X, y

# Load dataset
data = pd.read_csv('Data2.csv')
X, y = preprocess_dataset(data)

# Save processed features and labels to Excel
X.to_excel('Processed_Features.xlsx', index=False)
if y is not None:
    y.to_excel('Processed_Labels.xlsx', index=False)

print("Hasil preprocessing telah disimpan ke dalam file Excel:")
print("1. Processed_Features.xlsx untuk fitur (X)")
if y is not None:
    print("2. Processed_Labels.xlsx untuk label (y)")
