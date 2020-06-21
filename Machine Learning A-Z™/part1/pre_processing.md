### Importing the libraries.
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

### Importing Datasets
```python
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

### Taking Care of missing Dataset
```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X[:, 1:])
X[:, 1:] = imputer.transform(X[:, 1:])
```

### Encoding Categorical Data

#### Encoding the Independent variable
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformer=[("encoder", OneHotEncoder(), [0])], remainder="passthrough")

X = np.array(ct.fit_transform(X))
```

#### Encoding the Dependent Variable
```python
from sklearn.preprocessing import LabelEnconder
le = LabelEncoder()
y = le.fit_transform(y)
```
### Splitting the dataset into the Training set and Test set

```python
from sklearn.model_selection import train_test_split
X_train, Y_train, X_test, Y_test = train_test_split(X, y, test_size=0.2, random_state=1)
```

### Feature Scaling
#### There are two types of feature scaling
- Standardisation   
  Xstand = (x - mean(x)) / standard deviation (X)
- Normalisation   
  Xnorm = (X - min(X)) / (max(X) - min(X))
  
 ```python
 from sklearn.preprocessing import StandardScaler
 standard_scaler = StandardScaler()
 X_train[:, 3:] = standard_scaler.fit_transfor(X_train[:, 3:])
 X_test[:, 3:] = standard_scaler.transform(X_test[:, 3:])
 ```
