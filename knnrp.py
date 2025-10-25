import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("insurance.csv")



# Creates a dataframe using the drop method,
# which has two parameters:
# The first parameter tells which labels to remove
# (Columns Name or
# The second parameter tells whether to remove a row index or
# a column name. axis=1 means we want to remove a column.
features = df.drop("charges",axis=1)


# Creates a dataframe from just one column
targets = df["charges"]

print(features.head())


print(targets.head())

features_train, features_test, targets_train, targets_test = train_test_split(features, targets, test_size=0.2, random_state=10)

categorical_features = ["sex", "smoker", "region"]


numeric_features = ["age", "bmi", "children"]

preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(), categorical_features ) , ("num", MinMaxScaler(), numeric_features)
]
)

preprocessed_features_train = preprocessor.fit_transform(features_train)

preprocessed_features_test = preprocessor.transform(features_test)



knnr = KNeighborsRegressor(n_neighbors=8)
knnr.fit(preprocessed_features_train, targets_train) 

predictions = knnr.predict(preprocessed_features_test)



#Previous code omitted. 

#Compute the Mean absolute error (MAE):
mae = mean_absolute_error(targets_test, predictions)
print(f"MAE: {mae:.3f}")

#Compute the Mean Squared Error (MSE):
mse = mean_squared_error(targets_test, predictions)
print(f"MSE: {mse:.3f}")

#Compute the Root Mean Squared Error (RMSE):
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.3f}")







