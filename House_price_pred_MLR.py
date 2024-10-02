import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset

df = pd.read_csv('kc_house_data.csv')

# Display the first few rows of the dataset

print(df.head())

# Display the summary statistics of the dataset

print(df.describe())

# Check for missing values

print(df.isnull().sum())

#Drop unnecessary columns

df.drop(['id', 'date'], axis=1, inplace=True)

# Understand distribution with seaborn

#with sns.plotting_context('notebook',font_scale=2.0):
    #fig= sns.pairplot(df[['sqft_lot', 'sqft_above', 'price', 'sqft_living','bedrooms']],hue='bedrooms',height=5,palette='tab20')
    #plt.show()

# Seperate the dataset into dependent and independent

X = df.iloc[:, 1:].values

y = df.iloc[:, 0].values

# Split the dataset into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training Multiple Linear Regression

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)

# Predicting the Test set results

y_pred = regressor.predict(X_test)


#Backward Elimination
import statsmodels.api as sm

def backward_elimination(X, y, SL):
    numVars = X.shape[1]  # Number of columns (variables)
    remaining_columns = list(range(numVars))  # Track the indices of columns in X
    
    while numVars > 0:
        regressor_OLS = sm.OLS(y, X).fit()  # Fit the OLS model
        max_pval = max(regressor_OLS.pvalues)  # Get the maximum p-value
        
        if max_pval > SL:  # Check if the max p-value is greater than significance level (SL)
            max_pval_idx = regressor_OLS.pvalues.argmax()  # Get index of the variable with max p-value
            
            # Remove the column with the highest p-value
            X = np.delete(X, max_pval_idx, 1)  # Remove column from X
            del remaining_columns[max_pval_idx]  # Remove from the list of remaining columns
            numVars -= 1  # Decrease the number of variables
        else:
            break  # Stop if all p-values are below SL
    
    return X, regressor_OLS, remaining_columns

# Example usage
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]]  # Initial features
X_Modeled, final_model, remaining_columns = backward_elimination(X_opt, y, SL)

# Print the columns that remain after backward elimination
print("Remaining columns after backward elimination:", remaining_columns)

print(final_model.summary())