import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ... reading Data ...
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# ... correlation from SalePrice ...
(train.corr()**2)['SalePrice'].sort_values(ascending=False)[:20]

# ... normal distribution ...
train['SalePrice'] = np.log1p(train['SalePrice']) # logarithm to normalize

# ... combine test and train data ...
all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)

# ... missing values ...
Na_columns = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
              'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
              'PoolQC', 'Fence', 'MiscFeature', 'MasVnrType', 'MSZoning']

for item in Na_columns:
    all_data[item] = all_data[item].fillna('None')

# ... data cleaning ...
# ... some features consist of numbers that are actually categories so we will convert to str ...
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['OverallQual'] = all_data['OverallQual'].astype(str)
all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)
all_data['YearBuilt'] = all_data['YearBuilt'].astype(str)
all_data['YearRemodAdd'] = all_data['YearRemodAdd'].astype(str)
all_data['GarageYrBlt'] = all_data['GarageYrBlt'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)


# filling with default values
all_data['Functional'] = all_data['Functional'].fillna('Typ')
all_data['Utilities'] = all_data['Utilities'].fillna('AllPub')
all_data['SaleType'] = all_data['SaleType'].fillna('Oth')
all_data['Electrical'] = all_data['Electrical'].fillna('SBrkr')

# if data is missing there is none
all_data['LotFrontage'] = all_data['LotFrontage'].fillna(0)
all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)
all_data['BsmtFullBath'] = all_data['BsmtFullBath'].fillna(0)
all_data['BsmtHalfBath'] = all_data['BsmtHalfBath'].fillna(0)
all_data['BsmtFinSF1'] = all_data['BsmtFinSF1'].fillna(0)
all_data['BsmtFinSF2'] = all_data['BsmtFinSF2'].fillna(0)
all_data['TotalBsmtSF'] = all_data['TotalBsmtSF'].fillna(0)
all_data['GarageCars'] = all_data['GarageCars'].fillna(0)
all_data['GarageYrBlt'] = all_data['GarageYrBlt'].fillna(0)
all_data['GarageArea'] = all_data['GarageArea'].fillna(0)
all_data['BsmtUnfSF'] = all_data['BsmtUnfSF'].fillna(0)

# filling with most common values
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

# basement + floor 1 + floor 2
all_data['TotalSQFT'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
# strings -> concatenate
all_data['YearBuilt/remodelled'] = (all_data['YearBuilt'] + all_data['YearRemodAdd'])
all_data['BSMT'] = all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBathroom'] = (all_data['FullBath'] + all_data['HalfBath'] + all_data['BsmtFullBath'] + all_data['BsmtHalfBath'])


# ... there are ordinal scales wich will be converted to numbers (excellent - good - poor is 3-2-1) ...
# ... removing duplicates from list ...
def Dup(x):
    return list(dict.fromkeys(x))


columns_2_order = ['LotShape', 'Utilities', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                   'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'KitchenQual', 'FireplaceQu', 'GarageFinish',
                   'GarageQual', 'GarageCond', 'PoolQC', 'Fence']

# ... converting original categorical features to scales ...
for item in columns_2_order:
    n = len(Dup(all_data[item]))
    sample_dic = {}
    vlist = []
    nlist = []

    for val in Dup(all_data[item]):
        vlist.append(val)
        nlist.append(n)
        n -= 1
    sample_dic = dict(zip(vlist, nlist))
    all_data[item] = all_data[item].map(sample_dic)

# ... add more columns
all_data['hasPool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['has2ndFloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['hasGarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['hasBsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['hasFireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

final_features = pd.get_dummies(all_data).reset_index(drop=True)
final_features.drop('Id', axis=1, inplace=True)


x = final_features[:1460]
y = pd.DataFrame(data=train['SalePrice'])
# ... we have too many colomns. Which categories contribute the most? ...
# ... PCA = Python's Scikit-learn Library ...
sc = StandardScaler()
x = sc.fit_transform(x)
pca_com = 250
pca = PCA(n_components=1212)  # 978
x = pca.fit_transform(x)

explained_variance = pca.explained_variance_ratio_
# ... order columns by their variance and take the first 250 ...
var_matrix = (pd.concat([pd.DataFrame(list(final_features.columns), columns=['Factors']),
                         pd.DataFrame(list(explained_variance ** 2), columns=['^2 Variance'])],
                        axis=1).sort_values(by='^2 Variance', ascending=False)[0:pca_com])

list(var_matrix['Factors'])
x_new = final_features[list(var_matrix['Factors'])]


x_train = x_new[:1460]
x_test = x_new[1460:]

# ... linear regression ...
# ... for checking the model ...
x_train_T, x_test_T, y_train_T, y_test_T = train_test_split(x_train, y, test_size=0.4)

# ... ridge regression ...
model = linear_model.Ridge(alpha=20)
model = model.fit(x_train_T, y_train_T)
predictions_T = model.predict(x_train_T)

y_hat_test = model.predict(x_test_T)

# ... evaluation ...
# ... inaccuracy precent ...
print('Mean Absolute Error:', mean_absolute_error(y_test_T, y_hat_test))
print('Mean Squared Error:', mean_squared_error(y_test_T, y_hat_test))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test_T, y_hat_test)))
print('Mean Mean Squared Logarithmic Error:', np.sqrt(mean_squared_error(y_test_T, y_hat_test)))


# ... prediction ...
y_pred = model.predict(x_test)  # with logarithmic values
pred = pd.DataFrame(y_pred, columns=['SalePrice'])
df1 = pd.concat([y, pred], axis=1)
# reverse logarithmic values
regResults = np.expm1(y_pred)

# ... create df for results ...
regResults = pd.DataFrame(regResults, columns=['SalePrice'])
test['Id'].shape, regResults.shape
id_df = pd.DataFrame(test['Id'])

submission = pd.concat([id_df, regResults], axis=1)
submission.to_csv('prediction.csv', index=False)
