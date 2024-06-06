###### ------------------------- ######
######   1. IMPORT LIBRARIES     ######
###### ------------------------- ######

#General
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from chembl_webresource_client.new_client import new_client

#For Feature Engineering
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import AllChem
from sklearn.feature_selection import VarianceThreshold
import math

#Import scaling methods and evaulation metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Import models to evaluate
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, RandomizedSearchCV, learning_curve
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import RFE

###### ------------------------- ######
######    2. USER PARAMETERS     ######
###### ------------------------- ######

#If set to true, this will obtain data from CHMEBL serve. If false, the a raw datafile will be read from the input_dir set below
fetch_chembl = True

input_dir = ""

raw_data_filename = "raw_data_erbb1_ic50.csv"

#Specify models, scalers, and parameters to be evaluated
models = [
    SVR(kernel='rbf'),
    XGBRegressor(random_state=34, n_estimators=100, learning_rate=0.1),
    DecisionTreeRegressor(random_state=34),
    RandomForestRegressor(random_state=34, n_estimators=100),
    GradientBoostingRegressor(random_state=34),
    AdaBoostRegressor(random_state=34),
    MLPRegressor(hidden_layer_sizes=(50, 100, 50), random_state=34, max_iter=300),
    KNeighborsRegressor()
]

#Cross Validation Parameters
num_sp = 5
num_rep = 1
kfold = RepeatedKFold(n_splits=num_sp, n_repeats=num_rep, random_state=34)

###### ------------------------- ######
######     3. FUNCTIONS          ######
###### ------------------------- ######

#FUNCTION 1: Converts the nM of a compound to the -log10(m)
def logm(nm):
    m  = nm/1000000000
    m = -math.log10(m)
    return m

#FUNCTION 2: Creates new features based on the canononical SMILES from data
def mol_descriptors(smiles):
    molecules = [Chem.MolFromSmiles(i) for i in smiles]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    desc_names = calc.GetDescriptorNames()

    molecular_descriptors = []
    for mol in molecules:
        #add hydrogen to the molecules
        mol = Chem.AddHs(mol)
        #Calculate the molecular descriptors for each molecule
        descriptors = calc.CalcDescriptors(mol)
        molecular_descriptors.append(descriptors)
    return molecular_descriptors, desc_names

#FUNCTION 3: Generate morgan fingerprint features
def morgan_fpts(data):
    Morgan_fpts = []
    for i in data:
        mol = Chem.MolFromSmiles(i)
        fpts =  AllChem.GetMorganFingerprintAsBitVect(mol,2,3072)
        mfpts = np.array(fpts)
        Morgan_fpts.append(mfpts)
    return np.array(Morgan_fpts)

#FUNCTION 4: TRAIN, PERFORM CROSS-VALIDATION, AND EVALUATE MODELS ON TEST SET
def train_evaluate_model_with_cv(model, X_train, X_test, y_train, y_test):


    print(f"Performing cross-validation for {model.__class__.__name__}...")

    #Perform Cross-validation scores
    cv_scores_mae = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_absolute_error').mean()
    cv_scores_mse = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error').mean()
    cv_scores_r2 = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2').mean()

    print(f"Cross-validation for {model.__class__.__name__} completed.")

    #Append results to df
    results_df = pd.DataFrame({
        #Model identification
        'Model': [model.__class__.__name__],

        #Extract mean CV metrics as well
        'CV_MAE': [cv_scores_mae],
        'CV_MSE': [cv_scores_mse],
        'CV_R^2': [cv_scores_r2]
    })

    return results_df

#FUNCTION 5: Plot learning curves for initial algorythm testing
def plot_learning_curve(model, X, y, cv, n_jobs, train_sizes = np.linspace(.1, 1.0, 5)):
    
    model_name = type(model).__name__
    print(f"Computing learning curve for {model_name}...")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(model, X, y, cv=kfold, n_jobs=n_jobs,train_sizes= train_sizes, return_times= True)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    plt.figure()

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.xlabel("Training size")
    plt.ylabel("Score")
    plt.title(f"Learning Curve ({model.__class__.__name__})")
    plt.legend(loc="best")
    
    return plt

###### ------------------------- ######
######  4. DATA PREPROCESSING    ######
###### ------------------------- ######


### EXTRACTING DATA FROM CHEMBL  ###

if fetch_chembl:
    # searching and selecting Erbb1 as the drug target
    target_query = new_client.target.search("Erbb1")
    targets = pd.DataFrame(target_query)
    print("Data set options:")
    print(targets)

    # select the first option as selected_query
    selected_query = targets.target_chembl_id[0]
    print("Selected dataset:")
    print(selected_query)

    # selecting the desired data set
    activity = new_client.activity
    erbb1_ic50 = activity.filter(target_chembl_id=selected_query).filter(standard_type="IC50")
    erbb1_df = pd.DataFrame(erbb1_ic50)
    print("Data acquired:")
    print(erbb1_df.head())
    
    erbb1_df.to_csv(input_dir + "raw_data_erbb1_ic50.csv", index=False)

else:
    #If fetch_chembl was set to false, open file from input_dir
    os.chdir(input_dir)
    
    #Load data
    erbb1_df = pd.read_csv(raw_data_filename)
    print(erbb1_df.shape)

#selecting desired columns
selected_columns = ['canonical_smiles', 'molecule_chembl_id','bao_label', 'standard_units', 'standard_value','data_validity_comment']
erbb1_df = erbb1_df[selected_columns]
print(erbb1_df.head())

#removing any rows with missing standard values (IC50)
erbb1_df = erbb1_df[erbb1_df.standard_value.notna()]
print(erbb1_df.shape)

#removing missing cananonical smiles
erbb1_df = erbb1_df[erbb1_df.canonical_smiles.notna()]
print(erbb1_df.shape)

#checking out the out of range values
erbb1_df_err = erbb1_df[erbb1_df['data_validity_comment'] == 'Outside typical range' ]
print(erbb1_df_err.loc[:,'standard_value'].head())

#checking the max and min for the standard value of erbb1_df_err
erbb1_df_err['standard_value'] = pd.to_numeric(erbb1_df_err['standard_value'])
print(erbb1_df_err['standard_value'].max())
print(erbb1_df_err['standard_value'].min())

#filtering for data_validity_comment = None
erbb1_df = erbb1_df[erbb1_df['data_validity_comment'].isnull()]
print(erbb1_df.shape)

#looking at range of standard_values
erbb1_df['standard_value'] = pd.to_numeric(erbb1_df['standard_value'])
print(erbb1_df['standard_value'].max())
print(erbb1_df['standard_value'].min())

#looking at the bao labels (experimental design) and looking at the number of observations for each bao label
print(erbb1_df['bao_label'].unique())
print(erbb1_df['bao_label'].value_counts())

#Selecting 'single protein format' boa labels
subset_pb_erbb1_df = erbb1_df[erbb1_df['bao_label'] == 'single protein format']
print(subset_pb_erbb1_df.head())

#Selecting 'cell-based format' boa labels
subset_cb_erbb1_df = erbb1_df[erbb1_df['bao_label'] == 'cell-based format']
print(subset_cb_erbb1_df.head())

subset_pb_erbb1_df = subset_pb_erbb1_df[['canonical_smiles', 'standard_value']]
subset_pb_erbb1_duplicate_mean_df = subset_pb_erbb1_df.groupby('canonical_smiles').mean().reset_index()
print(subset_pb_erbb1_duplicate_mean_df.shape)
print(subset_pb_erbb1_df.shape)
print(subset_pb_erbb1_duplicate_mean_df['canonical_smiles'].value_counts())

subset_cb_erbb1_df = subset_cb_erbb1_df[['canonical_smiles', 'standard_value']]
subset_cb_erbb1_duplicate_mean_df = subset_cb_erbb1_df.groupby('canonical_smiles').mean().reset_index()
print(subset_cb_erbb1_duplicate_mean_df.shape)
print(subset_cb_erbb1_df.shape)
print(subset_cb_erbb1_duplicate_mean_df['canonical_smiles'].value_counts())

#adding a assay type column where 1 indicates protein based assay type
subset_pb_erbb1_duplicate_mean_df["assay_type"] = 1
print(subset_pb_erbb1_duplicate_mean_df.head())

#adding a assay type column where 0 indicates cell based assay type
subset_cb_erbb1_duplicate_mean_df["assay_type"] = 0
print(subset_cb_erbb1_duplicate_mean_df.head())

#combining the protein-based and cell-based dataframes into one dataframe 
df_cb_pb = pd.concat([subset_pb_erbb1_duplicate_mean_df,subset_cb_erbb1_duplicate_mean_df], axis = 0)
df_cb_pb.reset_index(inplace=True)
print(df_cb_pb)

#APPLE FUNCTION 1 - Creating a new column called '-log(M)' which contains the -log(M) of the 'standard_value' column
df_cb_pb['-log(M)'] = df_cb_pb['standard_value'].apply(logm)
print(df_cb_pb.head())

#check to make sure we don't have NA's in our columns
print(df_cb_pb.isna().sum())

#subsetting for canonical_smiles, assay type and  -log(M)
final_df = df_cb_pb[['canonical_smiles', 'assay_type', '-log(M)']]
print(final_df)

final_df.to_csv("erbb1_bothassay_neglog10_ic50.csv", index = False)

###### ------------------------ ######
###### 5. FEATURE ENGINEERING   ######
###### ------------------------ ######

data = final_df
print(data.head())

#APPLY FUNCTION 2 - Generate molecular descriptors from canonical smiles
molecular_descriptors, desc_names = mol_descriptors(data["canonical_smiles"])

#creating a dataframe for the moleculare descriptors
data_descriptors = pd.DataFrame(molecular_descriptors, columns=desc_names)
print(data_descriptors.head())

#Eliminate single value columns 
num_unique_col = data_descriptors.nunique()

#getting a record of single-value columns
col_to_del = [i for i,v in enumerate(num_unique_col) if v == 1 ]

#drop single value columns
data_descriptors.drop(data_descriptors.columns[col_to_del], axis=1, inplace=True)

#checking the dimensions of the data
print(data_descriptors.shape)

##dropping columns with variance less than 1

#define variance threshold
var_threshold = VarianceThreshold(1)

#fit the variance threshold to the data
var_threshold.fit(data_descriptors)

#get the boolen expression of columns that had variance above 1
feature_mask = var_threshold.get_support()

#use the mask to get the specfic column names
selected_columns = data_descriptors.columns[feature_mask]

#filter data_descriptors for the specific columns
data_descriptors = data_descriptors[selected_columns]

print(data_descriptors.shape)

#Using the min-max approach to scale the features
scaler = MinMaxScaler()

#scaling the data
data_descriptors_scaled = scaler.fit_transform(data_descriptors)

data_descriptors_scaled_pd = pd.DataFrame(data_descriptors_scaled)

data_descriptors_scaled_pd.columns = data_descriptors.columns

print(data_descriptors_scaled_pd.head())

print(data_descriptors_scaled_pd.head())
print(data_descriptors_scaled_pd.columns)

#APPLY FUNCTION 3 - Generate morgan fingerprints from canoncial smiles
Morgan_fpts = morgan_fpts(data['canonical_smiles'])
print(Morgan_fpts.shape)

Morgan_fingerprints = pd.DataFrame(Morgan_fpts,columns=['Col_{}'.format(i) for i in range(Morgan_fpts.shape[1])])
print(Morgan_fingerprints.head())

#looking at the number of unique finger print columns 
unique_finger_prints = Morgan_fingerprints.nunique()
print(unique_finger_prints)

#get a list of single value columns
fingerprint_columns_to_delete = [i for i,v in enumerate(unique_finger_prints) if v == 1]

#drop single value columns
Morgan_fingerprints.drop(Morgan_fingerprints.columns[fingerprint_columns_to_delete], axis=1, inplace=True)
print(Morgan_fingerprints.shape)

#filter low variance 
vt = VarianceThreshold(threshold=0.1).set_output(transform="pandas")
Morgan_fingerprints = vt.fit_transform(Morgan_fingerprints)
print(Morgan_fingerprints.shape)

Morgan_fingerprints.to_csv("cb_pb_fingerprints.csv")

descriptors_fingerprints = pd.concat([data_descriptors_scaled_pd, Morgan_fingerprints], axis = 1)
print(descriptors_fingerprints.head())

#add the response variable
descriptors_fingerprints["assay_type"] = data["assay_type"]
descriptors_fingerprints["standardized_ic50"] = data["-log(M)"]
print(descriptors_fingerprints.head())

#seeing if any columns have 'na' values
nan_mask = descriptors_fingerprints.isna().any(axis = 1)

#a sequence of booleans — select all rows where True
rows_with_nan = descriptors_fingerprints[nan_mask]

print(rows_with_nan)


#dropping column with 'na'
descriptors_fingerprints = descriptors_fingerprints.dropna()
print(descriptors_fingerprints.head())

#saving the data to a csv file
descriptors_fingerprints.to_csv("df_pb_cb_for_model_building.csv", index=False)

###### ----------------------------------- ######
###### 6. MODEL TRAINING AND EVALUATIONS   ######
###### ----------------------------------- ######

## ---- DATA SPLITTING AND NORMALIZATION ---- ##

df = pd.read_csv("df_pb_cb_for_model_building.csv")
df.dropna(inplace=True)

#Separate predictors and response
features = df.columns[:-1]
X = df[features]
y = df.iloc[:, -1] 

#Split data before scaling to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34)

#Initialize dictionary to store results
results_dict = {}

## ---- APPLY FUNCTION TO TRAIN AND EVALUATE MODELS ---- ##

#APPLY FUNCTION 4 - Train and evaluate models
for model in models:
    result = train_evaluate_model_with_cv(model, X_train, X_test, y_train, y_test)
    key = model.__class__.__name__ 
    results_dict[key] = result

#Bind all results into a single DataFrame for viewing
evaluations_cv = pd.concat(results_dict.values(), ignore_index=True)

#Print the DataFrame to quickly view
print(evaluations_cv)

#Write the evaluations with cross-validation results to csv
evaluations_cv.to_csv('evaluations_with_cv.csv', index=False)

#APPLY FUNCTION 5: Plot learning curve for initial model testing.
for model in models:
    plot = plot_learning_curve(model, X, y, cv = kfold, n_jobs=-1)
    plot.show()

###### --------------------------------------- ######
###### 7. HYPER-PARAMETER OPTIMIZATION WITH RF ######
###### --------------------------------------- ######

### --- Recursive Feature Elimination --- ###

the_model = RandomForestRegressor()

rfe = RFE(the_model, n_features_to_select=50)
rfe_fit = rfe.fit(X_train,y_train)
selected_features = X.columns[rfe_fit.support_]
print("Selected features:", selected_features)

#Subset a new X_train to only include selected features from RFE
X_train_rfe = X_train[selected_features]

## ---- Optimization ---- ##
grid = dict()
grid['random_state'] = [34]
#grid['n_estimators'] = [100, 250, 500, 750, 1000]
grid['max_features'] = [5,10,15,20,25,30,'sqrt', 'log2']
grid['max_depth'] = [5,10,15,20,25,30]
#grid['min_samples_split'] = [2, 4, 6, 8]
grid['min_samples_leaf'] = [5, 10, 15,20,25,30]

param_search = RandomizedSearchCV(the_model, param_distributions=grid, n_iter=100, n_jobs=-1, cv=kfold, scoring='neg_root_mean_squared_error')

search_result = param_search.fit(X_train_rfe, y_train)

print("Best: %f using %s" % (np.absolute(search_result.best_score_), search_result.best_params_))
#means = search_result.cv_results_['mean_test_score']
#stds = search_result.cv_results_['std_test_score']
#params = search_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
#print("%f (%f) with: %r" % (mean, stdev, params))

#Extract best features for incorporation into model fitting
best_params = search_result.best_params_

###### ------------------------------------ ######
###### 8. FITTING FINAL OPTIMIZED RF MODEL  ######
###### ------------------------------------ ######

### ----- Model fitting with optimized parameters ---- ###
final_forest = RandomForestRegressor(
    random_state=34,
    n_estimators=500,
    max_features=best_params['max_features'],
    max_depth=best_params['max_depth'],
    #min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf']
)
final_forest.fit(X_train_rfe,y_train)

##APPLY FUNCTION 5: Evaluate with learning curve
lc_plot = plot_learning_curve(final_forest, X, y, cv=kfold, n_jobs=-1)
lc_plot.show()


#Ensure only RFE selected features are in testing dataset
X_test_rfe = X_test[selected_features]

##Make predictions on held out test set and evaluate
y_pred = final_forest.predict(X_test_rfe)

test_mae = mean_absolute_error(y_test, y_pred)
test_mse = mean_squared_error(y_test, y_pred)
test_rmse = np.sqrt(test_mse)  
test_r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error:{test_mae}")
print(f"Mean Squared Error: {test_mse}")
print(f"Root Mean Squared Error:{test_rmse}")
print(f"R-squared: {test_r2}")

test_results = pd.DataFrame({
        #Model identification
        'Model': ["Optimized RF"],

        #Extract mean CV metrics as well
        'MAE': [test_mae],
        'MSE': [test_mse],
        'R^2': [test_r2],
        'RMSE': [test_rmse]
    })

test_results.to_csv("test_results.csv")
feature_ranking = (pd.DataFrame({"Features": X_test_rfe.columns, "Feature importance" : final_forest.feature_importances_}))
feature_ranking.to_csv("final_feature_importance.csv")

#Put results in a data frame
df_results = pd.DataFrame({'Actual': y_test, 'Predicited': y_pred})
df_results = df_results.head(30)

#Plotting results
df_results.plot(kind='bar', figsize=(10, 6))
plt.show()
