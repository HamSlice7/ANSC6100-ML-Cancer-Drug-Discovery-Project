# ANSC6100-ML-Cancer-Drug-Discovery-Project
By Edicon Chan, Leon Edmiiz, Jacob Hambly and Zach Ribau 

>>List of Changeable Script Parameters
- input_dir: Directory path where input files are located, and where output files will be written.
- models: List of regression models to be evaluated.
- num_sp: Number of splits for cross-validation.
- num_rep: Number of repetitions for cross-validation.
Other hyperparameters within models and training functions.

>>Input Files

Option 1: Read in provided csv file

The following raw data file can be read in by setting the input_dir variable to the directory containing the file. 
- raw_data_erbb1_ic50.csv: CSV file containing the data on EGFR protein inhibitors, including canonical smiles and IC50 values.

Option 2: Fetch data from CHEMBL
Alternatively, by setting the fetch_chembl variable to TRUE, one can obtain the data directly from the CHEMBL database.

>>Output Files Generated

The following files will be written directly in the input_dir that was set at the beginning of the script. Files highlighted in orange and intermediate outputs that are used in subsequent steps within the script.
- erbb1_bothassay_neglog10_ic50.csv: Processed dataset with transformed IC50 values.
- cb_pb_fingerprints.csv: Molecular fingerprints data.
- df_pb_cb_for_model_building.csv: Final dataset used for model training.
- evaluations_with_cv.csv: Evaluation metrics from cross-validation.
- test_results.csv: Final test results for the optimized model.
- final_feature_importance.csv: Feature importance from the optimized RandomForest model.

List of Custom Functions
1. logm: Converts IC50 values from nM to -log(M).
2. mol_descriptors: Generates molecular descriptors from SMILES.
3. morgan_fpts: Generates Morgan fingerprints from SMILES.
4. train_evaluate_model_with_cv: Trains models from list of models and performs cross-validation.
5. plot_learning_curve: Plots the learning curve for models from list of models.
