from flask import Flask, render_template, redirect, request, url_for
import pandas as pd
import numpy as np
import joblib
import random
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from itertools import combinations
from scipy import stats

app = Flask(__name__, template_folder="templates")
verts = pd.read_excel("verts.xlsx")
# add see and max min prediction
data = verts.iloc[:-5]

vert_data_M = verts[verts["Sex"] == 'M']
vert_data_M = vert_data_M.iloc[0:vert_data_M.shape[0], 3:27]

vert_data_F = verts[verts["Sex"] == 'F']
vert_data_F = vert_data_F.iloc[0:vert_data_F.shape[0], 3:27]

vert_data_A = verts.iloc[0:153, 3:27]

################################################################
genders = ['M', 'F', 'UD']

results_rf = {}
r2_results = {}
se_results = {}

mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
#r2_scorer = make_scorer(r2_score)

data_combined = pd.concat([data[data['Sex'] == gender] for gender in genders])

y_combined = data_combined['Sum_Verts']
X_combined = data_combined.drop(columns=['Sum_Verts', 'ID', 'Age_mean', 'Sex'])



for gender in genders:
    y = data[data['Sex'] == gender]['Sum_Verts']
    X = data[data['Sex'] == gender].drop(columns=['Sum_Verts', 'ID', 'Age_mean', 'Sex'])
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = cross_val_score(rf, X, y, cv=kf, scoring=mse_scorer)

    rf.fit(X, y)

    results_rf[gender] = {
        "Mean Squared Error (Cross-Validation)": -np.mean(mse_scores)}
    
    model_filename = f"random_forest_{gender}.joblib"
    joblib.dump(rf, model_filename)
    
    model_filename = f"random_forest_{gender}.joblib"
    joblib.dump(rf, model_filename)

spine_features = X_combined.columns.tolist()
for num_features in range(1, len(spine_features) + 1):
    r2_values = []
    se_values = []

    for _ in range(5):  # Run 10 times
        sampled_features = random.sample(spine_features, num_features)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        
        rf.fit(X_combined[sampled_features], y_combined)
        predictions = rf.predict(X_combined[sampled_features])
        
        r2_value = r2_score(y_combined, predictions)
        r2_values.append(r2_value)
        
        se = np.std(predictions) / np.sqrt(len(predictions))  # Standard Error formula
        se_values.append(se)

    # Store the minimum R² and SE for this number of features
    r2_results[num_features] = np.min(r2_values)
    se_results[num_features] = np.min(se_values)

joblib.dump(r2_results, 'r2_results.joblib')
joblib.dump(se_results, 'se_results.joblib')

#rf_male = joblib.load('random_forest_M.joblib')
#rf_female = joblib.load('random_forest_F.joblib')
#rf_unknown = joblib.load('random_forest_UD.joblib')


numeric_columns = data.select_dtypes(include=[np.number]).columns
data_numeric = data[numeric_columns]

### save the mean value
mean_values_by_gender = data.groupby('Sex')[numeric_columns].mean()
overall_mean = data[numeric_columns].mean()
mean_values_dict = pd.DataFrame({
    'M': mean_values_by_gender.loc['M'],
    'F': mean_values_by_gender.loc['F'],
    'UD': overall_mean    ####### Here we use the mean of overall to fill the nan value for the UD gender.
})
mean_values_dict = mean_values_dict.T

joblib.dump(mean_values_dict, "mean_values.joblib")

mean_values_by_gender = joblib.load("mean_values.joblib")



mean_values = X_combined.mean()
std_values = X_combined.std()
max_values = round((mean_values + 2 * std_values),2)
min_values = round((mean_values - 2 * std_values),2)


def predict_sum_verts(user_input, user_gender):
    if user_gender not in ['M', 'F', 'UD']:
        raise ValueError("Invalid gender! Please input 'M', 'F', or 'UD'.")
#####################################################################################################################
    ##### If you need to contorl the input to have different decimal places, please use this part
    
    # for key in user_input:
    #     if not pd.isnull(user_input[key]): 
    #         user_input[key] = round(float(user_input[key]), 2)     #### Convert input into two decimal palces.
#####################################################################################################################
    
    #non_null_features = [key for key, value in user_input.items() if (value != 0).bool()]

    #num_non_null_features = len(non_null_features)
    num_non_null_features = 24
   
    if num_non_null_features < 14:
        print("Warning: Insufficient data. Please provide at least 14 non-null features for accurate prediction.")
    
    gender_means = mean_values_by_gender.loc[user_gender]
    
    spine_features = ['C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6',
                      'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'L1', 'L2', 'L3', 'L4', 'L5']

    for feature in spine_features:
        #print(user_input[feature][0])
        if feature not in user_input or int(user_input[feature][0]) == 0:
            #print("Flag")
            num_non_null_features -=1
            user_input[feature] = gender_means[feature]
        else:
            if (user_input[feature] < min_values[feature]).all() or (user_input[feature] > max_values[feature]).all():
                print(f"Warning: The value for '{feature}' is out of bounds! "
                      f"Expected range: [{min_values[feature]}, {max_values[feature]}], "
                      f"but got: {user_input[feature]}.")
    #print(num_non_null_features)
    input_df = pd.DataFrame([user_input], columns=spine_features)

    model_filename = f"random_forest_{user_gender}.joblib"
    model = joblib.load(model_filename)

    prediction = model.predict(input_df)

    # R² and SE
    r_squared = r2_results.get(num_non_null_features, None)
    se = se_results.get(num_non_null_features, None)

    predictions = model.predict(X_combined[spine_features])  
    std_dev = np.std(predictions)
    critical_value = stats.t.ppf(0.975, df=len(predictions) - 1)
    margin_of_error = critical_value * (std_dev / np.sqrt(len(predictions)))

    lower_bound = prediction[0] - margin_of_error
    upper_bound = prediction[0] + margin_of_error

    return round(prediction[0], 1), round(r_squared,4), round(se,4), (round(lower_bound, 1), round(upper_bound, 1))




all_mse = []


################################################################


def total_sampling(data, t):
    # Ensure t is in range
    if t > 23 or t < 1:
        raise ValueError("Sample size t must be between 1 and 23.")
    
    # Sampling number of targets
    sampled_columns = random.sample(range(23), t)
    vert_sampled = data.iloc[:, sampled_columns]
    
    # Everything but sum
    vert_sampled_counter = data.iloc[:, 23:24]
    
    vert_combined = pd.concat([vert_sampled.reset_index(drop=True), vert_sampled_counter.reset_index(drop=True)], axis=1)
    vert_combined.columns = list(vert_combined.columns[:-1]) + ["Sum_Verts"]

    # Train model
    X = vert_combined.iloc[:, :-1]  # All but the last column
    y = vert_combined["Sum_Verts"]
    X = sm.add_constant(X)  # Add a constant for intercept
    lm_model = sm.OLS(y, X).fit() #ordinary least squares methods for regression
    
    predictions = lm_model.predict(X)
    residuals = vert_combined["Sum_Verts"] - predictions
    vert_sampling_sse = np.sum(residuals**2)  # Sum of Squared Errors
    vert_sampling_r2 = lm_model.rsquared_adj
    
    n = len(y)  # Number of observations
    p = len(lm_model.params)  # Number of parameters (including intercept)
    see = np.sqrt(vert_sampling_sse / (n - p))

    max_estimate = np.max(predictions)
    min_estimate = np.min(predictions)
    vertebrae_names = ["C2", "C3", "C4", "C5", "C6", "C7",
                       "T1", "T2", "T3", "T4", "T5", "T6", "T7", 
                       "T8", "T9", "T10", "T11", "T12", 
                       "L1", "L2", "L3", "L4", "L5"]
    
    variable_set = [1 if name in vert_combined.columns else 0 for name in vertebrae_names]
    # Return model and other stats
    return {
        "estimate" : predictions.mean(), 
        "model": lm_model,
        "SSE": vert_sampling_sse,
        "R2": vert_sampling_r2,
        "SEE": see,
        "max_estimate": max_estimate,
        "min_estimate": min_estimate,
        "variable_set": variable_set
    }

results_A = total_sampling(vert_data_A, t=10)
results_F = total_sampling(vert_data_F, t=10)
results_M = total_sampling(vert_data_M, t=10)

vertebrae_names = ["C2", "C3", "C4", "C5", "C6", "C7",
                    "T1", "T2", "T3", "T4", "T5", "T6", "T7", 
                    "T8", "T9", "T10", "T11", "T12", 
                    "L1", "L2", "L3", "L4", "L5"] 
new_data = vert_data_A.iloc[0, :23].to_frame().T  # Sample new data




vertices = []
#new_data = pd.DataFrame([0]*23).transpose()
#render_template("index.html")
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/input", methods=["POST"])
def input():
    #vertIN = request.form['Vertebrae']
    vertices = []
    sex = ''
    regression = ''
    vertices.append(request.form['Vertebrae'])
    vertices.append(request.form['Vertebrae2'])
    vertices.append(request.form['Vertebrae3'])
    vertices.append(request.form['Vertebrae4'])
    vertices.append(request.form['Vertebrae5'])
    vertices.append(request.form['Vertebrae6'])
    vertices.append(request.form['Vertebrae7'])
    vertices.append(request.form['Vertebrae8'])
    vertices.append(request.form['Vertebrae9'])
    vertices.append(request.form['Vertebrae10'])
    vertices.append(request.form['Vertebrae11'])
    vertices.append(request.form['Vertebrae12'])
    vertices.append(request.form['Vertebrae13'])
    vertices.append(request.form['Vertebrae14'])
    vertices.append(request.form['Vertebrae15'])
    vertices.append(request.form['Vertebrae16'])
    vertices.append(request.form['Vertebrae17'])
    vertices.append(request.form['Vertebrae18'])
    vertices.append(request.form['Vertebrae19'])
    vertices.append(request.form['Vertebrae20'])
    vertices.append(request.form['Vertebrae21'])
    vertices.append(request.form['Vertebrae22'])
    vertices.append(request.form['Vertebrae23'])
    print("done")
    for i in range(23):
        if vertices[i] == '':
            vertices[i]=0
        new_data.iloc[:,i] = float(vertices[i])
    
    sex = request.form['Sex']
    #print(sex)
    regression = request.form['Reg']
    return redirect(url_for("getPrediction", sex=sex, regression=regression))
#print(sex)
@app.route("/predict", methods=["GET"])
def getPrediction():
    #for i in range(23):
    #    print(new_data.iloc[:,i])
    new_data_imputed = new_data.copy()
    vert_data = 0
    results = 0
    sex = request.args.get('sex')
    regression = request.args.get('regression')
    
    #total_sampling(vert_data_A, t=10)
    #print(sex)
    if regression == "Linear":
        if sex == 'Male': #selecting correct model and data based on sex
            vert_data = vert_data_M 
            results = results_M
        elif sex == 'Female':
            vert_data = vert_data_F
            results = results_F
        else:
            vert_data = vert_data_A 
            results = results_A
        for i in range(new_data_imputed.shape[1]):
            if (new_data_imputed.iloc[:, i] == 0).any():
                mean_value = vert_data.iloc[:, i].mean()  # Calculate mean
                new_data_imputed.iloc[:, i] = mean_value
        X_columns = results["model"].model.exog_names  
        new_data_imputed = new_data_imputed.reindex(columns=X_columns, fill_value=0)
        predictions = results["model"].predict(new_data_imputed)
        print(results['estimate'])
        print(predictions[0])
        print(results['R2'])
        print(results['SEE'])
        print(results['max_estimate'])
        print(results['min_estimate'])
        return render_template("predict.html", prediction = predictions[0], r2 = results['R2'], sse = results['SSE'], see = results['SEE'])
    else:
        if sex == 'Male': #selecting correct model and data based on sex
            vert_data = vert_data_M 
            gender = 'M'
            #results = results_M
        elif sex == 'Female':
            vert_data = vert_data_F
            gender = 'F'
            #results = results_F
        else:
            vert_data = vert_data_A 
            gender = 'UD'
            #results = results_A
        user_input_data = {
        'C2': new_data_imputed.iloc[:, 0],
        'C3': new_data_imputed.iloc[:, 1],
        'C4': new_data_imputed.iloc[:, 2],
        'C5': new_data_imputed.iloc[:, 3],
        'C6': new_data_imputed.iloc[:, 4],
        'C7': new_data_imputed.iloc[:, 5],
        'T1': new_data_imputed.iloc[:, 6],
        'T2': new_data_imputed.iloc[:, 7],
        'T3': new_data_imputed.iloc[:, 8],
        'T4': new_data_imputed.iloc[:, 9],
        'T5': new_data_imputed.iloc[:, 10],
        'T6': new_data_imputed.iloc[:, 11],
        'T7': new_data_imputed.iloc[:, 12],
        'T8': new_data_imputed.iloc[:, 13],
        'T9': new_data_imputed.iloc[:, 14],
        'T10': new_data_imputed.iloc[:, 15],
        'T11': new_data_imputed.iloc[:, 16],
        'T12': new_data_imputed.iloc[:, 17],
        'L1': new_data_imputed.iloc[:, 18],
        'L2': new_data_imputed.iloc[:, 19],
        'L3': new_data_imputed.iloc[:, 20],
        'L4': new_data_imputed.iloc[:, 21],
        'L5': new_data_imputed.iloc[:, 22],
        }
        print("RF")
        predicted_sum_verts, r_squared, standard_errors, confidence_interval = predict_sum_verts(user_input_data, gender)
        #predicted_sum_verts = predict_sum_verts(user_input_data, sex)
        return render_template("predict.html", prediction = predicted_sum_verts, r2 = r_squared, sse =  results_rf[gender]["Mean Squared Error (Cross-Validation)"] *23, see = standard_errors)

if __name__ == '__main__':
    app.run(debug=True)