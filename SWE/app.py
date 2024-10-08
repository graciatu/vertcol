from flask import Flask, render_template, redirect, request, url_for
import pandas as pd
import numpy as np
import random
import statsmodels.api as sm
app = Flask(__name__, template_folder="templates")
verts = pd.read_excel("verts.xlsx")

vert_data_M = verts[verts["Sex"] == 'M']
vert_data_M = vert_data_M.iloc[0:vert_data_M.shape[0], 3:27]

vert_data_F = verts[verts["Sex"] == 'F']
vert_data_F = vert_data_F.iloc[0:vert_data_F.shape[0], 3:27]

vert_data_A = verts.iloc[0:153, 3:27]

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
    
    vertebrae_names = ["C2", "C3", "C4", "C5", "C6", "C7",
                       "T1", "T2", "T3", "T4", "T5", "T6", "T7", 
                       "T8", "T9", "T10", "T11", "T12", 
                       "L1", "L2", "L3", "L4", "L5"]
    
    variable_set = [1 if name in vert_combined.columns else 0 for name in vertebrae_names]
    # Return model and other stats
    return {
        "model": lm_model,
        "SSE": vert_sampling_sse,
        "R2": vert_sampling_r2,
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
    #print(sex)
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
            new_data_imputed.iloc[:, i] = mean_value  # Replace 0 values with vertebra mean
    X_columns = results["model"].model.exog_names  
    new_data_imputed = new_data_imputed.reindex(columns=X_columns, fill_value=0)
    predictions = results["model"].predict(new_data_imputed)
    return render_template("predict.html", prediction = predictions[0], r2 = results['R2'], sse = results['SSE'])

if __name__ == '__main__':
    app.run(debug=True)