from flask import Flask , render_template , request
import pandas as pd
import joblib
from datetime import datetime, date
import numpy as np

models = {
    "student_performance":{
        'name':'student performance predictor' ,
        'model':joblib.load(r"models/student_performance_model.pkl")
    } , 

    "employee_salary":{
        'name':"employee salary predictor" ,
        'model':joblib.load(r"models/Employee_Salary_Predictor.pkl")
    } , 

    "calory_burned":{
        'name':"calory burned predictor" ,
        'model':joblib.load(r"models/Calories Burned Predictor.pkl")
    } ,

    "home_loan":{
        "name":'home loan approval predictor' ,
        "model":joblib.load(r'models/Home_Loan_approval_Predcitor.pkl')
    } ,

    "diabetes_risk":{
        "name":'diabetes risk predictor' ,
        'model':joblib.load(r'models/diabetes_predictor.pkl')
    } ,

    "customer_churn":{
        'name': 'customer churn predictor' ,
        'model':joblib.load(r'models/Clean_Customer_churn.pkl')
    } ,

    "customer_segmentation":{
        'name':'customer segmentation predictor' ,
        'model':joblib.load(r'models/customer_segmentation.pkl')
    } ,

    "house_clustering":{
        'name':"house clustering predictor" ,
        'model':joblib.load(r'models/house_clustering_pipeline.pkl')
    }
}

df1 = pd.read_csv(r"models\Cleaned_Employee_dataset.csv")
job_titles = sorted(df1["Job Title"].unique())

df2 = pd.read_csv(r'models\Clean_Customer_Churn.csv')
internet_service = sorted(df2['InternetService'].unique())
contract = sorted(df2['Contract'].unique())
paymentmethod = sorted(df2['PaymentMethod'].unique())

df3 = pd.read_csv(r'models\Market Basket Rules.csv')
product_name = sorted(df3['antecedents'].unique())

df4 = pd.read_csv(r'models\Symptoms_rules.csv')
antecedents = sorted(df4['antecedents'].unique())

app = Flask(__name__)

@app.route('/')
def home():
   return render_template("index.html")

@app.route('/student_performance' , methods=['GET' , 'POST'])
def student_performance():
    model = models['student_performance']['model']
    prediction = None
    if request.method=='POST':
        try:
            study_hours = float(request.form['study_hours'])
            previous_scores = float(request.form['previous_scores'])
            ex_act = request.form['ex_act']
            sleep_hours = float(request.form['sleep_hours'])
            sqp = int(request.form['sqp'])

            df = pd.DataFrame([{
                "Hours Studied": study_hours,
                "Previous Scores": previous_scores,
                "Extracurricular Activities": ex_act,
                "Sleep Hours": sleep_hours,
                "Sample Question Papers Practiced": sqp
            }])

            prediction = round(model.predict(df)[0] , 2)
        
        except Exception as e:
            prediction = f"Error: {e}"

    

    return render_template("student_performance.html" , prediction=prediction)

@app.route('/employee_salary' , methods=['GET' ,'POST'])
def employee_salary():
    model = models['employee_salary']['model']
    prediction=None
    if request.method=='POST':
        try:
            job_title = request.form['job_title']
            age = int(request.form['age'])
            exp = float(request.form['exp'])
            gender = request.form['gender']
            edu = request.form['edu']

            df = pd.DataFrame([{
                "Age": float(age),
                "Gender": gender,
                "Education Level": edu,
                "Job Title": job_title,
                "Years of Experience": float(exp)
            }])


            prediction = round(model.predict(df)[0] , 2)

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('employee_salary.html',prediction=prediction , job_titles=job_titles)

@app.route('/calory_burned' , methods=['GET' , 'POST'])
def calory_burned():
    model = models['calory_burned']['model']
    prediction = None
    if request.method=='POST':
        try:
            age = int(request.form['age'])
            height = float(request.form['height'])
            weight = float(request.form['weight'])
            body_temp = float(request.form['body_temp'])
            exercise_time = int(request.form['exercise_time'])
            gender = request.form['gender']
            heart_rate = int(request.form['heart_rate'])

            df = pd.DataFrame([{
                "Age":age ,
                "Height":height ,
                "Weight":weight ,
                "Body_Temp":body_temp ,
                "Duration": exercise_time ,
                "Gender":gender ,
                "Heart_Rate":heart_rate
            }])

            prediction = round(model.predict(df)[0] , 2)

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('calory_burned.html' , prediction=prediction)

@app.route('/home_loan' , methods=['GET' , 'POST'])
def home_loan():
    model = models['home_loan']['model']
    prediction = None
    if request.method=='POST':
        try:
            Gender = request.form['gender']
            Married = request.form['married']
            Dependents = request.form['dependent']
            Education = request.form['education']
            Self_Employed = request.form['self_employed']
            ApplicantIncome = float(request.form['applicant_income'])
            CoapplicantIncome = float(request.form['coapplicant_income'])
            LoanAmount = float(request.form['loan_amount'])
            Loan_Amount_Term = int(request.form['loan_amount_term'])
            Credit_History = float(request.form['credit_history'])
            Property_Area = request.form['property_type']

            df = pd.DataFrame([{
                "Gender":Gender ,
                "Married":Married ,
                "Dependents":Dependents ,
                "Education":Education ,
                "Self_Employed":Self_Employed ,
                "ApplicantIncome":ApplicantIncome ,
                "CoapplicantIncome":CoapplicantIncome ,
                "LoanAmount":LoanAmount ,
                "Loan_Amount_Term":Loan_Amount_Term ,
                "Credit_History":Credit_History ,
                "Property_Area":Property_Area
            }])

            result = model.predict(df)[0]
            prediction = int(result)
            print(prediction)

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('home_loan.html' , prediction=prediction)

@app.route('/diabetes_risk' , methods=['GET' , 'POST'])
def diabetes_risk():
    model = models['diabetes_risk']['model']
    prediction = None
    if request.method=='POST':
        try:
            pregnancies = float(request.form['pregnancies'])
            glucoselevel = float(request.form['glucoselevel'])
            bloodpressure = float(request.form['bloodpressure'])
            skinthickness = float(request.form['skinthickness'])
            insulinlevel = float(request.form['insulinlevel'])
            bmi = float(request.form['bmi'])
            diabetespedigreefunc = float(request.form['diabetespedigreefunc'])
            age = float(request.form['age'])

            df = pd.DataFrame([{
                "Pregnancies":pregnancies ,
                "Glucose":glucoselevel ,
                "BloodPressure": bloodpressure ,
                "SkinThickness": skinthickness ,
                "Insulin":	insulinlevel ,
                "BMI":	bmi ,
                "DiabetesPedigreeFunction":	diabetespedigreefunc ,
                "Age": age
            }])

            prediction = int(model.predict(df)[0])

        except Exception as e:
            prediction=f'Error: {e}'

    return render_template('diabetes_risk.html' , prediction=prediction)

@app.route('/customer_churn' , methods=['GET' , 'POST'])
def customer_churn():
    model = models['customer_churn']['model']
    prediction = None
    if request.method=='POST':
        try:
            Gender = request.form['gender']
            SeniorCitizen = int(request.form['seniorcitizen'])
            Partner = int(request.form['partner'])
            Dependents = int(request.form['dependents'])
            Tenure = int(request.form['tenure'])
            PhoneService = int(request.form['phoneservice'])
            MultipleLines = int(request.form['multiplelines'])
            InternetService = request.form['internetservice']
            OnlineSecurity = int(request.form['onlinesecurity'])
            OnlineBackup = int(request.form['onlinebackup'])
            DeviceProtection = int(request.form['deviceprotection'])
            TechSupport = int(request.form['techsupport'])
            StreamingTv = int(request.form['streamingtv'])
            StreamingMovies = int(request.form['streamingmovies'])
            Contract = request.form['contract']
            PaperlessBilling = int(request.form['paperlessbilling'])
            PaymentMethod = request.form['paymentmethod']
            MonthlyCharges = float(request.form['monthlycharges'])
            TotalCharges = float(request.form['totalcharges'])

            df = pd.DataFrame([{

                "gender":Gender ,	
                "SeniorCitizen":SeniorCitizen ,	
                "Partner":Partner , 	
                "Dependents":Dependents ,	
                "tenure":Tenure	,
                "PhoneService":PhoneService ,	
                "MultipleLines":MultipleLines ,	
                "InternetService":InternetService ,	
                "OnlineSecurity":OnlineSecurity	,
                "OnlineBackup":OnlineBackup ,
                "DeviceProtection":DeviceProtection ,
                "TechSupport":TechSupport ,
                "StreamingTV":StreamingTv ,	
                "StreamingMovies":StreamingMovies ,
                "Contract":Contract ,
                "PaperlessBilling":PaperlessBilling ,
                "PaymentMethod":PaymentMethod ,	
                "MonthlyCharges":MonthlyCharges ,
                "TotalCharges":TotalCharges

            }])

            prediction = int(model.predict(df)[0])

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('customer_churn.html' , prediction=prediction , internet_service=internet_service , paymentmethod=paymentmethod , contract=contract)

@app.route('/customer_segmentation' , methods=['GET' , 'POST'])
def customer_segmentation():
    model = models['customer_segmentation']['model']
    prediction = None
    if request.method=="POST":
        try:
            last_purchase = request.form['last_purchase']
            num_purchase = int(request.form['num_purchase'])
            total_amount = float(request.form['total_amount'])

            today = date.today()
            recency = (today - datetime.strptime(last_purchase, "%Y-%m-%d").date()).days
            frequency = num_purchase
            monetary = total_amount

            df=pd.DataFrame([{
                'Recency':recency ,
                'Frequency':frequency ,
                'Monetary':monetary
            }])

            cluster_map = {
                0: "Potential Customer",
                1: "Lost Customer",
                2: "Loyal Customer",
                3: "At Risk Customer"
            }

            result = int(model.predict(df)[0])
            prediction = cluster_map.get(result , "Unknown Category")

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('customer_segmentation.html' , prediction=prediction)

@app.route('/market_basket' , methods=['GET' , 'POST'])
def market_basket():
    recommendations=None
    item = None
    if request.method=='POST':
        try:
            item = request.form['item']
            filtered = df3[df3['antecedents']==item].sort_values(by='lift' , ascending=False).head(3)
            
            recommendations = []
            for _, row in filtered.iterrows():
                recommendations.append({
                    "consequent": row["consequents"],
                    "support": round(row["support"], 4),
                    "confidence": round(row["confidence"], 4),
                    "lift": round(row["lift"], 4)
                })

        except Exception as e:
            recommendations = f'Error: {e}'

    return render_template('market_basket.html' , product_name=product_name , recommendations=recommendations , item=item)
    
@app.route('/symptoms_rules', methods=['GET', 'POST'])
def symptoms_rules():
    recommendations = None
    disease = None
    symptoms = None
    filtered2 = []

    disease_name = sorted(df4['Disease'].unique())

    if request.method == 'POST':
        disease = request.form.get('disease')
        symptoms = request.form.get('symptoms')

        if disease and not symptoms:
            filtered2 = df4[df4['Disease'] == disease]['antecedents'].unique()
            return render_template(
                "symptoms_rules.html",
                disease_name=disease_name,
                disease=disease,
                symptoms=symptoms,
                filtered2=filtered2,
                recommendations=None
            )

        if disease and symptoms:
            filtered1 = df4[
                (df4['Disease'] == disease) &
                (df4['antecedents'] == symptoms)
            ].sort_values(by='lift', ascending=False)

            filtered2 = df4[df4['Disease'] == disease]['antecedents'].unique()

            recommendations = []
            for _, row in filtered1.iterrows():
                recommendations.append({
                    "consequent": row["consequents"],
                    "support": round(row["support"], 4),
                    "confidence": round(row["confidence"], 4),
                    "lift": round(row["lift"], 4)
                })

    return render_template(
        "symptoms_rules.html",
        disease_name=disease_name,
        disease=disease,
        symptoms=symptoms,
        filtered2=filtered2,
        recommendations=recommendations
    )

@app.route('/house_clustering' , methods=['GET' , "POST"])
def house_clustering():
    model = models['house_clustering']['model']
    prediction = None
    if request.method=="POST":
        try:
            house_age = datetime.now().year - int(request.form['yr_built'])
            yr_since_renovated = datetime.now().year - int(request.form['yr_renovated'])
            floors = float(request.form['floors'])
            waterfront = int(request.form['waterfront'])
            view = int(request.form['view'])
            condition = int(request.form['condition'])
            grade = int(request.form['grade'])
            bedrooms = float(request.form['bedrooms'])
            bathrooms = float(request.form['bathrooms'])
            sqft_living = np.log1p(float(request.form['sqft_living']))
            sqft_lot = np.log1p(float(request.form['sqft_lot']))
            sqft_above = np.log1p(float(request.form['sqft_above']))
            sqft_basement = np.log1p(float(request.form['sqft_basement']))
            price = np.log1p(int(request.form['price']))
            totalrooms = bedrooms + bathrooms
            livingratio = sqft_living / sqft_lot
            basement_ratio = sqft_basement / sqft_living
            above_ratio = sqft_above / sqft_living
            price_per_sqft = price / sqft_living

            df = pd.DataFrame([{
                "floors":floors ,
                "waterfront":waterfront ,
                "view":view ,
                "condition":condition ,
                "grade":grade ,
                "house_age":house_age ,
                "yr_since_renovated":yr_since_renovated ,
                'Total Rooms':totalrooms ,
                'living ratio':livingratio ,
                'basement_ratio':basement_ratio ,
                'above_ratio':above_ratio ,
                'price_per_sqft':price_per_sqft
            }])

            cluster_map = {
            0: "Medium Expensive House",
            1: "Medium-Large House",
            2: "Compact / Affordable House",
            3: "Premium House"
            }

            result = int(model.predict(df)[0])
            prediction = cluster_map.get(result , "Unknown Category")

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('house_clustering.html' , prediction=prediction)
    
if __name__=="__main__":
    app.run(debug=True)