import sys

from flask import Flask, request, render_template, jsonify
from src.CreditcardDefaultPrediction.pipelines.prediction_pipeline import CustomDataset, PredictPipeline
from src.CreditcardDefaultPrediction.exception import CustomException
from src.CreditcardDefaultPrediction.logger import logging

app = Flask(__name__)

try:

    @app.route('/',methods = ['GET','POST'])
    def predict_datapoints():
        if request.method == 'GET':
            return render_template('form.html')

        else:

            empty_fields = [field for field, value in request.form.items() if not value]
            if empty_fields:
                error_message = "All fields are required. Please fill out the form completely."
                return render_template('form.html', error_message=error_message)
            

            data = CustomDataset(LIMIT_BAL=float(request.form.get('LIMIT_BAL')),
                                                SEX=int(request.form.get('SEX')),
                                                EDUCATION=int(request.form.get('EDUCATION')),
                                                MARRIAGE=int(request.form.get('MARRIAGE')),
                                                AGE=int(request.form.get('AGE')),
                                                PAY_0=int(request.form.get('PAY_0')),
                                                PAY_2=int(request.form.get('PAY_2')),
                                                PAY_3=int(request.form.get('PAY_3')),
                                                PAY_4=int(request.form.get('PAY_4')),
                                                PAY_5=int(request.form.get('PAY_5')),
                                                PAY_6=int(request.form.get('PAY_6')),
                                                BILL_AMT1=float(request.form.get('BILL_AMT1')),
                                                BILL_AMT2=float(request.form.get('BILL_AMT2')),
                                                BILL_AMT3=float(request.form.get('BILL_AMT3')),
                                                BILL_AMT4=float(request.form.get('BILL_AMT4')),
                                                BILL_AMT5=float(request.form.get('BILL_AMT5')),
                                                BILL_AMT6=float(request.form.get('BILL_AMT6')),
                                                PAY_AMT1=float(request.form.get('PAY_AMT1')),
                                                PAY_AMT2=float(request.form.get('PAY_AMT2')),
                                                PAY_AMT3=float(request.form.get('PAY_AMT3')),
                                                PAY_AMT4=float(request.form.get('PAY_AMT4')),
                                                PAY_AMT5=float(request.form.get('PAY_AMT5')),
                                                PAY_AMT6=float(request.form.get('PAY_AMT6'))
                            )
        
        
        final_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()

        prediction = predict_pipeline.predict(final_data)

        result = prediction.tolist()


        if result[0] == 1:
            string = "The credit card holder will be Defaulter in the next month"
        else:
            string = "The Credit card holder will not be Defaulter in the next month"

        print(string)

        return render_template('result.html', final_result=string)
    
except Exception as e:
    logging.info("An exception has occured in home route.")
    raise CustomException(e, sys)
    



if __name__ == "__main__":
    app.run(host="0.0.0.0")

'''if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080)'''
