# Loan-approval-prection
Loan approval prediction: hackathon by analytics vidhya

Steps to create API
1) Run the model.py file in terminal (make sure train data filled.csv path is correctly provided in model.py)
python model.py
2) If model.py is run successfully then model.plk file will be created in the directory
3) Run server.py and provide port number along with it
python server.py 12345
4) If server .py is run successfully then API is set up in http://127.0.0.1:12345/predict

To test API, Please provide the following json data
input:
[
    {"Credit_History": 1, "Total_Income": "100000", "LoanAmount": "10000"},
    {"Credit_History": 1, "Total_Income": "5000", "LoanAmount": "100000000"},
    {"Credit_History": 0, "Total_Income": "2000", "LoanAmount": "100000"},
    {"Credit_History": 0, "Total_Income": "10000000", "LoanAmount": "100"}
]

output:
{
  "prediction": "['Y', 'N', 'N', 'Y']"
}
