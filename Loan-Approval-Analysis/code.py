# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
 



# code starts here
# path
df = path
bank = pd.read_csv(df)
bank
# categorical_var = df.select_dtypes(include = 'object')
categorical_var = bank.select_dtypes(include = 'object')
categorical_var
numerical_var = bank.select_dtypes(include = 'number')
numerical_var

# code ends here


# --------------
# code starts here
# print(bank)
bank.drop(['Loan_ID'],inplace=True,axis=1)
banks=bank
banks.isnull().sum()
# print(df)
bank_mode = banks.mode()
# bank_mode
banks.fillna("bank_mode",inplace=True)
print(banks)
#code ends here


# --------------
# Code starts here
import numpy as np
import pandas as pandas


avg_loan_amount = pd.pivot_table(banks,index=['Gender','Married','Self_Employed'],values='LoanAmount',aggfunc=np.mean)
avg_loan_amount



# code ends here 


# --------------
# code starts here




loan_approved_se =banks[(banks.Self_Employed=='Yes') & (banks.Loan_Status =='Y')]['Loan_Status'].count()

loan_approved_nse = banks[(banks.Self_Employed=='No') & (banks.Loan_Status=='Y')]['Loan_Status'].count()
Loan_Status=banks.Loan_Status.count()


# percentage_se = 
percentage_se=(loan_approved_se/Loan_Status)*100

percentage_nse=(loan_approved_nse/Loan_Status)*100
# code ends here


# --------------
# code starts here
loan_term=banks['Loan_Amount_Term'].apply(lambda x: x/12)
loan_term

# big_loan_term=
big_loan_term=loan_term.apply(lambda x: x>=25).value_counts().loc[True]

# code ends here


# --------------
# code starts here
loan_groupby=banks.groupby('Loan_Status')
# loan_groupby
loan_groupby=loan_groupby['ApplicantIncome','Credit_History']
loan_groupby
mean_values=loan_groupby.mean()

# code ends here


