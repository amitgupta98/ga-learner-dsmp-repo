# --------------
#Importing header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(path)
print(data)
loan_status=data['Loan_Status'].value_counts()
loan_status.plot()

#Code starts here


# --------------
#Code starts here
property_and_loan=data.groupby(['Property_Area','Loan_Status']).size().unstack()
property_and_loan
property_and_loan.plot(kind='bar', stacked=False)
plt.xlabel('Property Area')
plt.ylabel('Loan Status')
plt.xticks(rotation=45)





# # Label X-axes and Y-axes
# plt.xlabel('Type 1')
# plt.ylabel('Frequency')
# # Rotate X-axes labels
# plt.xticks(rotation=45)


# --------------
#Code starts here
education_and_loan=data.groupby(['Education','Loan_Status']).size().unstack()
property_and_loan
education_and_loan.plot(kind='bar')
plt.xlabel('Education Status')
plt.ylabel('Loan Status')
plt.xticks(rotation=45)


# --------------
#Code starts here

# graduate
graduate=data[data['Education']=='Graduate']

# not_graduate
not_graduate=data[data['Education']=='Not Graduate']


graduate.plot(kind='density')
# plt.label('Graduate')
not_graduate.plot(kind='density')




#Code ends here

#For automatic legend display
plt.legend()


# --------------
#Code starts here
fig ,(ax_1,ax_2,ax_3)=plt.subplots(nrows=3, ncols=1)
plt.scatter(data['ApplicantIncome'], data['LoanAmount'],label='ax_1')
plt.title('ApplicantIncome')
plt.scatter(data['CoapplicantIncome'],data['LoanAmount'],label='ax_2')
plt.title('Coapplicant Income')
data['TotalIncome']=data['ApplicantIncome']+data['CoapplicantIncome']
plt.scatter(data['TotalIncome'],data['LoanAmount'],label='ax_3')
plt.title('Total Income')






