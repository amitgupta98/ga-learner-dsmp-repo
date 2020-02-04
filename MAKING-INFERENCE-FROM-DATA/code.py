# --------------
import pandas as pd
import scipy.stats as stats
import math
import numpy as np
import warnings

warnings.filterwarnings('ignore')
#Sample_Size
sample_size=2000

#Z_Critical Score
z_critical = stats.norm.ppf(q = 0.95)  


# path        [File location variable]

#Code starts here
data=pd.read_csv(path)
data_sample=data.sample(n=sample_size,random_state=0)
sample_mean=data_sample['installment'].mean()
sample_std=data_sample['installment'].std()
margin_of_error=z_critical*sample_std/math.sqrt(sample_size)
confidence_interval=(sample_mean-margin_of_error,sample_mean+margin_of_error)
true_mean=data['installment'].mean()
true_mean



# --------------
import matplotlib.pyplot as plt
import numpy as np

#Different sample sizes to take
sample_size=np.array([20,50,100])

#Code starts here
fig ,axes =plt.subplots(nrows = 3 , ncols = 1)
for i in range(len(sample_size)):
    m=[]
    for j in range(1,1000):
        data['installment'].sample(n=sample_size[i])
        data['installment'].mean()
        data['installment'].append(m)
    mean_series=pd.Series(m)
    mean_series.hist()



# --------------
#Importing header files

from statsmodels.stats.weightstats import ztest

#Code starts here
data['int.rate']=[x.strip('%') for x in data['int.rate']]
data['int.rate']=data['int.rate'].astype(float)
data['int.rate']=data['int.rate']/100
z_statistic, p_value = ztest(data[data['purpose']=='small_business']['int.rate'],value=data['int.rate'].mean(),alternative='larger')
print("Z-statistics = ",z_statistic)
print("p-value = ",p_value)




# --------------
#Importing header files
from statsmodels.stats.weightstats import ztest

#Code starts here
x1=data[data['paid.back.loan']=='No']['installment']
x2=data[data['paid.back.loan']=='Yes']['installment']
z_statistic, p_value = ztest(x1,x2,value=0,alternative='two-sided')
print("Z-statistics = ",z_statistic)
print("p-value = ",p_value)
# def twoSampZ(X1, X2, mudiff, sd1, sd2, n1, n2):
#     from numpy import sqrt, abs, round
#     from scipy.stats import norm
#     pooledSE = sqrt(sd1**2/n1 + sd2**2/n2)
#     z = ((X1 - X2) - mudiff)/pooledSE
#     pval = 2*(1 - norm.cdf(abs(z)))
#     return round(z, 3), round(pval, 4)
# print (z, p)



# --------------
#Importing header files
from scipy.stats import chi2_contingency

#Critical value 
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 6)   # Df = number of variable categories(in purpose) - 1

#Code starts here
yes=data[data['paid.back.loan']=='Yes']['purpose'].value_counts()
no=data[data['paid.back.loan']=='No']['purpose'].value_counts()
observed=pd.concat([yes.transpose(),no.transpose()],keys=['Yes','No'],axis=1)
chi2, p, dof, ex=stats.chi2_contingency(observed)
print('chi-square statistic =',chi2)
print('critical value =',critical_value)
# observed



