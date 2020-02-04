# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here
df=pd.read_csv(path)
p_a=len(df[df['fico']>700])/len(df['fico'])
# p_b=df[df['purpose']=='debt_consolidation']
df1=df[df['purpose']=='debt_consolidation']
p_b=len(df1)/len(df['purpose'])
p_a_b=(p_a and p_b)/p_a
result=p_a_b == p_a  
result
# code ends here


# --------------
# code starts here
prob_lp=df[df['paid.back.loan']=='Yes'].shape[0]/df.shape[0]
prob_cs=df[df['credit.policy']=='Yes'].shape[0]/df.shape[0]
new_df=df[df['paid.back.loan']=='Yes']

prob_pd_cs=new_df[new_df['credit.policy']=='Yes'].shape[0]/new_df.shape[0]
bayes=(prob_pd_cs*prob_lp)/prob_cs
bayes



# code ends here


# --------------
# code starts here
# ax=df.plot.bar(x='purpose' , rot=0)
# ax=df['purpose']
# ax.plot.bar()
df1=df[df['paid.back.loan']=='No']
df1.plot.bar()
# code ends here


# --------------
# code starts here
inst_median=df['installment'].median()
inst_mean=df['installment'].mean()
df['installment'].hist()
df['log.annual.inc'].hist()


# code ends here


