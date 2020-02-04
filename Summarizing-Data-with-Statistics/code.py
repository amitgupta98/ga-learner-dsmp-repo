# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#path of the data file- path

#Code starts here 
data=pd.read_csv(path)
data['Gender'].replace('-','Agender',inplace=True)
gender_count=data['Gender'].value_counts()
gender_count.plot(kind='bar')


# --------------
#Code starts here
alignment=data['Alignment'].value_counts()
alignment.plot.pie(label='Character Alignment')



# --------------
#Code starts here
sc_df=data[['Strength','Combat']]
# sc_smean=data['Strength'].mean()
# sc_cmean=data['Combat'].mean()
# n=len(sc_df)
sc_covariance=sc_df.cov().iloc[0,1]
# sc_covariance=1/n*(sc_df['Strength']-sc_smean)*(sc_df['Combat']-sc_cmean)
sc_strength=sc_df['Strength'].std()
sc_combat=sc_df['Combat'].std()
sc_pearson=sc_covariance/(sc_strength*sc_combat)

ic_df=data[['Intelligence','Combat']]
ic_covariance=ic_df.cov().iloc[0,1]
# ic_imean=ic_df['Intelligence'].mean()
# ic_cmean=ic_df['Combat'].mean()
# n=len(ic_df)
# ic_covariance=1/n*(ic_df['Intelligence']-ic_imean)*(sc_df['Combat']-ic_cmean)
ic_intelligence=ic_df['Intelligence'].std()
ic_combat=ic_df['Combat'].std()
ic_pearson=ic_covariance/(ic_intelligence*ic_combat)


# --------------
#Code starts here
total_high=data['Total'].quantile(0.99)
super_best=data[data['Total']>total_high]
super_best_names=data['Name'].tolist()
super_best_names


# --------------
#Code starts here
fig, (ax_1, ax_2, ax_3)= plt.subplots(1,3, figsize=(20,10))

ax_1=plt.boxplot(data['Intelligence'])
ax_2=plt.boxplot(data['Speed'])
ax_3=plt.boxplot(data['Power'])


