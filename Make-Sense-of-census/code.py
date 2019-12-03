# --------------
# Importing header files
import numpy as np
# print(path)

# Path of the file has been stored in variable called 'path'

#New record
new_record=[[50,  9,  4,  1,  0,  0, 40,  0]]

#Code starts here
census = np.array([])
data=np.genfromtxt(path, delimiter=",", skip_header=1)
census = np.concatenate((data , new_record))
# print(census)



# --------------
#Code starts here
age = census[:,0]
max_age = np.max(age)
min_age = np.min(age)
age_mean = np.mean(age)
age_std = np.std(age)


# --------------
# Code starts here
# race_0 = np.array([])
# race_1 = np.array([])
# race_2 = np.array([])
# race_3 = np.array([])
# race_4 = np.array([])
# for i in range(0,census.shape[0]):
#     if int(census[i,2]) == 0:
#         race_0 = np.concatenate(race_0 , np.array([census[i , :]]))
#     elif int(census[i,2]) == 1:
#         race_1 = np.concatenate(race_1 , np.array([census[i , :]]))
#     elif int(census[i,2]) == 2:
#         race_2 = np.concatenate(race_2 , np.array([census[i , :]]))
#     elif int(census[i,2]) == 3:
#         race_3 = np.concatenate(race_3 , np.array([census[i , :]]))
#     else:
#         race_4 = np.concatenate(race_4 , np.array([census[i , :]]))

# print('r0 \n' , race_0)
# print(census[0 , :])
# len_0 , len_1 , len_2 , len_3 , len_4 = len(race_0) , len(race_1) , len(race_2) , len(race_3) , len(race_4)
# minority_race = np.min(np.array([len_0 , len_1 , len_2 , len_3 , len_4]))
# race_0 = np.array([])
# for i in range(0,census.shape[0]):
#      if int(census[i,2]) == 0:
#          race_0 = np.append(race_0 , np.array([census[i , :]]))
race_0=census[census[:,2]==0]
race_1=census[census[:,2]==1]
race_2=census[census[:,2]==2]
race_3=census[census[:,2]==3]
race_4=census[census[:,2]==4]

len_0=len(race_0)
len_1=len(race_1)
len_2=len(race_2)
len_3=len(race_3)
len_4=len(race_4)

Race_list=[len_0,len_1,len_2,len_3,len_4]

minority_race=Race_list.index(min(Race_list))
print(minority_race)





# --------------
#Code starts here
senior_citizens = census[census[:,0]>60]
working_hours_sum = senior_citizens.sum(axis=0)[6]
senior_citizens_len = len(senior_citizens)
avg_working_hours = (working_hours_sum)/(senior_citizens_len)
print(avg_working_hours)


# --------------
#Code starts here
high = census[census[:,1]>10]
low = census[census[:,1]<=10]
avg_pay_high = high.mean(axis=0)[7]
avg_pay_low = low.mean(axis=0)[7]
avg_pay_high,avg_pay_low.mean()


