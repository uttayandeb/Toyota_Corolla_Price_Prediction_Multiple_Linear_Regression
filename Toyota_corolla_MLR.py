#Consider only the below columns and prepare a prediction model for predicting Price.
#Corolla<-Corolla[c("Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight")]



############################ Multilinear Regression #################

#importing required packages 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import csv

# loading the data
Toyota_Corolla = pd.read_csv("C:\\Users\\home\\Desktop\\Data Science Assignments\\Python_codes\\Multiple_linear_regression\\Toyota_Corolla.csv", encoding= 'unicode_escape')

#csv.reader(open('C:\\Users\\home\\Desktop\\Data Science Assignments\\Python_codes\\Multiple_linear_regression\\Toyota_Corolla.csv', newline='', encoding='utf-8'))



######################### Data Cleaning #############################

#Only consider the mentioned columns
#Toyota_Corolla_new=pd.DataFrame(Toyota_Corolla["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"])


Toyota_Corolla_new = Toyota_Corolla.filter(["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"], axis=1)
#we can also use the drop fucntion


# to get top 6 rows
Toyota_Corolla_new.head(6) # 

Toyota_Corolla_new.shape  #(1436, 9)
Toyota_Corolla_new.dtypes# all int type

 #number of null values

Toyota_Corolla_new.info()# so there are no null values in the data

Toyota_Corolla_new.columns

# number of unique values of column cd
Toyota_Corolla_new.nunique()



################## EDA(Exploratory Data analysis) ##################

#1st moment business decision
Toyota_Corolla_new.mean()

Toyota_Corolla_new.median()

Toyota_Corolla_new.mode()

#2nd moment busines decision
Toyota_Corolla_new.var()  
                 
Toyota_Corolla_new.std()            



# 3rd and 4th moment business decision
Toyota_Corolla_new.skew()

Toyota_Corolla_new.kurt()

#### Graphical representation   #########
                  
plt.hist(Toyota_Corolla_new.Price)
plt.boxplot(Toyota_Corolla_new.Price)#we have lots of outliers


plt.hist(Toyota_Corolla_new.Age_08_04)
plt.boxplot(Toyota_Corolla_new.Age_08_04)

plt.plot(Toyota_Corolla_new.Price,Toyota_Corolla_new.HP,"bo");plt.xlabel("Price");plt.ylabel("Horse power")
#so we can c that the price exactly does not depends on the HP since its not linear

plt.plot(Toyota_Corolla_new.Price,Toyota_Corolla_new.KM,"bo");plt.xlabel("Price");plt.ylabel(" Accumulated Kilometers")


plt.plot(Toyota_Corolla_new.Price,Toyota_Corolla_new.cc,"bo");plt.xlabel("Price");plt.ylabel(" Cylinder volume")


plt.plot(Toyota_Corolla_new.Price,Toyota_Corolla_new.Doors,"bo");plt.xlabel("Price");plt.ylabel(" Doors")


plt.plot(Toyota_Corolla_new.Price,Toyota_Corolla_new.Gears,"bo");plt.xlabel("Price");plt.ylabel(" Gears")


plt.plot(Toyota_Corolla_new.Price,Toyota_Corolla_new.Quarterly_Tax,"bo");plt.xlabel("Price");plt.ylabel(" Quarterly_Tax")



plt.plot(Toyota_Corolla_new.Price,Toyota_Corolla_new.Weight,"bo");plt.xlabel("Price");plt.ylabel("weight")


plt.plot(Toyota_Corolla_new.Price,Toyota_Corolla_new.Age_08_04,"bo");plt.xlabel("Price");plt.ylabel("Age in Months")
















Toyota_Corolla_new.Price.corr(Toyota_Corolla_new.Age_08_04) # -0.8765904971436391 # correlation value between X and Y

### or ### table format
Toyota_Corolla_new.corr() 
#        Price  Age_08_04        KM  ...     Gears  Quarterly_Tax    Weight
#Price          1.000000  -0.876590 -0.569960  ...  0.063104       0.219197  0.581198
#Age_08_04     -0.876590   1.000000  0.505672  ... -0.005364      -0.198431 -0.470253
#KM            -0.569960   0.505672  1.000000  ...  0.015023       0.278165 -0.028598
#HP             0.314990  -0.156622 -0.333538  ...  0.209477      -0.298432  0.089614
#cc             0.126389  -0.098084  0.102683  ...  0.014629       0.306996  0.335637
#Doors          0.185326  -0.148359 -0.036197  ... -0.160141       0.109363  0.302618
#Gears          0.063104  -0.005364  0.015023  ...  1.000000      -0.005452  0.020613
#Quarterly_Tax  0.219197  -0.198431  0.278165  ... -0.005452       1.000000  0.626134
#Weight         0.581198  -0.470253 -0.028598  ...  0.020613       0.626134  1.000000

#[9 rows x 9 columns]         

#or using numpy
np.corrcoef(Toyota_Corolla_new.Price,Toyota_Corolla_new.KM)

import seaborn as sns
sns.pairplot(Toyota_Corolla_new)

# Correlation matrix 
correlation=Toyota_Corolla_new.corr()









                             
# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model

         
############ Preparing MLR model  ####################
                
model1 = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=Toyota_Corolla_new).fit() # regression model

# Getting coefficients of variables               
model1.params

# Summary
model1.summary()# Adj. R-squared:                  0.863




#### preparing different models based on each column

# preparing model based only on Age_08_04
model_a=smf.ols('Price~Age_08_04',data = Toyota_Corolla_new).fit()  
model_a.summary() #  Adj. R-squared:                  0.768
# p-value <0.05 .. It is significant 

# Preparing model based only on KM
model_k=smf.ols('Price~KM',data = Toyota_Corolla_new).fit()  
model_k.summary() #  Adj. R-squared:                  0.324

# Preparing model based only on HP
model_h=smf.ols('Price~HP',data = Toyota_Corolla_new).fit()  
model_h.summary() #Adj. R-squared:                  0.099

# Preparing model based only on cc
model_cc=smf.ols('Price~cc',data = Toyota_Corolla_new).fit()  
model_cc.summary() #  Adj. R-squared:                  0.015

# Preparing model based only on Doors
model_D=smf.ols('Price~Doors',data = Toyota_Corolla_new).fit()  
model_D.summary() # Adj. R-squared:                  0.034

# Preparing model based only on Gears
model_g=smf.ols('Price~Gears',data =Toyota_Corolla_new).fit()  
model_g.summary() #  Adj. R-squared:                  0.003

# Preparing model based only on Quarterly_Tax
model_q=smf.ols('Price~Quarterly_Tax',data = Toyota_Corolla_new).fit()  
model_q.summary() #   Adj. R-squared:                  0.047

# Preparing model based only on Weight
model_W=smf.ols('Price~Weight',data = Toyota_Corolla_new).fit()  
model_W.summary() #   Adj. R-squared:                  0.337





# Preparing model based only on Age_08_04, Weight & KM
model_aWK=smf.ols('Price~Age_08_04+Weight + KM',data =Toyota_Corolla_new).fit()  
model_aWK.summary() # Adj. R-squared:                  0.848
# Both coefficients p-value is significant... 


# Checking whether data has any influential values 
# influence index plots

import statsmodels.api as sm
sm.graphics.influence_plot(model1)
# index 80,960 is showing high influence so we can exclude that entire row
Toyota_Corolla_new2=Toyota_Corolla_new.drop(Toyota_Corolla_new.index[[80,960]],axis=0)


# Studentized Residuals = Residual/standard deviation of residuals





# X => A B C D 
# X.drop(["A","B"],axis=1) # Dropping columns 
# X.drop(X.index[[5,9,19]],axis=0)

#X.drop(["X1","X2"],aixs=1)
#X.drop(X.index[[0,2,3]],axis=0)


# Preparing model                  
model1_new = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=Toyota_Corolla_new2).fit()   

# Getting coefficients of variables        
model1_new.params

# Summary
model1_new.summary() #   Adj. R-squared:                  0.873,little bit increased
# Confidence values 99%
print(model1_new.conf_int(0.01)) # 99% confidence level


# Predicted values of price
price_pred = model1_new.predict(Toyota_Corolla_new2[["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]])
price_pred

Toyota_Corolla_new2.head()


# calculating VIF's values of independent variables
rsq_Age_08_04 = smf.ols('Age_08_04~KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=Toyota_Corolla_new2).fit().rsquared  
vif_Age_08_04= 1/(1-rsq_Age_08_04)#1.9368015852756546
print(vif_Age_08_04) 

rsq_KM = smf.ols('KM~Age_08_04+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=Toyota_Corolla_new2).fit().rsquared  
vif_KM = 1/(1-rsq_KM) #1.9096957519128952

rsq_HP = smf.ols('HP~Age_08_04+KM+cc+Doors+Gears+Quarterly_Tax+Weight',data=Toyota_Corolla_new2).fit().rsquared  
vif_HP = 1/(1-rsq_HP) #1.5986159573520944


rsq_cc = smf.ols('cc~Age_08_04+KM+HP+Doors+Gears+Quarterly_Tax+Weight',data=Toyota_Corolla_new2).fit().rsquared  
vif_cc = 1/(1-rsq_cc) #2.8236807345026476

rsq_Doors = smf.ols('Doors~Age_08_04+KM+cc+HP+Gears+Quarterly_Tax+Weight',data=Toyota_Corolla_new2).fit().rsquared  
vif_Doors = 1/(1-rsq_Doors) #1.1846106894939212



rsq_Gears = smf.ols('Gears~Age_08_04+KM+cc+Doors+HP+Quarterly_Tax+Weight',data=Toyota_Corolla_new2).fit().rsquared  
vif_Gears = 1/(1-rsq_Gears) #1.1009477991964105


rsq_Quarterly_Tax = smf.ols('Quarterly_Tax~Age_08_04+KM+cc+Doors+Gears+HP+Weight',data=Toyota_Corolla_new2).fit().rsquared  
vif_Quarterly_Tax = 1/(1-rsq_Quarterly_Tax) #2.9632454185292407

rsq_Weight = smf.ols('Weight~Age_08_04+KM+cc+Doors+Gears+HP+Quarterly_Tax',data=Toyota_Corolla_new2).fit().rsquared  
vif_Weight = 1/(1-rsq_Weight) #3.3134504413782593


#rsq_speed1 = smf.ols('speed~cd+multi+premium',data=Computer_Data_new).fit().rsquared  
#vif_speed1 = 1/(1-rsq_speed1)#0.07116024326964443
 




##################### Storing vif values in a data frame ###################

d1 = {'Variables':["Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"],'VIF':[vif_Age_08_04,vif_KM,vif_HP,vif_cc,vif_Doors,vif_Gears,vif_Quarterly_Tax,vif_Weight]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As there is no  higher VIF value

# Added varible plot 
sm.graphics.plot_partregress_grid(model1_new)

# added varible plot for weight is not showing any significance 

######################## final model ###################################
final_model= smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=Toyota_Corolla_new2).fit()
final_model.params
final_model.summary() # Adj. R-squared:                  0.873
# As we can see that r-squared value has increased

price_pred = final_model.predict(Toyota_Corolla_new2)

import statsmodels.api as sm
# added variable plot for the final model
sm.graphics.plot_partregress_grid(final_model)


######  Linearity #########
# Observed values VS Fitted values
plt.scatter(Toyota_Corolla_new2.Price,price_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

# Residuals VS Fitted Values 
plt.scatter(price_pred,final_model.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")


########    Normality plot for residuals ######
# histogram
plt.hist(final_model.resid_pearson) # Checking the standardized residuals are normally distributed

# QQ plot for residuals 
import pylab          
import scipy.stats as st

# Checking Residuals are normally distributed
st.probplot(final_model.resid_pearson, dist="norm", plot=pylab)


############ Homoscedasticity #######

# Residuals VS Fitted Values 
plt.scatter(price_pred,final_model.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")



### Splitting the data into train and test data 

from sklearn.model_selection import train_test_split
Toyota_Corolla_train,Toyota_Corolla_test  = train_test_split(Toyota_Corolla_new2,test_size = 0.2) # 20% size

# preparing the model on train data 

model_train = smf.ols("Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight",data=Toyota_Corolla_train).fit()

# train_data prediction
train_pred = model_train.predict(Toyota_Corolla_train)

# train residual values 
train_resid  = train_pred - Toyota_Corolla_train.Price

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))

# prediction on test data set 
test_pred = model_train.predict(Toyota_Corolla_test)

# test residual values 
test_resid  = test_pred - Toyota_Corolla_test.Price

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))
