#setwd(normalizePath(dirname(R.utils::commandArgs(asValues=TRUE)$"f")))
#source("../../../scripts/h2o-r-test-setup.R")
#source("h2o-r/scripts/h2o-r-test-setup.R")

library(h2o)
library(MASS)
library(glmnet)

h2o.init()

# read in the training data stored as a CSV file.  Each row contains the predictor
# and the last element is the response value
data1 = read.csv("training_set.csv",header=FALSE)
#plot(data1$V1,data1$V2)

# change dataframe into a matrix
mat1 = as.matrix(data1)
dataSize = dim(mat1)
# get column names of data frame and change the last one to 
# Y
allColnames = colnames(data1)
allColnames[dataSize[2]] = "Y"
colnames(data1) = allColnames

mat1_standardized = scale(mat1)
data1_standardized = as.data.frame(mat1_standardized)
colnames(data1_standardized) = colnames(data1)

# change response back to be non-standardized
data1_standardized$Y = data1$Y
# generate GLM model here from R
model1 = glm(Y~.,data=data1)  # no standardization
model1_standardized = glm(Y~., data=data1_standardized)

mat1 = mat1[1:dataSize[1],1:dataSize[2]-1]
mat1_standardized = mat1_standardized[1:dataSize[1],1:dataSize[2]-1]
 
# t-distribution parameter
dof = dataSize[1]-dataSize[2]

temp = matrix(1,nrow = dataSize[1], ncol = 1)
# need to be careful, when we have only 1 predictor, things get strange
if (dataSize[2] == 2) {
  mat2 = rbind(t(temp),mat1)
  mat2_standardized = rbind(t(temp),mat1_standardized)
  X_MAT = t(mat2)
  X_MAT_S = t(mat2_standardized)
} else {
  X_MAT = cbind(temp,mat1)
  X_MAT_S = cbind(temp,mat1_standardized)
}

mat4 = ginv(t(X_MAT)%*%X_MAT)
Weight = mat4%*%t(X_MAT)%*%data1$Y
predictY = X_MAT%*%Weight
delta = predictY-as.matrix(data1$Y)
mysd = t(delta)%*%delta/dof
# if you use mysd, you will get the same SE as in R
se = mysd*diag(mat4)
seqrt = sqrt(se)

# do the same with standardized inputs/response parameter
mat4_standardized = ginv(t(X_MAT_S)%*%X_MAT_S)
Weight_S = mat4_standardized%*%t(X_MAT_S)%*%data1_standardized$Y
predictY_S = X_MAT_S%*%Weight_S
delta_S = predictY_S-as.matrix(data1_standardized$Y)
mysd_S = t(delta_S)%*%delta_S/dof
# if you use mysd, you will get the same SE as in R
se_S = mysd_S*diag(mat4_standardized)
seqrt_S = sqrt(se_S)


# with the standard errors, we can now calculate the p-values
pvalue_R = coef(summary(model1))[,4]
pvalue_R_standardized = coef(summary(model1_standardized))[,4]
pvalue_theory = pvalue_R
pvalue_theory_standardized = pvalue_R

# run h2o glm and get the pValues
trainingset.hex = as.h2o(data1)
if (dataSize[2] == 2) {
  cars.glm = h2o.glm(x=1,y=dataSize[2],training_frame=trainingset.hex,compute_p_values = TRUE, lambda=0,alpha=0,standardize = FALSE)
  cars.glm.standardized = h2o.glm(x=1,y=dataSize[2],training_frame=trainingset.hex,compute_p_values = TRUE, lambda=0,alpha=0,standardize = TRUE)
} else {
  cars.glm = h2o.glm(x=1:(dataSize[2]-1),y=dataSize[2],training_frame=trainingset.hex,compute_p_values = TRUE, lambda=0,alpha=0,standardize = FALSE)
  cars.glm.standardized = h2o.glm(x=1:(dataSize[2]-1),y=dataSize[2],training_frame=trainingset.hex,compute_p_values = TRUE, lambda=0,alpha=0,standardize = TRUE)
  }

pVal = cars.glm@model$coefficients_table[1:dataSize[2],5]
pVal_standardized = cars.glm.standardized@model$coefficients_table[1:dataSize[2],5]
pvalue_h2oR = pvalue_R #get the row names and stuff
pvalue_h2oR_standardized = pvalue_R

for (ind in 1:dataSize[2])
{
  se = seqrt[ind]
  weight = Weight[ind]
  tval = abs(weight/se)
  
  # look up p-value off t-distribution with dof
  pvalue_theory[ind] = 2*pt(tval, df=dof,lower.tail=FALSE)

  se = seqrt_S[ind]
  weight = Weight_S[ind]
  tval = abs(weight/se)
  
  pvalue_theory_standardized[ind]=2*pt(tval, df=dof,lower.tail=FALSE)
  
  pvalue_h2oR[ind] = pVal[ind]
  pvalue_h2oR_standardized[ind] = pVal_standardized[ind]

}
