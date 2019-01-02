---
title: "Data Mining_HW1"
author: "Youfei Zhang"
---
require(glmnet)
library(dplyr)
library(tidyr)
library(data.table)

## Standardize scientific notations
options(scipen = 999)

## Set Today's Date
tdate <- strftime(Sys.Date(), "_%Y-%m-%d") # for naming files

## Question 1: L2 linear regression 

## for dataset 1: train-100-10.csv, test-100-10.csv
train1 = read.csv(file = "Desktop/code/CISC6930_DM/hw/hw1/train-100-10.csv", sep = ",")
test1 = read.csv(file = "Desktop/code/CISC6930_DM/hw/hw1/test-100-10.csv", sep = ",")

x.train1 = model.matrix(y~.-1, data = train1)
y.train1 = train1$y
x.test1 = model.matrix(y~.-1, data = test1)
y.test1 = test1$y

lambdas = (seq(0:150))
fit.ridge1 = glmnet(x.train1, y.train1, alpha = 0, lambda = lambdas)

pred.tr1 = predict(fit.ridge1, x.train1)
mse.tr1 = apply((pred.tr1 - y.train1)^2, 2, mean) # column-wised 

pred.test1 = predict(fit.ridge1, x.test1)
mse.test1 = apply((y.test1 - pred.test1)^2, 2, mean)
# rmse.test1 = sqrt(apply((y.test1 - pred.test1)^2, 2, mean))
# best.lam = fit.ridge1$lambda[order(rmse.test1)[1]]
                  
par(mar = c(5,4,4,4))
plot(log(fit.ridge1$lambda), mse.tr1, type = "l", col = "red", xlab = "Log(lambda)", ylab = "MSE")
mtext("Train MSE",side = 2,line = 2,col = 2)
par(new = TRUE)
plot(log(fit.ridge1$lambda), mse.test1, type = 'l', axes = F, col = "blue", xlab = " ", ylab = " ")
axis(side = 4)
mtext("Test MSE",side = 4,line = 2,col = 4)

######
## for dataset 2: train-100-100.csv, test-100-100.csv
train2 = read.csv('/Users/Scarlet/Desktop/code/CISC6930_DM/hw/hw1/train-100-100.csv', header = TRUE, sep = ",")
test2 = read.csv("/Users/Scarlet/Desktop/code/CISC6930_DM/hw/hw1/test-100-100.csv", header = TRUE, sep = ",")

x.train2 = model.matrix(y~.-1, data = train2)
y.train2 = train2$y
x.test2 = model.matrix(y~.-1, data = test2)
y.test2 = test2$y

lambdas = (seq(0:150))
fit.ridge2 = glmnet(x.train2, y.train2, alpha = 0, lambda = lambdas)

pred.tr2 = predict(fit.ridge2, x.train2)
mse.tr2 = apply((pred.tr2 - y.train2)^2, 2, mean) # column-wised 

pred.test2 = predict(fit.ridge2, x.test2)
mse.test2 = apply((y.test2 - pred.test2)^2, 2, mean)


par(mar = c(5,4,4,4))
plot(log(fit.ridge2$lambda), mse.tr2, type = "l", col = "red", xlab = "Log(lambda)", ylab = "MSE")
mtext("Train MSE",side = 2,line = 2,col = 2)
par(new = TRUE)
plot(log(fit.ridge2$lambda), mse.test2, type = 'l', axes = F, col = "blue", xlab = " ", ylab = " ")
axis(side = 4)
mtext("Test MSE",side = 4,line = 2,col = 4)

######
## for dataset 3: train-1000-100.csv, test-1000-100.csv
train3 = read.csv('/Users/Scarlet/Desktop/code/CISC6930_DM/hw/hw1/train-1000-100.csv', header = TRUE, sep = ",")
test3 = read.csv('/Users/Scarlet/Desktop/code/CISC6930_DM/hw/hw1/test-1000-100.csv', header = TRUE, sep = ",")

x.train3 = model.matrix(y~.-1, data = train3)
y.train3 = train3$y
x.test3 = model.matrix(y~.-1, data = test3)
y.test3 = test3$y

lambdas = (seq(0:150))
fit.ridge3 = glmnet(x.train3, y.train3, alpha = 0, lambda = lambdas)

pred.tr3 = predict(fit.ridge3, x.train3)
mse.tr3 = apply((pred.tr3 - y.train3)^2, 2, mean) # column-wised 

pred.test3 = predict(fit.ridge3, x.test3)
mse.test3 = apply((y.test3 - pred.test3)^2, 2, mean)

par(mar = c(5,4,4,4))
plot(log(fit.ridge3$lambda), mse.tr3, type = "l", col = "red", xlab = "Log(lambda)", ylab = "MSE")
mtext("Train MSE",side = 2,line = 2,col = 2)
par(new = TRUE)
plot(log(fit.ridge3$lambda), mse.test3, type = 'l', axes = F, col = "blue", xlab = " ", ylab = " ")
axis(side = 4)
mtext("Test MSE",side = 4,line = 2,col = 4)

## for train-1000-100.csv: 
## taking the first 50, 100, and 150 instances
train3_50 = train3[0:50,]
train3_100 = train3[0:100,]
train3_150 = train3[0:150,]

## for dataset 3 - 50 instances
x.train3_50 = model.matrix(y~.-1, data = train3_50)
y.train3_50 = train3_50$y

fit.ridge3_50 = glmnet(x.train3_50, y.train3_50, alpha = 0, lambda = lambdas)

pred.tr3_50 = predict(fit.ridge3_50, x.train3_50)
mse.tr3_50 = apply((pred.tr3_50 - y.train3_50)^2, 2, mean)

pred.test3_50 = predict(fit.ridge3_50, x.test3)
mse.test3_50 = apply((pred.test3_50 - y.test3)^2, 2, mean)

par(mar = c(5,4,4,4))
plot(log(fit.ridge3_50$lambda), mse.tr3_50, type = "l", col = "red", xlab = "Log(lambda)", ylab = "MSE")
mtext("Train MSE",side = 2,line = 2,col = 2)
par(new = TRUE)
plot(log(fit.ridge3_50$lambda), mse.test3_50, type = 'l', axes = F, col = "blue", xlab = " ", ylab = " ")
axis(side = 4)
mtext("Test MSE",side = 4,line = 2,col = 4)

## for dataset 3 - 100 instances
x.train3_100 = model.matrix(y~.-1, data = train3_100)
y.train3_100 = train3_100$y

fit.ridge3_100 = glmnet(x.train3_100, y.train3_100, alpha = 0, lambda = lambdas)

pred.tr3_100 = predict(fit.ridge3_100, x.train3_100)
mse.tr3_100 = apply((pred.tr3_100 - y.train3_100)^2, 2, mean)

pred.test3_100 = predict(fit.ridge3_100, x.test3)
mse.test3_100 = apply((pred.test3_100 - y.test3)^2, 2, mean)

par(mar = c(5,4,4,4))
plot(log(fit.ridge3_100$lambda), mse.tr3_100, type = "l", col = "red", xlab = "Log(lambda)", ylab = "MSE")
mtext("Train MSE",side = 2,line = 2,col = 2)
par(new = TRUE)
plot(log(fit.ridge3_100$lambda), mse.test3_100, type = 'l', axes = F, col = "blue", xlab = " ", ylab = " ")
axis(side = 4)
mtext("Test MSE",side = 4,line = 2,col = 4)

## for dataset 3 - 150 instances
x.train3_150 = model.matrix(y~.-1, data = train3_150)
y.train3_150 = train3_150$y

fit.ridge3_150 = glmnet(x.train3_150, y.train3_150, alpha = 0, lambda = lambdas)

pred.tr3_150 = predict(fit.ridge3_150, x.train3_150)
mse.tr3_150 = apply((pred.tr3_150 - y.train3_150)^2, 2, mean)

pred.test3_150 = predict(fit.ridge3_150, x.test3)
mse.test3_150 = apply((pred.test3_150 - y.test3)^2, 2, mean)

par(mar = c(5,4,4,4))
plot(log(fit.ridge3_150$lambda), mse.tr3_150, type = "l", col = "red", xlab = "Log(lambda)", ylab = "MSE")
mtext("Train MSE",side = 2,line = 2,col = 2)
par(new = TRUE)
plot(log(fit.ridge3_150$lambda), mse.test3_150, type = 'l', axes = F, col = "blue", xlab = " ", ylab = " ")
axis(side = 4)
mtext("Test MSE",side = 4,line = 2,col = 4)


## (a) Which lambda gives the least test set MSE?

## for dataset 1: train-100-10.csv, test-100-10.csv
opt_lambda_1 = fit.ridge1$lambda[order(mse.test1)[1]] ## put in ascending order
opt_lambda_1 # 3
coef(fit.ridge1, s = opt_lambda_1)
fit.ridge1$lambda[3]

## for dataset 2: train-100-100.csv, test-100-100.csv
opt_lambda_2 = fit.ridge2$lambda[order(mse.test2)[1]]
opt_lambda_2 # 7
coef(fit.ridge2, s = opt_lambda_2)
mse.test2['s3'] # 8.5265 
fit.ridge1$lambda[7]

## for dataset 3: train-1000-100.csv, test-1000-100.csv
opt_lambda_3 = fit.ridge3$lambda[order(mse.test3)[1]]
opt_lambda_3 
coef(fit.ridge3, s = opt_lambda_3)
mse.test3['s3'] 

## for dataset 3: 50 instances
opt_lambda_3_50 = fit.ridge3_50$lambda[order(mse.test3_50)[1]]
opt_lambda_3_50 
coef(fit.ridge3_50, s = opt_lambda_3_50)

## for dataset 3: 100 instances
opt_lambda_3_100 = fit.ridge3_100$lambda[order(mse.test3_100)[1]]
opt_lambda_3_100 
coef(fit.ridge3_100, s = opt_lambda_3_100)

## for dataset 3: 150 instances
opt_lambda_3_150 = fit.ridge3_150$lambda[order(mse.test3_150)[1]]
opt_lambda_3_150 
coef(fit.ridge3_150, s = opt_lambda_3_150)


## (b) Additional graph with lambda ranging from 1 to 150 

## draw a plot of coefficients
par(mfrow = c(3,1))

## for dataset2: 100-100
plot(fit.ridge2, xvar = "lambda", label = TRUE)

## for dataset3: 50(1000)-100
plot(fit.ridge3_50, xvar = "lambda", label = TRUE)
plot(fit10, main = "100-100")

## for dataset3: 100(1000)-100
plot(fit.ridge3_100, xvar = "lambda", label = TRUE)

# plot(fit.ridge1, xvar = "lambda", label = TRUE)
# plot(fit.ridge2, xvar = "lambda", label = TRUE)
# plot(fit.ridge3, xvar = "lambda", label = TRUE)
# plot(fit.ridge3_50, xvar = "lambda", label = TRUE)
# plot(fit.ridge3_100, xvar = "lambda", label = TRUE)
# plot(fit.ridge3_150, xvar = "lambda", label = TRUE)

## (c) Why lambda = 0 gives abnormally large MSE? 

## when lambda - 0, ridge regression is the same as full least squares 
## all the parameters are fit into the model without any penality 
## so the MSE is the largest 

##########################

## Question 2: 10-fold Cross-validation

## (a) What's the best choice of lambda given CV

## set seed for reproducibility 
set.seed(10)
## define lambda 
lambdas = (seq(0:150))

## for dataset 1: train-100-10.csv, test-100-10.csv
cv.ridge1 = cv.glmnet(x.train1, y.train1, alpha = 0, lambda = lambdas)
plot(cv.ridge1)
cv_opt_lambda1 = cv.ridge1$lambda.min
cv_opt_lambda1 
cv.mse.min1 <- cv.ridge1$cvm[cv.ridge1$lambda == cv.ridge1$lambda.min]
cv.mse.min1

# cv.pred.test1 = predict(cv.ridge1, s = cv_opt_lambda1, x.test1)
# cv.mse.test1 = apply((y.test1 - cv.pred.test1)^2, 2, mean)
# cv.mse.test1 # 6.241207

## for dataset 2: train-100-100.csv, test-100-100.csv
cv.ridge2 = cv.glmnet(x.train2, y.train2, alpha = 0, lambda = lambdas)
plot(cv.ridge2)
cv_opt_lambda2 = cv.ridge2$lambda.min
cv_opt_lambda2 
cv.mse.min2 <- cv.ridge2$cvm[cv.ridge2$lambda == cv.ridge2$lambda.min]
cv.mse.min2 

## for dataset 3: train-1000-100.csv, test-1000-100.csv
cv.ridge3 = cv.glmnet(x.train3, y.train3, alpha = 0, lambda = lambdas)
plot(cv.ridge3)
cv_opt_lambda3 = cv.ridge3$lambda.min
cv_opt_lambda3 # 1
cv.mse.min3 <- cv.ridge3$cvm[cv.ridge3$lambda == cv.ridge3$lambda.min]
cv.mse.min3 # 6.31163

## for dataset 3: 50 instances
cv.ridge3_50 = cv.glmnet(x.train3_50, y.train3_50, alpha = 0, lambda = lambdas)
plot(cv.ridge3_50)
cv_opt_lambda3_50 = cv.ridge3_50$lambda.min
cv_opt_lambda3_50 # 27
cv.mse.min3_50 <- cv.ridge3_50$cvm[cv.ridge3_50$lambda == cv.ridge3_50$lambda.min]
cv.mse.min3_50 # 8.891025

## for dataset 3: 100 instances
cv.ridge3_100 = cv.glmnet(x.train3_100, y.train3_100, alpha = 0, lambda = lambdas)
plot(cv.ridge3_100)
cv_opt_lambda3_100 = cv.ridge3_100$lambda.min
cv_opt_lambda3_100 # 12
cv.mse.min3_100 <- cv.ridge3_100$cvm[cv.ridge3_100$lambda == cv.ridge3_100$lambda.min]
cv.mse.min3_100 # 7.806104

## for dataset 3: 150 instances
cv.ridge3_150 = cv.glmnet(x.train3_150, y.train3_150, alpha = 0, lambda = lambdas)
plot(cv.ridge3_150)
cv_opt_lambda3_150 = cv.ridge3_150$lambda.min
cv_opt_lambda3_150 # 13
cv.mse.min3_150 <- cv.ridge3_150$cvm[cv.ridge3_150$lambda == cv.ridge3_150$lambda.min]
cv.mse.min3_150 # 8.193494

## (b) How do the values for lambda and MSE from CV compare to 
## the ones from 1(a)?
## see R.markdown

## (c) What are the drawbacks of CV? 
## see R.markdown

## (d) What are the factors affecting CV? 
## see R.markdown


## 3. learning curve with fixed lambda = 1, 25, 150 
## prepare package 
library(caret)

# define features and responses
x.train3 = model.matrix(y~.-1, data = train3)
y.train3 = train3$y
x.test3 = model.matrix(y~.-1, data = test3)
y.test3 = test3$y

# Run algorithms using 10-fold cross validation with 3 repeats
trainControl <- trainControl(method="repeatedcv", number = 10, repeats = 3)
metric <- "RMSE"

## When lambda = 1 
# create empty data frame to store MSE
learnCurve1 <- data.frame(m = integer(21), 
                         trainRMSE = integer(21),
                         testRMSE = integer(21))

for (i in 3:21){
  learnCurve1$m[i] <- i
  
  # train learning algorithm with size i
  fit.ridge <- train(y~., data = train3[1:i,], method = "glmnet", metric = metric, lambda = lambda1, 
                     tuneGrid = expand.grid(alpha = 0, lambda = 1), trControl=trainControl)
  learnCurve1$trainRMSE[i] <- fit.ridge$results$RMSE
  
  # use trained parameters to predict on test data
  prediction <- predict(fit.ridge, newdata = x.test3)
  rmse <- postResample(prediction, y.test3)
  learnCurve1$cvRMSE[i] <- rmse[1]
}

# plot learning curves of training set size vs. error measure
# for training set and test set
plot(log(learnCurve1$trainRMSE),type = "o",col = "red", xlab = "Training set size",
     ylab = "Error (RMSE)", main = "Learning Curve")
lines(log(learnCurve1$cvRMSE), type = "o", col = "blue")
legend('topright', c("Train error", "Test error"), lty = c(1,1), lwd = c(2.5, 2.5),
       col = c("red", "blue"))

dev.off()


## whem lambda = 25
learnCurve25 <- data.frame(m = integer(21), 
                           trainRMSE = integer(21),
                           testRMSE = integer(21))

for (i in 3:21){
  learnCurve25$m[i] <- i
  
  # train learning algorithm with size i
  fit.ridge <- train(y~., data = train3[1:i,], method = "glmnet", metric = metric, lambda = 25, tuneGrid = expand.grid(alpha = 0, lambda = 25), trControl=trainControl)
  learnCurve25$trainRMSE[i] <- fit.ridge$results$RMSE
  
  # use trained parameters to predict on test data
  prediction <- predict(fit.ridge, newdata = x.test3)
  rmse <- postResample(prediction, y.test3)
  learnCurve25$cvRMSE[i] <- rmse[1]
}

# plot learning curves
plot(log(learnCurve25$trainRMSE),type = "o",col = "red", xlab = "Training set size",
     ylab = "Error (RMSE)", main = "Learning Curve")
lines(log(learnCurve25$cvRMSE), type = "o", col = "blue")
legend('topright', c("Train error", "Test error"), lty = c(1,1), lwd = c(2.5, 2.5),
       col = c("red", "blue"))

dev.off()


## whem lambda = 150
learnCurve150 <- data.frame(m = integer(21), 
                           trainRMSE = integer(21),
                           testRMSE = integer(21))

for (i in 3:21){
  learnCurve150$m[i] <- i
  
  # train learning algorithm with size i
  fit.ridge <- train(y~., data = train3[1:i,], method = "glmnet", metric = metric, lambda = 25, tuneGrid = expand.grid(alpha = 0, lambda = 25), trControl=trainControl)
  learnCurve150$trainRMSE[i] <- fit.ridge$results$RMSE
  
  # use trained parameters to predict on test data
  prediction <- predict(fit.ridge, newdata = x.test3)
  rmse <- postResample(prediction, y.test3)
  learnCurve150$cvRMSE[i] <- rmse[1]
}

# plot learning curves
plot(log(learnCurve150$trainRMSE),type = "o",col = "red", xlab = "Training set size",
     ylab = "Error (RMSE)", main = "Learning Curve")
lines(log(learnCurve150$cvRMSE), type = "o", col = "blue")
legend('topright', c("Train error", "Test error"), lty = c(1,1), lwd = c(2.5, 2.5),
       col = c("red", "blue"))




