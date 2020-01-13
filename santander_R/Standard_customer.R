rm(list=ls(all=T))
setwd("C:/Users/Spathak/Desktop/standar-R")
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees')
lib = x[! x %in% installed.packages()[,'Package']]
if (length(lib))
  install.packages(lib, dependencies = TRUE)
lapply(x, require, character.only = TRUE)
rm(x,lib)

train_df = read.csv('train.csv')
#### removing instant variable from train_df as it is unique every single time ##########
train_df = train_df[,c(2:202)] 
head(train_df)
#Dimension of train data
dim(train_df)
#### converting target variable into factor ######
train_df$target<-as.factor(train_df$target)
#Count of target classes
table(train_df$target)
#Percenatge counts of target classes
table(train_df$target)/length(train_df$target)*100
#Bar plot for count of target classes
plot1<-ggplot(train_df,aes(target))+theme_bw()+geom_bar(stat='count',fill='red')
grid.arrange(plot1, ncol=1)
#Distribution of train attributes from 3 to 102
for (var in names(train_df)[c(2:102)]){
  target<-train_df$target
  plot<-ggplot(train_df, aes(x=train_df[[var]],fill=target)) +
    geom_density(kernel='gaussian') + ggtitle(var)+theme_classic()
  print(plot)
}
#Distribution of train attributes from 103 to 202
for (var in names(train_df)[c(103:202)]){
  target<-train_df$target
  plot<-ggplot(train_df, aes(x=train_df[[var]], fill=target)) +
    geom_density(kernel='gaussian') + ggtitle(var)+theme_classic()
  print(plot)
}

#We can observed that their is a considerable number of features which are significantly have different distributions 
#for two target variables. For example like var_0,var_1,var_9,var_198 var_180 etc.
############
#We can observed that their is a considerable number of features which are significantly have same distributions for t
#wo target variables. For example like var_3,var_7,var_10,var_171,var_185 etc.


#Finding the missing values in train data
missing_val<-data.frame(missing_val=apply(train_df,2,function(x){sum(is.na(x))}))
missing_val<-sum(missing_val)
missing_val

### There are no Missing Value in the Data set###

############################Outlier Analysis#############################################
train_df[,c(2:50)] %>% gather() %>% ggplot(aes(value)) +                     # Plot the values
  facet_wrap(~ key, scales = "free") +                     # In separate panels
  geom_density()
train_df[,c(50:98)] %>% gather() %>% ggplot(aes(value)) +                     # Plot the values
  facet_wrap(~ key, scales = "free") +                     # In separate panels
  geom_density()
train_df[,c(98:146)] %>% gather() %>% ggplot(aes(value)) +                     # Plot the values
  facet_wrap(~ key, scales = "free") +                     # In separate panels
  geom_density()
train_df[,c(146:194)] %>% gather() %>% ggplot(aes(value)) +                     # Plot the values
  facet_wrap(~ key, scales = "free") +                     # In separate panels
  geom_density()
train_df[,c(194:201)] %>% gather() %>% ggplot(aes(value)) +                     # Plot the values
  facet_wrap(~ key, scales = "free") +                     # In separate panels
  geom_density()

##################################Feature Selection################################################
library(reshape2)
train_correlations=cor(train_df[,c(2:201)])
train_correlations = melt(train_correlations)
ggplot(data = train_correlations, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile()

### there is very little correlation between the variables#############

##################################Feature Scaling################################################

for(i in colnames(train_df)[-1]){
  print(i)
  train_df[,i] = (train_df[,i] - min(train_df[,i]))/
    (max(train_df[,i] - min(train_df[,i])))
}
rm(i)
###################################Model Development#######################################
#Clean the environment
rmExcept("train_df")

#Divide data into train and test using stratified sampling method
set.seed(1234)
train.index = createDataPartition(train_df$target, p = .80, list = FALSE)
train = train_df[ train.index,]
test  = train_df[-train.index,]
rm(train.index)
library(ROSE)
#Random Oversampling Examples(ROSE)
set.seed(699)
train.rose = ROSE(target~., data =train,seed=32)$data
#target classes in balanced train data
table(train.rose$target)

#naive Bayes
library(e1071)

#Develop model
NB_model = naiveBayes(target ~ .,data = train)
NB_model_ROSE = naiveBayes(target ~ .,data = train.rose)

#predict on test cases #raw
NB_Predictions = predict(NB_model, test[,2:201], type = 'class')
NB_Predictions_ROSE = predict(NB_model_ROSE, test[,2:201], type = 'class')

#Look at confusion matrix unsampled
Conf_matrix = table(observed = test[,1], predicted = NB_Predictions)
confusionMatrix(Conf_matrix)
FN = Conf_matrix[1,2]
TP = Conf_matrix[2,2]
TN = Conf_matrix[1,1]
FP =Conf_matrix[2,1]
FNR = (FN/(FN+TP))*100
FPR = (FP/(FP+TP))*100

#Look at confusion matrix ROSE
Conf_matrix = table(observed = test[,1], predicted = NB_Predictions_ROSE)
confusionMatrix(Conf_matrix)
#this is performing really bad with sampled data

###Random Forest
RF_model = randomForest(target ~ ., train, importance = TRUE, ntree = 200)
RF_ROSE = randomForest(target ~ .,data = train.rose)

##predictions
RF_Predictions = predict(RF_model, test[,-1])
RF_Predictions_ROSE = predict(RF_ROSE, test[,-1])

## confusion matrix
Conf_matrix = table(observed = test[,1], predicted = RF_Predictions)
confusionMatrix(Conf_matrix)
FN = Conf_matrix[1,2]
TP = Conf_matrix[2,2]
TN = Conf_matrix[1,1]
FP =Conf_matrix[2,1]
FNR = (FN/(FN+TP))*100
FPR = (FP/(FP+TP))*100

#confusion matrix rf rose
Conf_matrix = table(observed = test[,1], predicted = RF_Predictions_ROSE)
confusionMatrix(Conf_matrix)
FN = Conf_matrix[1,2]
TP = Conf_matrix[2,2]
TN = Conf_matrix[1,1]
FP =Conf_matrix[2,1]
FNR = (FN/(FN+TP))*100
FPR = (FP/(FP+TP))*100

#Logistic Regression
logit_model = glm(target ~ ., data = train, family = "binomial")
LR_ROSE = glm(target ~ ., data = train.rose, family = "binomial")


#summary of the model
summary(logit_model)

#summarye LR Rose
summary(LR_ROSE)


#predict using logistic regression
logit_Predictions = predict(logit_model, newdata = test[-1], type = "response")
logit_Predictions = ifelse(logit_Predictions > 0.5, 1, 0)
##Evaluate the performance of classification model
ConfMatrix_LR = table(test$target, logit_Predictions)
confusionMatrix(ConfMatrix_LR)
FN = ConfMatrix_LR[1,2]
TP = ConfMatrix_LR[2,2]
TN = ConfMatrix_LR[1,1]
FP =ConfMatrix_LR[2,1]
FNR = (FN/(FN+TP))*100
FPR = (FP/(FP+TP))*100

#predict using logistic regression ROSE
logit_Predictions = predict(LR_ROSE, newdata = test[-1], type = "response")
logit_Predictions = ifelse(logit_Predictions > 0.5, 1, 0)
##Evaluate the performance of classification model ROSE
ConfMatrix_LR = table(test$target, logit_Predictions)
confusionMatrix(ConfMatrix_LR)
FN = ConfMatrix_LR[1,2]
TP = ConfMatrix_LR[2,2]
TN = ConfMatrix_LR[1,1]
FP =ConfMatrix_LR[2,1]
FNR = (FN/(FN+TP))*100
FPR = (FP/(FP+TP))*100




##Decision tree for classification
#Develop Model on training data
C50_model = C5.0(target ~., train, trials = 100, rules = TRUE)
#Summary of DT model
summary(C50_model)
C50_Predictions = predict(C50_model, test[,-1])
Conf_matrix = table(observed = test[,1], predicted = C50_Predictions)
confusionMatrix(Conf_matrix)
FN = Conf_matrix[1,2]
TP = Conf_matrix[2,2]
TN = Conf_matrix[1,1]
FP =Conf_matrix[2,1]
FNR = (FN/(FN+TP))*100
FPR = (FP/(FP+TP))*100

##Decision tree for classification ROSE
#Develop Model on training data
C50_model.rose = C5.0(target ~., train.rose, trials = 100, rules = TRUE)
#Summary of DT model
summary(C50_model.rose)
C50_Predictions = predict(C50_model.rose, test[,-1])
Conf_matrix = table(observed = test[,1], predicted = C50_Predictions)
confusionMatrix(Conf_matrix)
FN = Conf_matrix[1,2]
TP = Conf_matrix[2,2]
TN = Conf_matrix[1,1]
FP =Conf_matrix[2,1]
FNR = (FN/(FN+TP))*100
FPR = (FP/(FP+TP))*100

# as we can see out of all the model LR_rose is performing better so we will go with that
#loading Test Data
test_df = read.csv('test.csv')
test = test_df[,c(1:201)] 
logit_Predictions = predict(LR_ROSE, newdata = test, type = "response")
test_df$target=logit_Predictions
write.csv(test_df,'R_submission.csv')





