# Redhat Kaggle Competition #
# Stephanie Kirmer #
# August 2016 #


setwd("U:/Programming/R/Kaggle/Redhat")

#Libraries
require(randomForest)
require(caret)
require(dplyr)
require(data.table)
require(ggplot2)
library(pROC)
library(stringr)
library(dummies)
library(Metrics)
library(kernlab)
library(FeatureHashing)
library(xgboost)

#Load datasets
test_act <- read.csv("act_test.csv", stringsAsFactors=F)
train_act <- read.csv("act_train.csv", stringsAsFactors=F)
people <- read.csv("people.csv", stringsAsFactors=F)



#Format the people table ####
people2 <- as.data.table(people, keep.rownames = FALSE)

reformat <- c("char_10", "char_11",  "char_12","char_13","char_14","char_15","char_16","char_17","char_18","char_19","char_20","char_21",
              "char_22","char_23","char_24","char_25","char_26","char_27","char_28","char_29","char_30","char_31","char_32","char_33",
              "char_34","char_35","char_36","char_37")

for(var in reformat) people2[, var := as.numeric(as.logical(get(var))), with=FALSE] #Woooo anonymous functions! Get our dummies right.

#Other features are a little different, so let's set those up as numerics
#people2[char_1=="type 1", char_1n := 1][char_1=="type 2", char_1n := 2] #Hey put it all on one line! One way to do it.

move_types <- c("char_1","char_2","char_3", "char_4", "char_5", "char_6", "char_7", "char_8", "char_9") 
for(var in move_types) people2[, var := as.numeric(gsub("type ","",(get(var)))), with=FALSE] # A different approach when there are tons of types to deal with

people2[, group_1 := as.numeric(gsub("group ","",(group_1)))]
people2[, person_num := as.numeric(gsub("ppl_","",(people_id)))]

#Make sure the dates are in date format
people2[, date := as.Date(date)]


# Format the activity train table ####
train_act2 <- as.data.table(train_act, keep.rownames = FALSE)
move_types <- c("activity_category","char_1","char_2","char_3", "char_4", "char_5", "char_6", "char_7", "char_8", "char_9", "char_10") 
for(var in move_types) train_act2[, var := as.numeric(gsub("type ","",(get(var)))), with=FALSE] # A different approach when there are tons of types to deal with
for(var in move_types) train_act2[is.na(get(var)), var := 0, with=FALSE] #Fill in the missing with zeros

#Make sure the dates are in date format
train_act2[, date := as.Date(date)]

train_act2[, activity_num := as.numeric(gsub("act2_","",(activity_id)))]
train_act2[is.na(activity_num), activity_num := 0] #Fill in the missing with zeros


#Merge people and activity train just to see what starts coming together

people_act_train <- merge(people2, train_act2, by="people_id", suffixes = c("_ppl", "_act"))
people_act_train[is.na(outcome), outcome := 0] #Fill in the missing with zeros
names(people_act_train)


# Format the activity test table ####
test_act2 <- as.data.table(test_act, keep.rownames = FALSE)
move_types <- c("activity_category","char_1","char_2","char_3", "char_4", "char_5", "char_6", "char_7", "char_8", "char_9", "char_10") 
for(var in move_types) test_act2[, var := as.numeric(gsub("type ","",(get(var)))), with=FALSE] # A different approach when there are tons of types to deal with
for(var in move_types) test_act2[is.na(get(var)), var := 0, with=FALSE] #Fill in the missing with zeros

#Make sure the dates are in date format
test_act2[, date := as.Date(date)]

test_act2[, activity_num := as.numeric(gsub("act2_","",(activity_id)))]
test_act2[is.na(activity_num), activity_num := 0] #Fill in the missing with zeros


#Merge people and activity test just to see what starts coming together

people_act_test <- merge(people2, test_act2, by="people_id", suffixes = c("_ppl", "_act"))





model.matrix( ~ char_23, people_act_train)

fit.rf <- train(as.factor(outcome)~char_23, data=people_act_train, method="rf", trControl=control)


# Initial model- adjusting here is going to improve model- this uses everything except what's listed and removes the intercept


##### USING TECHNIQUE BY LEWIS ######
# Hash training set to sparse matrix x_train
b <- 2 ^ 20
f <- ~ . - people_id - person_num - activity_id - activity_num - date_act - date_ppl - 1 

X_train <- hashed.model.matrix(f, people_act_train, b) #Creating a model matrix - uses the formula inside f object to choose variables to include
# Model matrix takes all the possible variations and expands them out so you can run through them all. Also does interactions. "Hashed" means it is less space/time consuming.
#Hashed model matrix actually means that the data manipulation done above is not strictly necessary.
dtrain  <- xgb.DMatrix(X_train, label = people_act_train$outcome)


# Validate xgboost then re-train with all data
#xgboost: extreme gradient boosting
# 
set.seed(6859)
unique_p <- unique(people_act_train$people_id) # Get the unique list of people
valid_p  <- unique_p[sample(1:length(unique_p), 30000)] # choose from the unique list of people, random row, repeat 30k times

valid <- which(people_act_train$people_id %in% valid_p) # Take the list of people that was created in the random selection and identify their records in the merged dataset
model <- (1:length(people_act_train$people_id))[-valid] # model is the row numbers from the merged dataset MINUS the sample selected above
#length(valid) + length(model) == total frame size
#This is basically just partitioning the sample

param <- list(objective = "binary:logistic", eval_metric = "auc",
              booster = "gblinear", eta = 0.01) #Inputs needed for the xgboost modeling interface

dmodel  <- xgb.DMatrix(X_train[model, ], label = people_act_train$outcome[model]) #Training sample
dvalid  <- xgb.DMatrix(X_train[valid, ], label = people_act_train$outcome[valid]) #Test sample

m1 <- xgb.train(data = dmodel, param, nrounds = 100, #Train the xgboost model for the training sample
                watchlist = list(model = dmodel, valid = dvalid)) #The output it will print- first column is going to be the training sample, second is going to be the test sample

m2 <- xgb.train(data = dtrain, param, nrounds = 110, #Train the xgboost model for the entire sample
                watchlist = list(train = dtrain)) #The output it will print- only one sample here, so it's just that.


#m2 is the optimized model in the end, which you use to predict off of.

#Doing the same process as above for the test set
X_test <- hashed.model.matrix(f, people_act_test, b)
dtest  <- xgb.DMatrix(X_test)



prediction <- predict(m2, dtest)
sub <- data.frame(activity_id = people_act_test$activity_id, outcome = prediction)
write.csv(sub, file = "outcome_test2.csv", row.names = F)