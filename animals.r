# Kaggle project - shelter animals #
# Stephanie Kirmer #
# July 2016 #


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


#Load datasets
sample_submit <- read.csv("U:/Programming/R/Kaggle/Animals/sample_submission.csv")
train2 <- read.csv("U:/Programming/R/Kaggle/Animals/train.csv", stringsAsFactors=F)
test2 <- read.csv("U:/Programming/R/Kaggle/Animals/test.csv", stringsAsFactors = F)

#Objective: predict each different outcome for each animal and combine predictions into a single file


train <- train2
# Data Manipulation ####

#Black animals tougher to adopt out
train$anyblack[grepl("Black", train$Color)] <- 1 
train$anyblack[is.na(train$anyblack)] <- 0

#How about white?
train$anywhite[grepl("White", train$Color)] <- 1 
train$anywhite[is.na(train$anywhite)] <- 0

#Are cat breeds anything? Some look special or different.
train$catoriental[grepl("Abyssinian", train$Breed) |grepl("Balinese", train$Breed) |grepl("Bengal", train$Breed) |
                         grepl("Burmese", train$Breed) |grepl("Bombay", train$Breed) |grepl("Javanese", train$Breed)
                  |grepl("Siamese", train$Breed) |grepl("Japanese", train$Breed)|grepl("Tonkinese", train$Breed)] <- 1
train$catoriental[is.na(train$catoriental)] <- 0

train$catlookspecial[grepl("Snowshoe", train$Breed) |grepl("Manx", train$Breed) |grepl("Bengal", train$Breed) |
                       grepl("Ocicat", train$Breed) |grepl("Munchkin", train$Breed) |grepl("Bobtail", train$Breed)] <- 1
train$catlookspecial[is.na(train$catlookspecial)] <- 0

train$catlonghair[grepl("Persian", train$Breed) |grepl("Longhair", train$Breed) |grepl("Himalayan", train$Breed) |
                         grepl("Maine Coon", train$Breed) |grepl("Angora", train$Breed) |grepl("Ragdoll", train$Breed)] <- 1
train$catlonghair[is.na(train$catlonghair)] <- 0

#Pit bulls and 'aggressive' breeds hard to adopt out
train$breedrisk[grepl("Pit Bull", train$Breed) | grepl("Staffordshire", train$Breed) | grepl("Rottweiler", train$Breed)
                | grepl("Doberman", train$Breed)] <- 1
train$breedrisk[is.na(train$breedrisk)] <- 0

#interest in small dog breeds?
train$breedsmol[grepl("Toy", train$Breed) | grepl("Chihuahua", train$Breed) | grepl("Pomeranian", train$Breed)
                | grepl("Pug", train$Breed)| grepl("Dachshund", train$Breed)| grepl("Miniature", train$Breed)] <- 1
train$breedsmol[is.na(train$breedsmol)] <- 0

#Super popular breeds in USA (Just checked Wikipedia for AKC list)
train$breedpopular[grepl("Labrador", train$Breed) | grepl("Yorkshire", train$Breed) | grepl("German Shepherd", train$Breed)
                | grepl("Golden Retriever", train$Breed)| grepl("Beagle", train$Breed)| grepl("Dachshund", train$Breed)
                | grepl("Boxer", train$Breed)| grepl("Poodle", train$Breed)| grepl("Shih", train$Breed)| grepl("Schnauzer", train$Breed)] <- 1
train$breedpopular[is.na(train$breedpopular)] <- 0


#Young animals might have different results
train$under1yr[grepl("months", train$AgeuponOutcome) | grepl("days", train$AgeuponOutcome)] <- 1
train$under1yr[is.na(train$under1yr)] <- 0

#Senior animals also different?
age_yrs <- str_split_fixed(str_trim(train$AgeuponOutcome), " ", 2)
train <- cbind(train, age_yrs)

train <- as.data.table(train, keep.rownames = F)
train[train$"2" == "years", age_in_years := train$"1"] 
train[train$"2" == "year", age_in_years := train$"1"] 
train[is.na(age_in_years), age_in_years := 0] 
train <- as.data.frame(train)

train$age_in_years <- as.numeric(train$age_in_years)

train$senior[train$age_in_years > 9] <- 1
train$senior[is.na(train$senior)] <- 0

#Gender
train$male[grepl("Male", train$SexuponOutcome)] <- 1 
train$male[is.na(train$male)] <- 0

#Fixed Status
train$fixed[grepl("Neutered", train$SexuponOutcome) | grepl("Spayed", train$SexuponOutcome)] <- 1 
train$fixed[is.na(train$fixed)] <- 0


#Time of year 
str(train$DateTime)
train$DateTime_2 <- as.Date(train$DateTime)
train$month <- month(train$DateTime_2)
train$quarter <- quarter(train$DateTime_2)

train$spring[train$month %in% c(3,4,5,6)] <- 1
train$spring[is.na(train$spring)] <- 0

train$xmas[train$month %in% c(11,12,1)] <- 1
train$xmas[is.na(train$xmas)] <- 0

#Cat vs dog
train$cat[train$AnimalType=="Cat"] <- 1
train$cat[is.na(train$cat)] <- 0

#Has a name?
train$named[train$Name!=""] <- 1
train$named[is.na(train$named)] <- 0


#Make our outcome variables
outcomes <- dummy(train$OutcomeType)
train <- cbind(train, outcomes)
head(train)
names(train)



# Partition to see how well things work for different outcome options ####
outcome <- train$OutcomeTypeAdoption

partition <- createDataPartition(y=outcome,
                                 p=.5,
                                 list=F)
training <- train[partition,]
testing <- train[-partition,]


#Experiments with the GLM  ####

# Predict using the test set -  GLM this time
glm_model_15 <- glm(factor(OutcomeTypeDied) ~ 
                     #anyblack+
                     #anywhite+
                     #catlookspecial+
                     catlonghair+
                     #catoriental+
                     #cat+
                     #breedrisk+
                     #breedsmol+
                     #breedpopular+
                     named+
                     #under1yr+
                     #senior+
                     #male+
                     fixed+
                     #month+
                     #spring+
                     #xmas+
                     cat:named+
                     #cat:xmas+
                     #anyblack:cat+
                     #cat:under1yr+
                     cat:senior#+
                   #male:fixed
                   ,
                   data = training, family=binomial)


summary(glm_model_15)
prediction <- predict(glm_model_15, testing, type="response")
model_output <- cbind(testing, prediction)

#Test with logloss ####
LogLossBinary <- function(actual, predicted, eps = 1e-15) {
     predicted = pmin(pmax(predicted, eps), 1-eps)
     - (sum(actual * log(predicted) + (1 - actual) * log(1 - predicted))) / length(actual)
   }

LogLossBinary(model_output$OutcomeTypeDied, model_output$prediction)





# Test with ROC  ####
everything_roc <- roc(glm_model_15$y, glm_model_15$fitted.values)
plot(everything_roc)



test <- test2
# Data Manipulation test set ####

#Black animals tougher to adopt out
test$anyblack[grepl("Black", test$Color)] <- 1 
test$anyblack[is.na(test$anyblack)] <- 0

#How about white?
test$anywhite[grepl("White", test$Color)] <- 1 
test$anywhite[is.na(test$anywhite)] <- 0

#Are cat breeds anything? Some look special or different.
test$catoriental[grepl("Abyssinian", test$Breed) |grepl("Balinese", test$Breed) |grepl("Bengal", test$Breed) |
                    grepl("Burmese", test$Breed) |grepl("Bombay", test$Breed) |grepl("Javanese", test$Breed)
                  |grepl("Siamese", test$Breed) |grepl("Japanese", test$Breed)|grepl("Tonkinese", test$Breed)] <- 1
test$catoriental[is.na(test$catoriental)] <- 0

test$catlookspecial[grepl("Snowshoe", test$Breed) |grepl("Manx", test$Breed) |grepl("Bengal", test$Breed) |
                       grepl("Ocicat", test$Breed) |grepl("Munchkin", test$Breed) |grepl("Bobtail", test$Breed)] <- 1
test$catlookspecial[is.na(test$catlookspecial)] <- 0

test$catlonghair[grepl("Persian", test$Breed) |grepl("Longhair", test$Breed) |grepl("Himalayan", test$Breed) |
                    grepl("Maine Coon", test$Breed) |grepl("Angora", test$Breed) |grepl("Ragdoll", test$Breed)] <- 1
test$catlonghair[is.na(test$catlonghair)] <- 0

#Pit bulls and 'aggressive' breeds hard to adopt out
test$breedrisk[grepl("Pit Bull", test$Breed) | grepl("Staffordshire", test$Breed) | grepl("Rottweiler", test$Breed)
                | grepl("Doberman", test$Breed)] <- 1
test$breedrisk[is.na(test$breedrisk)] <- 0

#interest in small dog breeds?
test$breedsmol[grepl("Toy", test$Breed) | grepl("Chihuahua", test$Breed) | grepl("Pomeranian", test$Breed)
                | grepl("Pug", test$Breed)| grepl("Dachshund", test$Breed)| grepl("Miniature", test$Breed)] <- 1
test$breedsmol[is.na(test$breedsmol)] <- 0

#Super popular breeds in USA (Just checked Wikipedia for AKC list)
test$breedpopular[grepl("Labrador", test$Breed) | grepl("Yorkshire", test$Breed) | grepl("German Shepherd", test$Breed)
                   | grepl("Golden Retriever", test$Breed)| grepl("Beagle", test$Breed)| grepl("Dachshund", test$Breed)
                   | grepl("Boxer", test$Breed)| grepl("Poodle", test$Breed)| grepl("Shih", test$Breed)| grepl("Schnauzer", test$Breed)] <- 1
test$breedpopular[is.na(test$breedpopular)] <- 0


#Young animals might have different results
test$under1yr[grepl("months", test$AgeuponOutcome) | grepl("days", test$AgeuponOutcome)] <- 1
test$under1yr[is.na(test$under1yr)] <- 0

#Senior animals also different?
age_yrs <- str_split_fixed(str_trim(test$AgeuponOutcome), " ", 2)
test <- cbind(test, age_yrs)

test <- as.data.table(test, keep.rownames = F)
test[test$"2" == "years", age_in_years := test$"1"] 
test[test$"2" == "year", age_in_years := test$"1"] 
test[is.na(age_in_years), age_in_years := 0] 
test <- as.data.frame(test)

test$age_in_years <- as.numeric(test$age_in_years)

test$senior[test$age_in_years > 9] <- 1
test$senior[is.na(test$senior)] <- 0

#Gender
test$male[grepl("Male", test$SexuponOutcome)] <- 1 
test$male[is.na(test$male)] <- 0

#Fixed Status
test$fixed[grepl("Neutered", test$SexuponOutcome) | grepl("Spayed", test$SexuponOutcome)] <- 1 
test$fixed[is.na(test$fixed)] <- 0


#Time of year 
str(test$DateTime)
test$DateTime_2 <- as.Date(test$DateTime)
test$month <- month(test$DateTime_2)
test$quarter <- quarter(test$DateTime_2)

test$spring[test$month %in% c(3,4,5,6)] <- 1
test$spring[is.na(test$spring)] <- 0

test$xmas[test$month %in% c(11,12,1)] <- 1
test$xmas[is.na(test$xmas)] <- 0

#Cat vs dog
test$cat[test$AnimalType=="Cat"] <- 1
test$cat[is.na(test$cat)] <- 0

#Has a name?
test$named[test$Name!=""] <- 1
test$named[is.na(test$named)] <- 0


#Make our outcome variables
outcomes <- dummy(test$OutcomeType)
test <- cbind(test, outcomes)
head(test)
names(test)


outcome <- test$OutcomeTypeAdoption



# Predict with GLM ####
glm_model_test <- glm(factor(OutcomeTypeDied) ~ 
                        #anyblack+
                        #anywhite+
                        #catlookspecial+
                        catlonghair+
                        #catoriental+
                        #cat+
                        #breedrisk+
                        #breedsmol+
                        #breedpopular+
                        named+
                        #under1yr+
                        #senior+
                        #male+
                        fixed+
                        #month+
                        #spring+
                        #xmas+
                        cat:named+
                        #cat:xmas+
                        #anyblack:cat+
                        #cat:under1yr+
                        cat:senior#+
                      #male:fixed
                      , data = train, family=binomial)

prediction_glm <- predict(glm_model_test, test, type="response")
summary(glm_model_test)

#model_output_glm<- cbind(test, prediction_glm) #First prediction ONLY

model_output_glm <- cbind(model_output_glm, prediction_glm) #Use after the first prediction

model_output_glm$predict_OutcomeTypeDied <- model_output_glm$prediction_glm
model_output_glm$prediction_glm <- NULL
head(model_output_glm)


prediction_dataset <- cbind("ID" = model_output_glm$ID, "Adoption" = model_output_glm$predict_OutcomeTypeAdoption, 
                                           "Died" = model_output_glm$predict_OutcomeTypeDied,
                                           "Euthanasia" = model_output_glm$predict_OutcomeTypeEuthanasia, 
                                           "Return_to_owner" = model_output_glm$predict_OutcomeTypeReturn_to_owner,
                                           "Transfer" = model_output_glm$predict_OutcomeTypeTransfer)

write.csv(prediction_dataset, "U:/Programming/R/Kaggle/Animals/prediction3.csv", row.names = F)
