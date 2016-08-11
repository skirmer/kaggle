# Continuing work on the Kaggle Titanic Project
# Stephanie Kirmer
# July 2016

#Libraries
require(randomForest)
require(caret)
require(dplyr)
require(data.table)
require(ggplot2)
library(pROC)
library(stringr)

#Load datasets ####
train <- read.csv("U:/Programming/R/Kaggle/Titanic/train.csv", stringsAsFactors=F)
test <- read.csv("U:/Programming/R/Kaggle/Titanic/test.csv", stringsAsFactors = F)


# Format the files ####
summary(train)


#Names
lastname_firstname <- str_split_fixed(str_trim(train$Name), ",", 2)
firstname_title <- str_split_fixed(str_trim(lastname_firstname[,2]), "[.]", 2)
head(lastname_firstname)
head(firstname_title)

train <- cbind(train, "surname" = lastname_firstname[,1], "title" = firstname_title[,1], "firstname" = firstname_title[,2])

#titles
table(train$title)
train$elite_title[train$title %in% c("Capt", "Col", "Don", "Dr", "Jonkheer", "Lady", "Major", "Master", "Rev", "Sir", "the Countess")] <- 1
train$elite_title[is.na(train$elite_title)] <- 0

train$military[train$title %in% c("Capt", "col", "Major")] <- 1
train$military[is.na(train$military)] <- 0

train$title_consolidated <- as.character(train$title)
train$title_consolidated[train$title == "Mlle"] <- "Miss"
train$title_consolidated[train$title == "Ms"] <- "Miss"
train$title_consolidated[train$title == "Mme"] <- "Mrs"
train$title_consolidated[train$elite_title==1] <- "Rare"
table(train$title_consolidated)

train$title_cons_tree[train$title_consolidated=="Miss"] <- 1
train$title_cons_tree[train$title_consolidated=="Mrs"] <- 2
train$title_cons_tree[train$title_consolidated=="Mr"] <- 3
train$title_cons_tree[train$title_consolidated=="Rare"] <- 4

#language - if you speak English, more likely to survive?
train$unlikely_eng[train$title %in% c("Don", "Jonkheer", "Mlle", "Mme")] <- 1
train$unlikely_eng[is.na(train$unlikely_eng)]<-0

#Sex - female more likely to survive
train$sexn[train$Sex=="male"] <- 1
train$sexn[train$Sex=="female"] <- 0

#Single woman- is this important specifically?
train$singlefem[train$title_cons_tree==1] <- 1
train$singlefem[is.na(train$singlefem)] <- 0


#Family size
train$famsize <- train$SibSp+train$Parch
train$zerofam[train$famsize==0] <- 1
train$zerofam[is.na(train$zerofam)] <- 0

train$bigfam[train$famsize>6] <- 1
train$bigfam[is.na(train$bigfam)] <- 0


#Age - missing vals
train$missing_age[is.na(train$Age)] <- 1
train$missing_age[is.na(train$missing_age)] <- 0

agemissing <- filter(train, missing_age==1)

#Criteria to impute the age:
am_sibsp_med <- median(agemissing$SibSp)
am_parch_med <- median(agemissing$Parch)
am_pclass_med <- median(agemissing$Pclass)

age_impute <- filter(train, SibSp==am_sibsp_med, Parch==am_parch_med, Pclass==am_pclass_med, !is.na(Age))

train$Age[is.na(train$Age)] <- mean(age_impute$Age, na.rm=T)

#Solo man between 18 and 50- is this important specifically?
train$singlemale[train$sexn==1 & train$famsize == 0 & train$Age > 18 & train$Age < 50] <- 1
train$singlemale[is.na(train$singlemale)] <- 0

#Age - find children and elderly
train$child[train$Age < 17] <- 1
train$child[is.na(train$child)] <- 0

train$elder[train$Age > 60] <- 1
train$elder[is.na(train$elder)] <- 0


#Port of embarcation
train$port[train$Embarked=="C"] <- 1
train$port[train$Embarked=="Q"] <- 2
train$port[train$Embarked=="S"] <- 3
train$port[is.na(train$port)] <- 0

#Cabin
train$cabintype[grepl("A", train$Cabin)] <- 1
train$cabintype[grepl("B", train$Cabin)] <- 2
train$cabintype[grepl("C", train$Cabin)] <- 3
train$cabintype[grepl("D", train$Cabin)] <- 4
train$cabintype[grepl("E", train$Cabin)] <- 5
train$cabintype[grepl("F", train$Cabin)] <- 6
train$cabintype[grepl("G", train$Cabin)] <- 7
train$cabintype[is.na(train$cabintype)] <- 0

#Give a boost to a VERY high fare price
train$highfare[train$Fare >= 75] <- 1
train$highfare[is.na(train$highfare)] <- 0

#All alone child penalty
train$solochild[train$famsize == 0 & train$child ==1] <- 1
train$solochild[is.na(train$solochild)] <- 0



# Partition to see how well things work ####
partition <- createDataPartition(y=train$Survived,
                                 p=.5,
                                 list=F)
training <- train[partition,]
testing <- train[-partition,]




# ======================================================================= #

# ================= Model ================== #####
set.seed(150)

model_1 <- randomForest(factor(Survived) ~  #Class related
                          #Class related
                          port + 
                          cabintype +
                          highfare + 
                          Fare +
                          Pclass + 
                          
                          #Resilience
                          Age + 
                          solochild + 
                          #missing_age+
                          
                          #chivalry
                          child + 
                          sexn + 
                          military +
                          singlefem + 
                          
                          #dragdowns
                          famsize + 
                          bigfam + 
                          zerofam + 
                          SibSp + 
                          Parch + 
                          singlemale +  
                          
                          title_cons_tree, 
                                data = training)


# The section below that visualizes the importance is borrowed with thanks from Megan L. Ridsal. kaggle.com/mrisdal.
# Get importance
importance    <- importance(model_1)
varImportance <- data.frame(Variables = row.names(importance),
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

# Create a rank variable based on importance
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

# Use ggplot2 to visualize the relative importance of variables
ggplot(rankImportance, aes(x = reorder(Variables, Importance),
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') +
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip()


# Predict using the test set
prediction <- predict(model_1, testing)

table(prediction)
table(testing$Survived)
table(training$Survived)

model_output <- cbind(testing, prediction)
table(model_output$prediction, model_output$Survived, dnn=c("predict", "actual"))



# Take a look at the people who are being mis-labeled
errorset <- filter(model_output, prediction==1, Survived==0)
head(errorset)
table(errorset$singlemale, errorset$prediction, dnn=c("var", "prediction"))
summary(errorset)


# ======================================================================= #

# Predict using the test set -  GLM this time
glm_model_1 <- glm(factor(Survived) ~ 
                     #Class related
                     port + 
                     cabintype:sexn +
                     #cabintype+
                     #highfare
                     #Fare:sexn + 
                     #Fare +
                     Pclass + 
                     Pclass:sexn +
                     
                     #Resilience
                     #Age +
                     Age:sexn +
                     sexn:solochild + 
                     #missing_age+
                     
                     #chivalry
                     child + 
                     sexn + 
                     #military +
                     #elite_title +
                     #singlefem + 
                     
                     #dragdowns
                     famsize + 
                     zerofam:sexn+ 
                     sexn:SibSp + 
                     #Parch + 
                     #singlemale +  
                     
                     title_cons_tree,
                   data = training, family=binomial)

names(training)
prediction <- predict(glm_model_1, testing)
summary(glm_model_1)
#summary(prediction)
# table(testing$Survived)
# table(training$Survived)

model_output <- cbind(testing, prediction)
#table(model_output$prediction, model_output$Survived, dnn=c("predict", "actual"))

model_output$prediction_summary[model_output$prediction > 0] <- 1
model_output$prediction_summary[model_output$prediction < 0] <- 0

table(model_output$prediction_summary, model_output$Survived, dnn=c("predict", "actual"))


head(model_output,15)

# ======================================================================= #

# ------------------------------------- ROC - testing the quality of the models ------------------------------------ ####
#In plot, specificity = rate of false pos and sensitivity = rate of false neg

#do it with the glm
everything_roc <- roc(glm_model_1$y, glm_model_1$fitted.values)
plot(everything_roc)


# Do it with the random forest
rfModel <- as.vector(predict(model_1, testing, type="prob")[,1])
everything_roc <- roc(testing$Survived,rfModel)
plot(everything_roc)


# ------------------------------------------------------- #
# Prepare output for submission #####
# ------------------------------------------------------- #

# Format the files ####
summary(test)


#Names
lastname_firstname <- str_split_fixed(str_trim(test$Name), ",", 2)
firstname_title <- str_split_fixed(str_trim(lastname_firstname[,2]), "[.]", 2)
head(lastname_firstname)
head(firstname_title)

test <- cbind(test, "surname" = lastname_firstname[,1], "title" = firstname_title[,1], "firstname" = firstname_title[,2])

#titles
table(test$title)
test$elite_title[test$title %in% c("Capt", "Col", "Don","Dona", "Dr", "Jonkheer", "Lady", "Major", "Master", "Rev", "Sir", "the Countess")] <- 1
test$elite_title[is.na(test$elite_title)] <- 0

test$military[test$title %in% c("Capt", "col", "Major")] <- 1
test$military[is.na(test$military)] <- 0

test$title_consolidated <- as.character(test$title)
test$title_consolidated[test$title == "Mlle"] <- "Miss"
test$title_consolidated[test$title == "Ms"] <- "Miss"
test$title_consolidated[test$title == "Mme"] <- "Mrs"
test$title_consolidated[test$elite_title==1] <- "Rare"
table(test$title_consolidated)

test$title_cons_tree[test$title_consolidated=="Miss"] <- 1
test$title_cons_tree[test$title_consolidated=="Mrs"] <- 2
test$title_cons_tree[test$title_consolidated=="Mr"] <- 3
test$title_cons_tree[test$title_consolidated=="Rare"] <- 4



#language - if you speak English, more likely to survive?
test$unlikely_eng[test$title %in% c("Don", "Jonkheer", "Mlle", "Mme")] <- 1
test$unlikely_eng[is.na(test$unlikely_eng)]<-0

#Sex - female more likely to survive
test$sexn[test$Sex=="male"] <- 1
test$sexn[test$Sex=="female"] <- 0

#Single woman- is this important specifically?
test$singlefem[test$title_cons_tree==1] <- 1
test$singlefem[is.na(test$singlefem)] <- 0


#Family size
test$famsize <- test$SibSp+test$Parch
test$zerofam[test$famsize==0] <- 1
test$zerofam[is.na(test$zerofam)] <- 0

test$bigfam[test$famsize>6] <- 1
test$bigfam[is.na(test$bigfam)] <- 0


#Age - missing vals
test$missing_age[is.na(test$Age)] <- 1
test$missing_age[is.na(test$missing_age)] <- 0

agemissing <- filter(test, missing_age==1)

#Criteria to impute the age:
am_sibsp_med <- median(agemissing$SibSp)
am_parch_med <- median(agemissing$Parch)
am_pclass_med <- median(agemissing$Pclass)

age_impute <- filter(test, SibSp==am_sibsp_med, Parch==am_parch_med, Pclass==am_pclass_med, !is.na(Age))

test$Age[is.na(test$Age)] <- mean(age_impute$Age, na.rm=T)



#Solo man between 18 and 50- is this important specifically?
test$singlemale[test$sexn==1 & test$famsize == 0 & test$Age > 18 & test$Age < 50] <- 1
test$singlemale[is.na(test$singlemale)] <- 0



#Age - find children and elderly
test$child[test$Age < 17] <- 1
test$child[is.na(test$child)] <- 0

test$elder[test$Age > 60] <- 1
test$elder[is.na(test$elder)] <- 0


#Port of embarcation
test$port[test$Embarked=="C"] <- 1
test$port[test$Embarked=="Q"] <- 2
test$port[test$Embarked=="S"] <- 3
test$port[is.na(test$port)] <- 0

#Cabin
test$cabintype[grepl("A", test$Cabin)] <- 1
test$cabintype[grepl("B", test$Cabin)] <- 2
test$cabintype[grepl("C", test$Cabin)] <- 3
test$cabintype[grepl("D", test$Cabin)] <- 4
test$cabintype[grepl("E", test$Cabin)] <- 5
test$cabintype[grepl("F", test$Cabin)] <- 6
test$cabintype[grepl("G", test$Cabin)] <- 7
test$cabintype[is.na(test$cabintype)] <- 0

#Give a boost to a VERY high fare price
test$highfare[test$Fare >= 75] <- 1
test$highfare[is.na(test$highfare)] <- 0

test$Fare[is.na(test$Fare)]<-0

#All alone child penalty
test$solochild[test$famsize == 0 & test$child ==1] <- 1
test$solochild[is.na(test$solochild)] <- 0


# ======================================================================= #

# Predict with GLM ####

glm_model_2 <- glm(factor(Survived) ~ 
                     #Class related
                     port + 
                     cabintype:sexn +
                     #cabintype+
                     #highfare
                     #Fare:sexn + 
                     #Fare +
                     Pclass + 
                     Pclass:sexn +

                     #Resilience
                     #Age +
                     Age:sexn +
                     sexn:solochild + 
                     #missing_age+
                     
                     #chivalry
                     child + 
                     sexn + 
                     #military +
                     #elite_title +
                     #singlefem + 
                     
                     #dragdowns
                     famsize + 
                     zerofam:sexn+ 
                     sexn:SibSp + 
                     #Parch + 
                     #singlemale +  
                     
                     title_cons_tree
                   , data = train, family=binomial)

prediction_glm <- predict(glm_model_2, test)
summary(glm_model_2)

names(train)
model_output_glm<- cbind(test, prediction_glm)
#table(model_output_glm$prediction_glm, model_output_glm$Survived, dnn=c("predict", "actual"))

model_output_glm$prediction_glm_summary[model_output_glm$prediction_glm > 0] <- 1
model_output_glm$prediction_glm_summary[model_output_glm$prediction_glm < 0] <- 0

table(model_output_glm$prediction_glm_summary)



# ======================================================================= #

# Predict with RandomForest ####

model_2 <- randomForest(factor(Survived) ~  #Class related
                          port + 
                          cabintype:sexn +
                          #cabintype+
                          #highfare
                          #Fare:sexn + 
                          #Fare +
                          Pclass + 
                          Pclass:sexn +
                          
                          #Resilience
                          #Age +
                          Age:sexn +
                          sexn:solochild + 
                          #missing_age+
                          
                          #chivalry
                          child + 
                          sexn + 
                          #military +
                          #elite_title +
                          #singlefem + 
                          
                          #dragdowns
                          famsize + 
                          zerofam:sexn+ 
                          sexn:SibSp + 
                          #Parch + 
                          #singlemale +  
                          
                          title_cons_tree, 
                        data = train)

# Predict using the test set
prediction_rf <- predict(model_2, test)
prediction_rf <- as.data.frame(prediction_rf)

model_output_rf <- cbind(test, prediction_rf)

names(model_output_rf)


#Mixed models - look at the disagreements

model_combined_rf_glm <- merge(model_output_rf, model_output_glm[,c("PassengerId", "prediction_glm_summary")], by="PassengerId")
model_combined_rf_glm$prediction_rf <- as.numeric(as.character(model_combined_rf_glm$prediction_rf))

model_combined_rf_glm$conflict[model_combined_rf_glm$prediction_rf != model_combined_rf_glm$prediction_glm_summary] <- 1
model_combined_rf_glm$conflict[is.na(model_combined_rf_glm$conflict)] <- 0

#If there's a disagreement, set decision rules about the choice to take:
model_combined_rf_glm$Survived <- model_combined_rf_glm$prediction_rf


# REALLY high price goes with GLM
model_combined_rf_glm$Survived[model_combined_rf_glm$conflict==1 & model_combined_rf_glm$Fare > 200] <- model_combined_rf_glm$prediction_glm_summary

# Fancy titles goes with GLM
model_combined_rf_glm$Survived[model_combined_rf_glm$conflict==1 & model_combined_rf_glm$title_consolidated=="Rare"] <- model_combined_rf_glm$prediction_glm_summary



table(model_combined_rf_glm$prediction_rf, model_combined_rf_glm$prediction_glm_summary, dnn=c("rf", "glm"))
table(model_combined_rf_glm$Survived)

errorset_test <- filter(model_combined_rf_glm, prediction_rf != prediction_glm_summary)
head(errorset_test[errorset_test$sexn==1,], 20)



#Straight GLM prediction - better than the RF
glm_model <- model_combined_rf_glm[,c("PassengerId", "prediction_glm_summary")]
glm_model <- dplyr::rename(glm_model, "Survived"=prediction_glm_summary)
write.csv(glm_model, "U:/Programming/R/Kaggle/Titanic/glm_mod2.csv", row.names = F)


#Straight RF prediction
rf_model <- model_combined_rf_glm[,c("PassengerId", "prediction_rf")]
rf_model <- dplyr::rename(rf_model, "Survived"=prediction_rf)
write.csv(rf_model, "U:/Programming/R/Kaggle/Titanic/rf_mod2.csv", row.names = F)
