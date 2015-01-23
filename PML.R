library(caret)
library(dplyr)
#download.file('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv','training.csv','curl')
#download.file('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv','testing.csv','curl')
set.seed(1234)
train<-read.csv('training.csv')
test<-read.csv('testing.csv')
train[,-160]<-sapply(train[,-160],function(x){as.numeric(as.character(x))})
## shows the proportion of NA in each variable:
# colSums(sapply(train,is.na))/nrow(train) 
# see the proportion of NA in each variable


train<-train[,-c(1:6)]
#test<-test[,-c(1:6)]
allNA<-sapply(train,function(x)sum(is.na(x)))>0
#test<-test[,!allNA]
train<-train[,!allNA]

inTrain<-createDataPartition(y<-train$classe,p=0.75,list=FALSE)
training<-train[inTrain,]
testing<-train[-inTrain,]

modelFit<-train(classe~.,method='rf',data=training)
confusionMatrix(testing$classe,predict(modelFit,testing))


cr <- trainControl(
        method = "cv",
        number = 5,
        )
modelFit<-train(classe~.,method='rf',trControl=cr,data=training)


answerList<-data.frame(id=test$problem_id,predict=predict(modelFit,test))

pml_write_files = function(x){
                n = length(x)
                for(i in 1:n){
                        filename = paste0("problem_id_",i,".txt")
                        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
                }
        }
       
pml_write_files(answerList[,2])
