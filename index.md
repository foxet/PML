# Human Activity Recognition
foxet  
2015年1月22日  

___note:  I'm not a native english speaker, and I'm learning english, machine learning and markdown. So, I'm sorry that my report is going to torchure you. :)__   
        
-----------------------------------------------------------------------------------------------

### Introduction
This project is to built an machine learning algorithm to discriminating between different human activities. The experimental data is from accelerometers on the belt, forearm, arm, and dumbell of 6 participants who were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here:  
http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).    

#### Data
 * The training data for this project are available here:  
 https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv  

 * The test data are available here:  
 https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv  
 
 * The data for this project come from this source:   
 http://groupware.les.inf.puc-rio.br/har.   


```r
library(caret)
library(dplyr)
train<-read.csv('training.csv')
test<-read.csv('testing.csv')
set.seed(1234)
```

```r
train[,-160]<-sapply(train[,-160],function(x){as.numeric(as.character(x))})
```


---------------------------------------------------------------------------------------------

#### Prepocess 
* Remove unrelated variables: "X" "user_name", "raw_timestamp_part_1","raw_timestamp_part_2"     "cvtd_timestamp","new_window":


```r
train<-train[,-c(1:6)]
```

* Remove 'NA' variables: Some variables contain a large proportion of NA(>95%). Imputation can't do well in this case. In addition, all these variables are secondary measurement such as standard Deviation, mean value and kurtosis. their information can be reflected by the other predictors.


```r
allNA<-sapply(train,function(x)sum(is.na(x)))>0
train<-train[,!allNA]
```


* Data Splitting: separate data set into two sets, called the training set and the testing set. The function approximator fits a function and tuning the model parameters using the training set only. Then the fitted model is asked to predict the output values for the data in the testing set. 



```r
inTrain<-createDataPartition(y<-train$classe,p=0.75,list=FALSE)
training<-train[inTrain,]
testing<-train[-inTrain,]
```

---------------------------------------------------------------------------------------------

#### Model Tuning


```r
modelFit<-train(classe~.,method='rf',data=training)
```

* Using randomforest provide by caret package to fit the model. Train function use 5-fold cross-validation to tune the parameter 'mtry' of candidate models. Then it returen the best model with best parameter.


```r
cr <- trainControl(
        method = "cv",
        number = 5,
        )
modelFit<-train(classe~.,method='rf',trControl=cr,data=training)
```


```r
modelFit
```

```
## Random Forest 
## 
## 14718 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## 
## Summary of sample sizes: 11774, 11775, 11776, 11773, 11774 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD   Kappa SD    
##    2    0.9942926  0.9927803  0.0010023193  0.0012675393
##   27    0.9970784  0.9963045  0.0006175835  0.0007811626
##   53    0.9937488  0.9920928  0.0015319991  0.0019375435
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```


---------------------------------------------------------------------------------------------

#### In-Sample Error Evaluation  

```r
print(modelFit$results[2,])
```

```
##   mtry  Accuracy     Kappa   AccuracySD      KappaSD
## 2   27 0.9970784 0.9963045 0.0006175835 0.0007811626
```

```r
print(modelFit$finalModel$confusion)
```

```
##      A    B    C    D    E  class.error
## A 4184    0    0    0    1 0.0002389486
## B    8 2837    3    0    0 0.0038623596
## C    0    4 2563    0    0 0.0015582392
## D    0    0    9 2402    1 0.0041459370
## E    0    0    0    5 2701 0.0018477458
```
  
  
  The model's performance on training set is great. However, because of overfitting, accuracy on the training set is usually optimistic, a better estimate should come from an independent test set. 
 

---------------------------------------------------------------------------------------------
  
#### Out-Of-Sample Error evaluation    


```r
confusionMatrix(testing$classe,predict(modelFit,testing))
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
## 
## Attaching package: 'randomForest'
## 
## The following object is masked from 'package:dplyr':
## 
##     combine
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    0    0    0    0
##          B    0  949    0    0    0
##          C    0    0  855    0    0
##          D    0    0    0  804    0
##          E    0    0    0    0  901
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9992, 1)
##     No Information Rate : 0.2845     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

Out-Of-Sample Error is the error rate from a new data set. I used the independent testing set to evaluate this error. This is called 'holdout method', the simplest cross validation method. The advantage of this method is that it is usually preferable to the residual method and takes no longer to compute. But I also know this method can increase variance.   
The prediction on the testing set is surprising, even better than the training set, that's unexpected.  


---------------------------------------------------------------------------------------------
#### END
