######################################################################
# 08_model_rf.R
# Random Forest: training, evaluation, AUC/AUPRC, Youden threshold
#
# Assumes: train.df / valid.df / test.df exist from previous scripts
######################################################################

library(randomForest)
library(caret)
library(pROC)
library(precrec)


# Local copies + target recoding (neg/pos)

train_rf <- train.df
valid_rf <- valid.df
test_rf  <- test.df

train_rf$loan_status <- factor(ifelse(train_rf$loan_status == "1", "pos", "neg"),
                               levels = c("neg","pos"))
valid_rf$loan_status <- factor(ifelse(valid_rf$loan_status == "1", "pos", "neg"),
                               levels = c("neg","pos"))
test_rf$loan_status  <- factor(ifelse(test_rf$loan_status  == "1", "pos", "neg"),
                               levels = c("neg","pos"))

table(train_rf$loan_status)
table(valid_rf$loan_status)


# Training control (repeated CV, ROC metric)

ctrl_rf <- trainControl(
  method          = "repeatedcv",
  number          = 5,
  repeats         = 3,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  verboseIter     = TRUE
)


# Class weights for imbalance (pos vs neg)

w_pos <- sum(train_rf$loan_status == "neg") / sum(train_rf$loan_status == "pos")
w_vec <- ifelse(train_rf$loan_status == "pos", w_pos, 1)


# Train Random Forest model

set.seed(3030)
mod_rf <- caret::train(
  loan_status ~ . - row_id,
  data       = train_rf,
  method     = "rf",
  trControl  = ctrl_rf,
  metric     = "ROC",
  weights    = w_vec,
  tuneLength = 4
)

print(mod_rf)


# Predicted probabilities for all splits

p_train_rf <- predict(mod_rf, newdata = train_rf, type = "prob")[, "pos"]
p_valid_rf <- predict(mod_rf, newdata = valid_rf, type = "prob")[, "pos"]
p_test_rf  <- predict(mod_rf, newdata  = test_rf,  type = "prob")[, "pos"]

stopifnot(length(p_train_rf) == nrow(train_rf))
stopifnot(length(p_valid_rf) == nrow(valid_rf))
stopifnot(length(p_test_rf)  == nrow(test_rf))


# ROC + AUC for train / valid / test

roc_train_rf <- pROC::roc(
  response  = train_rf$loan_status,
  predictor = p_train_rf,
  levels    = c("neg","pos"),
  quiet     = TRUE
)

roc_valid_rf <- pROC::roc(
  response  = valid_rf$loan_status,
  predictor = p_valid_rf,
  levels    = c("neg","pos"),
  quiet     = TRUE
)

roc_test_rf <- pROC::roc(
  response  = test_rf$loan_status,
  predictor = p_test_rf,
  levels    = c("neg","pos"),
  quiet     = TRUE
)

auc_train_rf <- as.numeric(pROC::auc(roc_train_rf))
auc_valid_rf <- as.numeric(pROC::auc(roc_valid_rf))
auc_test_rf  <- as.numeric(pROC::auc(roc_test_rf))

auc_train_rf
auc_valid_rf
auc_test_rf


# AUPRC for train / valid / test

ev_train_rf <- precrec::evalmod(precrec::mmdata(p_train_rf, train_rf$loan_status))
ev_valid_rf <- precrec::evalmod(precrec::mmdata(p_valid_rf, valid_rf$loan_status))
ev_test_rf  <- precrec::evalmod(precrec::mmdata(p_test_rf,  test_rf$loan_status))

aucs_train <- precrec::auc(ev_train_rf)
aucs_valid <- precrec::auc(ev_valid_rf)
aucs_test  <- precrec::auc(ev_test_rf)

auprc_train_rf <- aucs_train[aucs_train$curvetypes == "PRC", "aucs"]
auprc_valid_rf <- aucs_valid[aucs_valid$curvetypes == "PRC", "aucs"]
auprc_test_rf  <- aucs_test[aucs_test$curvetypes == "PRC", "aucs"]

auprc_train_rf
auprc_valid_rf
auprc_test_rf


# Youden threshold on VALID + confusion matrix

thr_rf <- as.numeric(pROC::coords(
  roc_valid_rf,
  x           = "best",
  ret         = "threshold",
  best.method = "youden"
))
thr_rf

pred_class_rf_valid <- ifelse(p_valid_rf >= thr_rf, "pos", "neg")

cm_rf_valid <- caret::confusionMatrix(
  factor(pred_class_rf_valid, levels = c("neg","pos")),
  factor(valid_rf$loan_status, levels = c("neg","pos")),
  positive = "pos"
)

cm_rf_valid


# Summary metrics table (TRAIN / VALID / TEST)

metrics_rf <- data.frame(
  model   = "RandomForest",
  dataset = c("train","valid","test"),
  AUC_ROC = c(auc_train_rf, auc_valid_rf, auc_test_rf),
  AUPRC   = c(auprc_train_rf, auprc_valid_rf, auprc_test_rf),
  threshold_youden = thr_rf
)

metrics_rf


# ROC plot – VALID

plot(
  roc_valid_rf,
  main        = "Random Forest - ROC (VALID)",
  legacy.axes = TRUE,
  print.auc   = TRUE
)
abline(a = 0, b = 1, lty = 2)

