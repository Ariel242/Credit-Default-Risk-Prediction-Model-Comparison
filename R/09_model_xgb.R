######################################################################
# 09_model_xgb.R
# XGBoost (xgbTree): training, evaluation, AUC/AUPRC, Youden threshold
#
# Assumes: train.df / valid.df / test.df exist from previous scripts
######################################################################

library(xgboost)
library(caret)
library(pROC)
library(precrec)


# Local copies + target recoding (neg/pos)

train_xgb <- train.df
valid_xgb <- valid.df
test_xgb  <- test.df


train_xgb$loan_status <- factor(
  ifelse(train_xgb$loan_status %in% c("1","pos"), "pos", "neg"),
  levels = c("neg","pos")
)
valid_xgb$loan_status <- factor(
  ifelse(valid_xgb$loan_status %in% c("1","pos"), "pos", "neg"),
  levels = c("neg","pos")
)
test_xgb$loan_status <- factor(
  ifelse(test_xgb$loan_status %in% c("1","pos"), "pos", "neg"),
  levels = c("neg","pos")
)

table(train_xgb$loan_status)
table(valid_xgb$loan_status)


# trainControl + class weights

ctrl_xgb <- trainControl(
  method          = "repeatedcv",
  number          = 5,
  repeats         = 3,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

w_pos <- sum(train_xgb$loan_status == "neg") / sum(train_xgb$loan_status == "pos")
w_vec <- ifelse(train_xgb$loan_status == "pos", w_pos, 1)


# Train XGBoost (xgbTree) model

set.seed(4040)
mod_xgb <- caret::train(
  loan_status ~ . - row_id,
  data       = train_xgb,
  method     = "xgbTree",
  trControl  = ctrl_xgb,
  metric     = "ROC",
  weights    = w_vec,
  tuneLength = 3
)

print(mod_xgb)


# Predicted probabilities for all splits

p_train_xgb <- predict(mod_xgb, newdata = train_xgb, type = "prob")[, "pos"]
p_valid_xgb <- predict(mod_xgb, newdata = valid_xgb, type = "prob")[, "pos"]
p_test_xgb  <- predict(mod_xgb, newdata  = test_xgb,  type = "prob")[, "pos"]

stopifnot(length(p_train_xgb) == nrow(train_xgb))
stopifnot(length(p_valid_xgb) == nrow(valid_xgb))
stopifnot(length(p_test_xgb)  == nrow(test_xgb))


# ROC + AUC for train / valid / test

roc_train_xgb <- pROC::roc(
  response  = train_xgb$loan_status,
  predictor = p_train_xgb,
  levels    = c("neg","pos"),
  quiet     = TRUE
)

roc_valid_xgb <- pROC::roc(
  response  = valid_xgb$loan_status,
  predictor = p_valid_xgb,
  levels    = c("neg","pos"),
  quiet     = TRUE
)

roc_test_xgb <- pROC::roc(
  response  = test_xgb$loan_status,
  predictor = p_test_xgb,
  levels    = c("neg","pos"),
  quiet     = TRUE
)

auc_train_xgb <- as.numeric(pROC::auc(roc_train_xgb))
auc_valid_xgb <- as.numeric(pROC::auc(roc_valid_xgb))
auc_test_xgb  <- as.numeric(pROC::auc(roc_test_xgb))

auc_train_xgb
auc_valid_xgb
auc_test_xgb

# thr Youden on valid
thr_xgb <- as.numeric(pROC::coords(
  roc_valid_xgb,
  x           = "best",
  ret         = "threshold",
  best.method = "youden"
))
thr_xgb


# AUPRC for train / valid / test

ev_train_xgb <- precrec::evalmod(precrec::mmdata(p_train_xgb, train_xgb$loan_status))
ev_valid_xgb <- precrec::evalmod(precrec::mmdata(p_valid_xgb, valid_xgb$loan_status))
ev_test_xgb  <- precrec::evalmod(precrec::mmdata(p_test_xgb,  test_xgb$loan_status))

aucs_train <- precrec::auc(ev_train_xgb)
aucs_valid <- precrec::auc(ev_valid_xgb)
aucs_test  <- precrec::auc(ev_test_xgb)

auprc_train_xgb <- aucs_train[aucs_train$curvetypes == "PRC", "aucs"]
auprc_valid_xgb <- aucs_valid[aucs_valid$curvetypes == "PRC", "aucs"]
auprc_test_xgb  <- aucs_test[aucs_test$curvetypes == "PRC", "aucs"]

auprc_train_xgb
auprc_valid_xgb
auprc_test_xgb


# Class predictions (VALID) at Youden threshold + confusion matrix

pred_class_xgb_valid <- ifelse(p_valid_xgb >= thr_xgb, "pos", "neg")

cm_xgb_valid <- caret::confusionMatrix(
  factor(pred_class_xgb_valid, levels = c("neg","pos")),
  factor(valid_xgb$loan_status, levels = c("neg","pos")),
  positive = "pos"
)

cm_xgb_valid


# Summary metrics table (TRAIN / VALID / TEST)

metrics_xgb <- data.frame(
  model   = "XGBoost_xgbTree",
  dataset = c("train","valid","test"),
  AUC_ROC = c(auc_train_xgb, auc_valid_xgb, auc_test_xgb),
  AUPRC   = c(auprc_train_xgb, auprc_valid_xgb, auprc_test_xgb),
  threshold_youden = thr_xgb
)

metrics_xgb


# ROC VALID plot

plot(
  roc_valid_xgb,
  main        = "XGBoost - ROC (VALID)",
  legacy.axes = TRUE,
  print.auc   = TRUE
)
abline(a = 0, b = 1, lty = 2)

