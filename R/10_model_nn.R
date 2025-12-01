######################################################################
# 10_model_nn.R
# Neural network (nnet): training, evaluation, AUC/AUPRC, Youden threshold
#
# Assumes: train.df / valid.df / test.df exist from previous scripts
######################################################################

library(nnet)
library(caret)
library(pROC)
library(precrec)


# Local copies + target recoding (neg/pos)

train_nn <- train.df
valid_nn <- valid.df
test_nn  <- test.df

train_nn$loan_status <- factor(
  ifelse(train_nn$loan_status %in% c("1","pos"), "pos", "neg"),
  levels = c("neg","pos")
)
valid_nn$loan_status <- factor(
  ifelse(valid_nn$loan_status %in% c("1","pos"), "pos", "neg"),
  levels = c("neg","pos")
)
test_nn$loan_status <- factor(
  ifelse(test_nn$loan_status %in% c("1","pos"), "pos", "neg"),
  levels = c("neg","pos")
)

table(train_nn$loan_status)
table(valid_nn$loan_status)


#Training control (repeated CV)

ctrl_nn <- trainControl(
  method          = "repeatedcv",
  number          = 3,
  repeats         = 3,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)


# Class weights for imbalance

w_pos <- sum(train_nn$loan_status == "neg") / sum(train_nn$loan_status == "pos")
w_vec <- ifelse(train_nn$loan_status == "pos", w_pos, 1)


# Tuning grids

nn_grid_1 <- expand.grid(
  size  = c(3, 6),
  decay = 0
)

nn_grid_2 <- expand.grid(
  size  = c(5, 10),
  decay = c(0.01, 0.1)
)

nn_grid_3 <- expand.grid(
  size  = c(7, 12),
  decay = c(0, 0.01, 0.1)
)

nn_grid_4 <- expand.grid(
  size  = c(10, 15),
  decay = c(0.001, 0.005, 0.01)
)

nn_grid_5 <- expand.grid(
  size  = c(15, 20),
  decay = c(0, 0.001, 0.01, 0.1)
)

nn_grid_6 <- expand.grid(
  size  = c(20, 25),
  decay = c(0.0001, 0.001, 0.01, 0.1)
)

nn_grid_7 <- expand.grid(
  size  = c(25, 30),
  decay = c(0, 0.001, 0.01, 0.1)
)


# Train neural network model

set.seed(5050)
mod_nn <- caret::train(
  loan_status ~ . - row_id,
  data       = train_nn,
  method     = "nnet",
  trControl  = ctrl_nn,
  metric     = "ROC",
  weights    = w_vec,
  tuneGrid   = nn_grid_5,
  trace      = TRUE,  # מציג פרטי אימון
  maxit      = 120
)

print(mod_nn)


# Predicted probabilities for all splits

p_train_nn <- predict(mod_nn, newdata = train_nn, type = "prob")[, "pos"]
p_valid_nn <- predict(mod_nn, newdata = valid_nn, type = "prob")[, "pos"]
p_test_nn  <- predict(mod_nn, newdata = test_nn,  type = "prob")[, "pos"]

stopifnot(length(p_train_nn) == nrow(train_nn))
stopifnot(length(p_valid_nn) == nrow(valid_nn))
stopifnot(length(p_test_nn)  == nrow(test_nn))


# ROC + AUC for train / valid / test

roc_train_nn <- pROC::roc(
  response  = train_nn$loan_status,
  predictor = p_train_nn,
  levels    = c("neg","pos"),
  quiet     = TRUE
)

roc_valid_nn <- pROC::roc(
  response  = valid_nn$loan_status,
  predictor = p_valid_nn,
  levels    = c("neg","pos"),
  quiet     = TRUE
)

roc_test_nn <- pROC::roc(
  response  = test_nn$loan_status,
  predictor = p_test_nn,
  levels    = c("neg","pos"),
  quiet     = TRUE
)

auc_train_nn <- as.numeric(pROC::auc(roc_train_nn))
auc_valid_nn <- as.numeric(pROC::auc(roc_valid_nn))
auc_test_nn  <- as.numeric(pROC::auc(roc_test_nn))

auc_train_nn
auc_valid_nn
auc_test_nn

# thr Youden on valid
thr_nn <- as.numeric(pROC::coords(
  roc_valid_nn,
  x           = "best",
  ret         = "threshold",
  best.method = "youden"
))
thr_nn


# AUPRC for train / valid / test

ev_train_nn <- precrec::evalmod(precrec::mmdata(p_train_nn, train_nn$loan_status))
ev_valid_nn <- precrec::evalmod(precrec::mmdata(p_valid_nn, valid_nn$loan_status))
ev_test_nn  <- precrec::evalmod(precrec::mmdata(p_test_nn,  test_nn$loan_status))

aucs_train <- precrec::auc(ev_train_nn)
aucs_valid <- precrec::auc(ev_valid_nn)
aucs_test  <- precrec::auc(ev_test_nn)

auprc_train_nn <- aucs_train[aucs_train$curvetypes == "PRC", "aucs"]
auprc_valid_nn <- aucs_valid[aucs_valid$curvetypes == "PRC", "aucs"]
auprc_test_nn  <- aucs_test[aucs_test$curvetypes == "PRC", "aucs"]

auprc_train_nn
auprc_valid_nn
auprc_test_nn


# Class predictions (VALID) + confusion matrix

pred_class_nn_valid <- ifelse(p_valid_nn >= thr_nn, "pos", "neg")

cm_nn_valid <- caret::confusionMatrix(
  factor(pred_class_nn_valid, levels = c("neg","pos")),
  factor(valid_nn$loan_status, levels = c("neg","pos")),
  positive = "pos"
)

cm_nn_valid


# Summary metrics table (TRAIN / VALID / TEST)

metrics_nn <- data.frame(
  model   = "NeuralNet_nnet",
  dataset = c("train","valid","test"),
  AUC_ROC = c(auc_train_nn, auc_valid_nn, auc_test_nn),
  AUPRC   = c(auprc_train_nn, auprc_valid_nn, auprc_test_nn),
  threshold_youden = thr_nn
)

metrics_nn


# ROC VALID plot

plot(
  roc_valid_nn,
  main        = "Neural Net (nnet) - ROC (VALID)",
  legacy.axes = TRUE,
  print.auc   = TRUE
)
abline(a = 0, b = 1, lty = 2)

