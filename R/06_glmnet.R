######################################################################
# 06_glmnet.R
# Elastic Net logistic regression with CV + Youden threshold
#
# Assumes: train.df / valid.df / test.df are preprocessed (scripts 00–03)
######################################################################

library(glmnet)
library(caret)
library(pROC)

# Local copies + target recoding (neg/pos)

train_glmnet <- train.df
valid_glmnet <- valid.df
test_glmnet  <- test.df


train_glmnet$loan_status <- factor(ifelse(train_glmnet$loan_status == "1", "pos", "neg"),
                                   levels = c("neg","pos"))
valid_glmnet$loan_status <- factor(ifelse(valid_glmnet$loan_status == "1", "pos", "neg"),
                                   levels = c("neg","pos"))
test_glmnet$loan_status  <- factor(ifelse(test_glmnet$loan_status  == "1", "pos", "neg"),
                                   levels = c("neg","pos"))

table(train_glmnet$loan_status)
table(valid_glmnet$loan_status)

# TrainControl: Repeated CV (AUC metric)

ctrl_glmnet <- trainControl(
  method          = "repeatedcv",  # קרוס-ולידציה חוזרת
  number          = 5,
  repeats         = 3,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,  # ROC
  savePredictions = "final"
)

# Class weights (imbalance handling)

w_pos <- sum(train_glmnet$loan_status == "neg") / sum(train_glmnet$loan_status == "pos")
w_vec <- ifelse(train_glmnet$loan_status == "pos", w_pos, 1)

# Train GLMNET Logistic Model

set.seed(2025)
mod_glmnet <- caret::train(
  loan_status ~ . - row_id,
  data       = train_glmnet,
  method     = "glmnet",
  family     = "binomial",
  trControl  = ctrl_glmnet,
  metric     = "ROC",
  weights    = w_vec,
  tuneLength = 15
)

print(mod_glmnet)
plot(mod_glmnet)  

# Predicted probabilities for all splits

p_train_glmnet <- predict(mod_glmnet, newdata = train_glmnet, type = "prob")[, "pos"]
p_valid_glmnet <- predict(mod_glmnet, newdata = valid_glmnet, type = "prob")[, "pos"]
p_test_glmnet  <- predict(mod_glmnet, newdata = test_glmnet,  type = "prob")[, "pos"]

stopifnot(length(p_train_glmnet) == nrow(train_glmnet))
stopifnot(length(p_valid_glmnet) == nrow(valid_glmnet))
stopifnot(length(p_test_glmnet)  == nrow(test_glmnet))


roc_train_glmnet <- pROC::roc(
  response  = train_glmnet$loan_status,
  predictor = p_train_glmnet,
  levels    = c("neg","pos"),
  quiet     = TRUE
)

roc_valid_glmnet <- pROC::roc(
  response  = valid_glmnet$loan_status,
  predictor = p_valid_glmnet,
  levels    = c("neg","pos"),
  quiet     = TRUE
)

roc_test_glmnet <- pROC::roc(
  response  = test_glmnet$loan_status,
  predictor = p_test_glmnet,
  levels    = c("neg","pos"),
  quiet     = TRUE
)

auc_train_glmnet <- as.numeric(pROC::auc(roc_train_glmnet))
auc_valid_glmnet <- as.numeric(pROC::auc(roc_valid_glmnet))
auc_test_glmnet  <- as.numeric(pROC::auc(roc_test_glmnet))

auc_train_glmnet
auc_valid_glmnet
auc_test_glmnet

# Youden thr on valid

thr_glmnet <- as.numeric(pROC::coords(
  roc_valid_glmnet,
  x           = "best",
  ret         = "threshold",
  best.method = "youden"
))
thr_glmnet

# AUPRC for train / valid / test

ev_train_glmnet <- precrec::evalmod(precrec::mmdata(p_train_glmnet, train_glmnet$loan_status))
ev_valid_glmnet <- precrec::evalmod(precrec::mmdata(p_valid_glmnet, valid_glmnet$loan_status))
ev_test_glmnet  <- precrec::evalmod(precrec::mmdata(p_test_glmnet,  test_glmnet$loan_status))

aucs_train <- precrec::auc(ev_train_glmnet)
aucs_valid <- precrec::auc(ev_valid_glmnet)
aucs_test  <- precrec::auc(ev_test_glmnet)

auprc_train_glmnet <- aucs_train[aucs_train$curvetypes == "PRC", "aucs"]
auprc_valid_glmnet <- aucs_valid[aucs_valid$curvetypes == "PRC", "aucs"]
auprc_test_glmnet  <- aucs_test[aucs_test$curvetypes == "PRC", "aucs"]

auprc_train_glmnet
auprc_valid_glmnet
auprc_test_glmnet

# Confusion matrix for VALID at Youden threshold

pred_class_glmnet_valid <- ifelse(p_valid_glmnet >= thr_glmnet, "pos", "neg")

cm_glmnet_valid <- caret::confusionMatrix(
  factor(pred_class_glmnet_valid, levels = c("neg","pos")),
  factor(valid_glmnet$loan_status, levels = c("neg","pos")),
  positive = "pos"
)

cm_glmnet_valid

# Summary performance metrics table

metrics_glmnet <- data.frame(
  model   = "GLMNET_ElasticNet",
  dataset = c("train","valid","test"),
  AUC_ROC = c(auc_train_glmnet, auc_valid_glmnet, auc_test_glmnet),
  AUPRC   = c(auprc_train_glmnet, auprc_valid_glmnet, auprc_test_glmnet),
  threshold_youden = thr_glmnet
)

metrics_glmnet


# Feature Selection

coef_mat <- as.matrix(
  coef(mod_glmnet$finalModel, s = mod_glmnet$bestTune$lambda)
)

coef_tbl <- data.frame(
  variable = rownames(coef_mat),
  coef     = as.numeric(coef_mat)
)

coef_tbl_nonzero <- coef_tbl[coef_tbl$coef != 0, ]
coef_tbl_nonzero <- coef_tbl_nonzero[order(-abs(coef_tbl_nonzero$coef)), ]

print("Top 20 important features (non-zero coefficients):")
print(head(coef_tbl_nonzero, 20))

# ROC plot for VALID with Youden point

plot(roc_valid_glmnet, main = "ROC – VALID (GLMNET, threshold = Youden)")

coords_youden_glmnet <- pROC::coords(
  roc_valid_glmnet,
  x            = thr_glmnet,
  input        = "threshold",
  ret          = c("specificity", "sensitivity")
)

points(
  x = 1 - coords_youden_glmnet["specificity"],
  y = coords_youden_glmnet["sensitivity"],
  pch = 19, cex = 1.2
)
