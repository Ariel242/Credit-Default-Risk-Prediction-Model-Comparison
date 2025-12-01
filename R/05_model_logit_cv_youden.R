######################################################################
# 05_model_logit_cv_youden.R
# Logistic regression with CV, class weights, and Youden threshold
#
# Assumes:
#   - Preprocessed train.df / valid.df / test.df are available
#     from previous scripts (00–03).
######################################################################


# Local copies + target recoding to pos/neg

train_cv <- train.df
valid_cv <- valid.df
test_cv  <- test.df


train_cv$loan_status <- factor(ifelse(train_cv$loan_status == "1", "pos", "neg"),
                               levels = c("neg","pos"))
valid_cv$loan_status <- factor(ifelse(valid_cv$loan_status == "1", "pos", "neg"),
                               levels = c("neg","pos"))
test_cv$loan_status  <- factor(ifelse(test_cv$loan_status  == "1", "pos", "neg"),
                               levels = c("neg","pos"))

table(train_cv$loan_status)
table(valid_cv$loan_status)

# Training control for CV (AUC-based)

ctrl <- trainControl(
  method          = "repeatedcv",
  number          = 5,
  repeats         = 3,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,   # ROC, Sens, Spec
  savePredictions = "final"
)

# Class weights for positive class (pos)

w_pos <- sum(train_cv$loan_status == "neg") / sum(train_cv$loan_status == "pos")
w_vec <- ifelse(train_cv$loan_status == "pos", w_pos, 1)

# Train logistic regression with caret + CV

set.seed(95)
glm_cv <- caret::train(
  loan_status ~ . - row_id,
  data      = train_cv,
  method    = "glm",
  family    = binomial(),
  trControl = ctrl,
  metric    = "ROC",
  weights   = w_vec
)

glm_cv


# Predicted probabilities for all splits

p_train_cv <- predict(glm_cv, newdata = train_cv, type = "prob")[, "pos"]
p_valid_cv <- predict(glm_cv, newdata = valid_cv, type = "prob")[, "pos"]
p_test_cv  <- predict(glm_cv, newdata = test_cv,  type = "prob")[, "pos"]

stopifnot(length(p_train_cv) == nrow(train_cv))
stopifnot(length(p_valid_cv) == nrow(valid_cv))
stopifnot(length(p_test_cv)  == nrow(test_cv))


# ROC, AUC for train / valid / test + Youden threshold (VALID)

roc_train_cv <- pROC::roc(response  = train_cv$loan_status,
                          predictor = p_train_cv,
                          levels    = c("neg","pos"),
                          quiet     = TRUE)

roc_valid_cv <- pROC::roc(response  = valid_cv$loan_status,
                          predictor = p_valid_cv,
                          levels    = c("neg","pos"),
                          quiet     = TRUE)

roc_test_cv  <- pROC::roc(response  = test_cv$loan_status,
                          predictor = p_test_cv,
                          levels    = c("neg","pos"),
                          quiet     = TRUE)

auc_train_cv <- as.numeric(pROC::auc(roc_train_cv))
auc_valid_cv <- as.numeric(pROC::auc(roc_valid_cv))
auc_test_cv  <- as.numeric(pROC::auc(roc_test_cv))

auc_train_cv
auc_valid_cv
auc_test_cv


thr_youden <- as.numeric(pROC::coords(
  roc_valid_cv,
  x           = "best",
  ret         = "threshold",
  best.method = "youden"
))

thr_youden


# AUPRC for train / valid / test

ev_train_cv <- precrec::evalmod(precrec::mmdata(p_train_cv, train_cv$loan_status))
ev_valid_cv <- precrec::evalmod(precrec::mmdata(p_valid_cv, valid_cv$loan_status))
ev_test_cv  <- precrec::evalmod(precrec::mmdata(p_test_cv,  test_cv$loan_status))

aucs_train <- precrec::auc(ev_train_cv)
aucs_valid <- precrec::auc(ev_valid_cv)
aucs_test  <- precrec::auc(ev_test_cv)

auprc_train_cv <- aucs_train[aucs_train$curvetypes == "PRC", "aucs"]
auprc_valid_cv <- aucs_valid[aucs_valid$curvetypes == "PRC", "aucs"]
auprc_test_cv  <- aucs_test[aucs_test$curvetypes == "PRC", "aucs"]

auprc_train_cv
auprc_valid_cv
auprc_test_cv


# Class predictions using Youden threshold + confusion matrix (VALID)

pred_cls_train <- ifelse(p_train_cv >= thr_youden, "pos", "neg")
pred_cls_valid <- ifelse(p_valid_cv >= thr_youden, "pos", "neg")
pred_cls_test  <- ifelse(p_test_cv  >= thr_youden, "pos", "neg")

cm_glm_valid <- caret::confusionMatrix(
  data      = factor(pred_cls_valid,             levels = c("neg","pos")),
  reference = factor(valid_cv$loan_status,       levels = c("neg","pos")),
  positive  = "pos"
)

cm_glm_valid

# שיעור יעד - בקרה
table(valid_cv$loan_status)


# Key performance metrics on VALID (at Youden threshold)

acc_valid    <- cm_glm_valid$overall["Accuracy"]
kappa_valid  <- cm_glm_valid$overall["Kappa"]

sens_valid   <- cm_glm_valid$byClass["Sensitivity"]
spec_valid   <- cm_glm_valid$byClass["Specificity"]
prec_valid   <- cm_glm_valid$byClass["Pos Pred Value"]  # Precision
f1_valid     <- cm_glm_valid$byClass["F1"]
balacc_valid <- cm_glm_valid$byClass["Balanced Accuracy"]

youden_valid <- as.numeric(sens_valid + spec_valid - 1)

round(
  c(
    Threshold        = thr_youden,
    Accuracy         = acc_valid,
    Kappa            = kappa_valid,
    Sensitivity      = sens_valid,
    Specificity      = spec_valid,
    Precision        = prec_valid,
    F1               = f1_valid,
    BalancedAccuracy = balacc_valid,
    Youden_J         = youden_valid
  ),
  3
)


# Summary metrics table (TRAIN / VALID / TEST)

metrics_glm_cv <- data.frame(
  model   = "Logit_CV_Youden",
  dataset = c("train","valid","test"),
  AUC_ROC = c(auc_train_cv, auc_valid_cv, auc_test_cv),
  AUPRC   = c(auprc_train_cv, auprc_valid_cv, auprc_test_cv),
  threshold_youden = thr_youden
)

metrics_glm_cv


# ROC for VALID with Youden point highlighted

plot(roc_valid_cv, main = "ROC – VALID (GLM CV, threshold = Youden)")

coords_youden <- pROC::coords(
  roc_valid_cv,
  x            = thr_youden,
  input        = "threshold",
  ret          = c("specificity", "sensitivity")
)

points(
  x = 1 - coords_youden["specificity"],
  y = coords_youden["sensitivity"],
  pch = 19, cex = 1.2
)

