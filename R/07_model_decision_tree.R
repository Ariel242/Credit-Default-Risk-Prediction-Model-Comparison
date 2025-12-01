######################################################################
# 07_model_conditional_inference_tree.R
# Conditional inference tree (party): training and evaluation
#
# Assumes: train.df / valid.df / test.df exist from previous scripts
######################################################################

library(party)
library(pROC)
library(precrec)

# Local copies + target levels (0/1)

train_tree <- train.df
valid_tree <- valid.df
test_tree  <- test.df

train_tree$loan_status <- factor(as.character(train_tree$loan_status),
                                 levels = c("0","1"))
valid_tree$loan_status <- factor(as.character(valid_tree$loan_status),
                                 levels = c("0","1"))
test_tree$loan_status  <- factor(as.character(test_tree$loan_status),
                                 levels = c("0","1"))


# Full tree (unpruned)

tr_full <- party::ctree(loan_status ~ . - row_id, data = train_tree)
plot(tr_full)


# Simpler, more interpretable tree

tr_simple <- party::ctree(
  loan_status ~ . - row_id,
  data     = train_tree,
  controls = party::ctree_control(maxdepth = 4, minbucket = 100)
)

plot(tr_simple, type = "simple", gp = grid::gpar(fontsize = 3))


# Predicted probabilities (full tree) for all splits

p_train_tree <- sapply(predict(tr_full, newdata = train_tree, type = "prob"), "[", 2)
p_valid_tree <- sapply(predict(tr_full, newdata = valid_tree, type = "prob"), "[", 2)
p_test_tree  <- sapply(predict(tr_full, newdata = test_tree,  type = "prob"), "[", 2)

stopifnot(length(p_train_tree) == nrow(train_tree))
stopifnot(length(p_valid_tree) == nrow(valid_tree))
stopifnot(length(p_test_tree)  == nrow(test_tree))


# ROC + AUC for train / valid / test

roc_train_tree <- pROC::roc(
  response  = train_tree$loan_status,
  predictor = p_train_tree,
  levels    = c("0","1"),
  quiet     = TRUE
)

roc_valid_tree <- pROC::roc(
  response  = valid_tree$loan_status,
  predictor = p_valid_tree,
  levels    = c("0","1"),
  quiet     = TRUE
)

roc_test_tree <- pROC::roc(
  response  = test_tree$loan_status,
  predictor = p_test_tree,
  levels    = c("0","1"),
  quiet     = TRUE
)

auc_train_tree <- as.numeric(pROC::auc(roc_train_tree))
auc_valid_tree <- as.numeric(pROC::auc(roc_valid_tree))
auc_test_tree  <- as.numeric(pROC::auc(roc_test_tree))

auc_train_tree
auc_valid_tree
auc_test_tree


#  AUPRC for train / valid / test

ev_train_tree <- precrec::evalmod(precrec::mmdata(p_train_tree, train_tree$loan_status))
ev_valid_tree <- precrec::evalmod(precrec::mmdata(p_valid_tree, valid_tree$loan_status))
ev_test_tree  <- precrec::evalmod(precrec::mmdata(p_test_tree,  test_tree$loan_status))

aucs_train <- precrec::auc(ev_train_tree)
aucs_valid <- precrec::auc(ev_valid_tree)
aucs_test  <- precrec::auc(ev_test_tree)

auprc_train_tree <- aucs_train[aucs_train$curvetypes == "PRC", "aucs"]
auprc_valid_tree <- aucs_valid[aucs_valid$curvetypes == "PRC", "aucs"]
auprc_test_tree  <- aucs_test[aucs_test$curvetypes == "PRC", "aucs"]

auprc_train_tree
auprc_valid_tree
auprc_test_tree

# Class predictions (cutoff = 0.5) + confusion matrix (VALID)

pred_valid_tree_cls <- ifelse(p_valid_tree >= 0.5, "1", "0")

cm_tree_valid <- caret::confusionMatrix(
  data      = factor(pred_valid_tree_cls, levels = c("0","1")),
  reference = valid_tree$loan_status,
  positive  = "1"
)

cm_tree_valid


# Summary metrics table (TRAIN / VALID / TEST)

metrics_ctree <- data.frame(
  model   = "CTree_full",
  dataset = c("train","valid","test"),
  AUC_ROC = c(auc_train_tree, auc_valid_tree, auc_test_tree),
  AUPRC   = c(auprc_train_tree, auprc_valid_tree, auprc_test_tree),
  cutoff  = 0.5
)

metrics_ctree


# ROC curve for VALID

plot(roc_valid_tree,
     col  = "red",
     lwd  = 2,
     main = paste("CTree ROC (AUC =", round(auc_valid_tree, 4), ") – VALID"))
abline(a = 0, b = 1, lty = 2, col = "gray40")

