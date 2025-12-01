##########################################################################
# 04_model_logit_baseline.R
# Baseline logistic regression (GLM) for credit default prediction
#
# Assumes:
#   - 00_load_packages.R
#   - 01_load_and_clean.R
#   - 02_data_split_imput.R
#   - 03_data_transforms.R
# have been run and train.df / valid.df / test.df exist in the workspace.
##########################################################################


# Local copies + target factor levels #

train_log <- train.df
valid_log <- valid.df
test_log  <- test.df

train_log$loan_status <- factor(train_log$loan_status, levels = c("0","1"))
valid_log$loan_status <- factor(valid_log$loan_status, levels = c("0","1"))
test_log$loan_status  <- factor(test_log$loan_status,  levels = c("0","1"))

# Class weights for imbalance #

w_pos <- sum(train_log$loan_status == "0") / sum(train_log$loan_status == "1")
w_log <- ifelse(train_log$loan_status == "1", w_pos, 1)

# Fit baseline logistic regression model #

log_reg <- glm(
  loan_status ~ . - row_id,
  data    = train_log,
  family  = binomial,
  weights = w_log
)

summary(log_reg)

# Odds ratios (OR) and confidence intervals #

co <- summary(log_reg)$coefficients
or <- exp(co[, 1])
ci <- exp(confint.default(log_reg))

or_table <- cbind(
  OR      = or,
  CI_low  = ci[, 1],
  CI_high = ci[, 2],
  p_value = co[, 4]
)
or_table

# Multicollinearity #

vif_vals <- car::vif(log_reg)
vif_vals



# Unified evaluation helper: ROC, AUPRC, confusion matrix #

evaluate_logit <- function(model,
                           train_df,
                           valid_df,
                           test_df,
                           target_col = "loan_status",
                           positive   = "1",
                           cutoff     = 0.5,
                           model_name = "Logit_baseline") {
  
  
  # Ensure target factor levels
  train_df[[target_col]] <- factor(train_df[[target_col]], levels = c("0","1"))
  valid_df[[target_col]] <- factor(valid_df[[target_col]], levels = c("0","1"))
  test_df[[target_col]]  <- factor(test_df[[target_col]],  levels = c("0","1"))
  
  y_train <- train_df[[target_col]]
  y_valid <- valid_df[[target_col]]
  y_test  <- test_df[[target_col]]
  
  # Ensure target factor levels 
  p_train <- predict(model, newdata = train_df, type = "response")
  p_valid <- predict(model, newdata = valid_df, type = "response")
  p_test  <- predict(model, newdata = test_df,  type = "response")
  
  # AUC ROC
  roc_train <- pROC::roc(y_train, p_train, levels = c("0","1"), quiet = TRUE)
  roc_valid <- pROC::roc(y_valid, p_valid, levels = c("0","1"), quiet = TRUE)
  roc_test  <- pROC::roc(y_test,  p_test,  levels = c("0","1"), quiet = TRUE)
  
  auc_train <- as.numeric(pROC::auc(roc_train))
  auc_valid <- as.numeric(pROC::auc(roc_valid))
  auc_test  <- as.numeric(pROC::auc(roc_test))
  
  # AUPRC
  mm_train <- precrec::mmdata(p_train, y_train)
  mm_valid <- precrec::mmdata(p_valid, y_valid)
  mm_test  <- precrec::mmdata(p_test,  y_test)
  
  ev_train <- precrec::evalmod(mm_train)
  ev_valid <- precrec::evalmod(mm_valid)
  ev_test  <- precrec::evalmod(mm_test)
  
  auprc_train <- precrec::auc(ev_train)[precrec::auc(ev_train)$curvetypes == "PRC", "aucs"]
  auprc_valid <- precrec::auc(ev_valid)[precrec::auc(ev_valid)$curvetypes == "PRC", "aucs"]
  auprc_test  <- precrec::auc(ev_test)[precrec::auc(ev_test)$curvetypes == "PRC", "aucs"]
  
  # Classes for confusion matrix
  pred_valid_class <- factor(ifelse(p_valid >= cutoff, "1", "0"),
                             levels = c("0","1"))
  
  cm_valid <- caret::confusionMatrix(
    data      = pred_valid_class,
    reference = y_valid,
    positive  = "1"
  )
  
  # Summary metrics table
  metrics_df <- data.frame(
    model   = model_name,
    dataset = c("train", "valid", "test"),
    AUC_ROC = c(auc_train, auc_valid, auc_test),
    AUPRC   = c(auprc_train, auprc_valid, auprc_test),
    cutoff  = cutoff
  )
  
  list(
    metrics   = metrics_df,
    cm_valid  = cm_valid,
    roc_train = roc_train,
    roc_valid = roc_valid,
    roc_test  = roc_test
  )
}



# Run baseline model evaluation & basic ROC plot #

res_logit <- evaluate_logit(
  model      = log_reg,
  train_df   = train_log,
  valid_df   = valid_log,
  test_df    = test_log,
  target_col = "loan_status",
  positive   = "1",
  cutoff     = 0.5,
  model_name = "Logit_baseline"
)

logit_metrics  <- res_logit$metrics
logit_cm_valid <- res_logit$cm_valid

logit_metrics
logit_cm_valid

# ROC plot for VALID set
plot(res_logit$roc_valid,
     col  = "blue",
     lwd  = 2,
     main = paste("Logit baseline – VALID ROC (AUC =", round(logit_metrics$AUC_ROC[logit_metrics$dataset=="valid"], 4), ")"))
abline(a = 0, b = 1, lty = 2, col = "gray40")
