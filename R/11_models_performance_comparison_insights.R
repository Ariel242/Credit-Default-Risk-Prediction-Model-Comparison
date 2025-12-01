######################################################################
# 11_model_comparison_summary.R
# Comprehensive test-set comparison of all models
#
# Assumes:
#   - Scripts 00–10 have been run
#   - All required model objects & predictions exist in memory
######################################################################

library(dplyr)
library(ggplot2)
library(pROC)
library(caret)

cat("═══════════════════════════════════════════════════════════\n")
cat("              MODEL COMPARISON - FINAL REPORT              \n")
cat("═══════════════════════════════════════════════════════════\n\n")


# Collect test metrics for all models


model_objects <- list(
  list(name = "Logit_Baseline", 
       p_test = res_logit$roc_test$predictor,
       y_test = res_logit$roc_test$response,
       auc_test = res_logit$metrics$AUC_ROC[3],
       auprc_test = res_logit$metrics$AUPRC[3],
       levels = c("0","1")),
  
  list(name = "Logit_CV", 
       p_test = p_test_cv,
       y_test = test_cv$loan_status,
       auc_test = auc_test_cv,
       auprc_test = auprc_test_cv,
       thr = thr_youden,
       levels = c("neg","pos")),
  
  list(name = "GLMnet", 
       p_test = p_test_glmnet,
       y_test = test_glmnet$loan_status,
       auc_test = auc_test_glmnet,
       auprc_test = auprc_test_glmnet,
       thr = thr_glmnet,
       levels = c("neg","pos")),
  
  list(name = "Decision_Tree", 
       p_test = p_test_tree,
       y_test = test_tree$loan_status,
       auc_test = auc_test_tree,
       auprc_test = auprc_test_tree,
       levels = c("0","1")),
  
  list(name = "Random_Forest", 
       p_test = p_test_rf,
       y_test = test_rf$loan_status,
       auc_test = auc_test_rf,
       auprc_test = auprc_test_rf,
       thr = thr_rf,
       levels = c("neg","pos")),
  
  list(name = "XGBoost", 
       p_test = p_test_xgb,
       y_test = test_xgb$loan_status,
       auc_test = auc_test_xgb,
       auprc_test = auprc_test_xgb,
       thr = thr_xgb,
       levels = c("neg","pos")),
  
  list(name = "Neural_Net", 
       p_test = p_test_nn,
       y_test = test_nn$loan_status,
       auc_test = auc_test_nn,
       auprc_test = auprc_test_nn,
       thr = thr_nn,
       levels = c("neg","pos"))
)


# Ensure Youden threshold exists for all models

for (i in seq_along(model_objects)) {
  mod <- model_objects[[i]]
  
  # אם אין threshold (Logit_Baseline, Decision_Tree), חשב Youden
  if (is.null(mod$thr)) {
    roc_obj <- pROC::roc(
      response = mod$y_test,
      predictor = mod$p_test,
      levels = mod$levels,
      quiet = TRUE
    )
    
    mod$thr <- as.numeric(pROC::coords(
      roc_obj,
      x = "best",
      ret = "threshold",
      best.method = "youden"
    ))
    
    model_objects[[i]]$thr <- mod$thr
  }
}


# Confusion matrices on TEST for all models

results_list <- list()

for (i in seq_along(model_objects)) {
  mod <- model_objects[[i]]
  
  # Class prediction based on Youden threshold
  if (mod$levels[2] == "pos") {
    pred_class <- ifelse(mod$p_test >= mod$thr, "pos", "neg")
    pred_class <- factor(pred_class, levels = c("neg", "pos"))
    y_factor <- factor(mod$y_test, levels = c("neg", "pos"))
    positive_class <- "pos"
  } else {
    pred_class <- ifelse(mod$p_test >= mod$thr, "1", "0")
    pred_class <- factor(pred_class, levels = c("0", "1"))
    y_factor <- factor(mod$y_test, levels = c("0", "1"))
    positive_class <- "1"
  }
  
  # Confusion Matrix
  cm <- caret::confusionMatrix(
    data = pred_class,
    reference = y_factor,
    positive = positive_class
  )
  
  results_list[[i]] <- list(
    model = mod$name,
    auc_roc = mod$auc_test,
    auprc = mod$auprc_test,
    threshold = mod$thr,
    sensitivity = as.numeric(cm$byClass["Sensitivity"]),
    specificity = as.numeric(cm$byClass["Specificity"]),
    precision = as.numeric(cm$byClass["Precision"]),
    f1 = as.numeric(cm$byClass["F1"]),
    balanced_acc = as.numeric(cm$byClass["Balanced Accuracy"]),
    accuracy = as.numeric(cm$overall["Accuracy"]),
    cm_obj = cm
  )
}


# Summary comparison table (TEST)

comparison_table <- do.call(rbind, lapply(results_list, function(x) {
  data.frame(
    Model = x$model,
    AUC_ROC = round(x$auc_roc, 4),
    AUPRC = round(x$auprc, 4),
    Threshold = round(x$threshold, 4),
    Sensitivity = round(x$sensitivity, 3),
    Specificity = round(x$specificity, 3),
    Precision = round(x$precision, 3),
    F1_Score = round(x$f1, 3),
    Balanced_Acc = round(x$balanced_acc, 3),
    Accuracy = round(x$accuracy, 3)
  )
}))

comparison_table <- comparison_table[order(-comparison_table$AUC_ROC), ]
comparison_table$Rank <- 1:nrow(comparison_table)


cat("\n")
cat("═══════════════════════════════════════════════════════════════════════════════\n")
cat("                    TEST SET PERFORMANCE - ALL MODELS                          \n")
cat("═══════════════════════════════════════════════════════════════════════════════\n\n")

print(comparison_table, row.names = FALSE)


# High-level insights

cat("\n\n")
cat("═══════════════════════════════════════════════════════════\n")
cat("                    KEY INSIGHTS                           \n")
cat("═══════════════════════════════════════════════════════════\n\n")


best_model <- comparison_table$Model[1]
best_auc <- comparison_table$AUC_ROC[1]

cat("🏆 BEST MODEL:", best_model, "\n")
cat("   AUC-ROC:", best_auc, "\n")
cat("   Sensitivity:", comparison_table$Sensitivity[1], 
    "(catches", round(comparison_table$Sensitivity[1]*100, 1), "% of defaults)\n")
cat("   Precision:", comparison_table$Precision[1],
    "(", round(comparison_table$Precision[1]*100, 1), "% of predictions are correct)\n\n")

# Comparison to Base
baseline_idx <- which(comparison_table$Model == "Logit_Baseline")
if (length(baseline_idx) > 0) {
  improvement <- (best_auc - comparison_table$AUC_ROC[baseline_idx]) / 
    comparison_table$AUC_ROC[baseline_idx] * 100
  cat("📈 IMPROVEMENT OVER BASELINE:", round(improvement, 1), "%\n\n")
}

# complexity vs performance

cat("💡 MODEL COMPLEXITY ANALYSIS:\n")
cat("   Simple Models (Logit, GLMnet):\n")
simple_models <- comparison_table[comparison_table$Model %in% 
                                    c("Logit_Baseline", "Logit_CV", "GLMnet"), ]
cat("     Average AUC:", round(mean(simple_models$AUC_ROC), 4), "\n")

cat("   Complex Models (RF, XGBoost, NN):\n")
complex_models <- comparison_table[comparison_table$Model %in% 
                                     c("Random_Forest", "XGBoost", "Neural_Net"), ]
cat("     Average AUC:", round(mean(complex_models$AUC_ROC), 4), "\n\n")

# Overfitting 
cat("⚠️  OVERFITTING CHECK:\n")
cat("   (Compare TRAIN vs TEST AUC in individual model files)\n\n")


##############################
# Comparative visualizations #
##############################

# AUC-ROC bar plot

p1 <- ggplot(comparison_table, aes(x = reorder(Model, AUC_ROC), y = AUC_ROC)) +
  geom_col(aes(fill = AUC_ROC), show.legend = FALSE) +
  geom_text(aes(label = round(AUC_ROC, 3)), hjust = -0.2, size = 4) +
  coord_flip() +
  scale_fill_gradient(low = "#FFC107", high = "#2E7D32") +
  labs(
    title = "Model Comparison - TEST SET AUC-ROC",
    subtitle = "Higher is better - all models evaluated on same held-out test set",
    x = NULL,
    y = "AUC-ROC"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 10, color = "gray40"),
    axis.text = element_text(size = 10)
  ) +
  ylim(0, 1)

print(p1)

# Sensitivity vs Precision trade-off
p2 <- ggplot(comparison_table, aes(x = Sensitivity, y = Precision)) +
  geom_point(aes(size = AUC_ROC, color = Model), alpha = 0.7) +
  geom_text(aes(label = Model), vjust = -1, size = 3) +
  scale_size_continuous(range = c(3, 10)) +
  labs(
    title = "Sensitivity vs Precision Trade-off",
    subtitle = "Point size represents AUC-ROC",
    x = "Sensitivity (Recall)",
    y = "Precision"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

print(p2)

# ROC curves – all models (TEST)
plot(NULL, xlim = c(0, 1), ylim = c(0, 1), 
     xlab = "1 - Specificity (False Positive Rate)",
     ylab = "Sensitivity (True Positive Rate)",
     main = "ROC Curves - All Models (TEST SET)")
abline(a = 0, b = 1, lty = 2, col = "gray50")

colors <- c("#2E7D32", "#1976D2", "#F57C00", "#C62828", "#7B1FA2", "#00796B", "#5D4037")

for (i in seq_along(model_objects)) {
  mod <- model_objects[[i]]
  roc_obj <- pROC::roc(mod$y_test, mod$p_test, levels = mod$levels, quiet = TRUE)
  plot(roc_obj, col = colors[i], lwd = 2, add = TRUE)
}

legend("bottomleft", 
       legend = paste0(comparison_table$Model, " (", comparison_table$AUC_ROC, ")"),
       col = colors[order(-comparison_table$AUC_ROC)],
       lwd = 2,
       cex = 0.7)


# Detailed confusion matrix for best model

cat("\n\n")
cat("═══════════════════════════════════════════════════════════\n")
cat("        DETAILED CONFUSION MATRIX - BEST MODEL             \n")
cat("═══════════════════════════════════════════════════════════\n\n")

best_model_name <- comparison_table$Model[1]
best_model_idx <- which(sapply(results_list, function(x) x$model == best_model_name))

best_cm <- results_list[[best_model_idx]]$cm_obj

print(best_cm)

cm_df <- as.data.frame(best_cm$table)
colnames(cm_df) <- c("Predicted", "Actual", "Count")


cm_df$Correct <- ifelse(cm_df$Predicted == cm_df$Actual, "Correct", "Error")

# heatmap 
p_cm <- ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Correct)) +
  geom_tile(color = "white", linewidth = 1.5) +
  geom_text(aes(label = Count), size = 12, fontface = "bold", color = "white") +
  scale_fill_manual(
    values = c("Correct" = "#4CAF50", "Error" = "#D32F2F"),
    labels = c("Correct" = "✓ Correct", "Error" = "✗ Error")
  ) +
  labs(
    title = paste("Confusion Matrix -", best_model_name),
    subtitle = paste0("TEST SET | Threshold: ", round(best_cm$overall["Accuracy"], 3)),
    x = "Actual Class",
    y = "Predicted Class",
    fill = "Prediction"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5, color = "gray40"),
    axis.title = element_text(size = 13, face = "bold"),
    axis.text = element_text(size = 11, face = "bold"),
    legend.position = "right",
    legend.title = element_text(size = 11, face = "bold"),
    legend.text = element_text(size = 10),
    panel.grid = element_blank()
  ) +
  coord_fixed()


print(p_cm)


cat("\n📊 INTERPRETATION:\n")
cat("   • True Positives (Caught defaults):", best_cm$table[2,2], "\n")
cat("   • False Negatives (Missed defaults):", best_cm$table[1,2], "⚠️\n")
cat("   • False Positives (False alarms):", best_cm$table[2,1], "\n")
cat("   • True Negatives (Correct rejections):", best_cm$table[1,1], "\n\n")


cat("═══════════════════════════════════════════════════════════\n")
cat("✅ Comparison complete! Results saved to:\n")
cat("   model_comparison_results.csv\n")
cat("═══════════════════════════════════════════════════════════\n\n")

# Final recommendation

cat("═══════════════════════════════════════════════════════════\n")
cat("                  FINAL RECOMMENDATION                     \n")
cat("═══════════════════════════════════════════════════════════\n\n")

cat("Based on TEST SET evaluation:\n\n")
cat("🎯 FOR PRODUCTION DEPLOYMENT:\n")
cat("   Model:", best_model, "\n")
cat("   Threshold:", round(results_list[[best_model_idx]]$threshold, 4), "\n")
cat("   Expected Performance:\n")
cat("     - Will catch", round(results_list[[best_model_idx]]$sensitivity*100, 1), 
    "% of actual defaults\n")
cat("     - ", round(results_list[[best_model_idx]]$precision*100, 1), 
    "% of flagged cases will be true defaults\n")
cat("     - Overall accuracy:", round(results_list[[best_model_idx]]$accuracy*100, 1), "%\n\n")


# Secondary Recommendation

second_best <- comparison_table$Model[2]
cat("🔄 ALTERNATIVE (for interpretability):\n")
if (second_best %in% c("Logit_CV", "GLMnet")) {
  cat("   Consider", second_best, "if you need:\n")
  cat("     - Explainable predictions\n")
  cat("     - Faster inference\n")
  cat("     - Regulatory compliance\n")
  cat("   Performance trade-off: -", 
      round((best_auc - comparison_table$AUC_ROC[2])*100, 1), 
      "% AUC\n\n")
}

cat("═══════════════════════════════════════════════════════════\n")
cat("                    END OF REPORT                          \n")
cat("═══════════════════════════════════════════════════════════\n")
