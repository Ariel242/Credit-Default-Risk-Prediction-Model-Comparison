# Winsorization of monetary variables (for regression / NN) #

# Winsorization helper: learn from train, apply to all splits #
wins_apply <- function(tr, va, te, col, p=.01){
  q <- quantile(tr[[col]], c(p,1-p), na.rm=TRUE)
  tr[[col]] <- pmin(pmax(tr[[col]], q[1]), q[2])
  va[[col]] <- pmin(pmax(va[[col]], q[1]), q[2])
  te[[col]] <- pmin(pmax(te[[col]], q[1]), q[2])
  list(tr=tr, va=va, te=te)
}

num_cols <- c("person_income","loan_amnt","loan_int_rate","loan_percent_income")
for(cl in num_cols){
  out <- wins_apply(train.df, valid.df, test.df, cl); train.df <- out$tr; valid.df <- out$va; test.df <- out$te
}

# Log transforms for skewed variables #

for(cl in c("loan_amnt","person_income")){
  train.df[[cl]] <- log1p(train.df[[cl]]);
  valid.df[[cl]] <- log1p(valid.df[[cl]])
  test.df[[cl]]  <- log1p(test.df[[cl]])
}

# Align factor levels across train / valid / test #

align_factor_levels <- function(tr, va, te) {
  fact_cols <- union(names(Filter(is.factor, tr)),
                     union(names(Filter(is.factor, va)),
                           names(Filter(is.factor, te))))
  for (cl in fact_cols) {
    lv <- union(levels(factor(tr[[cl]])),
                union(levels(factor(va[[cl]])),
                      levels(factor(te[[cl]]))))
    tr[[cl]] <- factor(tr[[cl]], levels = lv)
    va[[cl]] <- factor(va[[cl]], levels = lv)
    te[[cl]] <- factor(te[[cl]], levels = lv)
  }
  list(tr=tr, va=va, te=te)
}
.tmp <- align_factor_levels(train.df, valid.df, test.df)
train.df <- .tmp$tr; valid.df <- .tmp$va; test.df <- .tmp$te


# סCentering & scaling for models sensitive to feature scale #

num_all <- c("person_age","person_income","person_emp_length","loan_amnt",
             "loan_int_rate","loan_percent_income","cb_person_cred_hist_length")
pp <- caret::preProcess(train.df[, num_all], method=c("center","scale"))
train.df[, num_all] <- predict(pp, train.df[, num_all])
valid.df[, num_all] <- predict(pp, valid.df[, num_all])
test.df[, num_all]  <- predict(pp, test.df[, num_all])


# Row ID stability and integrity checks #

nrow(train.df) + nrow(valid.df) + nrow(test.df) == nrow(cr)
cr$row_id <- seq_len(nrow(cr)) 

all_idx   <- seq_len(nrow(cr))
hold_idx  <- setdiff(all_idx, idx_train)
valid_idx <- hold_idx[idx_valid]
test_idx  <- setdiff(hold_idx, valid_idx)

train.df$row_id <- cr$row_id[idx_train]
valid.df$row_id <- cr$row_id[valid_idx]
test.df$row_id  <- cr$row_id[test_idx]

# Overlap / coverage checks #

all( intersect(train.df$row_id, valid.df$row_id) == integer(0) )
all( intersect(train.df$row_id,  test.df$row_id) == integer(0) )
all( intersect(valid.df$row_id,  test.df$row_id) == integer(0) )
setequal(c(train.df$row_id, valid.df$row_id, test.df$row_id), cr$row_id)

# Final check: target class balance after all processing #

prop.table(table(cr$loan_status))
prop.table(table(train.df$loan_status))
prop.table(table(valid.df$loan_status))
prop.table(table(test.df$loan_status))

