###################################
# Train / validation / test split #
###################################

set.seed(68)
idx_train <- createDataPartition(cr$loan_status, p = 0.7, list = FALSE)
train.df <- cr[idx_train, ]
hold.df <- cr[-idx_train, ]

set.seed(96)
idx_valid <- createDataPartition(hold.df$loan_status, p = 0.5, list = FALSE)
valid.df <- hold.df[idx_valid, ]
test.df <- hold.df[-idx_valid, ]


# בדיקת שמירת יחסים במשתנה היעד
prop.table(table(cr$loan_status))
prop.table(table(train.df$loan_status))
prop.table(table(valid.df$loan_status))
prop.table(table(test.df$loan_status))

p <- sapply(list(all = cr$loan_status,
                 train = train.df$loan_status,
                 valid = valid.df$loan_status,
                 test = test.df$loan_status),
            function(x) prop.table(table(x))["1"])

max(abs(p - p[1]))

dim(valid.df)
dim(train.df)
dim(test.df)


# Missing value handling #

colSums(is.na(cr))

miss_cols <- names(which(colSums(is.na(train.df)) > 0))


# 1.  Missing flags (1 = was NA in original)

for (cl in miss_cols) {
  train.df[[paste0("was_na_", cl)]] <- as.integer(is.na(train.df[[cl]]))
  valid.df[[paste0("was_na_", cl)]] <- as.integer(is.na(valid.df[[cl]]))
  test.df[[paste0("was_na_", cl)]] <- as.integer(is.na(test.df[[cl]]))
}

# 2. Split columns with NA into numeric vs categorical

is_num_miss <- vapply(train.df[miss_cols], is.numeric, logical(1))
num_miss <- miss_cols[is_num_miss]

is_cat_miss <- vapply(
  train.df[miss_cols],
  function(x) is.factor(x) || is.character(x),
  logical(1)
)
cat_miss <- miss_cols[is_cat_miss]

# 3. Numeric imputation by median

medians <- vapply(train.df[num_miss], \(x) median(x, na.rm = TRUE), numeric(1))
for (nm in num_miss) {
  train.df[[nm]][is.na(train.df[[nm]])] <- medians[[nm]]
  valid.df[[nm]][is.na(valid.df[[nm]])] <- medians[[nm]]
  test.df[[nm]][is.na(test.df[[nm]])] <- medians[[nm]]
}

# 4. Categorical imputation by 'Missing' level

for (cm in cat_miss) {
  # אחוד רמות + הוספת "Missing"
  lvls <- union(levels(factor(train.df[[cm]])), c("Missing"))
  train.df[[cm]] <- factor(train.df[[cm]], levels = lvls)
  valid.df[[cm]] <- factor(valid.df[[cm]], levels = lvls)
  test.df[[cm]] <- factor(test.df[[cm]], levels = lvls)
  
  train.df[[cm]][is.na(train.df[[cm]])] <- "Missing"
  valid.df[[cm]][is.na(valid.df[[cm]])] <- "Missing"
  test.df[[cm]][is.na(test.df[[cm]])]   <- "Missing"
}

# 5. Final sanity checks

stopifnot(!anyNA(train.df[miss_cols]), !anyNA(valid.df[miss_cols]), !anyNA(test.df[miss_cols]))
colSums(is.na(train.df))

# Target factor re-leveling for caret

train.df$loan_status <- factor(train.df$loan_status, levels = c("1","0"))
valid.df$loan_status <- factor(valid.df$loan_status, levels = c("1","0"))
test.df$loan_status  <- factor(test.df$loan_status,  levels = c("1","0"))

