#####################
# 01_load_and_clean #
#####################

setwd("C:/Users/ariel/Desktop/תיק עבודות- אנליטיקה/ניתוח הגירת אשראי")
cr <- read.csv("credit_risk_dataset.csv", header = T)

summary(cr); head(cr); str(cr)


# Factor & type cleaning

cr$person_home_ownership <- factor(tolower(trimws(cr$person_home_ownership)))
cr$loan_intent           <- factor(tolower(trimws(cr$loan_intent)))

cr$loan_grade <- factor(
  toupper(trimws(cr$loan_grade)),
  levels  = c("A","B","C","D","E","F","G"),
  ordered = TRUE
)

cr$loan_status <- factor(cr$loan_status, levels = c(0,1), labels = c("0","1"))

cr$cb_person_default_on_file <- factor(
  toupper(trimws(cr$cb_person_default_on_file)),
  levels = c("N","Y")
)

str(cr)


######################################################################
# Handling problematic values (employment & credit history length)
######################################################################


clean_tenure <- function(x, max_years = 60L) {
  x <- suppressWarnings(as.numeric(x))
  x[!is.finite(x)] <- NA
  x[x < 0] <- NA
  # ערכים לא סבירים לשנים
  x[x > max_years] <- NA
  x
}

cr$person_emp_length            <- clean_tenure(cr$person_emp_length,            max_years = 60)
cr$cb_person_cred_hist_length   <- clean_tenure(cr$cb_person_cred_hist_length,   max_years = 60)


