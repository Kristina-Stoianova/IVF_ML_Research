### HELPER FUNCTIONS

library(dplyr)
library(stringr)
library(janitor)
library(stringr)

#standardize missing values to N/A
standardise_missing_values <- function(df) {
  missing_vals <- c("", "na", "n/a", "missing", "_", "not on bbs", "incomplete chart", "incomplete data", "see comments", "-", "tftc")
  df %>%
    dplyr::mutate(
      across(
      where(is.character),
      ~ {
        x <- stringr::str_squish(.x) #remove trailing/leading empty spaces
        x_lower <- stringr::str_to_lower(x) #
        x[x_lower %in% missing_vals] <- NA
        x
      }
    )
  )
}

#clean categorical variables/strings 
clean_categorical <- function(x){
  x <- stringr::str_replace_all(x, "[()]", "")        #remove brackets
  x <- stringr::str_replace_all(x, "/", " + ")        #/ to +
  x <- stringr::str_replace_all(x, ",\\s*", " + ")    #commas to +
  x <- stringr::str_replace_all(x, "\\band\\b", " + ") #"and" to +
  x <- stringr::str_squish(x)                         #remove extra spaces
  x <- stringr::str_replace_all(x, "\\+\\s*\\+", "+") #remove duplicate +
  x <- stringr::str_replace_all(x, "^[+\\s]+|[+\\s]+$", "") #trim separators
  return(x)
}

#clean baseline hormones 
clean_baseline <- function(x) {
  x %>%
    stringr::str_remove("\\s*\\(.*\\)") %>%  #remove "(date)"
    as.numeric() %>% #convert to numeric
    round(1) #round to 1dp
}

#clean sperm parameters
clean_sperm_numeric <- function(x) {
  x %>%
    stringr::str_squish() %>%                  # remove extra spaces
    stringr::str_remove("%") %>%               # remove % if present
    as.numeric() %>%                          # convert to numeric
    round(1)                                  # round to 1dp
}

#Convert Y/N columns to numeric 
convert_binary_numeric <- function(x){
  ifelse(x == "Y", 1,
         ifelse(x == "N", 0, NA))
}

