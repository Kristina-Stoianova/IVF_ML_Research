### HELPER FUNCTIONS

library(dplyr)
library(stringr)
library(janitor)
library(stringr)

#standardize missing values to N/A
standardise_missing_values <- function(df) {
  missing_vals <- c(
    "", "na", "n/a", "missing", "_", "not on bbs",
    "incomplete chart", "incomplete data", "see comments",
    "-", "tftc", ".", "null", "none", "pending", "not specified"
  )
  df %>%
    dplyr::mutate(
      across(
        where(is.character),
        ~ {
          x <- stringr::str_squish(.x)  #trim spaces
          x_lower <- stringr::str_to_lower(x)
          x[!is.na(x_lower) & x_lower %in% missing_vals] <- NA
          x
        }
      )
    )
}

#clean funding
clean_funding <- function(x) {
  x %>%
    stringr::str_squish() %>%
    stringr::str_remove("^\\d+\\s*[-–]\\s*")
}

#clean protocol type
clean_protocol_type <- function(x) {
  x_clean <- x %>%
    stringr::str_squish() %>%
    stringr::str_to_lower() %>%
    stringr::str_replace_all("[^a-z0-9 ]", " ")  #Removes + symbols
  
  dplyr::case_when(
    stringr::str_detect(x_clean, "egg donor|donor") ~ "Egg_donor",
    stringr::str_detect(x_clean, "fertility preservation") ~ "Fertility_preservation",
    stringr::str_detect(x_clean, "long agonist") ~ "Long_agonist",
    stringr::str_detect(x_clean, "short") ~ "Short_antagonist",
    TRUE ~ "Other"
  )
}

#clean doses in protocol
extract_fsh_dose <- function(x) {
  as.numeric(stringr::str_extract(x, "(?<=rfsh\\s)\\d+|(?<=fsh\\s)\\d+"))
}
extract_hmg_dose <- function(x) {
  as.numeric(stringr::str_extract(x, "(?<=hmg\\s)\\d+"))
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
