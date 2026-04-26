library(tidyverse)
library(here)

here::here()

#Load cleaned data
freshdata <- read_csv(
  here("data", "processed", "freshdata_cleaned.csv")
)


#Function to convert Y/N columns to numeric 
convert_binary_numeric <- function(x){
  ifelse(x == "Y", 1,
         ifelse(x == "N", 0, NA))
}

## Convert binary categorical variables to numeric

freshdata <- freshdata %>%
  mutate(
    across(
      c(PGD, Male_factor, Sperm_mobility),
      convert_binary_numeric
    )
  )

freshdata <- freshdata %>%
  mutate(
    Funding = case_when(
      Funding == "NHS" ~ 1,
      Funding == "PP" ~ 0,
      TRUE ~ NA_real_
    )
  )

#DEFINE FILTERS

#First cycle IVF only
filter_first_cycle <- function(df){
  df %>% filter(Cycle_no == 1)
}

#Standard IVF protocols - short and long (no egg donor, fertility preservation etc.)
filter_standard_ivf <- function(df) {
  df %>%
    filter(
      Protocol_type %in% c("Short Antagonist", "Long Agonist"),
    )
}

##Short, long, egg donors
filter_protocol_ivf <- function(df) {
  df %>% 
    filter(
      Protocol_type %in% c("Short Antagonist", "Long Agonist", "Egg Donor"),
    )
}

#Datasets
#1. first cycle only
firstcycle_data <- freshdata %>%
  filter_first_cycle()


"Semen_volume_ml",
freshdata <- freshdata %>%
  select(
    -contains("sperm"),
    -any_of(drop_cols)
  )


#2. first cycle standard long and short protocols only
standard_firstcycle_data <- firstcycle_data %>%
  filter_standard_ivf()

#3. first cycle long, short, egg donor protocols
firstcycle_data1 <- firstcycle_data %>%
  filter_protocol_ivf


#Save 

write_csv(firstcycle_data, here("data","processed","firstcycle_data.csv"))

write_csv(standard_firstcycle_data, here("data","processed","standard_firstcycle_data.csv"))

write_csv(firstcycle_data1, here("data", "processed", "firstcycle_data1.csv"))



