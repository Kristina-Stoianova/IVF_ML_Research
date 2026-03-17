library(tidyverse)
library(here)

here::here()

#Load cleaned data
freshdata <- read_csv(
  here("data", "processed", "freshdata_cleaned.csv")
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

#Datasets
#1. first cycle only
firstcycle_data <- freshdata %>%
  filter_first_cycle()

#2. first cycle standard long and short protocols only
standard_firstcycle_data <- firstcycle_data %>%
  filter_standard_ivf()


#Save

write_csv(firstcycle_data, here("data","processed","firstcycle_data.csv"))

write_csv(standard_firstcycle_data, here("data","processed","standard_firstcycle_data.csv"))



