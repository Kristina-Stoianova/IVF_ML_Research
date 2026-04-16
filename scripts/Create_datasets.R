library(tidyverse)
library(here)

here::here()

#Load cleaned data
freshdata <- read_csv(
  here("data", "processed", "freshdata_cleaned1.csv")
)

#DEFINE FILTERS

#First cycle IVF only
filter_first_cycle <- function(df){
  df %>% filter(Cycle_no == 1)
}

#Standard IVF protocols - short antagonist protocol
filter_short_protocol <- function(df) {
  df %>%
    filter(
      Protocol_type %in% c("Short Antagonist"),
    )
}


#Datasets
#1. first cycle only
allfirstcycle_data <- freshdata %>%
  filter_first_cycle()

#2. first cycle, short protocols
shortprotocol <- allfirstcycle_data %>%
  filter_short_protocol()


#Save 

write_csv(allfirstcycle_data, here("data","processed","allfirstcycle_data.csv"))

write_csv(shortprotocol, here("data","processed","allshort_protocol_data.csv"))