library(tidyverse)
library(here)
here::here()

source(here("scripts", "cleaning_helpers.R"))
source(here("scripts", "cleaning_functions_parsers.R"))

data100 <- read_csv(
  here("data", "processed", "freshdata_cleaned.csv")
)
data500 <- read_csv(
  here("data", "raw", "Final_Fresh_Merged.csv")
)

#Renaming
data500 <- data500 %>%
  rename(Unique_number = UniqueNbr)
data500 <- data500 %>%
  rename(AMH = AntiMullerianHormoneResult)
data500 <- data500 %>%
  rename(Final_follicles_lessthan_11.9 = `TotalFol>11.9mm_onLastScan`)

#Clean
data100 <- data100 %>%
  mutate(Unique_number = trimws(as.character(Unique_number)))
data500 <- data500 %>%
  mutate(Unique_number = trimws(as.character(Unique_number)))

#make sure data = date
data500 <- data500 %>%
  mutate(CycleStartDate = as.Date(CycleStartDate))
      
#Get earliest cycle per patient --> 226 patients total in big dataset
data500_small <- data500 %>%
  arrange(Unique_number, CycleStartDate) %>%
  group_by(Unique_number) %>%
  slice(1) %>%
  ungroup() %>%
  select(Unique_number, BMI, AMH, Final_follicles_lessthan_11.9)
        
#join
df <- data100 %>%
  left_join(data500_small, by = "Unique_number")

str(df$AMH)
str(df$Final_follicles_lessthan_11.9)
str(df$BMI)

##reorganize
df <- df %>%
  relocate(
    AMH,
    .after = Baseline_AMH
  )

df <- df %>%
  relocate(
    BMI,
    .after = Age
  )

df <- df %>%
  relocate(
    Final_follicles_lessthan_11.9,
    .after = Final_right_follicles
  )

df$Baseline_AMH <- NULL
colnames(df)[colnames(df) == "AMH"] <- "Baseline_AMH"


print(sum(is.na(df$Baseline_AMH))) #17 missing 

write_csv(
  df,
  here("data", "processed", "freshdata_cleaned1.csv")
)

