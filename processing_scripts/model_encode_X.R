names(freshdata)


modelling1 <- freshdata %>%
  select(
    Age,
    Aetiology,
    Cycle_no,
    Infertility_duration,
    Baseline_total_follicles,
    Baseline_endometrium,
    Protocol_type,
    GnRH_antagonist_duration,
    GnRH_antagonist_total_dose,
    FSH_duration,
    FSH_total_dose,
    FSH_avg_daily,
    FSH_dose_changes,
    hMG_total_dose,
    hMG_duration,
    hMG_avg_daily,
    hCG_trigger_dose,
    GnRH_agonist_trigger_dose,
    Last_day_of_stim,
    No_mature_eggs
  )
 
print(unique(freshdata$Aetiology))

#Tubal Factors, DOR, Endometriosis, Uterine Factor, Endometriosis + Uterine Factor, Endometriosis + Tubal Factors, DOR + Endmetreosis + PCO = Female_factor
#Other, Same Sex Relationship, Fertility Preservaton, Single Woman, Surrogate = Other
#Unexplained = Unexplained
#No female factor = No_female_factor


# Verify all unique values
print(unique(freshdata$Aetiology))

modelling1 <- modelling1 %>%
  mutate(Aetiology = case_when(
    
    #Female factor group
    Aetiology %in% c("Tubal Factors", 
                     "DOR",
                     "Endometriosis",
                     "Uterine Factor",
                     "Endometriosis + Uterine Factor",
                     "Endometriosis + Tubal Factors",
                     "PCO",
                     "DOR + Endometriosis + PCO")  ~ "Female_factor",
    
    #Other group
    Aetiology %in% c("Other",
                     "Same Sex Relationship",
                     "Fertility Preservation",
                     "Single woman",        # lowercase w from your data
                     "Surrogate")           ~ "Other",
    
    #Unexplained
    Aetiology == "Unexplained"              ~ "Unexplained",
    
    #No female factor
    Aetiology %in% c("No Female Factor",   # capital F from your data
                     "No female factor")   ~ "No_female_factor",
  ))


# Numeric encoding
modelling1 <- modelling1 %>%
  mutate(Aetiology = case_when(
    Aetiology == "Female_factor"    ~ 1,
    Aetiology == "Unexplained"      ~ 2,
    Aetiology == "Other"            ~ 3,
    Aetiology == "No_female_factor" ~ 4,
    TRUE ~ NA_real_
  ))

str(modelling1$Aetiology)

filter_first_cycle <- function(df){
  df %>% filter(Cycle_no == 1)
}

modelling1 <- modelling1 %>%
  filter_first_cycle()

modelling1$Cycle_no <- NULL
modelling1$Protocol_type <- NULL

str(modelling1)

write.csv(modelling1, "modelling1.csv", row.names = FALSE)
