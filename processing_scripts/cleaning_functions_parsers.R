#Cleaning Aetiology
clean_aetiology <- function(x) {
  na_idx <- is.na(x)
  x <- str_squish(x)
  x <- str_to_lower(x)
  #Fix typos
  x <- str_replace_all(x, "relatonship", "relationship")
  x <- str_replace_all(x, "researve", "reserve")
  # Normalise separators to +
  x <- str_replace_all(x, ",|;|\\band\\b|/", "+")
  # Simple substitutions
  x <- str_remove_all(x, regex("fertility preservation\\s*", ignore_case = TRUE))
  x <- str_remove_all(x, "[()]")
  x <- str_replace_all(x, "diminished ovarian reserve|low ovarian reserve", "DOR")
  x <- str_remove_all(x, regex("ovulation disord\\w*\\s*\\+?\\s*", ignore_case = TRUE))
  x <- str_replace_all(x, "pcos?", "PCO")
  #Remove male terms and PGD entirely
  male_and_pgd <- paste(
    c("male factor other", "male factor", "azoospermia", "oligozoospermia",
      "oat", "teratoma", "terato", "sperm donor", "pgd"),
    collapse = "|"
  )
  x <- str_remove_all(x, regex(male_and_pgd, ignore_case = TRUE))
  #Remove empty brackets and stray punctuation
  x <- str_remove_all(x, "\\(\\s*\\)")
  #Clean up leftover + separators
  x <- str_replace_all(x, "(\\s*\\+\\s*)+", "+")   # collapse multiples
  x <- str_remove_all(x, "^\\+|\\+$")              # trim leading/trailing
  x <- str_replace_all(x, "\\+", " + ")            # standardise spacing
  x <- str_squish(x)
  # Empty = was male-only
  x[x == ""] <- "No Female Factor"
  # Title case, then restore acronyms
  x <- str_to_title(x)
  x <- str_replace_all(x, "\\bDor\\b", "DOR")
  x <- str_replace_all(x, "\\bPco\\b", "PCO")
  x[na_idx] <- NA
  return(x)
}

parse_protocol <- function(protocol, total_fsh, total_hmg) {
  x <- stringr::str_to_lower(stringr::str_squish(protocol))
  #Protocol type
  protocol_clean <- dplyr::case_when(
    stringr::str_detect(x, "egg donor|donor") ~ "Egg_donor",
    stringr::str_detect(x, "fertility preservation") ~ "Fertility_preservation",
    stringr::str_detect(x, "long agonist") ~ "Long_agonist",
    stringr::str_detect(x, "short antagonist") ~ "Short_antagonist",
    stringr::str_detect(x, "short") ~ "Short_antagonist",
    TRUE ~ "Other"
  )
  
  #Drug used parameters
  fsh_used <- ifelse(
    stringr::str_detect(x, "fsh|rfsh") |
      (!is.na(total_fsh) & total_fsh > 0),
    1, 0
  )
  hmg_used <- ifelse(
    stringr::str_detect(x, "hmg") |
      (!is.na(total_hmg) & total_hmg > 0),
    1, 0
  )
  #Starting dose
  fsh_dose <- as.numeric(
    stringr::str_extract(x, "(?<=rfsh\\s)\\d+|(?<=fsh\\s)\\d+")
  )
  
  hmg_dose <- as.numeric(
    stringr::str_extract(x, "(?<=hmg\\s)\\d+")
  )
  
  tibble::tibble(
    Protocol_clean = protocol_clean,
    FSH_used = fsh_used,
    hMG_used = hmg_used,
    FSH_starting_dose = fsh_dose,
    hMG_starting_dose = hmg_dose
  )
}


#Stimulation cleaning function: applied to FSH, Hmg, Fyramedel
clean_stim_string <- function(x) {
  x %>%
    #Remove leading/trailing + double spaces
    str_squish() %>%
    #Remove drug names (FSH + hMG)
    str_replace_all(regex("Gonal\\s*F\\s*=?\\s*", ignore_case = TRUE), "") %>%
    str_replace_all(regex("Menop\\w*\\s*", ignore_case = TRUE), "") %>%
    str_replace_all(regex("\\bF\\s*=", ignore_case = TRUE), "") %>%
    #Remove IU units
    str_replace_all(regex("iu", ignore_case = TRUE), "") %>%
    #Standardise dash characters
    str_replace_all("–", "-") %>%
    #Standardise D formatting
    str_replace_all("D(\\d+)-D(\\d+)", "D\\1-\\2") %>%
    #Fix parentheses spacing
    str_replace_all("\\(", " (") %>%
    str_replace_all("\\(\\s+", "(") %>%
    str_replace_all("\\s+\\)", ")") %>%
    #Add comma between protocol segments
    str_replace_all("\\)\\s*(?=\\d)", "), ") %>%
    #Replace & with comma
    str_replace_all("&", ",") %>%
    #Collapse repeated brackets
    str_replace_all("\\(+", "(") %>%
    str_replace_all("\\)+", ")") %>%
    #Remove empty brackets
    str_replace_all("\\(\\)", "") %>%
    #Clean spaces again after modifications
    str_squish() 
}

##Parser function to derive stimulation parameters: FSH, Hmg
parse_stimulation <- function(x, prefix) {
  
  if (is.na(x) || x == "" || x == "N/A") {
    return(tibble(
      !!paste0(prefix, "_used") := 0,
      !!paste0(prefix, "_total_dose") := 0,
      !!paste0(prefix, "_duration") := 0,
      !!paste0(prefix, "_initial_dose") := NA,
      !!paste0(prefix, "_start_day") := NA,
      !!paste0(prefix, "_final_dose") := NA,
      !!paste0(prefix, "_max_dose") := NA,
      !!paste0(prefix, "_min_dose") := NA,
      !!paste0(prefix, "_dose_changes") := 0,
      !!paste0(prefix, "_change_day_1") := NA,
      !!paste0(prefix, "_change_day_2") := NA,
      !!paste0(prefix, "_change_day_3") := NA,
      !!paste0(prefix, "_dose_delta") := NA,
      !!paste0(prefix, "_dose_trend") := NA,
      !!paste0(prefix, "_avg_daily") := NA
    ))
  }
  
  segments <- str_split(x, ",")[[1]] %>% str_trim()
  
  dose_data <- tibble()
  last_dose <- NA
  
  for (seg in segments) {
    
    dose_match <- str_extract(seg, "^\\d+\\.?\\d*")
    
    if (is.na(dose_match)) {
      dose <- last_dose
    } else {
      dose <- as.numeric(dose_match)
      last_dose <- dose
    }
    
    day_part <- str_extract(seg, "D\\d+(?:-\\d+)?")
    
    if (!is.na(day_part) && str_detect(day_part, "-")) {
      start_day <- as.numeric(str_extract(day_part, "\\d+"))
      end_day <- as.numeric(str_extract(day_part, "\\d+$"))
    } else {
      start_day <- as.numeric(str_extract(day_part, "\\d+"))
      end_day <- start_day
    }
    
    days <- seq(start_day, end_day)
    
    dose_data <- bind_rows(
      dose_data,
      tibble(day = days, dose = dose)
    )
  }
  
  dose_data <- dose_data %>% arrange(day)
  
  total_dose <- sum(dose_data$dose)
  duration_total <- nrow(dose_data)
  
  change_days <- dose_data %>%
    mutate(change = dose != lag(dose)) %>%
    filter(change == TRUE & !is.na(change)) %>%
    pull(day)
  
  tibble(
    !!paste0(prefix, "_used") := 1,
    !!paste0(prefix, "_total_dose") := total_dose,
    !!paste0(prefix, "_duration") := duration_total,
    !!paste0(prefix, "_initial_dose") := first(dose_data$dose),
    !!paste0(prefix, "_start_day") := min(dose_data$day),
    !!paste0(prefix, "_final_dose") := last(dose_data$dose),
    !!paste0(prefix, "_max_dose") := max(dose_data$dose),
    !!paste0(prefix, "_min_dose") := min(dose_data$dose),
    !!paste0(prefix, "_dose_changes") := length(change_days),
    !!paste0(prefix, "_change_day_1") := ifelse(length(change_days) >= 1, change_days[1], NA),
    !!paste0(prefix, "_change_day_2") := ifelse(length(change_days) >= 2, change_days[2], NA),
    !!paste0(prefix, "_change_day_3") := ifelse(length(change_days) >= 3, change_days[3], NA),
    !!paste0(prefix, "_dose_delta") := last(dose_data$dose) - first(dose_data$dose),
    !!paste0(prefix, "_dose_trend") := sign(last(dose_data$dose) - first(dose_data$dose)),
    !!paste0(prefix, "_avg_daily") := total_dose / duration_total
  )
}

#Cleaning trigger string
clean_trigger_string <- function(x) {
  x %>%
    str_squish() %>%
    na_if("") %>%
    na_if("N/A") %>%
    #Remove quotes
    str_replace_all('"', "") %>%
    #Remove IU units
    str_replace_all(regex("iu", ignore_case = TRUE), "") %>%
    #Fix common typos
    str_replace_all(
      regex("Busserillin|Busserelin|Buserellin", ignore_case = TRUE),
      "Buserelin"
    ) %>%
    #Standardize spelling
    str_replace_all(regex("Ovitrelle", ignore_case = TRUE), "Ovitrelle") %>%
    str_squish()
}

##Parser function to derive Fyremadel parameters (GnRH_antagonist medications) - less parameters required than FSH and hMG
parse_fyr <- function(x) {
  if (is.na(x) || x == "" || x == "N/A") {
    return(tibble(
      Fyr_used = 0,
      Fyr_start_day = NA,
      Fyr_end_day = NA,
      Fyr_duration = 0,
      Fyr_total_dose = 0
    ))
  }
  #allow decimal doses
  dose <- as.numeric(str_extract(x, "^\\d+\\.?\\d*"))
  #extract day range or single day
  day_part <- str_extract(x, "D\\d+(?:-\\d+)?")
  if (str_detect(day_part, "-")) {
    start_day <- as.numeric(str_extract(day_part, "\\d+"))
    end_day <- as.numeric(str_extract(day_part, "\\d+$"))
  } else {
    start_day <- as.numeric(str_extract(day_part, "\\d+"))
    end_day <- start_day
  }
  duration <- end_day - start_day + 1
  tibble(
    Fyr_used = 1,
    Fyr_start_day = start_day,
    Fyr_end_day = end_day,
    Fyr_duration = duration,
    Fyr_total_dose = dose * duration
  )
}

parse_day <- function(x) {
  x %>%
    stringr::str_squish() %>%
    stringr::str_extract("\\d+") %>%   # only keep numbers
    as.numeric()
}
