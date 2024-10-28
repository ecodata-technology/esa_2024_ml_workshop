
#### 1. Setup ####

# install pacman, if necessary
if (!require("pacman")) {install.packages("pacman")}

pacman::p_load(dplyr, tidyr, lubridate, sf, purrr,  # General data wrangling
               rinat, prism, tidycensus, tigris,    # Fetching iNat, PRISM, census, county data
               exactextractr, rlang,                # Zonal statistics   
               sfdep)                               # Neighbour features
               
# Some of our source data doesn't work well with S2 geometry, so disable it
sf_use_s2(FALSE)



#### 2. Fetch Data ####

# a. Fetch pest data from iNaturalist
pull_pest_data <- function() {
  
  inat = data.frame()
  
  for(year in 2015:2023) {
    temp = rinat::get_inat_obs(taxon_name = "Lycorma delicatula",
                               quality = "research",
                               year = year,
                               place_id = 42,  # Pennsylvania has place ID 42
                               maxresults = 10000) %>%
      mutate(year = year) %>%
      select(year, latitude, longitude, scientific_name)
    
    inat = inat %>% rbind(temp)
  }
  
  write.csv(inat, "1-data/inat.csv", row.names = F)
  
  return(inat)
  
}

# b. Fetch TIGER/Line county shapefiles
pull_county_data <- function() {
  
  tigris::counties(state="PA",resolution="20m") %>%
    rename(county = NAME) %>%
    mutate(county = tolower(county)) %>%
    select(county)
  
}

# c. Fetch raw PRISM data and save a copy to save time in future
pull_prism_data <- function() {
  
  # precipitation
  prism_set_dl_dir('1-data/prism/ppt'); get_prism_annual(type = 'ppt', years = c(2014:2022), keepZip=F)
  
  # tmin
  prism_set_dl_dir('1-data/prism/tmin'); get_prism_annual(type = 'tmin', years = c(2014:2022), keepZip=F)
  
  # tmax
  prism_set_dl_dir('1-data/prism/tmax'); get_prism_annual(type = 'tmax', years = c(2014:2022), keepZip=F)
  
}

# d. Fetch American Community Survey data
pull_census_data <- function() {
  
  acs = data.frame()
  
  for(year in 2014:2022) {
    temp = tidycensus::get_acs(geography = "county",
                  variables = c(tot_population="B01003_001", med_income = "B19013_001"),
                  year = year,
                  state = 'PA',
                  geometry = F,
                  survey = 'acs5') %>%
      mutate(year = year)
    
    acs = acs %>% rbind(temp)
  }
  
  acs = acs %>%
    mutate(county = gsub(" County, Pennsylvania", "", NAME),
           county = tolower(county)) %>%
    select(county, variable, estimate, year) %>%
    pivot_wider(names_from = variable, values_from = estimate)
  
  return(acs)
  
}



#### 3. Wrangle data ####

# a. Aggregate pest point data by county and year
aggregate_pest_data <- function(pests, polygons) {
  
  pests %>%
    st_as_sf(coords = c("longitude","latitude"), crs=4326) %>%
    # match pest CRS to county shapefile CRS
    st_transform(crs=4269) %>%
    st_join(polygons) %>%
    # some entries appear to come from outside PA, we'll just exclude them for this demo but normally we'd want to investigate
    filter(!is.na(county)) %>%
    as.data.frame() %>%
    group_by(year, county) %>%
    summarise(detections = n()) %>%
    ungroup()
  
}

# b. Map PRISM raster to county polygons with zonal statistics from exactextractr package

# Due to how PRISM data is structured we have to process each year-variable combination separately and collate the results
wrangle_prism <- function(prism_directory, counties) {
  
  datasets = expand_grid(year = c(2014:2022), var = c('ppt', 'tmin', 'tmax'))
  
  output = data.frame()
  
  df = for(i in 1:nrow(datasets)) {
    
    var = datasets$var[i]
    year = datasets$year[i]
    
    # Open the subdir
    dir = paste0("1-data/prism/",var)
    prism_set_dl_dir(dir)
    
    # Fetch the raster
    temp_raster = prism_archive_subset(type=var,temp_period='annual',years=year) %>% pd_stack()
    
    # Realign to counties
    # Warning message is a known issue in the github, it's ambiguous but the correct projection is indeed being used
    extract = exact_extract(
      temp_raster,
      counties,
      'mean'
    )
    
    collate = counties %>%
      st_drop_geometry(geometry) %>%
      cbind(extract) %>%
      mutate(year = year, variable = var) %>%
      dplyr::select(county, year, variable, value = extract)
    
    output = rbind(output, collate)
    
  }
  
  output = output %>% pivot_wider(names_from = variable, values_from = value)
  
  return(output)
  
}



#### 4. Feature engineering ####

# a. Calculate county centroids
calc_county_centroids <- function(counties) {
  
  counties %>%
    st_transform(crs=5070) %>%
    mutate(
      centroids = st_centroid(geometry),
      county_x = unlist(purrr::map(centroids,1)),
      county_y = unlist(purrr::map(centroids,2))
      ) %>%
    as.data.frame() %>%
    select(county, county_x, county_y)
  
}

# b. Calculate neighbour detections
calc_neighbour_counts <- function(dat, counties) {
  
  temp = counties %>%
    mutate(id = row_number(),
           nb = sfdep::st_contiguity(geometry)) %>%
    right_join(dat) %>%
    st_drop_geometry(geometry)
  
  out = temp %>% mutate(nb_detections = purrr::map2(nb, year,
                                                    ~ temp %>% filter(id %in% unlist(.x) & year == .y) %>% summarise(sum(detections))
                                                    )
                        ) %>%
    select(county, year, nb_detections) %>%
    unnest("nb_detections") %>%
    rename(neighbour_detections = 3)
  
  return(out)
  
}