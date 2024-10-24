
#### 1. Setup ####

source('2-scripts/functions.R')



#### 2. Fetch data ####

# This is a little slow, so we've already pulled the data
# pest = pull_pest_data()
pest = read.csv("1-data/inat.csv")

counties = pull_county_data()

# PRISM limits data requests to 2x per dataset per IP per day, so we've already pulled the data to be safe
# pull_prism_data()

census = pull_census_data()



#### 3. Wrangle data ####

pest_agg = aggregate_pest_data(pest, counties)

climate_agg = wrangle_prism("1-data/prism/", counties)

cleaned_dat = expand_grid(year = c(2014:2023), county = counties$county) %>%
  left_join(pest_agg) %>%
  left_join(climate_agg) %>%
  left_join(census)
  


#### 4. Feature engineering ####

final_dat = cleaned_dat %>%
  mutate(
    detections = coalesce(detections,0),
    presence = if_else(detections > 0, 'detection', 'no detection'),
    tmean = rowMeans(pick(tmin, tmax)),
    detections = as.integer(detections),
    presence = as.factor(presence)
  ) %>%
  left_join(calc_county_centroids(counties)) %>%
  left_join(calc_neighbour_counts(., counties)) %>%
  group_by(county) %>%
  arrange(year, .by_group=T) %>%
  mutate(presence_t1 = lead(presence)) %>%
  ungroup() %>%
  filter(year < 2023)



#### 5. Model Spec and CV ####

# a. Setup, load ML libraries and set aside data to predict on with fitted model

pacman::p_load(tidymodels, parsnip, finetune, parallel, doParallel, magrittr)

set.seed(42)

cores = parallel::detectCores()
workers = parallel::makeCluster(cores, type="PSOCK")
registerDoParallel(cores=workers)

new_dat = final_dat %>% filter(year==2022)
test_train_dat = final_dat %>% filter(year < 2022)


# b. Get test-train split and class ratio

prop = nrow(test_train_dat %>% filter(year==2021))/nrow(test_train_dat)
dat_split = initial_time_split(test_train_dat, prop=1-prop)

n_training = training(dat_split) %$% table(presence_t1)
class_ratio = as.numeric(n_training[2]/n_training[1])


# c. Specify recipe

recipe = recipes::recipe(presence_t1 ~ ., data = training(dat_split)) %>% # shouldnt matter what dataset goes in here, it just catalogues the column names etc
  step_dummy(presence) %>%
  update_role(county, new_role='ID') # exclude a couple of variables (but leave them in the dataframe)


# d. Specify model and designate hyperparameters for tuning

xgb_model =
  # specify that the model is XGBoost
  parsnip::boost_tree(
    trees = tune(), #add to tuning
    # Weight in favour of the minority ()
    mtry = tune(),
    sample_size = tune(),
    min_n = tune(),
    tree_depth = tune(),
    learn_rate = tune(),
    loss_reduction = tune()
  ) %>%
  set_engine("xgboost", scale_pos_weight := !!class_ratio) %>%
  set_mode("classification")


# e. Define workflow

xgb_workflow = workflows::workflow() %>%
  # add the model
  add_model(xgb_model) %>%
  # add the recipe
  add_recipe(recipe)


# f. Set up values for hyperparameter tuning

xgboost_params = dials::parameters(
  trees(),
  finalize(mtry(), training(dat_split)),
  sample_size=sample_prop(),
  min_n(),
  tree_depth(),
  learn_rate(),
  loss_reduction()
)

xgboost_grid = dials::grid_max_entropy(
  xgboost_params, 
  size = 60
)


# g. Run cross validation

cv_folds = vfold_cv(training(dat_split),v=5)

xgb_tuned = finetune::tune_race_anova(
  object=xgb_workflow,
  resamples = cv_folds,
  grid = xgboost_grid,
  # https://yardstick.tidymodels.org/articles/metric-types.html
  metrics = yardstick::metric_set(roc_auc, pr_auc, f_meas, kap),
  control = control_race(verbose_elim = TRUE)
)


# h. Examine results and extract the best parameter set to finalize the model

xgb_tuned %>% tune::show_best(metric="roc_auc")

param_final = xgb_tuned %>% tune::select_best(metric="roc_auc")
xgb_final = xgb_workflow %>% finalize_workflow(param_final)



#### 6. Model Fitting & Predictions ####

# a. Evaluate on test set
xgb_fit = xgb_final %>% last_fit(dat_split)


# b. Get test set performance
xgb_fit %>% collect_metrics()


# c. Extract test set predictions and check out confusion matrix
test_predictions = xgb_fit %>% collect_predictions()
test_predictions %>% conf_mat(truth = presence_t1, estimate = .pred_class)


# d. Generate predictions on the 'new' data
preds = new_dat %>% mutate(pred = predict(extract_workflow(xgb_fit),.,type="raw"),
                           pred_class = if_else(pred >= .5, 'detection', 'no detection'),
                           result = case_when(
                             pred_class == 'detection' & presence_t1 == 'detection' ~ 'True +',
                             pred_class == 'detection' & presence_t1 == 'no detection' ~ 'False +',
                             pred_class == 'no detection' & presence_t1 == 'detection' ~ 'False -',
                             pred_class == 'no detection' & presence_t1 == 'no detection' ~ 'True -',
                           ))



#### 7. Outputs & SHAP ####

# a. Setup

pacman::p_load(ggplot2, shapviz)

preds_sf = preds %>% left_join(counties)


# b. Basic mapping

# Let's look at the 'new' data
preds_sf %>% ggplot(aes(fill=pred, geometry=geometry)) +
  geom_sf(colour="black") +
  labs(
    title="SLF predicted presence in Pennsylvania",
    subtitle="By county, 2022",
    fill="Probability of SLF"
  ) +
  guides(fill=guide_colourbar(position="inside")) +
  scale_fill_continuous(type="viridis") +
  theme_void() +
  theme(
    legend.justification.inside=c(.9,1),
    legend.direction="horizontal"
  )

# And map the confusion matrix
preds_sf %>% ggplot(aes(fill=result, geometry=geometry)) +
  geom_sf(colour="black") +
  labs(
    title="SLF predicted presence in Pennsylvania",
    subtitle="By county, 2022",
    fill=""
  ) +
  guides(fill=guide_legend(position="inside")) +
  scale_discrete_manual("fill", values = c('red','orange','blue','darkgreen')) +
  theme_void() +
  theme(
    legend.justification.inside=c(.9,1),
    legend.direction="horizontal"
  )


# c. Feature importance/SHAP

final_model = fit(xgb_final, final_dat)
xgb_obj = pull_workflow_fit(final_model)$fit

shap = shapviz(xgb_obj,
               X_pred = bake(prep(recipe), has_role("predictor"), new_data = final_dat, composition = "matrix"),
               X = bake(prep(recipe), has_role("predictor"), new_data = final_dat),
               interactions = T
               )

sv_importance(shap)
sv_importance(shap, kind = "beeswarm")
sv_dependence(shap, v = c("neighbour_detections","tot_population","detections","med_income"))
sv_interaction(shap)
sv_dependence(shap, v = 'county_x', interactions=T)
