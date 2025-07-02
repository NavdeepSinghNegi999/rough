
# pip install optuna[visualization]


import optuna.visualization as vis

# 1. Optimization History (Objective vs. Trial)
vis.plot_optimization_history(study).show()

# 2. Hyperparameter Importance
vis.plot_param_importances(study).show()

# 3. Slice Plot (Effect of Each Param on Accuracy)
vis.plot_slice(study).show()

# 4. Parallel Coordinates (Hyperparam interaction)
vis.plot_parallel_coordinate(study).show()

# 5. Contour Plot (2D interaction plot)
vis.plot_contour(study, params=None).show()  # Optional: ['lr', 'dense_units_0']

# 6. Empirical Distribution of Parameters
vis.plot_edf(study).show()

print("Best Accuracy:", study.best_value)
print("Best Params:", study.best_params)

# Then plot
vis.plot_optimization_history(study).show()

