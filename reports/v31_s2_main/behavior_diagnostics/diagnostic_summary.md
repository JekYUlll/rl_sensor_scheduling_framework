# V3.1 S2 Behavior Diagnostics

## Sensor Configuration
| order | sensor_id | power_cost | startup_peak_power | warmup_steps | variables |
| --- | --- | --- | --- | --- | --- |
| 0 | met_station_core | 0.42 | 0.55 | 0 | wind_speed_ms|wind_direction_deg|air_temperature_c|relative_humidity|air_pressure_pa |
| 1 | radiometer_basic | 0.36 | 0.48 | 0 | solar_radiation_wm2 |
| 2 | surface_temp_ir | 0.38 | 0.5 | 1 | snow_surface_temperature_c |
| 3 | ultrasonic_anemometer_hd | 0.58 | 0.78 | 1 | wind_speed_ms|wind_direction_deg |
| 4 | shielded_thermo_hygro | 0.52 | 0.7 | 1 | air_temperature_c|relative_humidity |
| 5 | snow_particle_counter | 0.68 | 0.92 | 2 | snow_particle_mean_diameter_mm|snow_particle_mean_velocity_ms |
| 6 | laser_disdrometer | 0.82 | 1.15 | 4 | snow_particle_mean_diameter_mm|snow_particle_mean_velocity_ms |
| 7 | fc4_flux | 0.86 | 1.2 | 5 | snow_mass_flux_kg_m2_s |

## Feasible Subset Capacity
| budget | rank | feasible_count | max_feasible_size | steady_power | startup_peak_power | sensor_count | sensor_ids |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.65 | 1 | 58 | 3 | 1.16 | 1.53 | 3 | met_station_core|radiometer_basic|surface_temp_ir |
| 1.65 | 2 | 58 | 3 | 1.26 | 1.68 | 3 | radiometer_basic|surface_temp_ir|shielded_thermo_hygro |
| 1.65 | 3 | 58 | 3 | 1.3 | 1.73 | 3 | met_station_core|radiometer_basic|shielded_thermo_hygro |
| 1.65 | 4 | 58 | 3 | 1.32 | 1.76 | 3 | radiometer_basic|surface_temp_ir|ultrasonic_anemometer_hd |
| 1.65 | 5 | 58 | 3 | 1.32 | 1.75 | 3 | met_station_core|surface_temp_ir|shielded_thermo_hygro |
| 1.7 | 1 | 63 | 4 | 1.68 | 2.23 | 4 | met_station_core|radiometer_basic|surface_temp_ir|shielded_thermo_hygro |
| 1.7 | 2 | 63 | 4 | 1.16 | 1.53 | 3 | met_station_core|radiometer_basic|surface_temp_ir |
| 1.7 | 3 | 63 | 4 | 1.26 | 1.68 | 3 | radiometer_basic|surface_temp_ir|shielded_thermo_hygro |
| 1.7 | 4 | 63 | 4 | 1.3 | 1.73 | 3 | met_station_core|radiometer_basic|shielded_thermo_hygro |
| 1.7 | 5 | 63 | 4 | 1.32 | 1.76 | 3 | radiometer_basic|surface_temp_ir|ultrasonic_anemometer_hd |
| 1.75 | 1 | 66 | 4 | 1.68 | 2.23 | 4 | met_station_core|radiometer_basic|surface_temp_ir|shielded_thermo_hygro |
| 1.75 | 2 | 66 | 4 | 1.74 | 2.31 | 4 | met_station_core|radiometer_basic|surface_temp_ir|ultrasonic_anemometer_hd |
| 1.75 | 3 | 66 | 4 | 1.16 | 1.53 | 3 | met_station_core|radiometer_basic|surface_temp_ir |
| 1.75 | 4 | 66 | 4 | 1.26 | 1.68 | 3 | radiometer_basic|surface_temp_ir|shielded_thermo_hygro |
| 1.75 | 5 | 66 | 4 | 1.3 | 1.73 | 3 | met_station_core|radiometer_basic|shielded_thermo_hygro |
| 1.5 | 1 | 46 | 3 | 1.16 | 1.53 | 3 | met_station_core|radiometer_basic|surface_temp_ir |
| 1.5 | 2 | 46 | 3 | 1.26 | 1.68 | 3 | radiometer_basic|surface_temp_ir|shielded_thermo_hygro |
| 1.5 | 3 | 46 | 3 | 1.3 | 1.73 | 3 | met_station_core|radiometer_basic|shielded_thermo_hygro |
| 1.5 | 4 | 46 | 3 | 1.32 | 1.76 | 3 | radiometer_basic|surface_temp_ir|ultrasonic_anemometer_hd |
| 1.5 | 5 | 46 | 3 | 1.32 | 1.75 | 3 | met_station_core|surface_temp_ir|shielded_thermo_hygro |
| 1.55 | 1 | 48 | 3 | 1.16 | 1.53 | 3 | met_station_core|radiometer_basic|surface_temp_ir |
| 1.55 | 2 | 48 | 3 | 1.26 | 1.68 | 3 | radiometer_basic|surface_temp_ir|shielded_thermo_hygro |
| 1.55 | 3 | 48 | 3 | 1.3 | 1.73 | 3 | met_station_core|radiometer_basic|shielded_thermo_hygro |
| 1.55 | 4 | 48 | 3 | 1.32 | 1.76 | 3 | radiometer_basic|surface_temp_ir|ultrasonic_anemometer_hd |
| 1.55 | 5 | 48 | 3 | 1.32 | 1.75 | 3 | met_station_core|surface_temp_ir|shielded_thermo_hygro |
| 1.6 | 1 | 53 | 3 | 1.16 | 1.53 | 3 | met_station_core|radiometer_basic|surface_temp_ir |
| 1.6 | 2 | 53 | 3 | 1.26 | 1.68 | 3 | radiometer_basic|surface_temp_ir|shielded_thermo_hygro |
| 1.6 | 3 | 53 | 3 | 1.3 | 1.73 | 3 | met_station_core|radiometer_basic|shielded_thermo_hygro |
| 1.6 | 4 | 53 | 3 | 1.32 | 1.76 | 3 | radiometer_basic|surface_temp_ir|ultrasonic_anemometer_hd |
| 1.6 | 5 | 53 | 3 | 1.32 | 1.75 | 3 | met_station_core|surface_temp_ir|shielded_thermo_hygro |

## Truth Event Statistics
| budget | truth_event_rate_mean | truth_event_rate_std | truth_event_rate_count | truth_storm_rate_mean | truth_storm_rate_std | truth_storm_rate_count | truth_blowing_snow_active_rate_mean | truth_blowing_snow_active_rate_std | truth_blowing_snow_active_rate_count | truth_event_run_count_mean | truth_event_run_count_std | truth_event_run_count_count | truth_event_duration_mean_steps_mean | truth_event_duration_mean_steps_std | truth_event_duration_mean_steps_count | truth_event_duration_median_steps_mean | truth_event_duration_median_steps_std | truth_event_duration_median_steps_count | truth_event_fraction_512_mean_mean | truth_event_fraction_512_mean_std | truth_event_fraction_512_mean_count | truth_event_fraction_512_max_mean | truth_event_fraction_512_max_std | truth_event_fraction_512_max_count | truth_event_fraction_512_gt_0p75_mean | truth_event_fraction_512_gt_0p75_std | truth_event_fraction_512_gt_0p75_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.65 | 0.2892 | 0.0236 | 10 | 0.2892 | 0.0236 | 10 | 0.2892 | 0.0236 | 10 | 486.2 | 37.64 | 10 | 17.84 | 0.1405 | 10 | 18 | 0 | 10 | 0.2895 | 0.02365 | 10 | 0.7945 | 0.006753 | 10 | 0.09138 | 0.02936 | 10 |
| 1.7 | 0.2892 | 0.0236 | 10 | 0.2892 | 0.0236 | 10 | 0.2892 | 0.0236 | 10 | 486.2 | 37.64 | 10 | 17.84 | 0.1405 | 10 | 18 | 0 | 10 | 0.2895 | 0.02365 | 10 | 0.7945 | 0.006753 | 10 | 0.09138 | 0.02936 | 10 |
| 1.75 | 0.2892 | 0.0236 | 10 | 0.2892 | 0.0236 | 10 | 0.2892 | 0.0236 | 10 | 486.2 | 37.64 | 10 | 17.84 | 0.1405 | 10 | 18 | 0 | 10 | 0.2895 | 0.02365 | 10 | 0.7945 | 0.006753 | 10 | 0.09138 | 0.02936 | 10 |

## Policy Behavior Summary
| budget | policy | near_constant_sensors_mean | const_active_sensors_mean | const_off_sensors_mean | switches_per_step_mean | warmup_abort_rate_mean | power_mean_mean | event_rate_rollout_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.65 | aoi | 1 | 0 | 1 | 2.998 | 0.166 | 1.6 | 0.3794 |
| 1.65 | custom_ppo | 5.1 | 1.6 | 3.5 | 0.3054 | 0.02041 | 1.561 | 0.3794 |
| 1.65 | feasible_static_projected | 8 | 3 | 5 | 0.0004883 | 0 | 1.46 | 0.3794 |
| 1.65 | full_open_unconstrained | 8 | 8 | 0 | 0.001302 | 0 | 4.62 | 0.3794 |
| 1.65 | random | 0 | 0 | 0 | 3.004 | 0.2032 | 1.586 | 0.3794 |
| 1.65 | round_robin | 0 | 0 | 0 | 2 | 0.249 | 1.555 | 0.3794 |
| 1.7 | aoi | 1 | 0 | 1 | 3.647 | 0.4189 | 1.623 | 0.3794 |
| 1.7 | custom_ppo | 5.4 | 1.7 | 3.7 | 0.3368 | 0.01681 | 1.559 | 0.3794 |
| 1.7 | feasible_static_projected | 8 | 3 | 5 | 0.0004883 | 0 | 1.46 | 0.3794 |
| 1.7 | full_open_unconstrained | 8 | 8 | 0 | 0.001302 | 0 | 4.62 | 0.3794 |
| 1.7 | random | 0 | 0 | 0 | 3.288 | 0.3189 | 1.605 | 0.3794 |
| 1.7 | round_robin | 0 | 0 | 0 | 2 | 0.249 | 1.555 | 0.3794 |
| 1.75 | aoi | 1 | 0 | 1 | 3 | 0.009766 | 1.668 | 0.3794 |
| 1.75 | custom_ppo | 5.4 | 1.7 | 3.7 | 0.3415 | 0.02179 | 1.546 | 0.3794 |
| 1.75 | feasible_static_projected | 8 | 3 | 5 | 0.0004883 | 0 | 1.46 | 0.3794 |
| 1.75 | full_open_unconstrained | 8 | 8 | 0 | 0.001302 | 0 | 4.62 | 0.3794 |
| 1.75 | random | 0 | 0 | 0 | 3.506 | 0.4352 | 1.623 | 0.3794 |
| 1.75 | round_robin | 0 | 0 | 0 | 2 | 0.249 | 1.555 | 0.3794 |

## Event-Conditioned High-Latency Sensor Use
| budget | policy | sensor | selected_rate_mean | selected_rate_event_mean | selected_rate_non_event_mean | event_selection_lift_mean | active_rate_mean | warming_rate_mean | switches_per_step_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.65 | aoi | fc4_flux | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.65 | aoi | laser_disdrometer | 0.166 | 0.1665 | 0.1658 | 0.0007659 | 0 | 0.166 | 0.332 |
| 1.65 | aoi | snow_particle_counter | 0.834 | 0.8335 | 0.8342 | -0.0007659 | 0.667 | 0.167 | 0.3322 |
| 1.65 | custom_ppo | fc4_flux | 0.004069 | 0.005655 | 0.002784 | 0.002871 | 0.002018 | 0.002051 | 0.001139 |
| 1.65 | custom_ppo | laser_disdrometer | 0.709 | 0.6527 | 0.7425 | -0.08976 | 0.6007 | 0.1083 | 0.08275 |
| 1.65 | custom_ppo | snow_particle_counter | 0.2869 | 0.3416 | 0.2547 | 0.08689 | 0.2453 | 0.04159 | 0.08271 |
| 1.65 | feasible_static_projected | fc4_flux | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.65 | feasible_static_projected | laser_disdrometer | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.65 | feasible_static_projected | snow_particle_counter | 1 | 1 | 1 | 0 | 0.999 | 0.0009766 | 0.0001628 |
| 1.65 | full_open_unconstrained | fc4_flux | 1 | 1 | 1 | 0 | 0.9961 | 0.003906 | 0.0001628 |
| 1.65 | full_open_unconstrained | laser_disdrometer | 1 | 1 | 1 | 0 | 0.9971 | 0.00293 | 0.0001628 |
| 1.65 | full_open_unconstrained | snow_particle_counter | 1 | 1 | 1 | 0 | 0.999 | 0.0009766 | 0.0001628 |
| 1.65 | random | fc4_flux | 0.05742 | 0.05865 | 0.05692 | 0.001726 | 0 | 0.05742 | 0.1086 |
| 1.65 | random | laser_disdrometer | 0.1366 | 0.1376 | 0.1362 | 0.001383 | 9.766e-05 | 0.1365 | 0.2361 |
| 1.65 | random | snow_particle_counter | 0.806 | 0.8037 | 0.8068 | -0.003109 | 0.6472 | 0.1588 | 0.3161 |
| 1.65 | round_robin | fc4_flux | 0.125 | 0.1248 | 0.125 | -0.0001818 | 0 | 0.125 | 0.2498 |
| 1.65 | round_robin | laser_disdrometer | 0.125 | 0.125 | 0.1251 | -0.0001064 | 0 | 0.125 | 0.25 |
| 1.65 | round_robin | snow_particle_counter | 0.75 | 0.7502 | 0.7499 | 0.0002882 | 0.625 | 0.125 | 0.25 |
| 1.7 | aoi | fc4_flux | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.7 | aoi | laser_disdrometer | 0.3282 | 0.3289 | 0.3279 | 0.001011 | 0 | 0.3282 | 0.6563 |
| 1.7 | aoi | snow_particle_counter | 0.6718 | 0.6711 | 0.6721 | -0.001011 | 0.3432 | 0.3286 | 0.6564 |
| 1.7 | custom_ppo | fc4_flux | 0.001628 | 0.002113 | 0.001203 | 0.0009107 | 0.0006022 | 0.001025 | 0.001009 |
| 1.7 | custom_ppo | laser_disdrometer | 0.6905 | 0.6833 | 0.6943 | -0.01106 | 0.5763 | 0.1142 | 0.08268 |
| 1.7 | custom_ppo | snow_particle_counter | 0.3079 | 0.3146 | 0.3045 | 0.01015 | 0.266 | 0.04181 | 0.08327 |
| 1.7 | feasible_static_projected | fc4_flux | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.7 | feasible_static_projected | laser_disdrometer | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.7 | feasible_static_projected | snow_particle_counter | 1 | 1 | 1 | 0 | 0.999 | 0.0009766 | 0.0001628 |
| 1.7 | full_open_unconstrained | fc4_flux | 1 | 1 | 1 | 0 | 0.9961 | 0.003906 | 0.0001628 |
| 1.7 | full_open_unconstrained | laser_disdrometer | 1 | 1 | 1 | 0 | 0.9971 | 0.00293 | 0.0001628 |
| 1.7 | full_open_unconstrained | snow_particle_counter | 1 | 1 | 1 | 0 | 0.999 | 0.0009766 | 0.0001628 |
| 1.7 | random | fc4_flux | 0.1176 | 0.1199 | 0.1163 | 0.003622 | 0 | 0.1176 | 0.2099 |
| 1.7 | random | laser_disdrometer | 0.1918 | 0.1885 | 0.1939 | -0.005405 | 0.002051 | 0.1897 | 0.3068 |
| 1.7 | random | snow_particle_counter | 0.6906 | 0.6916 | 0.6898 | 0.001783 | 0.4762 | 0.2145 | 0.428 |
| 1.7 | round_robin | fc4_flux | 0.125 | 0.1248 | 0.125 | -0.0001818 | 0 | 0.125 | 0.2498 |
| 1.7 | round_robin | laser_disdrometer | 0.125 | 0.125 | 0.1251 | -0.0001064 | 0 | 0.125 | 0.25 |
| 1.7 | round_robin | snow_particle_counter | 0.75 | 0.7502 | 0.7499 | 0.0002882 | 0.625 | 0.125 | 0.25 |
| 1.75 | aoi | fc4_flux | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.75 | aoi | laser_disdrometer | 0.6548 | 0.6569 | 0.6538 | 0.003076 | 0.1562 | 0.4986 | 0.333 |
| 1.75 | aoi | snow_particle_counter | 0.3452 | 0.3431 | 0.3462 | -0.003076 | 0.1782 | 0.167 | 0.3332 |
| 1.75 | custom_ppo | fc4_flux | 0.0004883 | 0.0007749 | 0.0002345 | 0.0005405 | 0.000179 | 0.0003092 | 0.0002279 |
| 1.75 | custom_ppo | laser_disdrometer | 0.5949 | 0.5642 | 0.6153 | -0.05118 | 0.4667 | 0.1281 | 0.09468 |
| 1.75 | custom_ppo | snow_particle_counter | 0.4046 | 0.4351 | 0.3844 | 0.05064 | 0.3571 | 0.04756 | 0.09458 |
| 1.75 | feasible_static_projected | fc4_flux | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.75 | feasible_static_projected | laser_disdrometer | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.75 | feasible_static_projected | snow_particle_counter | 1 | 1 | 1 | 0 | 0.999 | 0.0009766 | 0.0001628 |
| 1.75 | full_open_unconstrained | fc4_flux | 1 | 1 | 1 | 0 | 0.9961 | 0.003906 | 0.0001628 |
| 1.75 | full_open_unconstrained | laser_disdrometer | 1 | 1 | 1 | 0 | 0.9971 | 0.00293 | 0.0001628 |
| 1.75 | full_open_unconstrained | snow_particle_counter | 1 | 1 | 1 | 0 | 0.999 | 0.0009766 | 0.0001628 |
| 1.75 | random | fc4_flux | 0.1722 | 0.1736 | 0.1715 | 0.002135 | 0.0001953 | 0.172 | 0.2904 |
| 1.75 | random | laser_disdrometer | 0.2528 | 0.2519 | 0.2537 | -0.001819 | 0.005371 | 0.2475 | 0.3793 |
| 1.75 | random | snow_particle_counter | 0.575 | 0.5745 | 0.5748 | -0.0003163 | 0.3277 | 0.2473 | 0.4936 |
| 1.75 | round_robin | fc4_flux | 0.125 | 0.1248 | 0.125 | -0.0001818 | 0 | 0.125 | 0.2498 |
| 1.75 | round_robin | laser_disdrometer | 0.125 | 0.125 | 0.1251 | -0.0001064 | 0 | 0.125 | 0.25 |
| 1.75 | round_robin | snow_particle_counter | 0.75 | 0.7502 | 0.7499 | 0.0002882 | 0.625 | 0.125 | 0.25 |

