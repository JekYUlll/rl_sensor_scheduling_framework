# V3.1 S2 Behavior Diagnostics

## Sensor Configuration
| order | sensor_id | power_cost | startup_peak_power | warmup_steps | variables |
| --- | --- | --- | --- | --- | --- |
| 0 | met_station_core | 0.22 | 0.3 | 0 | wind_speed_ms|wind_direction_deg|air_temperature_c|relative_humidity|air_pressure_pa |
| 1 | radiometer_basic | 0.14 | 0.2 | 0 | solar_radiation_wm2 |
| 2 | surface_temp_ir | 0.16 | 0.24 | 1 | snow_surface_temperature_c |
| 3 | ultrasonic_anemometer_hd | 0.52 | 0.72 | 1 | wind_speed_ms|wind_direction_deg |
| 4 | shielded_thermo_hygro | 0.44 | 0.6 | 1 | air_temperature_c|relative_humidity |
| 5 | snow_particle_counter | 0.78 | 1.02 | 2 | snow_particle_mean_diameter_mm|snow_particle_mean_velocity_ms |
| 6 | laser_disdrometer | 1.24 | 1.72 | 4 | snow_particle_mean_diameter_mm|snow_particle_mean_velocity_ms |
| 7 | fc4_flux | 1.46 | 2.02 | 5 | snow_mass_flux_kg_m2_s |

## Feasible Subset Capacity
| budget | rank | feasible_count | max_feasible_size | steady_power | startup_peak_power | sensor_count | sensor_ids |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.7 | 1 | 64 | 4 | 0.96 | 1.34 | 4 | met_station_core|radiometer_basic|surface_temp_ir|shielded_thermo_hygro |
| 1.7 | 2 | 64 | 4 | 1.04 | 1.46 | 4 | met_station_core|radiometer_basic|surface_temp_ir|ultrasonic_anemometer_hd |
| 1.7 | 3 | 64 | 4 | 1.26 | 1.76 | 4 | radiometer_basic|surface_temp_ir|ultrasonic_anemometer_hd|shielded_thermo_hygro |
| 1.7 | 4 | 64 | 4 | 1.3 | 1.76 | 4 | met_station_core|radiometer_basic|surface_temp_ir|snow_particle_counter |
| 1.7 | 5 | 64 | 4 | 1.32 | 1.82 | 4 | met_station_core|radiometer_basic|ultrasonic_anemometer_hd|shielded_thermo_hygro |
| 1.5 | 1 | 50 | 4 | 0.96 | 1.34 | 4 | met_station_core|radiometer_basic|surface_temp_ir|shielded_thermo_hygro |
| 1.5 | 2 | 50 | 4 | 1.04 | 1.46 | 4 | met_station_core|radiometer_basic|surface_temp_ir|ultrasonic_anemometer_hd |
| 1.5 | 3 | 50 | 4 | 1.26 | 1.76 | 4 | radiometer_basic|surface_temp_ir|ultrasonic_anemometer_hd|shielded_thermo_hygro |
| 1.5 | 4 | 50 | 4 | 1.3 | 1.76 | 4 | met_station_core|radiometer_basic|surface_temp_ir|snow_particle_counter |
| 1.5 | 5 | 50 | 4 | 1.32 | 1.82 | 4 | met_station_core|radiometer_basic|ultrasonic_anemometer_hd|shielded_thermo_hygro |
| 1.55 | 1 | 53 | 4 | 0.96 | 1.34 | 4 | met_station_core|radiometer_basic|surface_temp_ir|shielded_thermo_hygro |
| 1.55 | 2 | 53 | 4 | 1.04 | 1.46 | 4 | met_station_core|radiometer_basic|surface_temp_ir|ultrasonic_anemometer_hd |
| 1.55 | 3 | 53 | 4 | 1.26 | 1.76 | 4 | radiometer_basic|surface_temp_ir|ultrasonic_anemometer_hd|shielded_thermo_hygro |
| 1.55 | 4 | 53 | 4 | 1.3 | 1.76 | 4 | met_station_core|radiometer_basic|surface_temp_ir|snow_particle_counter |
| 1.55 | 5 | 53 | 4 | 1.32 | 1.82 | 4 | met_station_core|radiometer_basic|ultrasonic_anemometer_hd|shielded_thermo_hygro |
| 1.6 | 1 | 58 | 4 | 0.96 | 1.34 | 4 | met_station_core|radiometer_basic|surface_temp_ir|shielded_thermo_hygro |
| 1.6 | 2 | 58 | 4 | 1.04 | 1.46 | 4 | met_station_core|radiometer_basic|surface_temp_ir|ultrasonic_anemometer_hd |
| 1.6 | 3 | 58 | 4 | 1.26 | 1.76 | 4 | radiometer_basic|surface_temp_ir|ultrasonic_anemometer_hd|shielded_thermo_hygro |
| 1.6 | 4 | 58 | 4 | 1.3 | 1.76 | 4 | met_station_core|radiometer_basic|surface_temp_ir|snow_particle_counter |
| 1.6 | 5 | 58 | 4 | 1.32 | 1.82 | 4 | met_station_core|radiometer_basic|ultrasonic_anemometer_hd|shielded_thermo_hygro |

## Truth Event Statistics
| budget | truth_event_rate_mean | truth_event_rate_std | truth_event_rate_count | truth_storm_rate_mean | truth_storm_rate_std | truth_storm_rate_count | truth_blowing_snow_active_rate_mean | truth_blowing_snow_active_rate_std | truth_blowing_snow_active_rate_count | truth_event_run_count_mean | truth_event_run_count_std | truth_event_run_count_count | truth_event_duration_mean_steps_mean | truth_event_duration_mean_steps_std | truth_event_duration_mean_steps_count | truth_event_duration_median_steps_mean | truth_event_duration_median_steps_std | truth_event_duration_median_steps_count | truth_event_fraction_512_mean_mean | truth_event_fraction_512_mean_std | truth_event_fraction_512_mean_count | truth_event_fraction_512_max_mean | truth_event_fraction_512_max_std | truth_event_fraction_512_max_count | truth_event_fraction_512_gt_0p75_mean | truth_event_fraction_512_gt_0p75_std | truth_event_fraction_512_gt_0p75_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.7 | 0.2701 |  | 1 | 0.2701 |  | 1 | 0.2701 |  | 1 | 454 |  | 1 | 17.85 |  | 1 | 18 |  | 1 | 0.2657 |  | 1 | 0.8008 |  | 1 | 0.06897 |  | 1 |

## Policy Behavior Summary
| budget | policy | near_constant_sensors_mean | const_active_sensors_mean | const_off_sensors_mean | switches_per_step_mean | warmup_abort_rate_mean | power_mean_mean | event_rate_rollout_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.7 | aoi | 3 | 1 | 2 | 2.997 | 0 | 1.595 | 0.4085 |
| 1.7 | custom_ppo | 1 | 0 | 1 | 2.049 | 0.06641 | 1.552 | 0.4085 |
| 1.7 | feasible_static_projected | 8 | 4 | 4 | 0.000651 | 0 | 1.3 | 0.4085 |
| 1.7 | full_open_unconstrained | 8 | 8 | 0 | 0.001302 | 0 | 4.96 | 0.4085 |
| 1.7 | random | 1 | 0 | 1 | 2.869 | 0.167 | 1.589 | 0.4085 |
| 1.7 | round_robin | 2 | 1 | 1 | 1.751 | 0.125 | 1.5 | 0.4085 |

## Event-Conditioned High-Latency Sensor Use
| budget | policy | sensor | selected_rate_mean | selected_rate_event_mean | selected_rate_non_event_mean | event_selection_lift_mean | active_rate_mean | warming_rate_mean | switches_per_step_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.7 | aoi | fc4_flux | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.7 | aoi | laser_disdrometer | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.7 | aoi | snow_particle_counter | 1 | 1 | 1 | 0 | 0.999 | 0.0009766 | 0.0001628 |
| 1.7 | custom_ppo | fc4_flux | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.7 | custom_ppo | laser_disdrometer | 0.07633 | 0.06414 | 0.08476 | -0.02061 | 0.002767 | 0.07357 | 0.1159 |
| 1.7 | custom_ppo | snow_particle_counter | 0.9237 | 0.9359 | 0.9152 | 0.02061 | 0.8649 | 0.05876 | 0.116 |
| 1.7 | feasible_static_projected | fc4_flux | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.7 | feasible_static_projected | laser_disdrometer | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.7 | feasible_static_projected | snow_particle_counter | 1 | 1 | 1 | 0 | 0.999 | 0.0009766 | 0.0001628 |
| 1.7 | full_open_unconstrained | fc4_flux | 1 | 1 | 1 | 0 | 0.9961 | 0.003906 | 0.0001628 |
| 1.7 | full_open_unconstrained | laser_disdrometer | 1 | 1 | 1 | 0 | 0.9971 | 0.00293 | 0.0001628 |
| 1.7 | full_open_unconstrained | snow_particle_counter | 1 | 1 | 1 | 0 | 0.999 | 0.0009766 | 0.0001628 |
| 1.7 | random | fc4_flux | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.7 | random | laser_disdrometer | 0.1729 | 0.1821 | 0.1665 | 0.01559 | 0 | 0.1729 | 0.2871 |
| 1.7 | random | snow_particle_counter | 0.8271 | 0.8179 | 0.8335 | -0.01559 | 0.6826 | 0.1445 | 0.2873 |
| 1.7 | round_robin | fc4_flux | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.7 | round_robin | laser_disdrometer | 0.125 | 0.1247 | 0.1252 | -0.0005052 | 0 | 0.125 | 0.25 |
| 1.7 | round_robin | snow_particle_counter | 0.875 | 0.8753 | 0.8748 | 0.0005052 | 0.749 | 0.126 | 0.2502 |

