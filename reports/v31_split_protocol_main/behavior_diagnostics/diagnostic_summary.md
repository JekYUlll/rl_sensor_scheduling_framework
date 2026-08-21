# V3.1 Split-Protocol Behavior Diagnostics

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
| 1.65 | 0.2905 | 0.01127 | 10 | 0.2905 | 0.01127 | 10 | 0.2905 | 0.01127 | 10 | 1466 | 52.22 | 10 | 17.83 | 0.1084 | 10 | 17.9 | 0.3162 | 10 | 0.2902 | 0.0112 | 10 | 0.798 | 0.004802 | 10 | 0.08171 | 0.01925 | 10 |
| 1.7 | 0.2905 | 0.01127 | 10 | 0.2905 | 0.01127 | 10 | 0.2905 | 0.01127 | 10 | 1466 | 52.22 | 10 | 17.83 | 0.1084 | 10 | 17.9 | 0.3162 | 10 | 0.2902 | 0.0112 | 10 | 0.798 | 0.004802 | 10 | 0.08171 | 0.01925 | 10 |
| 1.75 | 0.2905 | 0.01127 | 10 | 0.2905 | 0.01127 | 10 | 0.2905 | 0.01127 | 10 | 1466 | 52.22 | 10 | 17.83 | 0.1084 | 10 | 17.9 | 0.3162 | 10 | 0.2902 | 0.0112 | 10 | 0.798 | 0.004802 | 10 | 0.08171 | 0.01925 | 10 |

## Policy Behavior Summary
| budget | policy | near_constant_sensors_mean | const_active_sensors_mean | const_off_sensors_mean | switches_per_step_mean | warmup_abort_rate_mean | power_mean_mean | event_rate_rollout_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.65 | aoi | 1 | 0 | 1 | 2.998 | 0.166 | 1.6 | 0.2776 |
| 1.65 | custom_ppo | 4.2 | 1.1 | 3.1 | 0.3693 | 0.02507 | 1.521 | 0.2776 |
| 1.65 | feasible_static_projected | 8 | 3 | 5 | 0.0004883 | 0 | 1.46 | 0.2776 |
| 1.65 | full_open_unconstrained | 8 | 8 | 0 | 0.001302 | 0 | 4.62 | 0.2776 |
| 1.65 | random | 0 | 0 | 0 | 3.004 | 0.2032 | 1.586 | 0.2776 |
| 1.65 | round_robin | 0 | 0 | 0 | 2 | 0.249 | 1.555 | 0.2776 |
| 1.65 | validation_selected_static | 8 | 3 | 5 | 0.0004883 | 0 | 1.46 | 0.2776 |
| 1.7 | aoi | 1 | 0 | 1 | 3.646 | 0.416 | 1.623 | 0.2776 |
| 1.7 | custom_ppo | 5 | 1.5 | 3.5 | 0.387 | 0.01945 | 1.533 | 0.2776 |
| 1.7 | feasible_static_projected | 8 | 3 | 5 | 0.0004883 | 0 | 1.46 | 0.2776 |
| 1.7 | full_open_unconstrained | 8 | 8 | 0 | 0.001302 | 0 | 4.62 | 0.2776 |
| 1.7 | random | 0 | 0 | 0 | 3.288 | 0.3189 | 1.605 | 0.2776 |
| 1.7 | round_robin | 0 | 0 | 0 | 2 | 0.249 | 1.555 | 0.2776 |
| 1.7 | validation_selected_static | 8 | 3 | 5 | 0.0004883 | 0 | 1.488 | 0.2776 |
| 1.75 | aoi | 1 | 0 | 1 | 3 | 0.009766 | 1.668 | 0.2776 |
| 1.75 | custom_ppo | 5.2 | 1.6 | 3.6 | 0.2567 | 0.01828 | 1.514 | 0.2776 |
| 1.75 | feasible_static_projected | 8 | 3 | 5 | 0.0004883 | 0 | 1.46 | 0.2776 |
| 1.75 | full_open_unconstrained | 8 | 8 | 0 | 0.001302 | 0 | 4.62 | 0.2776 |
| 1.75 | random | 0 | 0 | 0 | 3.506 | 0.4352 | 1.623 | 0.2776 |
| 1.75 | round_robin | 0 | 0 | 0 | 2 | 0.249 | 1.555 | 0.2776 |
| 1.75 | validation_selected_static | 8 | 3 | 5 | 0.0004883 | 0 | 1.488 | 0.2776 |

## Event-Conditioned High-Latency Sensor Use
| budget | policy | sensor | selected_rate_mean | selected_rate_event_mean | selected_rate_non_event_mean | event_selection_lift_mean | active_rate_mean | warming_rate_mean | switches_per_step_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.65 | aoi | fc4_flux | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.65 | aoi | laser_disdrometer | 0.166 | 0.1663 | 0.1659 | 0.0003255 | 0 | 0.166 | 0.332 |
| 1.65 | aoi | snow_particle_counter | 0.834 | 0.8337 | 0.8341 | -0.0003255 | 0.667 | 0.167 | 0.3322 |
| 1.65 | custom_ppo | fc4_flux | 8.138e-05 | 0.0002664 | 0 | 0.0002664 | 1.628e-05 | 6.51e-05 | 3.255e-05 |
| 1.65 | custom_ppo | laser_disdrometer | 0.4252 | 0.4262 | 0.422 | 0.004239 | 0.3183 | 0.1069 | 0.08711 |
| 1.65 | custom_ppo | snow_particle_counter | 0.5748 | 0.5735 | 0.578 | -0.004505 | 0.5309 | 0.04383 | 0.08714 |
| 1.65 | feasible_static_projected | fc4_flux | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.65 | feasible_static_projected | laser_disdrometer | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.65 | feasible_static_projected | snow_particle_counter | 1 | 1 | 1 | 0 | 0.999 | 0.0009766 | 0.0001628 |
| 1.65 | full_open_unconstrained | fc4_flux | 1 | 1 | 1 | 0 | 0.9961 | 0.003906 | 0.0001628 |
| 1.65 | full_open_unconstrained | laser_disdrometer | 1 | 1 | 1 | 0 | 0.9971 | 0.00293 | 0.0001628 |
| 1.65 | full_open_unconstrained | snow_particle_counter | 1 | 1 | 1 | 0 | 0.999 | 0.0009766 | 0.0001628 |
| 1.65 | random | fc4_flux | 0.05742 | 0.0578 | 0.05738 | 0.0004165 | 0 | 0.05742 | 0.1086 |
| 1.65 | random | laser_disdrometer | 0.1366 | 0.1353 | 0.1371 | -0.001831 | 9.766e-05 | 0.1365 | 0.2361 |
| 1.65 | random | snow_particle_counter | 0.806 | 0.8069 | 0.8055 | 0.001414 | 0.6472 | 0.1588 | 0.3161 |
| 1.65 | round_robin | fc4_flux | 0.125 | 0.1238 | 0.1255 | -0.001683 | 0 | 0.125 | 0.2498 |
| 1.65 | round_robin | laser_disdrometer | 0.125 | 0.1256 | 0.1248 | 0.0007989 | 0 | 0.125 | 0.25 |
| 1.65 | round_robin | snow_particle_counter | 0.75 | 0.7506 | 0.7497 | 0.0008837 | 0.625 | 0.125 | 0.25 |
| 1.65 | validation_selected_static | fc4_flux | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.65 | validation_selected_static | laser_disdrometer | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.65 | validation_selected_static | snow_particle_counter | 1 | 1 | 1 | 0 | 0.999 | 0.0009766 | 0.0001628 |
| 1.7 | aoi | fc4_flux | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.7 | aoi | laser_disdrometer | 0.328 | 0.3289 | 0.3277 | 0.001188 | 0 | 0.328 | 0.6559 |
| 1.7 | aoi | snow_particle_counter | 0.672 | 0.6711 | 0.6723 | -0.001188 | 0.3436 | 0.3285 | 0.656 |
| 1.7 | custom_ppo | fc4_flux | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.7 | custom_ppo | laser_disdrometer | 0.5078 | 0.502 | 0.5084 | -0.00641 | 0.3899 | 0.1179 | 0.08896 |
| 1.7 | custom_ppo | snow_particle_counter | 0.4922 | 0.498 | 0.4916 | 0.00641 | 0.4474 | 0.04476 | 0.08896 |
| 1.7 | feasible_static_projected | fc4_flux | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.7 | feasible_static_projected | laser_disdrometer | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.7 | feasible_static_projected | snow_particle_counter | 1 | 1 | 1 | 0 | 0.999 | 0.0009766 | 0.0001628 |
| 1.7 | full_open_unconstrained | fc4_flux | 1 | 1 | 1 | 0 | 0.9961 | 0.003906 | 0.0001628 |
| 1.7 | full_open_unconstrained | laser_disdrometer | 1 | 1 | 1 | 0 | 0.9971 | 0.00293 | 0.0001628 |
| 1.7 | full_open_unconstrained | snow_particle_counter | 1 | 1 | 1 | 0 | 0.999 | 0.0009766 | 0.0001628 |
| 1.7 | random | fc4_flux | 0.1176 | 0.1174 | 0.1179 | -0.0005311 | 0 | 0.1176 | 0.2099 |
| 1.7 | random | laser_disdrometer | 0.1918 | 0.1929 | 0.1912 | 0.001631 | 0.002051 | 0.1897 | 0.3068 |
| 1.7 | random | snow_particle_counter | 0.6906 | 0.6898 | 0.6909 | -0.0011 | 0.4762 | 0.2145 | 0.428 |
| 1.7 | round_robin | fc4_flux | 0.125 | 0.1238 | 0.1255 | -0.001683 | 0 | 0.125 | 0.2498 |
| 1.7 | round_robin | laser_disdrometer | 0.125 | 0.1256 | 0.1248 | 0.0007989 | 0 | 0.125 | 0.25 |
| 1.7 | round_robin | snow_particle_counter | 0.75 | 0.7506 | 0.7497 | 0.0008837 | 0.625 | 0.125 | 0.25 |
| 1.7 | validation_selected_static | fc4_flux | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.7 | validation_selected_static | laser_disdrometer | 0.2 | 0.2 | 0.2 | 0 | 0.1994 | 0.0005859 | 3.255e-05 |
| 1.7 | validation_selected_static | snow_particle_counter | 0.8 | 0.8 | 0.8 | 0 | 0.7992 | 0.0007813 | 0.0001302 |
| 1.75 | aoi | fc4_flux | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.75 | aoi | laser_disdrometer | 0.6548 | 0.6547 | 0.6548 | -8.693e-05 | 0.1562 | 0.4986 | 0.333 |
| 1.75 | aoi | snow_particle_counter | 0.3452 | 0.3453 | 0.3452 | 8.693e-05 | 0.1782 | 0.167 | 0.3331 |
| 1.75 | custom_ppo | fc4_flux | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.75 | custom_ppo | laser_disdrometer | 0.3823 | 0.3971 | 0.3794 | 0.01773 | 0.2777 | 0.1046 | 0.07882 |
| 1.75 | custom_ppo | snow_particle_counter | 0.6177 | 0.6029 | 0.6206 | -0.01773 | 0.578 | 0.03978 | 0.07886 |
| 1.75 | feasible_static_projected | fc4_flux | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.75 | feasible_static_projected | laser_disdrometer | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.75 | feasible_static_projected | snow_particle_counter | 1 | 1 | 1 | 0 | 0.999 | 0.0009766 | 0.0001628 |
| 1.75 | full_open_unconstrained | fc4_flux | 1 | 1 | 1 | 0 | 0.9961 | 0.003906 | 0.0001628 |
| 1.75 | full_open_unconstrained | laser_disdrometer | 1 | 1 | 1 | 0 | 0.9971 | 0.00293 | 0.0001628 |
| 1.75 | full_open_unconstrained | snow_particle_counter | 1 | 1 | 1 | 0 | 0.999 | 0.0009766 | 0.0001628 |
| 1.75 | random | fc4_flux | 0.1722 | 0.1727 | 0.1722 | 0.0004116 | 0.0001953 | 0.172 | 0.2904 |
| 1.75 | random | laser_disdrometer | 0.2528 | 0.2553 | 0.2517 | 0.003562 | 0.005371 | 0.2475 | 0.3793 |
| 1.75 | random | snow_particle_counter | 0.575 | 0.5721 | 0.576 | -0.003974 | 0.3277 | 0.2473 | 0.4936 |
| 1.75 | round_robin | fc4_flux | 0.125 | 0.1238 | 0.1255 | -0.001683 | 0 | 0.125 | 0.2498 |
| 1.75 | round_robin | laser_disdrometer | 0.125 | 0.1256 | 0.1248 | 0.0007989 | 0 | 0.125 | 0.25 |
| 1.75 | round_robin | snow_particle_counter | 0.75 | 0.7506 | 0.7497 | 0.0008837 | 0.625 | 0.125 | 0.25 |
| 1.75 | validation_selected_static | fc4_flux | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.75 | validation_selected_static | laser_disdrometer | 0.2 | 0.2 | 0.2 | 0 | 0.1994 | 0.0005859 | 3.255e-05 |
| 1.75 | validation_selected_static | snow_particle_counter | 0.8 | 0.8 | 0.8 | 0 | 0.7992 | 0.0007813 | 0.0001302 |

