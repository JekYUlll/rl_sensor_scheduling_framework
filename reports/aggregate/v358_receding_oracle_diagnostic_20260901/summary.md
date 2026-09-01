# V358 receding-oracle diagnostic

Four independent V357 scenes were replayed with an 8-step receding oracle over the same feasible action geometry.
Lower forecast loss is better. The oracle is a privileged structural diagnostic and is not a deployable policy.

| Seed | PD-PPO | Best static | Receding oracle | Oracle - PD-PPO | Oracle - static | Oracle action coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 6901 | 0.433684 | 0.393336 | 0.332579 | -0.101105 | -0.060757 | 22 |
| 6902 | 0.409496 | 0.349566 | 0.290917 | -0.118579 | -0.058649 | 22 |
| 6903 | 0.411990 | 0.357579 | 0.283921 | -0.128069 | -0.073658 | 22 |
| 6904 | 0.405754 | 0.397784 | 0.343437 | -0.062318 | -0.054347 | 22 |

Mean ordinary loss: PD-PPO 0.415231; best static 0.374566; receding oracle 0.312713.
Oracle beats best static on ordinary loss in 4/4 scenes and beats PD-PPO in 4/4 scenes.
The diagnostic establishes available dynamic value in these scenes; it does not establish that PD-PPO has learned it.
