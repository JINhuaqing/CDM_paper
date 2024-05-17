from easydict import EasyDict as edict

simu_settings = edict()

simu_settings.setting1 = edict({
    "rho": 0, 
    "is_homo": True, 
    "n": 10000, 
    "ntest": 1000,
    "err_type": "norm",
})

simu_settings.setting2 = edict({
    "rho": 0.0, 
    "is_homo": True, 
    "n": 10000, 
    "ntest": 1000,
    "err_type": "t",
})


simu_settings.setting3 = edict({
    "rho": 0.0, 
    "is_homo": True, 
    "n": 10000, 
    "ntest": 1000,
    "err_type": "nonlocal",
})

simu_settings.setting4 = edict({
    "rho": 0, 
    "is_homo": False, 
    "n": 10000, 
    "ntest": 1000,
    "err_type": "norm",
})


simu_settings.setting5 = edict({
    "rho": 0, 
    "is_homo": False, 
    "n": 10000, 
    "ntest": 1000,
    "err_type": "gamma",
})

simu_settings.setting6 = edict({
    "rho": 0.0, 
    "is_homo": False, 
    "n": 10000, 
    "ntest": 1000,
    "err_type": "nonlocal",
})
