

D_piperacillin = 0.88 * 10**(-6) # cm^2/s
T_ampicillin = 37 + 273.15 # K
T_piperacillin = 20 + 273.15 # K
DynViscosity_37 = 0.691 # mPa.s
DynViscosity_20 = 1.002 # mPa.s
MW_piperacillin = 517.5 # g/mol # 
MW_ampicillin = 349.4 # g/mol



D_ampicillin = D_piperacillin * (T_ampicillin/T_piperacillin) * (DynViscosity_20 / DynViscosity_37) * ( MW_piperacillin/MW_ampicillin) ** (1/3)
print("Diffusion coefficient of ampicillin at 37C: ", D_ampicillin, "cm^2/s")
