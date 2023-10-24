import pandas as pd
import numpy as np
from numpy import pi
import xarray as xr

wind_speed = pd.read_excel(r'Standard-actions-v1.xlsx',\
                            sheet_name='AS1288-2021-A.2-Windspeed', \
                            skiprows=[0])
wind_speed = wind_speed.fillna(method='ffill').set_index(['cat_prefix','wind_cat'])

pressure_coef = pd.read_excel(r'Standard-actions-v1.xlsx',\
                            sheet_name='AS1288-2021-A.2-KcCp,n', \
                            skiprows=[0])

df_Cp = pressure_coef.set_index(['structure','direction','s_edge','limit_state'])
df_Cp.columns.name = 'cat_prefix'
df_Cp = df_Cp.stack()
df_Cp.name = 'KcCpn'
ds_Cp = df_Cp.to_xarray()
#df_Cp.to_frame().reorder_levels(['cat_prefix','structure','direction','s_edge','limit_state']).sort_index()

wind_speed.columns.name = 'limit_state'
df_v = wind_speed.stack()
df_v.name = 'v'
ds_v = df_v.to_xarray()
rho_air = 1.2
ds_P = 0.5*rho_air*ds_v**2*ds_Cp/1e3 #kPa
ds_P.name='pressure'

t_glass_tolerance = [[3,4,5,6,8,10,12,15,19,25],[0.2,0.2,0.2,0.2,0.3,0.3,0.3,0.5,1,1.5]]
t_tol = lambda t: np.interp(t,*t_glass_tolerance)
#t_wtol = lambda t: mcerp.Uniform(t-t_tol(t),t+t_tol(t))
t_wtol = lambda t: t-t_tol(t)

def s_glass_capacity(t,edgeQ=False, glass_type='Toughened',surface_type='Untreated',duration_type='Short'):
    '''t=glass thickness in mm, returns ultimate design capacity in MPa'''
    # Glass type: c1 = (f't + min.surface compressive stress)/f't
    c1 = {'Ordinary annealed':1,'Heat-strengthed':1.6,'Toughened':2.5,'Wired':0.5}
    # Surface type: use the minimum glass thickness of the pattern
    c2 = {'Untreated':1,'Sand-blasted':0.4,'Acid-etched':1,'Patterned':1}
    # Duraction factorShort --> 3s, Medium --> 10min, Long --> 1year
    c3 = {'Short':1,'Medium':1,'Medium-annealed':0.72,'Long':0.5,'Long-annealed':0.31}
    phi = 0.67
    c = phi*c1[glass_type]*c2[surface_type]*c3[duration_type]

    if edgeQ:
        return c*(-7.88*np.log(t_wtol(t))+57.07)
    else:
        return c*(-9.85*np.log(t_wtol(t))+71.34)

def Pmax_strength(h,t,M_prime_max=None):
    '''Balustrade/cantilevered panel
       M_prime_max=Max bending moment per unit length of channel in Nm/m
       t=glass thickness in mm
       h=glass height in mm
       return max permissible ULS pressure in kPa'''
    if M_prime_max:
        return M_prime_max*2/(h*1e-3)**2*1e-3
    else:
        return s_glass_capacity(t)/3/(h/(t_wtol(t)))**2*1e3

def Lambdamax_strength(h,t,M_prime_max=None):
    '''Balustrade/cantilevered panel
       M_prime_max=Max bending moment per unit length of channel in Nm/m
       t=glass thickness in mm
       h=glass height in mm
       return max permissible ULS line load in kN/m'''
    if M_prime_max:
        return M_prime_max/h
    else:
        return Pmax_strength(h,t,M_prime_max=None)*h*1e-3/2

def alt2pressure(z):
    L = 0.00976
    g = 9.8
    M = 0.02896968
    R0 = 8.314462618
    T0 = 288.16
    p0 = 101325
    return p0*(1-L*z/T0)**(g*M/R0/L)

def k_pane(tx,tlist):
    return 1.25*tx**3/np.sum(np.array(tlist)**3)

def navier_xr(x,nu=0.3,N=20):
    '''
    Scenario 1a: UDL, SSSS
    m,n = 1,3,5...
    x = a/b
    returns alpha, beta in Roark's format
    alpha_x/alpha_y gives gradients in x and y direction
    '''
    ns = np.arange(1,N,2)
    dim = tuple(np.ones(len(x.shape),dtype=int))
    nn,mm = [xr.DataArray(a,dims=['i','j']) for a in np.meshgrid(ns,ns)]
    alpha_mns = (-1)**((nn+mm)/2-1)/(mm*nn*((mm/x)**2+nn**2)**2)
    betay_mns = alpha_mns*((nn*pi)**2+nu*(mm*pi/x)**2)
    betax_mns = alpha_mns*(nu*(nn*pi)**2+(mm*pi/x)**2)
    alpha_x_mns = alpha_mns*mm*pi/x
    alpha_y_mns = alpha_mns*nn*pi
    output={'alpha':alpha_mns.sum(axis=(0,1))*16*12*(1-nu**2)/np.pi**6, 
            'betay':betay_mns.sum(axis=(0,1))*16*6/pi**6, 
            'betax':betax_mns.sum(axis=(0,1))*16*6/pi**6,
            'alpha_x':alpha_x_mns.sum(axis=(0,1))*16*12*(1-nu**2)/np.pi**6,
            'alpha_y':alpha_y_mns.sum(axis=(0,1))*16*12*(1-nu**2)/np.pi**6}
    output['beta']=output['betay']*(x>1)+output['betax']*(x<=1)
    return output

def navier_pointload_xr(x,nu=0.3,N=20):
    '''
    Scenario 1b: Concentrated load P, SSSS
    m,n = 1,3,5...
    x = a/b
    returns alpha in Roark's format    
    '''
    ns = np.arange(1,N,2)
    dim = tuple(np.ones(len(x.shape),dtype=int))
    nn,mm = [xr.DataArray(a,dims=['i','j']) for a in np.meshgrid(ns,ns)]
    alpha_mns = 1/((mm/x)**2+nn**2)**2
    output={'alpha':alpha_mns.sum(axis=(0,1))*12*(1-nu**2)*4/np.pi**4/x}
    return output

def structural_analysis(q,a,b,t,type='1a',E=70e9):
    AR = a/b
    SR = b/t
    alpha, beta, alpha_x, alpha_y = [navier_xr(AR,nu=0.22)[coef] for coef in ['alpha','beta','alpha_x','alpha_y']]
    #y = alpha*q*SR**3*b/E
    return {'Smax': beta*q*SR**2,
            'ymax': alpha*q*SR**3*b/E,
            'theta_x': np.arctan(alpha_x*q*SR**3/E)*180/pi,
            'theta_y': np.arctan(alpha_y*q*SR**3/E)*180/pi}

def dgu_handling_deflection(acc,a,b,t,E=70e9,nu=0.22, rho=2500):
    '''
    acc = acceleration (m/s^2)
    units in m, kg, s, Pa
    '''
    m1 = rho*a*b*t
    P = m1*2*acc
    AR = a/b
    SR = b/t
    alpha_pointload = navier_pointload_xr(AR,nu=nu,N=50)['alpha']
    ymax_pointload = alpha_pointload*P*b**2/(E*t**3)
    
    q = rho*acc*t
    alpha_inertia = navier_xr(AR,nu=nu,N=50)['alpha']
    ymax_inertia = alpha_inertia*q*SR**3*b/E
    return xr.Dataset({'pointload':ymax_pointload, 
                       'inertia':ymax_inertia, 
                       'panel1':ymax_pointload + ymax_inertia,
                       'mass':m1,
                       'force':P})
# Parameters
q_W = ds_P*1e3
a = np.arange(100,4100,100)
b = np.arange(100,4100,100)
t = np.array([4,5,6,8,10,12,15])
t_spacer = np.arange(6,22,2)
a,b,t,t_spacer = [xr.DataArray(x,dims=[k],coords={k:x}) for k,x in zip(['a','b','t','t_spacer'],[a,b,t,t_spacer])]
coef_G = xr.DataArray(np.array([[[1,1,0,0],
                                 [0,0,0,0]],
                                [[1.2,0.9,0,0],
                                 [0,0,0,0]]]),
                      dims=['limit_state','structure','direction'],
                      coords={'limit_state':['SLS','ULS'],
                              'structure':['Roof','Wall'],
                              'direction':['Down','Up','In','Out']})
k_pane_eq = k_pane(6,[6,6])
rho=2500
g=9.8
q_G = t*g*rho/1e3
q = q_G*coef_G + q_W*k_pane_eq
q.name='pressure'

# Altitude Pressure
z = np.array([10,20,40,80,160,320,640,1280])
z = xr.DataArray(z,dims=['z'],coords={'z':z})
dp = alt2pressure(7)-alt2pressure(z)
dp.name='pressure'
result_dp = xr.Dataset(structural_analysis(dp,a/1e3,b/1e3,t/1e3))
result_dp['ymax'] = np.abs(result_dp.ymax*1000)
result_dp['Sratio'] = (result_dp['Smax']/s_glass_capacity(t)/1e6)
result_dp['ymax'].sel(z=1280,t=4)

# Wind Pressure
def make_xarray(k,x):
    if type(x) in [float, int]:
        return xr.DataArray(x,dims=[k],coords={k:[x]})
    elif type(x) == np.ndarray and x.dtype in ['int32','float64']:
        return xr.DataArray(x,dims=[k],coords={k:x})
    else:
        return x
def wind_pressure_deflection(a,b):
    '''
    a, b in mm
    '''
    a,b = [make_xarray(k,x) for k,x in zip(['a','b'],[a,b])]
    result = xr.Dataset(structural_analysis(q,a/1e3,b/1e3,t/1e3))
    yratio_spacer_max = xr.DataArray([1/3,1/2,1],dims=['spacer_ratio'],coords={'spacer_ratio':[1/3,1/2,1]})
    smax = np.abs(result.sel(limit_state='ULS').Smax/1e6).max(['direction','s_edge'])
    sratio = np.abs(result.sel(limit_state='ULS').Smax/1e6).max(['direction','s_edge'])/s_glass_capacity(t)
    yratio_span60 = (np.abs(result.sel(limit_state='SLS').ymax)/(xr.where(a>b,b,a)/1e3/60)).max(['direction','s_edge'])
    yratio_spacer = (np.abs(result.sel(limit_state='SLS').ymax)/(t_spacer/1e3)).max(['direction','s_edge'])
    yratio_combined = xr.where(yratio_span60<(yratio_spacer/yratio_spacer_max),yratio_spacer/yratio_spacer_max,yratio_span60)
    ymax = (np.abs(result.sel(limit_state='SLS').ymax*1000)).max(['direction','s_edge'])
    sratio.name = 'Sratio'
    yratio_span60.name = 'yratio_span60'
    yratio_spacer.name = 'yratio_spacer'
    yratio_combined.name = 'yratio_combined'
    ymax.name = 'ymax'
    result = xr.Dataset({'Sratio':sratio.drop('limit_state'),
                         'Smax':smax.drop('limit_state'),
                         'yratio_spacer':yratio_spacer.drop('limit_state'),
                         'yratio_span60':yratio_span60.drop('limit_state'),
                         'yratio_combined':yratio_combined.drop('limit_state'),
                         'ymax':ymax.drop('limit_state')})
    return result

def wind_pressure_deflection_summary(a,b):
    return wind_pressure_deflection(a,b).sel(
        cat_prefix=['N'],
        wind_cat=['N1', 'N2', 'N3', 'N4', 'N5', 'N6'],
        structure='Wall'
    ).max(
        ['t_spacer','spacer_ratio']
    ).to_dataframe(
    )[['Sratio','ymax']].droplevel(
        level=[0,1,3]
    ).unstack(level=1).round(2)