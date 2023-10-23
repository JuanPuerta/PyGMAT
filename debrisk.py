# Very important packages
from load_gmat import * # For the API to work from any directory, load_gmat.py has to be copied inside
from time import time

# Check usability and if not used, remove
import numpy as np
from datetime import datetime
from dateutil import parser
import spiceypy as spy
import pandas as pd
import math as mt
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style

# For converting TLEs
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv

# For plotting
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# Measure execution time
STATUS = time()
def elapsed_time():
    global STATUS
    end = time()
    print(f"{Fore.BLUE}Execution Time: {end - STATUS} s {Style.RESET_ALL}")
    #print(f"Execution time: {end - STATUS} s")
    STATUS = time()
    return end

# Satellite class definition

class Satellite:

    def __init__(self,object_name="satellite",
                 physical_prop=dict(DryMass=1,DragArea=1,Cr=2.2,Cd=2.2,SRPArea=1),
                 epoch = '01 Jan 2000 00:00:00.00',
                 state_vector=[0,0,0,0,0,0],
                 prop_time = 1,
                 potential=dict(Degree=0,Order=0),
                 drag=dict(model=None),
                 srp=dict(status='Off',Flux=1367,SRPModel='Spherical',NominalSun='149597870.691'),
                 pm=dict(objects='') # {object1, object2, ..}
                 ):
        
        self.object_name = object_name
        self.physical_prop = physical_prop
        self.epoch = epoch
        self.prop_time = prop_time
        self.state_vector = state_vector
        self.potential = potential
        self.drag = drag
        self.srp = srp
        self.pm = pm

    def gen_script(self):

        # File suffix
        suffix = ''

        # Basic options
        script_basics = f"""
%General Mission Analysis Tool(GMAT) Script
%Created: 2023-09-03 23:01:32

%----------------------------------------
%---------- Spacecraft
%----------------------------------------
Create Spacecraft {self.object_name};
GMAT {self.object_name}.DateFormat = UTCGregorian;
GMAT {self.object_name}.Epoch = '{self.epoch}';
GMAT {self.object_name}.CoordinateSystem = EarthMJ2000Eq;
GMAT {self.object_name}.DisplayStateType = Cartesian;
GMAT {self.object_name}.X = {self.state_vector[0]:.13f};
GMAT {self.object_name}.Y = {self.state_vector[1]:.13f};
GMAT {self.object_name}.Z = {self.state_vector[2]:.13f};
GMAT {self.object_name}.VX = {self.state_vector[3]:.13f};
GMAT {self.object_name}.VY = {self.state_vector[4]:.13f};
GMAT {self.object_name}.VZ = {self.state_vector[5]:.13f};

% Additional
GMAT {self.object_name}.AtmosDensityScaleFactor = 1;
GMAT {self.object_name}.ExtendedMassPropertiesModel = 'None';
GMAT {self.object_name}.NAIFId = -10000001;
GMAT {self.object_name}.NAIFIdReferenceFrame = -9000001;
GMAT {self.object_name}.OrbitColor = Red;
GMAT {self.object_name}.TargetColor = Teal;
GMAT {self.object_name}.OrbitErrorCovariance = [ 1e+70 0 0 0 0 0 ; 0 1e+70 0 0 0 0 ; 0 0 1e+70 0 0 0 ; 0 0 0 1e+70 0 0 ; 0 0 0 0 1e+70 0 ; 0 0 0 0 0 1e+70 ];
GMAT {self.object_name}.CdSigma = 1e+70;
GMAT {self.object_name}.CrSigma = 1e+70;
GMAT {self.object_name}.Id = 'SatId';
"""
        # Physical properties
        script_physical = f"""        
%----------------------------------------
%---------- Physical properties
%----------------------------------------
GMAT {self.object_name}.DryMass = {self.physical_prop['DryMass']};
GMAT {self.object_name}.DragArea = {self.physical_prop['DragArea']};
GMAT {self.object_name}.Cd = {self.physical_prop["Cd"]};
GMAT {self.object_name}.Cr = {self.physical_prop["Cr"]};
GMAT {self.object_name}.SRPArea = {self.physical_prop["SRPArea"]};
"""

        # Force model
        script_force = f"""
%----------------------------------------
%---------- ForceModels
%----------------------------------------
Create ForceModel DefaultProp_ForceModel;
GMAT DefaultProp_ForceModel.CentralBody = Earth;
GMAT DefaultProp_ForceModel.PrimaryBodies = {{Earth}};
GMAT DefaultProp_ForceModel.RelativisticCorrection = Off;
GMAT DefaultProp_ForceModel.ErrorControl = RSSStep;
"""
        
        # Particle masses
        if self.pm['objects']:
            script_pm = f"""
GMAT DefaultProp_ForceModel.PointMasses = {self.pm['objects']};
"""
            suffix += "pm_objects-"
        else:
            script_pm = ""
            suffix += "pm_none-"

        # Drag force
        if self.drag['model']:
            script_drag = f"""
GMAT DefaultProp_ForceModel.Drag.AtmosphereModel = {self.drag['model']};
GMAT DefaultProp_ForceModel.Drag.HistoricWeatherSource = 'ConstantFluxAndGeoMag';
GMAT DefaultProp_ForceModel.Drag.PredictedWeatherSource = 'ConstantFluxAndGeoMag';
GMAT DefaultProp_ForceModel.Drag.CSSISpaceWeatherFile = 'SpaceWeather-All-v1.2.txt';
GMAT DefaultProp_ForceModel.Drag.SchattenFile = 'SchattenPredict.txt';
GMAT DefaultProp_ForceModel.Drag.F107 = 150;
GMAT DefaultProp_ForceModel.Drag.F107A = 150;
GMAT DefaultProp_ForceModel.Drag.MagneticIndex = 3;
GMAT DefaultProp_ForceModel.Drag.SchattenErrorModel = 'Nominal';
GMAT DefaultProp_ForceModel.Drag.SchattenTimingModel = 'NominalCycle';
GMAT DefaultProp_ForceModel.Drag.DragModel = 'Spherical';
"""    
            suffix += "drag_atmos-"
        else:
            script_drag = f"""
GMAT DefaultProp_ForceModel.Drag = None;
"""
            suffix += "drag_none-"

        # Potential
        if (self.potential['Degree'] + self.potential['Order']) > 0:
            script_potential = f"""
GMAT DefaultProp_ForceModel.GravityField.Earth.Degree = {self.potential["Degree"]};
GMAT DefaultProp_ForceModel.GravityField.Earth.Order = {self.potential["Order"]};
GMAT DefaultProp_ForceModel.GravityField.Earth.StmLimit = 100;
GMAT DefaultProp_ForceModel.GravityField.Earth.PotentialFile = 'JGM2.cof';
GMAT DefaultProp_ForceModel.GravityField.Earth.TideModel = 'None';
"""
            suffix += "pot_oblate-"
        else:
            script_potential = f"""
GMAT DefaultProp_ForceModel.GravityField.Earth.Degree = 0;
GMAT DefaultProp_ForceModel.GravityField.Earth.Order = 0;
GMAT DefaultProp_ForceModel.GravityField.Earth.StmLimit = 100;
GMAT DefaultProp_ForceModel.GravityField.Earth.PotentialFile = 'JGM2.cof';
GMAT DefaultProp_ForceModel.GravityField.Earth.TideModel = 'None';
"""
            suffix += "pot_spherical-"

        # SRP
        if self.srp['status'] == 'On':
            suffix += "srp_on-"
            script_srp = f"""
GMAT DefaultProp_ForceModel.SRP = On;
GMAT DefaultProp_ForceModel.SRP.Flux = {self.srp["Flux"]};
GMAT DefaultProp_ForceModel.SRP.SRPModel = {self.srp["SRPModel"]};
GMAT DefaultProp_ForceModel.SRP.Nominal_Sun = {self.srp["NominalSun"]};
"""
        else:
            suffix += "srp_off-"
            script_srp = f"""
GMAT DefaultProp_ForceModel.SRP = Off;
"""  
            
        self.suffix = suffix.strip('-')

        # Propagator, subscribers, mission sequence
        script_add=f"""
%----------------------------------------
%---------- Propagators
%----------------------------------------
Create Propagator DefaultProp;
GMAT DefaultProp.FM = DefaultProp_ForceModel;
GMAT DefaultProp.Type = RungeKutta89;
GMAT DefaultProp.InitialStepSize = 60;
GMAT DefaultProp.Accuracy = 9.999999999999999e-12;
GMAT DefaultProp.MinStep = 0.001;
GMAT DefaultProp.MaxStep = 2700;
GMAT DefaultProp.MaxStepAttempts = 50;
GMAT DefaultProp.StopIfAccuracyIsViolated = true;

%----------------------------------------
%---------- Subscribers
%----------------------------------------
Create ReportFile ReportFile1;
GMAT ReportFile1.SolverIterations = Current;
GMAT ReportFile1.UpperLeft = [ 0.02950819672131148 0.05714285714285714 ];
GMAT ReportFile1.Size = [ 0.5975409836065574 0.7957142857142857 ];
GMAT ReportFile1.RelativeZOrder = 138;
GMAT ReportFile1.Maximized = false;
GMAT ReportFile1.Filename = 'G:\\My Drive\\Numerical\\PhD-JuanFPuerta\\Coding\\Python_GMAT_files\\ReportFile-{self.suffix}.txt'
GMAT ReportFile1.Add = {{{self.object_name}.UTCGregorian, {self.object_name}.UTCModJulian, {self.object_name}.Earth.Altitude, {self.object_name}.EarthMJ2000Eq.X, {self.object_name}.EarthMJ2000Eq.Y, {self.object_name}.EarthMJ2000Eq.Z, {self.object_name}.EarthMJ2000Eq.VX, {self.object_name}.EarthMJ2000Eq.VY, {self.object_name}.EarthMJ2000Eq.VZ, {self.object_name}.Earth.SMA, {self.object_name}.Earth.Latitude, {self.object_name}.Earth.Longitude, {self.object_name}.EarthMJ2000Eq.AOP, {self.object_name}.Earth.ECC, {self.object_name}.EarthMJ2000Eq.INC, {self.object_name}.EarthMJ2000Eq.RAAN, {self.object_name}.Earth.TA, {self.object_name}.Earth.MA, {self.object_name}.Earth.EA, {self.object_name}.DefaultProp_ForceModel.AtmosDensity, {self.object_name}.DefaultProp_ForceModel.AccelerationX, {self.object_name}.DefaultProp_ForceModel.AccelerationY, {self.object_name}.DefaultProp_ForceModel.AccelerationZ, {self.object_name}.Earth.RMAG, {self.object_name}.ElapsedSecs, {self.object_name}.EarthMJ2000Eq.RA, {self.object_name}.EarthMJ2000Eq.DEC}};
GMAT ReportFile1.WriteHeaders = true;
GMAT ReportFile1.LeftJustify = On;
GMAT ReportFile1.ZeroFill = Off;
GMAT ReportFile1.FixedWidth = true;
GMAT ReportFile1.Delimiter = ' ';
GMAT ReportFile1.ColumnWidth = 23;
GMAT ReportFile1.WriteReport = true;

%----------------------------------------
%---------- Mission Sequence
%----------------------------------------
BeginMissionSequence;
Propagate DefaultProp({self.object_name}) {{{self.object_name}.ElapsedSecs = {self.prop_time}}};
"""

        # Assembly
        script = script_basics + script_physical + script_force + script_pm + script_potential + script_drag + script_srp + script_add

        # Script name
        self.script_name = f"{self.object_name}-{self.suffix}.script"

        print(f"{Fore.LIGHTRED_EX}Saving script: {self.script_name} {Style.RESET_ALL}")
        f=open(self.script_name,"w")
        f.write(script)
        f.close()


    def run_script(self):
        elapsed_time()
        gmat.LoadScript(self.script_name)
        gmat.RunScript()
        gmat.Help(self.script_name) #Check Help object
        elapsed_time()

# Utilities Sub-Class Definition
""" Create a class where elapsed time of execution is calculated, the data from the TLE can be read, transform to a CSV file,
 Numpy Array, and plotting data for further analysis
"""
class Utility:

    def __init__(self):
        self.Satellite = Satellite
        
    # Read CSV TLE File, convert it to a Pandas Dataframe and get timelapse between TLEs
    def read_csv_tle(self, state_source):
        sv_final = pd.read_csv(state_source,index_col=0)
        elapsed_time()
        for i,dtime_str in enumerate(sv_final.datetime):
            dtime = datetime.strptime(dtime_str,"%Y-%m-%d %H:%M:%S.%f")
            sv_final.loc[i,'datetime']=dtime
        elapsed_time()
        ifinal=49  
        time_diff=(sv_final.iloc[ifinal,1]-sv_final.iloc[0,1])
        sec_time_diff=time_diff.total_seconds()
        print(f"{Fore.CYAN}Timelapse between Epochs (secs): {sec_time_diff} {Style.RESET_ALL}")
        return sv_final,sec_time_diff

        
    # Read TLEs and convert it/them
    def read_convert_TLE(self, tle_file):
        # Read TLE text file from object
        sv_final=np.empty((0,8))
        #tle_file=f"tle_Envisat.txt"
        #lines=[]
        with open(tle_file, 'r') as f:
            lines=f.readlines()
        for i in range(0,len(lines),2):
            line1=lines[i].strip()
            line2=lines[i+1].strip()
            satellite=twoline2rv(line1,line2,wgs84)
            epoch=satellite.epoch
            year=epoch.year
            month=epoch.month
            day=epoch.day
            hours=epoch.hour
            minutes=epoch.minute
            seconds=epoch.second+round(epoch.microsecond/1e6,3)
            position,velocity=satellite.propagate(year, month, day, hours, minutes, seconds)
            jday=367*year-mt.trunc(((7*(year+mt.trunc((month+9)/12)))/4))+mt.trunc(275*month/9)+day+1721013.5
            poday=(((((seconds/60)+minutes)/60)+hours)/24)
            utc_mdj=(jday-2430000.0)+poday # Modified Julian Date TT in GMAT
            utc_gmat=epoch.strftime('%d %b %Y %H:%M:%S.%f')[:-3] # string use .%f for more precision and [:-3] for 3 digit
            sv=np.array(position+velocity)#+sec_epoch)
            sv_sec_sv=np.append(utc_mdj,sv)#.astype(object)) # it was sec_epoch
            sv_utc_sec_sv=np.append(utc_gmat,sv_sec_sv.astype(object))
            sv_final=np.append(sv_final,sv_utc_sec_sv)
        
        sv_final = sv_final.reshape(-1,8)
        statevec_csv = pd.DataFrame(sv_final, columns=['datetime_str',
                                                    'datetime',
                                                    'x', 
                                                    'y',
                                                    'z', 
                                                    'vx', 
                                                    'vy', 
                                                    'vz'])
        statevec_csv.to_csv("state_vector.csv")
        return statevec_csv
    
    def plot_Scatter_GEO(self, txt_report_file):

        # Read the TXT report file data

        orbit_data = pd.read_fwf(txt_report_file)
        orbit_loc=orbit_data[[f"Envisat.Earth.RMAG",f"Envisat.Earth.Latitude",f"Envisat.Earth.Longitude"]]
        locations=np.array(orbit_loc[[f"Envisat.Earth.RMAG",f"Envisat.Earth.Latitude",f"Envisat.Earth.Longitude"]])
        #print(orbit_data,orbit_loc,locations)

        # Create a scattergeo plot
        Scatter_RA_DEC=go.Scattergeo(
            lat=locations[:,1],
            lon=locations[:,2],
            mode='markers',
            marker=dict(
                size=10,
                color=locations[:,0],   #'blue', # RMag
                symbol='circle',
                opacity=0.7
            )
        )

        Layout_SGeo=go.Layout(
            title='Object Positions',
            geo=dict(
                scope='world',
                projection_type='orthographic',
                #projection_type='equirectangular',
                showland=True,
                landcolor='rgb(217, 217, 217)',
                showcountries=True,
                countrycolor='rgb(255, 255, 255)'
            )
        )

        fig1=go.Figure(data=[Scatter_RA_DEC],layout=Layout_SGeo)
        fig1.show()


if __name__ == '__main__':
    print("Hola soy PyGMAT")
    S = Satellite()