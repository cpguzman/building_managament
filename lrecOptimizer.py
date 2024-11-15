'''@TODO LREC optimizer is considering that the initial state of charge (SoC) is known. 
In the future, it is needed to adapt for the case that the initial SoC is not known'''

from pyomo.opt import SolverFactory
import pyomo
import pyomo.opt
import pyomo.environ as pyo
import pandas as pd
import numpy as np
from db import scheduling, price, forecast, measurements
import dotenv
from static_classes import Config
import json
import pendulum
import glob
import os

# Adding API of Statics imports:
from db.member.site import House, Corporate, Building #House Object

from db.api.site.house_api import HouseAPI #API 
from db.api.site.building_api import BuildingAPI #API 

from db.api.connection import ConnectionConfig
from db.member.device import DeviceType

# Adding API info:
db_hostname = "daryl.inesc-id.pt"
db_port = 3307
db_username = "root"
db_password = "rootev4eu"
db_database = "dayahead"

# Connect to the database
connection_config = ConnectionConfig(host=db_hostname,
    port=db_port,
    user=db_username,
    password=db_password,
    database=db_database,
    charset="utf8mb4",

)

class LrecOptimizer():
    def __init__(self, building, config_, dynamic_folder, measurements_db, tariff_file=None):
        self.measurements_db = measurements_db
        self.building = building 
        self.dynamic_folder = dynamic_folder
        
        self.bess = self.pv_ = None

        if self.building.has_bess:
            self.bess = next((device for device in self.building.devices if device.device_type == DeviceType.BESS), None)

        if self.building.has_pv:
            self.pv_ = next((device for device in self.building.devices if device.device_type == DeviceType.PHOTOVOLTAIC), None)
        
        self.model = pyo.ConcreteModel()

        self.config_ = config_
        self.dynamic_infos = {} # Dictionary to store the data from all JSON files
        self.initialize_dynamic_dict()
        try:
            self.initialize_dynamic_dict()
        except:
            print("It was not possible to initialize the dynamic informations from EVs")

        self.tariff_file = tariff_file
        self.initialize_tariff()

        # Open Forecast db
        self.forecast_conn, self.forecast_cursor = forecast.forecast_open()

        # Open price db
        self.price_conn, self.price_cursor = price.price_open()

    # Method used to format the data in order to use with pyomo library.
    def process_1d_array(self, array):
        processed_results = [value[0] for value in array if len(value) == 1]
        processed_results = np.array(processed_results)
        return processed_results

    def initialize_dynamic_dict(self):
        #json_files = sorted(glob.glob(os.path.join(self.dynamic_folder, '*', '*.json')))
        json_files = sorted(glob.glob(os.path.join(self.dynamic_folder, '*.json')))
        # Iterate over the sorted JSON files
        for file_path in json_files:
            with open(file_path, "r") as file:
                json_data = json.load(file)
                
                
                file_name = os.path.basename(file_path)
                file_name_without_ext = os.path.splitext(file_name)[0]

                
                # Create a unique key by combining the name of the subfolder and the name of the file
                dict_key = f"{file_name_without_ext}"
                
                # Store the data in the dictionary using the unique key
                self.dynamic_infos[dict_key] = json_data
        
        self.departureDateTime = [pendulum.parse(value["departureDateTime"]) for value in self.dynamic_infos.values()]
        self.userType = [value["userType"] for value in self.dynamic_infos.values()]
        self.batV2G = [value["batV2G"] for value in self.dynamic_infos.values()]
        self.batMaxCapacity = [value["batMaxCapacity"] for value in self.dynamic_infos.values()]
        self.batMaxChargeRateAc = [value["batMaxChargeRateAc"] for value in self.dynamic_infos.values()]
        self.batEfficiencyCharge = [value["batEfficiencyCharge"] for value in self.dynamic_infos.values()]
        self.connectorsInUse = [value["qrCodeData"] for value in self.dynamic_infos.values()]
        
        print("User Type (userType):")
        print(self.userType)
        print("Battery V2G Capability (batV2G):")
        print(self.batV2G)
        print("Battery Max Capacity (batMaxCapacity):")
        print(self.batMaxCapacity)
        print("Battery Max Charge Rate AC (batMaxChargeRateAc):")
        print(self.batMaxChargeRateAc)
        print("Battery Efficiency Charge (batEfficiencyCharge):")
        print(self.batEfficiencyCharge)
        self.calculate_departure_time()

    def round_time(self, time):
        """
        Rounds the time down to the nearest multiple of 15.
        """
        minutes = time.minute
        rounded_minutes = minutes - (minutes % 15)
        return time.replace(minute=rounded_minutes, second=0, microsecond=0)

    def difference_in_multiples_of_15(self, time1, time2):
        """
        Calculates the difference between two times in multiples of 15 minutes.
        """
        rounded_time1 = self.round_time(time1)
        rounded_time2 = self.round_time(time2)
        difference = rounded_time2 - rounded_time1
        minutes = difference.total_seconds() / 60
        return int(minutes / 15)
    
    def calculate_departure_time(self):
        self.stepDepartureDateTime = [self.difference_in_multiples_of_15(pendulum.now().in_timezone('UTC'), timestamp) for timestamp in self.departureDateTime]
        print("Departure Date/Time (stepDepartureDateTime):")
        print(self.stepDepartureDateTime)

    # Method used to format the data in order to use with pyomo library.
    @classmethod
    def _auxDictionary(cls, a):
        temp_dictionary = {}
        if len(a.shape) == 3:
            for dim0 in np.arange(a.shape[0]):
                for dim1 in np.arange(a.shape[1]):
                    for dim2 in np.arange(a.shape[2]):
                        temp_dictionary[(dim0 + 1, dim1 + 1, dim2 + 1)] = a[dim0, dim1, dim2]
        elif len(a.shape) == 2:
            for dim0 in np.arange(a.shape[0]):
                for dim1 in np.arange(a.shape[1]):
                    temp_dictionary[(dim0 + 1, dim1 + 1)] = a[dim0, dim1]
        else:
            for dim0 in np.arange(a.shape[0]):
                temp_dictionary[(dim0 + 1)] = a[dim0]
        return temp_dictionary
    
    def initialize_tariff(self):
        if self.tariff_file:
            with open(str(self.tariff_file), 'r') as file:
                self.tariff = json.load(file)

    def _set_params(self):
        self.n_time = max(self.stepDepartureDateTime) # The optimization will end when the last car is gone.
        self.n_evs = len(self.dynamic_infos) # 1cp to 1ev. No empty cp will enter in Optimizer, only the occupied ones.
        self.cps = len(self.dynamic_infos)

        #****************************************FORECAST DB:*************************************
        self.timesteps = self.process_1d_array(forecast.forecast_get_latest(self.forecast_cursor, self.n_time, 'date'))
        pv_data = self.process_1d_array(forecast.forecast_get_latest(self.forecast_cursor, self.n_time, 'pv_production'))
        building_consumption = self.process_1d_array(forecast.forecast_get_latest(self.forecast_cursor, self.n_time, 'house_consumption'))
        is_curtailment = self.process_1d_array(forecast.forecast_get_latest(self.forecast_cursor, self.n_time, 'wind_curtailment'))
        is_congestion_consumption = self.process_1d_array(forecast.forecast_get_latest(self.forecast_cursor, self.n_time, 'congestion_consumption'))
        is_congestion_generation = self.process_1d_array(forecast.forecast_get_latest(self.forecast_cursor, self.n_time, 'congestion_generation'))  #Validate de data from forecasting DB        


        #****************************************PRICE DB:*************************************
        export_price_data = self.process_1d_array(price.price_get_latest(self.price_cursor, self.n_time, 'energy_price')) #Price data get in €/MWh

        for i, data in enumerate(export_price_data):
            export_price_data[i] = data / 1000       #Calculating price data in €/KWh
        
        print("\n\n*************************export_price_data = ", export_price_data)
        # To avoid that the code crashes if there is more information from one db than the other, lenght will be the minimum
        self.n_time = min(len(pv_data), len(export_price_data))

        self.timesteps = self.timesteps[:self.n_time]
        pv_data = pv_data[:self.n_time]
        building_consumption = building_consumption[:self.n_time]
        is_curtailment = is_curtailment[:self.n_time]
        is_congestion_consumption = is_congestion_consumption[:self.n_time]
        is_congestion_generation =  is_congestion_generation[:self.n_time]
        export_price_data = export_price_data[:self.n_time]


        # Variables representing time, electric vehicles, charging points, and shared stations.
        self.model.ev = pyo.Set(initialize = np.arange(1, self.n_evs + 1))
        self.model.t = pyo.Set(initialize = np.arange(1, self.n_time + 1))
        self.model.cp = pyo.Set(initialize = np.arange(1, self.cps + 1))

        # Initializing parameters with the forecast db data:
        self.model.pv = pyo.Param(self.model.t, initialize=self._auxDictionary(pv_data), 
                                  doc = 'Power generated by the solar power plant.') 
        self.model.pl = pyo.Param(self.model.t, initialize=self._auxDictionary(building_consumption)) 
        
        self.model.is_curtailment = pyo.Param(self.model.t, initialize=self._auxDictionary(is_curtailment)) 
        self.model.is_congestion_consumption = pyo.Param(self.model.t, initialize=self._auxDictionary(is_congestion_consumption)) 
        self.model.is_congestion_generation = pyo.Param(self.model.t, initialize=self._auxDictionary(is_congestion_generation)) 
        

        # Initializing parameters with the price db data:
        
        #****************Adding the IMPORT tariffs that come from config jsons***************.
        self.hours_timesteps = [str(int(item.strftime("%H"))) for item in self.timesteps]
        self.import_price_data = [self.tariff.get(hour) for hour in self.hours_timesteps] # Come from json
        self.model.import_price = pyo.Param(self.model.t, initialize=self._auxDictionary(np.array(self.import_price_data)),
                                            doc = 'Energy import price - in €/KWh') #Already in €/kWh 
        
        print("\n\n*************************import_price_data = ", self.import_price_data)
        # Max and Min import price
        self.model.highprice = pyo.Param(initialize=max(self.import_price_data),within=pyo.Reals,
                                         doc = 'Maximum price among the import prices')
        self.model.lowprice = pyo.Param(initialize=min(self.import_price_data),within=pyo.Reals,
                                        doc = 'Minimum price among the import prices')
        
        #****************************Adding the EXPORT tariffs that come from db****************************
        self.export_price_data = export_price_data * self.config_.export_price_threshold
        self.model.export_price = pyo.Param(self.model.t, initialize=self._auxDictionary(self.export_price_data),
                                            doc = 'Energy export price - in €/KWh')

        #***************************************EV PARAMETERS********************************************

        # connected (alpha) is a variable that represents when the car is connected (1) or not (0)
        connected = [[1 if j <= item else 0 for j in range(self.n_time)] for i, item in enumerate(self.stepDepartureDateTime)] #(ev,t)

        # Since in the Simulator, Manual did all with (t,ev) instead of (ev,t), I will transpose it.
        connected_t = [list(i) for i in zip(*connected)] #(t,ev)

        self.model.connected = pyo.Param(self.model.t, self.model.ev, initialize = self._auxDictionary(np.array(connected_t)),
                                         doc = 'Car connection state')
        self.model.departure_time = pyo.Param(self.model.ev, initialize =self._auxDictionary(np.array(self.stepDepartureDateTime)),
                                              doc = 'Departure time of cars, represented in discrete time steps')
        
        EEVmin_ = [4000 for i in range(self.n_evs)] #@TODO I do not know if the EEVmin should come from outside input such as EEVmax
        self.model.EEVmin = pyo.Param(self.model.ev, initialize =self._auxDictionary(np.array(EEVmin_)),
                                      doc = 'Minimum battery level for car operation')
        
        self.model.EEVmax = pyo.Param(self.model.ev, initialize =self._auxDictionary(np.array(self.batMaxCapacity)),
                                      doc = 'Maximum battery level from the cars')
        self.model.EVch_eff = pyo.Param(self.model.ev, initialize =self._auxDictionary(np.array(self.batEfficiencyCharge)),
                                       doc = 'EV Battery efficiency charge')
        self.model.PchmaxEV = pyo.Param(self.model.ev, initialize =self._auxDictionary(np.array(self.batMaxChargeRateAc)),
                                        doc = 'Maximum Charge Power Rate from the cars')
        self.model.PdchmaxEV = pyo.Param(self.model.ev, initialize =self._auxDictionary(np.array(self.batMaxChargeRateAc)),
                                         doc = 'Maximum Disharge Power Rate from the cars')
        
        default_target = [self.config_.default_target for i in range(self.n_evs)] # Default since in the future this come from the user
        self.model.target = pyo.Param(self.model.ev, initialize=self._auxDictionary(np.array(default_target)),
                                      doc = 'Percentage target of energy at which the car should depart.')


        # ***************************@TODO ADDED BY LARISSA - CP PARAMETERS***************************
        
        # Getting all cps in use from API        
        self.cps_api = [device for device in self.building.devices if device.device_type == DeviceType.CONNECTOR and device.connector_id in self.connectorsInUse]

        self.PchConnmax = [cp.max_power_charge_rate for cp in self.cps_api]
        self.PdchConnmax = [cp.max_power_discharge_rate for cp in self.cps_api]
        self.PchConnmin = [cp.min_power_charge_rate for cp in self.cps_api]
        self.PdchConnmin = [cp.min_power_discharge_rate for cp in self.cps_api]
        self.cp_charge_efficiency = [cp.charge_efficiency for cp in self.cps_api]
        self.cp_discharge_efficiency = [cp.discharge_efficiency for cp in self.cps_api]
        self.my_cp_id = [cp.connector_id for cp in self.cps_api] # This id is the connector id, and not the id from API.
        self.cp_v2g = [cp.v2g_available for cp in self.cps_api]

         # Imprimindo os valores de cada parâmetro
        
        print("\nV2G Availability (cp_v2g):")
        print(self.cp_v2g)
        
        print("\nMax Power Charge Rate (PchConnmax):")
        print(self.PchConnmax)
        print("\nMax Power Charge Rate (PchConnmax):")
        print(self.PdchConnmax)
        print("\nCharging Efficiency (charging_efficiency):")
        print(self.cp_charge_efficiency)
        print("\nDischarging Efficiency (discharging_efficiency):")
        print(self.cp_discharge_efficiency)
        print("\nConnector IDs (my_cp_id):")
        print(self.my_cp_id)

        self.model.v2g_conn = pyo.Param(self.model.cp, initialize =self._auxDictionary(np.array(self.cp_v2g)),
                                        doc = 'Boolean indicating whether the connectors support V2G functionality')
        self.model.PchConnmax = pyo.Param(self.model.cp, initialize =self._auxDictionary(np.array(self.PchConnmax)),
                                          doc = 'Maximum charging power rate of the connectors')  
        self.model.PdchConnmax = pyo.Param(self.model.cp, initialize =self._auxDictionary(np.array(self.PdchConnmax)),
                                           doc = 'Maximum discharging power rate of the connectors') 

        v2g_overall = [self.cp_v2g[i] * self.batV2G[i] for i in range(len(self.batV2G))]
        self.model.v2g_overall = pyo.Param(self.model.ev, initialize=self._auxDictionary(np.array(v2g_overall)),
                                            doc = 'If both connector and ev are v2g, so the car can discharge')
        
        # This because the ev 1 is always connected with cp 1 and so on.
        overall_efficiency_charge = [self.batEfficiencyCharge[i] * self.cp_charge_efficiency[i] for i in range(self.n_evs)]
        
        overall_efficiency_discharge = [self.batEfficiencyCharge[i] * self.cp_discharge_efficiency[i] for i in range(self.n_evs)]

        #@TODO By now, the EVs API does not have information about the Max Discharge Rate, even though the EV is v2g. So we consider it is the same as Max Charge Rate.
        overall_max_charge_rate = [min(self.batMaxChargeRateAc[i], self.PchConnmax[i]) for i in range(self.n_evs)]
        overall_max_discharge_rate = [min(self.batMaxChargeRateAc[i], self.PdchConnmax[i]) for i in range(self.n_evs)]
        overall_max_charge_rate = [min(self.batMaxChargeRateAc[i], self.PchConnmax[i]) for i in range(self.n_evs)]
        overall_max_discharge_rate = [min(self.batMaxChargeRateAc[i], self.PdchConnmax[i]) for i in range(self.n_evs)]
        
        self.model.overall_efficiency_charge = pyo.Param(self.model.ev, initialize=self._auxDictionary(np.array(overall_efficiency_charge)),
                                                         doc = 'Charging efficiency considering both connectors and evs.')
        self.model.overall_efficiency_discharge = pyo.Param(self.model.ev, initialize=self._auxDictionary(np.array(overall_efficiency_discharge)),
                                                            doc = 'Discharging efficiency considering both connectors and evs.')
        self.model.overall_max_charge_rate = pyo.Param(self.model.ev, initialize=self._auxDictionary(np.array(overall_max_charge_rate)),
                                                        doc = 'Maximum charging power rate considering both connectors and evs.')
        self.model.overall_max_discharge_rate = pyo.Param(self.model.ev, initialize=self._auxDictionary(np.array(overall_max_discharge_rate)),
                                                        doc = 'Maximum discharging power rate considering both connectors and evs.')
        
        #@TODO By now, the EVs does not return the Min Charge and Discharge Rate in the APP. So we will consider only the CP one.
        self.model.overall_min_charge_rate = pyo.Param(self.model.ev, initialize=self._auxDictionary(np.array(self.PchConnmin)),
                                                        doc = 'Maximum charging power rate considering both connectors and evs.')
        self.model.overall_min_discharge_rate = pyo.Param(self.model.ev, initialize=self._auxDictionary(np.array(self.PdchConnmin)),
                                                        doc = 'Maximum discharging power rate considering both connectors and evs.')
        
        # CONFIG PARAMETERS:
        self.model.penalty1 = self.config_.penalty1
        self.model.penalty2 = self.config_.penalty2
        self.model.DegCost = self.config_.DegCost
        self.model.m = self.config_.m
        self.model.discount = self.config_.discount
        self.model.multiplier = self.config_.multiplier
        self.model.dT = 1/self.config_.timestep

        #@TODO This parameter should come from another place.
        ps_values = [1] * self.n_time
        self.model.pscurtbuy = pyo.Param(self.model.t,  initialize=self._auxDictionary(np.array(ps_values)),
                                         doc = 'Price signal for wind curtailment') 
        self.model.pscongbuy = pyo.Param(self.model.t, initialize=self._auxDictionary(np.array(ps_values)),
                                         doc = 'Price signal for buy congestion management') 
        self.model.pscongsell = pyo.Param(self.model.t, initialize=self._auxDictionary(np.array(ps_values)),
                                          doc = 'Price signal for sell congestion management')

        self.model.pc = pyo.Param(initialize=self.building.contracted_power, doc = 'Contracted Power')

        # Open connection with measurements db
        handle_measurements = measurements.measurements_open(self.measurements_db)

        # Initializing socs based on the latest measurements on rt-db:
        try:
            socs = [measurements.measurements_get_last(handle_measurements, device_id=cp_id,metric="soc")[0] for cp_id in self.my_cp_id]
            socs = [item[4] for i, item in enumerate(socs)]
            print("\nUsing real soc: ", socs)  
        except:
            socs = []

        if len(socs) != self.n_evs:  # In case it is not capable of taking the soc from measurement db
            socs = [0.5 for i in range(self.n_evs)] #fake soc
            print("\nUsing a fake soc...")
        
        self.model.initial_soc = pyo.Param(self.model.ev, initialize =self._auxDictionary(np.array(socs)))

    def _set_vars(self):
        # EV:
        self.model.PEV_aux = pyo.Var(self.model.t, self.model.ev, domain=pyo.NonNegativeReals, initialize=0,
                                     doc = 'Auxiliar variable of power allocated for vehicles charging')
        self.model.PEVdc_aux = pyo.Var(self.model.t, self.model.ev,domain=pyo.NonNegativeReals, initialize=0,
                                       doc = 'Auxiliar variable of power allocated for vehicles discharging')
        self.model.PEV = pyo.Var(self.model.t, self.model.ev, domain=pyo.NonNegativeReals, initialize=0,
                                 doc = 'Power allocated by the optimizer for vehicles charging')
        self.model.PEVdc = pyo.Var(self.model.t, self.model.ev,domain=pyo.NonNegativeReals, initialize=0,
                                   doc = 'Power allocated by the optimizer for vehicles discharging')

        self.model.import_relax = pyo.Var(self.model.t, domain=pyo.NonNegativeReals, initialize=0,
                                          doc = 'Relax variable of grid import')
        self.model.export_relax = pyo.Var(self.model.t, domain=pyo.NonNegativeReals, initialize=0,
                                          doc = 'Relax variable of grid export')
        
        # EV:
        self.model.EEV = pyo.Var(self.model.t, self.model.ev, domain=pyo.NonNegativeReals, initialize=0,
                                 doc = 'State of Charge (SoC) of the electric vehicle battery')
        self.model.Etargetrelax = pyo.Var(self.model.t, self.model.ev, domain=pyo.NonNegativeReals, initialize=0,
                                          doc = 'Relax variable of EV target')
        self.model.grid_import = pyo.Var(self.model.t, domain=pyo.NonNegativeReals, initialize=0,
                                        doc = 'Grid import of the system')
                                          
        self.model.grid_export = pyo.Var(self.model.t, domain=pyo.NonNegativeReals, initialize=0,
                                        doc = 'Grid export of the system')

        # EV:
        self.model.ev_is_charging = pyo.Var(self.model.t, self.model.ev, domain=pyo.Binary, bounds=(0, 1), initialize=0,
                                            doc = 'Binary representing if the car is charging')
        self.model.ev_is_discharging = pyo.Var(self.model.t, self.model.ev, domain=pyo.Binary, bounds=(0, 1), initialize=0,
                                               doc = 'Binary representing if the car is discharging')         
        # SYSTEM:
        self.model.is_importing = pyo.Var(self.model.t, domain=pyo.Binary, bounds=(0, 1), initialize=0,
                                          doc = 'Binary representing if there is grid export')
        self.model.is_exporting = pyo.Var(self.model.t, domain=pyo.Binary, bounds=(0, 1), initialize=0,
                                          doc = 'Binary representing if there is grid import')
        
        self.model.Eminsocrelax = pyo.Var(self.model.t, self.model.ev, domain=pyo.NonNegativeReals, initialize=0,
                                          doc = 'Relax variable of EV State of Charge (SoC)')
        
        self.model.service_relax_wind = pyo.Var(self.model.t, self.model.ev, domain=pyo.NonNegativeReals, initialize=0,
                                                doc = 'Relax variable related to wind curtailment service')
        self.model.service_relax_congestion = pyo.Var(self.model.t, self.model.ev,  domain=pyo.NonNegativeReals, initialize=0,
                                                      doc = 'Relax variable related to congestion in production')
        self.model.service_relax_congestiongen = pyo.Var(self.model.t, self.model.ev, domain=pyo.NonNegativeReals, initialize=0,
                                                         doc = 'Relax variable related to congestion in generation')

        # @TODO Not being used yet.
        self.model.flex_binary = pyo.Var(self.model.t, domain=pyo.Binary, bounds=(0, 1), initialize=0)
        self.model.flex_load_ewh = pyo.Var(self.model.t, domain=pyo.NonNegativeReals, initialize=0)
        self.model.flex_load2 = pyo.Var(self.model.t, domain=pyo.NonNegativeReals, initialize=0)
        self.model.psnew = pyo.Var(self.model.t, domain=pyo.NonNegativeReals, initialize=0)

         
    
    def _set_constraints(self):

        def _opt_soc_charge(model, t, ev):
            if t == 1:
                return model.EEV[t,ev] == model.initial_soc[ev]*model.EEVmax[ev] + (model.PEV[t,ev]*model.dT)*(model.overall_efficiency_charge[ev]) \
                    - (model.PEVdc[t,ev]*model.dT)/(model.overall_efficiency_charge[ev]) 
            elif t > 1:
                return model.EEV[t,ev] == model.EEV[t-1,ev] + (model.PEV[t,ev]*model.dT)*(model.overall_efficiency_charge[ev]) - \
                    (model.PEVdc[t,ev]*model.dT)/(model.overall_efficiency_charge[ev]) 
        # Target.
        def _balance_energy_EVS3(model,t,ev): 
            if t == model.departure_time[ev]:
                return model.EEV[t,ev] + model.Etargetrelax[t,ev] >= model.EEVmax[ev]*model.target[ev] 
            return pyo.Constraint.Skip
        

        # EV charge rate (in W)        
        def _max_charge_pcaux(model, t, ev):
            return model.PEV_aux[t, ev] == model.connected[t, ev]* (model.overall_max_charge_rate[ev]) * model.ev_is_charging[t, ev]
        
        # EV charge power limit (in W)               
        def _max_charge_pc(model, t, ev):
            return model.PEV[t, ev] <= model.PEV_aux[t, ev]

        #@TODO FINALIZE THIS
        def _min_charge_pc(model, t, ev):
            return model.PEV[t, ev] >= model.overall_min_charge_rate[ev] * model.ev_is_charging[t, ev]
        
        # EV discharge limit operation (in W)
        def _max_discharge_pcaux(model, t, ev):
            
            return model.PEVdc_aux[t, ev] <= model.connected[t, ev] * (model.overall_max_discharge_rate[ev]) * model.ev_is_discharging[t, ev] * model.v2g_overall[ev]

        # EV discharge power limit (in W)
        def _max_discharge_pc(model, t, ev):

            return model.PEVdc[t, ev] <= model.PEVdc_aux[t, ev]  # ***Cindy***: Power discharguer (MaW) limit
        
        #******inactivated constraint due to we do not have information related to the initial EV soc*********************
        def _soc_charge(model, t, ev):
            """
            EV energy balance (in Wh)
            """
            if t > 1:
                return model.EEV[t, ev] - model.Etargetrelax[t, ev] == model.EEV[t - 1, ev] + (model.PEV[t, ev]*model.dT) * model.overall_efficiency_charge[ev]  - (model.PEVdc[t, ev]*model.dT)/ model.efficiency_discharge[t, ev]  - model.etrip[t, ev]
            else:
                return model.EEV[t, ev] - model.Etargetrelax[t, ev] == self.evs[ev-1].SOC + (model.PEV[t, ev]*model.dT) * model.overall_efficiency_charge[ev]  - (model.PEVdc[t, ev]*model.dT)/ model.efficiency_discharge[t, ev]  - model.etrip[t, ev]
                
        def _p_max(model, t):
            """
                Power limit the system (in W)
            """
            return model.grid_import[t] <= (model.pc) * model.is_importing[t] + model.import_relax[t] #In w (Power)
        
        def _p_max_export(model, t):
            """
                Export Power limit the system (in Wh)
            """

            return model.grid_export[t] <= (model.pc) * model.is_exporting[t] + model.export_relax[t] #In w (Power)


        def _soc_max(model, t, ev):
            """
                EV maximum capacity (in Wh)
            """
            return model.EEV[t, ev] <= model.EEVmax[ev]

        def _soc_min(model, t, ev):
            """
                EV minimum capacity (in Wh)
            """
            return model.EEV[t, ev] + model.Eminsocrelax[t, ev] >= model.EEVmin[ev]

        def _balance2(model, t):
            """
                Energy balance in the system (in W))
            """
            return model.grid_import[t] == sum([model.PEV[t, ev] - model.PEVdc[t, ev] for ev in model.ev]) + model.pl[t]*model.dT + model.grid_export[t] - model.pv[t]*model.dT #In w (Power) without controlable loads
            #return model.grid_import[t] == sum([model.PEV[t, ev] - model.PEVdc[t, ev] for ev in model.ev]) + model.pl[t]*model.dT + model.flex_load_ewh[t]*model.dT + model.grid_export[t] - model.pv[t]*model.dT #In w (Power) without controlable loads
        
        def _limit_dsch(model, t, ev):
            """
                EV minimum discharge (in Wh)
            """
            return model.PEVdc[t, ev] >= model.PdchmaxEV[ev] * 0.1 * model.connected[t, ev] * model.ev_is_discharging[t,ev] #In w (Power)

        def _balance(model, t, ev):
            """
                EV status
            """
            return model.ev_is_charging[t, ev] + model.ev_is_discharging[t, ev] <= 1

        def _balance_exp(model, t):
            """
                EV status
            """
            return model.is_importing[t] + model.is_exporting[t] <= 1

        

        #****************************FOR MANDATORY SERVICES*******************************************
        # services operation constraints
        def _service(model,t,ev):
            """
                Service operation constraints
            """
            if model.is_curtailment[t]: # for wind service 
                return model.PEV[t, ev] == model.PEV_aux[t, ev] - model.service_relax_wind[t, ev]
            elif model.is_congestion_generation[t]: # for power generation congestion  service
                return model.PEV[t, ev] == model.PEV_aux[t, ev] - model.service_relax_congestiongen[t, ev]          
            elif model.is_congestion_consumption[t]: # for power consumption congestion service
                return model.PEVdc[t, ev] == model.PEVdc_aux[t, ev] - model.service_relax_congestion[t, ev] 
                
            return pyo.Constraint.Skip

        def _service2(model,t,ev):
            """
                Service operation constraints
            """

            if model.is_curtailment[t]: # for wind service 
                return model.PEVdc[t, ev] == 0
            if model.is_congestion_generation[t]: # for power generation congestion  service
                return model.PEVdc[t, ev] == 0
            elif model.is_congestion_consumption[t]: # for power consumption cngestion service
                return model.PEV[t, ev] - model.service_relax_congestion[t, ev] == 0
            return pyo.Constraint.Skip

        #****************************FOR OPTIONAL SERVICES*******************************************
        def _service3(model,t,ev):
           
            if model.is_curtailment[t] and model.pscurtbuy[t] <= model.userthrecurt: # for wind service 
                return model.PEV[t, ev] == model.PEV_aux[t, ev] - model.service_relax_wind[t, ev]
            elif model.is_congestion_generation[t] and  model.pscongbuy[t] <= model.userthrebuy: # for power generation congestion  service
                 return model.PEV[t, ev] == model.PEV_aux[t, ev] - model.service_relax_congestiongen[t, ev]          
            elif model.is_congestion_consumption[t] and model.pscongsell[t] >= model.userthresell: # for power consumption congestion service
                return model.PEVdc[t, ev] == model.PEVdc_aux[t, ev] - model.service_relax_congestion[t, ev] 
            return pyo.Constraint.Skip 

       
        #For the case in that congestion gen is activated at hihg energy price
        def auxservice(model,t,ev):
            if model.is_congestion_generation[t] and model.import_price[t] == model.highprice:
                return model.psnew[t] == ((model.lowprice*model.pscongbuy[t])/model.highprice)/model.pscongbuy[t]
            elif  model.import_price[t] != model.highprice: 
                return model.psnew[t] == 1
            elif  model.import_price[t] == model.highprice: 
                return model.psnew[t] == 1
            return pyo.Constraint.Skip  


        def _service4(model,t,ev):
            """
                Service operation constraints
            """

            if model.is_curtailment[t] and model.pscurtbuy[t] <= model.userthrebuy: # for wind service 
                return model.PEVdc[t, ev] == 0
            if model.is_congestion_generation[t] and model.pscongbuy[t] <= model.userthrebuy: # for power generation congestion  service
                return model.PEVdc[t, ev] == 0
            elif model.is_congestion_consumption[t] and model.pscongsell[t] >= model.userthresell: # for power consumption cngestion service
                return model.PEV[t, ev] - model.service_relax_congestion[t, ev] == 0
            return pyo.Constraint.Skip
               


        # model.finish_soc = pyo.Constraint(model.t, rule=_finish_soc)
        self.model.max_charge_pc = pyo.Constraint(self.model.t, self.model.ev, rule=_max_charge_pc)
        self.model.max_discharge_pc = pyo.Constraint(self.model.t, self.model.ev, rule=_max_discharge_pc)
        self.model.min_charge_pc = pyo.Constraint(self.model.t, self.model.ev, rule=_min_charge_pc)
        self.model.soc_max = pyo.Constraint(self.model.t, self.model.ev, rule=_soc_max)
        self.model.soc_min = pyo.Constraint(self.model.t, self.model.ev, rule=_soc_min)
        # model.soc_charge = pyo.Constraint(self.model.t, model.ev, rule=_soc_charge)
        self.model.soc_charge = pyo.Constraint(self.model.t, self.model.ev, rule=_opt_soc_charge)
        self.model.limit_dsch = pyo.Constraint(self.model.t, self.model.ev, rule=_limit_dsch)
        self.model.p_max = pyo.Constraint(self.model.t, rule=_p_max)
        self.model.p_max_export = pyo.Constraint(self.model.t, rule=_p_max_export)
        self.model.balance2 = pyo.Constraint(self.model.t, rule=_balance2)
        self.model.balance = pyo.Constraint(self.model.t, self.model.ev, rule=_balance)
        self.model.balance_exp = pyo.Constraint(self.model.t, rule=_balance_exp)
        #model.binary_load = pyo.Constraint(model.t, rule=_binary_load)
        #model.binary_load2 = pyo.Constraint(model.t, rule=_binary_load2)
        #model.binary_load3 = pyo.Constraint(model.t, rule=_binary_load3)
        #model.service = pyo.Constraint(model.t, model.ev, rule=_service)
        #model.service2 = pyo.Constraint(model.t, model.ev, rule=_service2)
        self.model.service3 = pyo.Constraint(self.model.t, self.model.ev, rule=_service3)
        self.model.service4 = pyo.Constraint(self.model.t, self.model.ev, rule=_service4)
        self.model.auxservice = pyo.Constraint(self.model.t, self.model.ev, rule=auxservice)
        #model.servicesaux = pyo.Constraint(model.t, model.ev, rule=_servicesaux)
        #model.servicesaux2 = pyo.Constraint(model.t, model.ev, rule=_servicesaux2)
        self.model.max_charge_pcaux = pyo.Constraint(self.model.t, self.model.ev, rule=_max_charge_pcaux)
        self.model.max_discharge_pcaux = pyo.Constraint(self.model.t, self.model.ev, rule=_max_discharge_pcaux)
        self.model.balance_energy_EVS3 = pyo.Constraint(self.model.t, self.model.ev, rule = _balance_energy_EVS3)   

        def objFntwo(model):
                print("Opa, este é o meu self.config_.soc_penalty = ", self.config_.soc_penalty )
                """
                Objective function measured in cost (€)
                """
                relax = 0
                price_total = 0
                events = 0
                
                relax_soc = 0
                penalize_soc = 0
                penalize_charge = 0 
                curt_penalty = self.config_.curtailment_penalty
                congestion_penalty = self.config_.congestion_penalty
                for t in np.arange(1, len(model.t) + 1):
                    price_total += ((model.grid_import[t]*model.dT)/1000 * (model.import_price[t]*model.pscongbuy[t]*model.pscurtbuy[t]*model.psnew[t]*model.pscongsell[t])) -\
                          ((model.grid_export[t]*model.dT)) / 1000 * (model.export_price[t]) + (model.import_relax[t]*model.dT)/1000 * self.config_.pc_penalty  + \
                            (model.export_relax[t]*model.dT)/1000 * self.config_.pc_penalty # €
                    #events += (model.service_relax_wind[t]*model.dT)/1000 * curt_penalty + (model.service_relax_congestion[t]*model.dT)/1000 * congestion_penalty
                    
                    for ev in np.arange(1, len(model.ev) + 1):
                        relax_soc += (model.Eminsocrelax[t,ev])/1000 * self.config_.soc_penalty  # € #Dividi por *model.dT #***Comment from Cindy***: Here is ok because soc relax is on Wh, /1000 is required to convert for kWh
                        
                        events += (model.service_relax_wind[t,ev]*model.dT)/1000 * curt_penalty + (model.service_relax_congestion[t, ev]*model.dT)/1000 * congestion_penalty + \
                            (model.service_relax_congestiongen[t, ev]*model.dT)/1000 * congestion_penalty
                        #events += (model.service_relax_wind[t,ev]*model.dT)/1000 * curt_penalty 

                        relax +=  (model.Etargetrelax[t, ev]) / 1000 * self.config_.trip_penalty # €#***Comment from Cindy***: Here is necessary import_relax*model.dT to convert for kW, Etargetrelax is on kWh
                        # events += (model.PEV[t, ev]*model.dT) / 1000 * model.wind_service[t] * curt_penalty + (model.PEVdc[t, ev]*model.dT) / 1000 * model.congestion_service[t] * curt_penalty #€
                        
                        penalize_soc += ((model.EEVmax[ev] - model.EEV[t, ev])/(model.EEVmax[ev]+ 1)) * model.import_price[t] * 0.0001

                        penalize_charge += (model.PEVdc[t, ev]*model.dT)/1000 * model.import_price[t] * 0.01

                return relax + events + price_total + relax_soc + penalize_soc + penalize_charge
        
        self.model.obj = pyo.Objective(expr=objFntwo, sense=pyo.minimize)

    def ext_pyomo_vals(self, vals):
        # Create a pandas Series from the Pyomo values
        s = pd.Series(vals.extract_values(),
                    index=vals.extract_values().keys())
        # Check if the Series is multi-indexed, if so, unstack it
        if type(s.index[0]) == tuple:    # it is multi-indexed
            s = s.unstack(level=1)
        else:
            # Convert Series to DataFrame
            s = pd.DataFrame(s)
        return s
    
    def generate_dataframe(self):
        # Converting Pyomo variables into DataFrames
        folder = f'./results'

        if not os.path.exists(folder):
            os.makedirs(folder)

        ev_charge_df = self.ext_pyomo_vals(self.model.PEV)
        ev_charge_df.columns = ["ev_charge_"+ f'{cp_id}' for cp_id in self.my_cp_id]
        #ev_charge_df.to_csv(folder + '/ev_charge.csv')
        
        ev_discharge_df = self.ext_pyomo_vals(self.model.PEVdc)
        ev_discharge_df.columns = ["ev_discharge_"+ f'{cp_id}' for cp_id in self.my_cp_id]
        #ev_discharge_df.to_csv(folder + '/ev_discharge.csv')

        EEV_df = self.ext_pyomo_vals(self.model.EEV)
        EEV_df.columns = ["SOC_"+ f'{cp_id}' for cp_id in self.my_cp_id]
        #EEV_df.to_csv(folder + '/EEV.csv')
        
        grid_import_df = self.ext_pyomo_vals(self.model.grid_import) 
        grid_import_df.columns = ["grid_import"]
        #grid_import_df.to_csv(folder + '/grid_import.csv')
        
        grid_export_df = self.ext_pyomo_vals(self.model.grid_export)
        grid_export_df.columns = ["grid_export"]
        #grid_export_df.to_csv(folder + '/grid_export.csv')
        
        ev_is_charging_df = self.ext_pyomo_vals(self.model.ev_is_charging) 
        ev_is_charging_df.columns = ["is_charging_"+ f'{cp_id}' for cp_id in self.my_cp_id]
        #ev_is_charging_df.to_csv(folder + '/ev_is_charging.csv')
        
        ev_is_discharging_df = self.ext_pyomo_vals(self.model.ev_is_discharging) 
        ev_is_discharging_df.columns = ["is_discharging_"+ f'{cp_id}' for cp_id in self.my_cp_id]
        #ev_is_discharging_df.to_csv(folder + '/ev_is_discharging.csv')
        
        # ****************** DF USED TO EXPORT RESULTS TO SCHEDULE MYSQL DATABASE*******
        self.new_df = pd.concat([ev_charge_df, ev_discharge_df, grid_import_df, grid_export_df, ev_is_charging_df, ev_is_discharging_df, EEV_df], axis=1)
       
    
    def execute(self, path=None, outfile=None):
        self.model.write('res_V4_EC.lp',  io_options={'symbolic_solver_labels': True})
        conf = dotenv.dotenv_values()
        solver_ = conf["SOLVER"]
        path = conf["PATH_TO_SOLVER"]
        opt = pyo.SolverFactory(solver_, executable=path)
        if solver_ == 'cplex':
            opt.options['LogFile'] = 'res_community.log'
        elif solver_ == 'ipopt':
            opt.options['output_file'] = 'res_community.log'
        self._set_params()
        self._set_vars()
        self._set_constraints()
        
        results = opt.solve(self.model)
        results.write()
        self.generate_dataframe()
        handle = scheduling.scheduling_open(outfile)
        for j, cp_id in enumerate(self.my_cp_id): # To insert the data by CP.
            for i, (_,row) in enumerate(self.new_df.iterrows()): 
                site_state = "import" if row["grid_import"] > 0 else "export" if row["grid_export"] > 0 else "idle"
                
                ev_state = "unplugged" if self.stepDepartureDateTime[j] < _ else "charge" if round(row[f"ev_charge_{cp_id}"]) > 0 else "discharge" if round(row[f"ev_discharge_{cp_id}"]) > 0 else "idle"

                timestamp = self.timesteps[i].strftime('%Y-%m-%d %H:%M:%S')
                timestamp = pendulum.parse(timestamp).to_datetime_string()

                decision_power = row[F"is_charging_{cp_id}"] * row[f"ev_charge_{cp_id}"] + row[f"is_discharging_{cp_id}"] * row[f"ev_discharge_{cp_id}"]
                
                # If the car is not there, decision power is automatically 0 kW.
                if self.stepDepartureDateTime[j] < _:
                    decision_power = 0 
                    scheduling.scheduling_insert(handle, timestamp, cp_id, ev_state, round(row[f"SOC_{cp_id}"],2), round(decision_power,2), 0, 0, 0, self.building.id, site_state)
                    continue
                
                #print("cp_id", cp_id, "time", timestamp, "is_charging: ", row[f"is_charging_{cp_id}"], "is_discharging: ", row[f"is_discharging_{cp_id}"], round(decision_power,2))
                scheduling.scheduling_insert(handle, timestamp, cp_id, ev_state, round(row[f"SOC_{cp_id}"],2), round(decision_power,2), 0, 0, 0, self.building.id, site_state)   
        
# It is needed the .env
def run_scheduling(site_id):
    conf = dotenv.dotenv_values()
    path = conf["PATH_TO_SOLVER"]
    config_file = conf["CONFIG_FILE"]
    outfile = conf["OUTPUT_FILE"]
    tariff_file = conf["TARIFF_FILE"]
    dynamic_folder = conf["DYNAMIC_PATH"]
    inputfile = conf["INPUT_FILE"]

    api = BuildingAPI(connection_config)
    my_building = api.get(site_id)
    my_config = Config.Config(config_file)
    new_optimizer = LrecOptimizer(my_building, my_config, dynamic_folder, inputfile, tariff_file)
    new_optimizer.execute(path, outfile)

if __name__ == "__main__":
    run_scheduling(26)
    pass