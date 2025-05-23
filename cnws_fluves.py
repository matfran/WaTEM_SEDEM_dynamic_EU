"""
This script is a slight modification of a module from pywatemsedem which allows WaTEM/SEDEM
to be run through a python wrapper. The complete source code for the module can be found at:
    
    https://github.com/watem-sedem/pywatemsedem?tab=readme-ov-file
    
This module is included to allow the reproduction of the code associated with the manuscript.
However, all credit for the development of this W/S wrapper should be given to the authors
of the pywatemsedem code. 

"""



import os
import subprocess
import configparser
import logging
import shutil

default_opt = {}
default_opt['ConvertOutput'] = 0
default_opt['Manual outlet selection'] = 0
default_opt['Include sewers'] = 0
default_opt['UserProvidedKTC'] = 0
default_opt['Adjusted Slope'] = 0
default_opt['Buffer reduce Area'] = 1
default_opt['Force Routing'] = 0
default_opt['River Routing'] = 0
default_opt['Estimate clay content'] = 0
default_opt['Calculate Tillage Erosion'] = 0
default_opt['Create ktil map'] = 0
default_opt['L model'] = 'Desmet1996_Vanoost2003'
default_opt['S model'] = 'McCool1987' #'Nearing1997' #old_kul_run = mcool1989
default_opt['TC model'] = 'VanOost2000'
default_opt['Deposition_limited'] = 0

default_ebmopt = {}
default_ebmopt['Include buffers'] = 0
default_ebmopt['Include ditches'] = 0
default_ebmopt['Include dams'] = 0

default_vars = {}
default_vars['Parcel connectivity cropland'] = 90
default_vars['Parcel connectivity forest'] = 30
default_vars['Parcel trapping efficiency cropland'] = 0
default_vars['Parcel trapping efficiency forest'] = 75
default_vars['Parcel trapping efficiency pasture'] = 75
default_vars['Max kernel'] = 50
default_vars['Max kernel river'] = 100
default_vars['Bulk density'] = 1350
default_vars['LS correction'] = 0.7
default_vars['ktc low'] = 1
default_vars['ktc high'] = 3
default_vars['ktc limit'] = 0.01
default_vars['ktil default'] = 600
default_vars['ktil threshold'] = 0.01
default_vars['Clay content parent material'] = 28
default_vars['SewerInletEff'] = 100
default_vars['Deposition_limit_mm'] = 5.0 #5mm


default_calibration = {}
default_calibration['Calibrate'] = 0
default_calibration['KTcHigh_lower'] = 5.0
default_calibration['KTcHigh_upper'] = 40.0
default_calibration['KTcLow_lower'] = 1.0
default_calibration['KTcLow_upper'] = 20.0
default_calibration['steps'] = 12


default_out = {}
default_out['Write aspect'] = 0
default_out['Write Slope'] = 1
default_out['Write LS'] = 1
default_out['Write upstream area'] = 1
default_out['Output per VHA river segment'] = 0
default_out['Write routing table'] = 1
default_out['Write routing column/row'] = 1
default_out['Write RUSLE'] = 1
default_out['Write sediment export'] = 1
default_out['Write water erosion'] = 1

#create a class where neccessary variables are initiated 
class CNWS:
    def __init__(self):
        self.infolder = None
        self.outfolder = None

        self.catchm_name = ''

        # input files
        self.k = None
        self.lu = None
        self.c = None
        self.p = None
        self.r = None
        self.runoff = None 
        self.dem = None
        self.ini = None
        self.riviersegm = None
        self.outletmap = None
        self.ktcmap = None
        self.ditchmap = None
        self.dammap = None
        self.sewermap = None
        self.buffermap = None
        self.bufferdata = []
        self.forcedroutingdata = []
        self.adj_edges = None
        self.up_edges = None
        self.riverrouting = None

        self.version = 'WS'

        self.EBMOptions = {}
        self.ModelOptions = {}
        self.Variables = {}
        self.Calibration = {}
        self.Output = {}

        self.logger = logging.getLogger(__name__)

    def set_choices(self, opt=default_opt, ebmopt=default_ebmopt, variables=default_vars, cal = default_calibration, out=default_out):
        self.Output = out
        self.Variables = variables
        self.EBMOptions = ebmopt
        self.ModelOptions = opt
        self.Calibration = cal

    def create_ini(self):
        """
        Creates an ini-file for the scenario
        sets the self.ini argument
        :return: Nothing
        """

        self.logger.info("Creating ini-file...")
        self.ini = os.path.join(self.infolder, "ini_%s.ini" % self.catchm_name)
        self.ini_out = os.path.join(self.outfolder, "ini_%s.ini" % self.catchm_name)

        cfg = configparser.ConfigParser()

        cfg.add_section("Working directories")
        cfg.set("Working directories", "Input directory", str(self.infolder))
        cfg.set("Working directories", "Output directory", str(self.outfolder))

        cfg.add_section("Files")
        cfg.set("Files", "DTM filename", self.dem)
        cfg.set("Files", "P factor map filename", self.p)
        cfg.set("Files", "Parcel filename", self.lu)
        cfg.set("Files", "C factor map filename", self.c)
        cfg.set("Files", "K factor filename", self.k)
        #if 'dynamic' in self.ModelOptions["TC model"].lower():
        cfg.set("Files", "Runoff factor map filename", self.runoff)

        cfg.add_section("User Choices")
        cfg.set("User Choices", "Max kernel", str(self.Variables["Max kernel"]))
        cfg.set("User Choices", "Max kernel river", str(self.Variables["Max kernel river"]))
        cfg.set("User Choices", "Use R factor", "1")
        cfg.set("User Choices", "Simplified model version", "1")
        cfg.set("User Choices", "L model", self.ModelOptions["L model"])
        cfg.set("User Choices", "S model", self.ModelOptions["S model"])
        cfg.set("User Choices", "TC model", self.ModelOptions["TC model"])
        cfg.set("User Choices", "Adjusted Slope", str(self.ModelOptions["Adjusted Slope"]))
        cfg.set("User Choices", "Deposition_limited", str(self.ModelOptions["Deposition_limited"]))
        cfg.set("User Choices", "Buffer reduce Area", str(self.ModelOptions["Buffer reduce Area"]))

        cfg.add_section("Output maps")
        cfg.set("Output maps", "Write aspect", str(self.Output["Write aspect"]))
        cfg.set("Output maps", "Write LS factor", str(self.Output["Write LS"]))
        cfg.set("Output maps", "Write RUSLE", str(self.Output["Write RUSLE"]))
        cfg.set("Output maps", "Write sediment export", str(self.Output["Write sediment export"]))
        cfg.set("Output maps", "Write slope", str(self.Output["Write Slope"]))
        cfg.set("Output maps",  "Write upstream area", str(self.Output["Write upstream area"]))
        cfg.set("Output maps", "Write water erosion", str(self.Output["Write water erosion"]))
        cfg.set("Output maps", "Write routing table", str(self.Output["Write routing table"]))
        cfg.set("Output maps", "Write routing column/row", str(self.Output["Write routing column/row"]))

        cfg.add_section("Variables")
        cfg.set("Variables", "Parcel connectivity cropland", str(int(self.Variables["Parcel connectivity cropland"])))
        cfg.set("Variables", "Parcel connectivity forest", str(int(self.Variables["Parcel connectivity forest"])))
        cfg.set("Variables", "Parcel trapping efficiency cropland", str(int(self.Variables["Parcel trapping efficiency cropland"])))
        cfg.set("Variables", "Parcel trapping efficiency forest", str(int(self.Variables["Parcel trapping efficiency forest"])))
        cfg.set("Variables", "Parcel trapping efficiency pasture", str(int(self.Variables["Parcel trapping efficiency pasture"])))
        cfg.set("Variables", "R factor", str(self.r))
        cfg.set("Variables", "Endtime model", "0")
        cfg.set("Variables", "LS correction", str(self.Variables["LS correction"]))
        cfg.set("Variables", "Bulk density", str(int(self.Variables["Bulk density"])))
        cfg.set("Variables", "Deposition_limit_mm", str(int(self.Variables["Deposition_limit_mm"])))

        cfg.add_section("Calibration")
        cfg.set("Calibration", "Calibrate", str(int(self.Calibration["Calibrate"])))
        cfg.set("Calibration", "KTcHigh_lower", str(self.Calibration["KTcHigh_lower"]))
        cfg.set("Calibration", "KTcHigh_upper", str(self.Calibration["KTcHigh_upper"]))
        cfg.set("Calibration", "KTcLow_lower", str(self.Calibration["KTcLow_lower"]))
        cfg.set("Calibration", "KTcLow_upper", str(self.Calibration["KTcLow_upper"]))
        cfg.set("Calibration", "steps", str(int(self.Calibration["steps"])))



        cfg.set("User Choices", "Output per VHA river segment", str(self.Output["Output per VHA river segment"]))
        if self.Output["Output per VHA river segment"]:
            cfg.set("Files", "River segment filename", self.riviersegm)

        cfg.set("User Choices", "Manual outlet selection", str(self.ModelOptions["Manual outlet selection"]))
        if self.ModelOptions["Manual outlet selection"]:
            cfg.set("Files", "Outlet map filename", self.outletmap)

        if self.ModelOptions["UserProvidedKTC"]:
            cfg.set("User Choices", "Create ktc map", "0")
            cfg.set("Files", "ktc map filename", self.ktcmap)
        else:
            cfg.set("User Choices", "Create ktc map", "1")
            cfg.set("Variables", "ktc low", str(self.Variables["ktc low"]))
            cfg.set("Variables", "ktc high", str(self.Variables["ktc high"]))
            cfg.set("Variables", "ktc limit", str(self.Variables["ktc limit"]))

        cfg.set("User Choices", "Calculate Tillage Erosion", str(self.ModelOptions["Calculate Tillage Erosion"]))
        if self.ModelOptions["Calculate Tillage Erosion"]:
            cfg.set("User Choices", "Create ktil map", str(self.ModelOptions["Create ktil map"]))
            if self.ModelOptions["Create ktil map"] == 1:
                cfg.set("Variables", "ktil default", str(self.Variables["ktil default"]))
                cfg.set("Variables", "ktil threshold", str(self.Variables["ktil threshold"]))

        # Estimating clay content in model?
        cfg.set("User Choices", "Estimate clay content", str(self.ModelOptions["Estimate clay content"]))
        if self.ModelOptions["Estimate clay content"]:
            cfg.set("Variables", "Clay content parent material", str(int(self.Variables["Clay content parent material"])))

        # Grachten gebruiken?
        cfg.set("User Choices", "Include ditches", str(self.EBMOptions["Include ditches"]))
        if self.EBMOptions["Include ditches"]:
            cfg.set("Files", "Ditch map filename", self.ditchmap)

        # Geleidende dammen gebruiken?
        cfg.set("User Choices", "Include dams", str(self.EBMOptions["Include dams"]))
        if self.EBMOptions["Include dams"]:
            cfg.set("Files", "Dam map filename", self.dammap)

        # Using sewers
        cfg.set("User Choices", "Include sewers", str(self.ModelOptions["Include sewers"]))
        if self.ModelOptions["Include sewers"]:
            cfg.set("Variables", "Sewer exit", str(self.Variables["TrappedInSewers"]))
            cfg.set("Files", "Sewer map filename", self.sewermap)

        # Using buffers in model?
        cfg.set("User Choices", "Include buffers", str(self.EBMOptions["Include buffers"]))
        if self.EBMOptions["Include buffers"]:
            cfg.set("Files", "Buffer map filename", self.buffermap)
            cfg.set("Variables", "Number of buffers", str(len(self.bufferdata)))
            if len(self.bufferData) != 0:
                for buffer_ in self.bufferdata:
                    sectie = "Buffer %s" % buffer_[0]
                    cfg.add_section(sectie)
                    cfg.set(sectie, "Volume", str(buffer_[1]))
                    cfg.set(sectie, "Height dam", str(buffer_[2]))
                    cfg.set(sectie, "Height opening", str(buffer_[3]))
                    cfg.set(sectie, "Opening area", str(buffer_[4]))
                    cfg.set(sectie, "Discharge coefficient", str(buffer_[5]))
                    cfg.set(sectie, "Width dam", str(buffer_[6]))
                    cfg.set(sectie, "Trapping efficiency", str(buffer_[7]))
                    cfg.set(sectie, "Extension ID", str(buffer_[8]))

        # force routing
        cfg.set("User Choices", "Force Routing", str(self.ModelOptions["Force Routing"]))
        if self.ModelOptions["Force Routing"]:
            cfg.set("Variables", "Number of Forced Routing", str(len(self.forcedroutingdata)))
            for row in self.forcedroutingdata.itertuples():
                sectie = "Forced Routing %s" % getattr(row, "NR")
                cfg.add_section(sectie)
                cfg.set(sectie, "from col", str(int(getattr(row, "from col"))))
                cfg.set(sectie, "from row", str(int(getattr(row, "from row"))))
                cfg.set(sectie, "target col", str(int(getattr(row, "to col"))))
                cfg.set(sectie, "target row", str(int(getattr(row, "to row"))))

        # river routing
        if self.ModelOptions["River Routing"]:
            cfg.set("User Choices", "river routing", str(self.ModelOptions["River Routing"]))
            cfg.set("Files", "adjectant segments", self.adj_edges)
            cfg.set("Files", "upstream segments", self.up_edges)
            cfg.set("Files", "river routing filename", self.riverrouting)

        f = open(self.ini, "w")
        cfg.write(f)
        f.close()
        return
    
    def copy_ini(self, cnws_path):
        self.ini_path_new = os.path.join(self.infolder, "ini.ini")
        #self.ini_path_outcopy = os.path.join(self.outfolder, "ini.ini")
        shutil.copyfile(self.ini, self.ini_path_new)
        #shutil.copyfile(self.ini, self.ini_out)
        return

    def run_model(self, cnws_path):
        """
        Run the CN-WS model
        :return: Nothing
        """
        try:
            cmd_args = [cnws_path, str(self.ini_path_new)]
            #print(cmd_args)
            subprocess.check_call(cmd_args)
            self.logger.info("Modelrun finished!")
            shutil.copyfile(self.ini, self.ini_out)
        except subprocess.CalledProcessError as e:
            msg = "Failed to run CNWS-model"
            self.logger.error(msg)
            self.logger.error(e.cmd)
        return

    def get_output(self):
        """
        Makes a list of all files in the outputfolder
        :return: list with all files in outputfolder
        """
        outfiles = []
        for f in os.listdir(self.outfolder):
            outfiles.append(os.path.join(self.outfolder, f))
        return outfiles

