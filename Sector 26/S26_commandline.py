# Enables command line scripting for HXN microscope operation
# start this with /APSshare/anaconda/x86_64/bin/ipython -i S26_commandline.py

import sys
import epics
import epics.devices
import time
import datetime
import numpy as np
import os 
import math
import socket
from matplotlib import pyplot

# initalize logging object
logger = Logger()

# Define motors
fomx = epics.Motor('26idcnpi:m10.')
fomy = epics.Motor('26idcnpi:m11.')
fomz = epics.Motor('26idcnpi:m12.')
#samx = epics.Motor('26idcnpi:m16.')
samy = epics.Motor('26idcnpi:m17.')
#samz = epics.Motor('26idcnpi:m18.')
samth = epics.Motor('atto2:PIC867:1:m1.')
osax = epics.Motor('26idcnpi:m13.')
osay = epics.Motor('26idcnpi:m14.')
osaz = epics.Motor('26idcnpi:m15.')
condx = epics.Motor('26idcnpi:m5.')
#attox = epics.Motor('atto2:m3.')
#attoz = epics.Motor('atto2:m4.')
#attoz = epics.Motor('26idcNES:sm27.')
attox = epics.Motor('atto2:m4.')
attoz = epics.Motor('atto2:m3.')
#samchi = epics.Motor('atto2:m1.')
#samphi = epics.Motor('atto2:m2.')
objx = epics.Motor('26idcnpi:m1.')
xrfx = epics.Motor('26idcDET:m7.')
#piezox = epics.Motor('26idcSOFT:sm1.')
#piezoy = epics.Motor('26idcSOFT:sm2.')
#lensx = epics.Motor('4idcThor:m2.')
#lensy = epics.Motor('4idcThor:m3.')
chopy = epics.Motor('26idc:m7.')
chopx = epics.Motor('26idc:m8.')

#hybridx = epics.Device('26idcDEV:X_HYBRID_SP.VAL', attrs=('VAL','DESC'))
hybridx = epics.Device('26idcnpi:X_HYBRID_SP.VAL', attrs=('VAL','DESC'))
#hybridx = epics.Device('26idcnpi:m34.VAL', attrs=('VAL','DESC'))
hybridx.add_pv('26idcnpi:m34.RBV', attr='RBV')
#hybridy  = epics.Device('26idcDEV:Y_HYBRID_SP.VAL', attrs=('VAL','DESC'))
hybridy  = epics.Device('26idcnpi:Y_HYBRID_SP.VAL', attrs=('VAL','DESC'))
hybridy.add_pv('26idcnpi:m35.RBV', attr='RBV')
twotheta = epics.Motor('26idcSOFT:sm3.')
#twotheta = epics.Device('26idcDET:base:Theta.VAL', attrs=('VAL','DESC'))
#twotheta.add_pv('26idcDET:base:Theta_d', attr='RBV')
#dcmy = epics.Device('26idb:DAC1_1.VAL', attrs=('VAL','DESC'))
#dcmy.add_pv('26idcDET:base:Theta_d', attr='RBV')
not_epics_motors = [hybridx.NAME, hybridy.NAME, twotheta.NAME]

# Define zone plate in-focus position
optic_in_x = -1065.7
optic_in_y = -964.68
optic_in_z = 2441.5
# 10.4 kev below
#optic_in_x = -1060.8
#optic_in_y = -959.34
#optic_in_z = -1223
#optic_in_x = -1060
#optic_in_y = -953.6
#optic_in_z = -1663

# Define medipix3 in-beam position
mpx_in_x = -.15
mpx_in_y = 93.7

# Define genie camera in-beam position
genie_in_x = 6.3
genie_in_y = 133.2

# Define pin diode in-beam position
pind_in_x = -36.5
pind_in_y = 123.0



# Define movement functions
def mov(motor,position):
	if motor in [fomx, fomy, samy]:
		epics.caput('26idcnpi:m34.STOP',1)
		epics.caput('26idcnpi:m35.STOP',1)
		epics.caput('26idcSOFT:userCalc1.SCAN',0)
		epics.caput('26idcSOFT:userCalc3.SCAN',0)
	if motor.NAME in not_epics_motors:
		motor.VAL = position
		time.sleep(1)
		print(motor.DESC+"--->  "+str(motor.RBV))
	else:
		result = motor.move(position, wait=True)
		if result==0:
			time.sleep(0.5)
			print(motor.DESC+" ---> "+str(motor.RBV))
			fp = open(logbook,"a")
			fp.write(motor.DESC+" ---> "+str(motor.RBV)+"\n")
			fp.close
			epics.caput('26idcSOFT:userCalc1.SCAN',6)
			epics.caput('26idcSOFT:userCalc3.SCAN',6)
		else:
			print("Motion failed")	

def movr(motor,tweakvalue):
	if motor in [fomx, fomy, samy]:
		epics.caput('26idcnpi:m34.STOP',1)
		epics.caput('26idcnpi:m35.STOP',1)
	if ( (motor in [hybridx, hybridy]) and ( (abs(hybridx.RBV-hybridx.VAL)>100) or (abs(hybridy.RBV-hybridy.VAL)>100) ) ):
		print("Please use lock_hybrid() to lock piezos at current position first...")
		return
	if motor.NAME in not_epics_motors:
		motor.VAL = motor.VAL+tweakvalue
		time.sleep(1)
		print(motor.DESC+"--->  "+str(motor.RBV))
	else:
		result = motor.move(tweakvalue, relative=True, wait=True)
		if result==0:
			time.sleep(0.5)
			print(motor.DESC+" ---> "+str(motor.RBV))
			fp = open(logbook,"a")
			fp.write(motor.DESC+" ---> "+str(motor.RBV)+"\n")
			fp.close
		else:
			print("Motion failed")

def zp_in():
	print('Moving ZP to focal position...\n')
	epics.caput('26idcSOFT:userCalc1.SCAN',0);
	epics.caput('26idcSOFT:userCalc3.SCAN',0);
	epics.caput('26idbSOFT:userCalc3.SCAN',0);
	mov(fomx,optic_in_x)
	mov(fomy,optic_in_y)
	mov(fomz,optic_in_z)
	epics.caput('26idcSOFT:userCalc1.SCAN',5);
	epics.caput('26idcSOFT:userCalc3.SCAN',5);
	epics.caput('26idbSOFT:userCalc3.SCAN',5);

def zp_out():
	global mpx_in_x, mpx_in_y
	tempx = epics.caget('26idc:sft01:ph02:ao09.VAL')
	tempy = epics.caget('26idc:robot:Y1.VAL')
	temp2th = epics.caget('26idcDET:base:Theta.VAL')
	if ( (abs(mpx_in_x-tempx)<0.1) and (abs(mpx_in_y-tempy)<0.1) and (abs(temp2th)<1.0) ):
		print("Please use genie_in() to move medipix out of beam first...")
		return
	print('Moving ZP out of beam...\n')
	epics.caput('26idcSOFT:userCalc1.SCAN',0);
	epics.caput('26idcSOFT:userCalc3.SCAN',0);
	epics.caput('26idbSOFT:userCalc3.SCAN',0);
	mov(fomx,optic_in_x+3500.0)
	mov(fomy,optic_in_y)
	mov(fomz,-4000.0)
	epics.caput('26idcSOFT:userCalc1.SCAN',5);
	epics.caput('26idcSOFT:userCalc3.SCAN',5);
	epics.caput('26idbSOFT:userCalc3.SCAN',5);

def lock_hybrid():
	tempx = hybridx.RBV
	time.sleep(1)
	mov(hybridx,tempx)
	time.sleep(1)
	tempy = hybridy.RBV
	time.sleep(1)
	mov(hybridy,tempy)
	time.sleep(1)

def unlock_hybrid():
	tempx = hybridx.RBV
	tempy = hybridy.RBV
	print("before unlock: x = {0} and y = {1}".format(tempx, tempy))
	epics.caput('26idcnpi:m34.STOP',1)
	epics.caput('26idcnpi:m35.STOP',1)  
	mov(fomx,optic_in_x);
	mov(fomy,optic_in_y);
	time.sleep(1)
	tempx = hybridx.RBV
	tempy = hybridy.RBV
	print("after unlock: x = {0} and y = {1}".format(tempx, tempy))

def set_zp_in():
	global optic_in_x, optic_in_y, optic_in_z
	print("ZP X focal position set to: "+str(fomx.RBV))
	optic_in_x = fomx.RBV
	print("ZP Y focal position set to: "+str(fomy.RBV))
	optic_in_y = fomy.RBV
	print("ZP Z focal position set to: "+str(fomz.RBV))
	optic_in_z = fomz.RBV

def set_medipix_in():
	global mpx_in_x, mpx_in_y
	tempx = epics.caget('26idc:sft01:ph02:ao09.VAL')
	tempy = epics.caget('26idc:robot:Y1.VAL')
	print("Medipix X position set to: "+str(tempx))
	mpx_in_x = tempx
	print("Medipix Y position set to: "+str(tempy))
	mpx_in_y = tempy

def set_genie_in():
	global genie_in_x, genie_in_y
	tempx = epics.caget('26idc:sft01:ph02:ao09.VAL')
	tempy = epics.caget('26idc:robot:Y1.VAL')
	print("Genie X position set to: "+str(tempx))
	genie_in_x = tempx
	print("Genie Y position set to: "+str(tempy))
	genie_in_y = tempy

def beamstop_in():
	print('Moving downstream beamstop in...\n')
	mov(objx,-500)
	time.sleep(1)
	epics.caput('26idcnpi:m1.STOP',1)

def beamstop_out():
	print('Moving downstream beamstop out...\n')
	#mov(objx,-2495)
	mov(objx,-1500)
	time.sleep(1)
	epics.caput('26idcnpi:m1.STOP',1)

def prism_in():
	print('Moving prism for on-axis microscope in...\n')
	mov(condx,-35000)

def prism_out():
	print('Moving prism out...\n')
	mov(condx,-7056)
	#condy = -1302 10/15/2018

def xrf_in():
	print('Moving inboard XRF detector in...\n')
	mov(xrfx,-265) #CHECK COLLIMATOR FLUSH TO FRONT FACE

def xrf_out():
	print('Moving inboard XRF detector out...\n')
	mov(xrfx,-400)

def chopper_in():
	print('Moving chopper in...\n')
	mov(chopy,5.7) #chopper x 13.4 coarse alignment

def chopper_out():
	print('Moving chopper to beam pass through...\n')
	mov(chopy,4.7) #chopper x 13.4 coarse alignment

def pixirad_in():
	print('Moving pixirad detector on beam axis...\n')
	temp2th = epics.caget('26idcDET:base:Theta.VAL')
	tempgam = epics.caget('26idcDET:robot:Gamma.VAL')
	epics.caput('26idc:sft01:ph02:ao09.VAL',0.3)
	epics.caput('26idc:robot:Y1.VAL',92.0)
	time.sleep(1)
	epics.caput('26idcDET:base:Theta.VAL',temp2th)
	time.sleep(1)
	epics.caput('26idcDET:robot:Gamma.VAL',tempgam)

def medipix_in():
	global mpx_in_x,mpx_in_y,optic_in_x
	temp2th = epics.caget('26idcDET:base:Theta.VAL')
	tempgam = epics.caget('26idcDET:robot:Gamma.VAL')
	if ( (abs(optic_in_x-fomx.RBV)>3000.0) and (abs(temp2th)<1.0) ):
		print("Please use zp_in() to block the direct beam first...")
		return
	print('Moving medipix 3 detector on beam axis...\n')
	epics.caput('26idc:sft01:ph02:ao09.VAL',mpx_in_x)
	epics.caput('26idc:robot:Y1.VAL',mpx_in_y)   
	time.sleep(1)
	epics.caput('26idcDET:base:Theta.VAL',temp2th)
	time.sleep(1)
	epics.caput('26idcDET:robot:Gamma.VAL',tempgam)

def genie_in():
	global genie_in_x,genie_in_y
	print('Moving Genie detector on beam axis...\n')
	temp2th = epics.caget('26idcDET:base:Theta.VAL')
	tempgam = epics.caget('26idcDET:robot:Gamma.VAL')
	if ( (abs(temp2th)>0.05) or (abs(tempgam)>0.05) ):
		print("**Warning**  you are not imaging the direct beam - move two theta and gamma to zero to do this.")
	epics.caput('26idc:sft01:ph02:ao09.VAL',genie_in_x)
	epics.caput('26idc:robot:Y1.VAL',genie_in_y)
	time.sleep(1)
	epics.caput('26idcDET:base:Theta.VAL',temp2th)
	time.sleep(1)
	epics.caput('26idcDET:robot:Gamma.VAL',tempgam)

def pind_in():
	global pind_in_x,pind_in_y
	print('Moving pin diode detector on beam axis...\n')
	temp2th = epics.caget('26idcDET:base:Theta.VAL')
	tempgam = epics.caget('26idcDET:robot:Gamma.VAL')
	epics.caput('26idc:sft01:ph02:ao09.VAL',pind_in_x)
	epics.caput('26idc:robot:Y1.VAL',pind_in_y)
	time.sleep(1)
	epics.caput('26idcDET:base:Theta.VAL',temp2th)
	time.sleep(1)
	epics.caput('26idcDET:robot:Gamma.VAL',tempgam)

# Link to scan records, patched to avoid overwriting PVs
scanrecord = "26idbSOFT"
temp1 = epics.caget(scanrecord+':scan1.T1PV')
temp2 = epics.caget(scanrecord+':scan1.T2PV')
temp3 = epics.caget(scanrecord+':scan1.T3PV')
temp4 = epics.caget(scanrecord+':scan1.T4PV')
temp5 = epics.caget(scanrecord+':scan1.NPTS')
sc1 = epics.devices.Scan(scanrecord+":scan1")
time.sleep(1)
sc1.T1PV=temp1
sc1.T2PV=temp2
sc1.T3PV=temp3
sc1.T4PV=temp4
sc1.NPTS=temp5
time.sleep(1)
temp1 = epics.caget(scanrecord+':scan2.T1PV')
temp2 = epics.caget(scanrecord+':scan2.T2PV')
temp3 = epics.caget(scanrecord+':scan2.T3PV')
temp4 = epics.caget(scanrecord+':scan2.T4PV')
temp5 = epics.caget(scanrecord+':scan2.NPTS')
time.sleep(1)
sc2 = epics.devices.Scan(scanrecord+":scan2")
sc2.T1PV=temp1
sc2.T2PV=temp2
sc2.T3PV=temp3
sc2.T4PV=temp4
sc2.NPTS=temp5
logbook = epics.caget(scanrecord+':saveData_fileSystem',as_string=True)+'/'+epics.caget(scanrecord+':saveData_subDir',as_string=True)+'/logbook.txt'

# Turn on/off detectors and set exposure times
def detectors(det_list):
	numdets = np.size(det_list)
	if(numdets<1 or numdets>4):
		print("Unexpected number of detectors")
	else:
		sc1.T1PV = ''
		sc1.T2PV = ''
		sc1.T3PV = ''
		sc1.T4PV = ''
		for ii in range(numdets):
			if det_list[ii]=='scaler':
				exec('sc1.T'+str(ii+1)+'PV = \'26idc:3820:scaler1.CNT\'')
			if det_list[ii]=='xrf':
				exec('sc1.T'+str(ii+1)+'PV = \'26idcXMAP:EraseStart\'')
			if det_list[ii]=='xrf_hscan':
				exec('sc1.T'+str(ii+1)+'PV = \'26idbSOFT:scanH.EXSC\'')
			if det_list[ii]=='andor':
				exec('sc1.T'+str(ii+1)+'PV = \'26idcNEO:cam1:Acquire\'')
			if det_list[ii]=='ccd':
				exec('sc1.T'+str(ii+1)+'PV = \'26idcCCD:cam1:Acquire\'')
			if det_list[ii]=='pixirad':
				exec('sc1.T'+str(ii+1)+'PV = \'dp_pixirad_xrd75:cam1:Acquire\'')
			#if det_list[ii]=='pilatus':
                # exec('sc1.T'+str(ii+1)+'PV = \'S18_pilatus:cam1:Acquire\'')
			if det_list[ii]=='pilatus':
				exec('sc1.T'+str(ii+1)+'PV = \'dp_pilatusASD:cam1:Acquire\'')
			#if det_list[ii]=='pilatus':
			#    exec('sc1.T'+str(ii+1)+'PV = \'S33-pilatus1:cam1:Acquire\'')
			if det_list[ii]=='medipix':
				exec('sc1.T'+str(ii+1)+'PV = \'QMPX3:cam1:Acquire\'')
				#exec('sc1.T'+str(ii+1)+'PV = \'dp_pixirad_msd1:cam1:MultiAcquire\'')
			if det_list[ii]=='vortex':
				exec('sc1.T'+str(ii+1)+'PV = \'dp_vortex_xrd77:mca1EraseStart\'')

def count_time(dettime):
	det_trigs = [sc1.T1PV, sc1.T2PV, sc1.T3PV, sc1.T4PV]
	if '26idc:3820:scaler1.CNT' in det_trigs:
		epics.caput("26idc:3820:scaler1.TP",dettime)
	if ('26idcXMAP:EraseStart' in det_trigs) or ('26idbSOFT:scanH.EXSC' in det_trigs):
		epics.caput("26idcXMAP:PresetReal",dettime)
	if '26idcNEO:cam1:Acquire' in det_trigs:
		epics.caput("26idcNEO:cam1:Acquire",0)
		time.sleep(0.5)
		epics.caput("26idcNEO:cam1:AcquireTime",dettime)
		epics.caput("26idcNEO:cam1:ImageMode","Fixed")
	if '26idcCCD:cam1:Acquire' in det_trigs:
		epics.caput("26idcCCD:cam1:Acquire",0)
		time.sleep(0.5)
		epics.caput("26idcCCD:cam1:AcquireTime",dettime)
		epics.caput("26idcCCD:cam1:ImageMode","Fixed")
		time.sleep(0.5)
		epics.caput("26idcCCD:cam1:Initialize",1)
	if 'dp_pixirad_xrd75:cam1:Acquire' in det_trigs:
		epics.caput("dp_pixirad_xrd75:cam1:AcquireTime",dettime)
	if 'dp_pilatusASD:cam1:Acquire' in det_trigs:
		epics.caput("dp_pilatusASD:cam1:AcquireTime",dettime)
   # if 'dp_pilatus4:cam1:Acquire' in det_trigs:
       # epics.caput("dp_pilatus4:cam1:AcquireTime",dettime)
	if 'QMPX3:cam1:Acquire' in det_trigs:
		epics.caput("QMPX3:cam1:AcquirePeriod",dettime*1000)
		#epics.caput("QMPX3:cam1:AcquirePeriod",500)
		#epics.caput("QMPX3:cam1:NumImages",np.round(dettime/0.5))
   # if 'S33-pilatus1:cam1:Acquire' in det_trigs:
       # epics.caput("S33-pilatus1:cam1:AcquireTime",dettime)
   # if 'S18_pilatus:cam1:Acquire' in det_trigs:
       # epics.caput("S18_pilatus:cam1:AcquireTime",dettime)
	# if 'dp_pixirad_msd1:MultiAcquire' in det_trigs:
   #     epics.caput("dp_pixirad_msd1:cam1:AcquireTime",dettime)
   # if 'dp_pixirad_msd1:cam1:Acquire' in det_trigs:
   #     epics.caput("dp_pixirad_msd1:cam1:AcquireTime",dettime)
	if 'dp_vortex_xrd77:mca1EraseStart' in det_trigs:
		epics.caput("dp_vortex_xrd77:mca1.PRTM",dettime)

def prescan(scanArgs):
	scannum = epics.caget(scanrecord+':saveData_scanNumber',as_string=True)
	print("scannum is {0}".format(scannum))
	pathname = epics.caget(scanrecord+':saveData_fullPathName',as_string=True)
	detmode = epics.caget("QMPX3:cam1:ImageMode");
	savemode = epics.caget("QMPX3:TIFF1:EnableCallbacks")
	if( detmode == 2 ):
		print("Warning - Medipix is in continuous acquisition mode - changing this to single")
		epics.caput("QMPX3:cam1:ImageMode",0)
		time.sleep(1)
	if( savemode == 0 ):
		print("Warning - Medipix is not saving images - enabling tiff output")
		epics.caput("QMPX3:TIFF1:EnableCallbacks",1)
		time.sleep(1)
	if( epics.caget('PA:26ID:SCS_BLOCKING_BEAM.VAL') ):
		print("Warning - C station shutter is closed - opening shutter")
		epics.caput("PC:26ID:SCS_OPEN_REQUEST.VAL",1)
		time.sleep(2)

	#!UPDATE! - do we want the save paths to be set this way?
	epics.caput("QMPX3:TIFF1:FilePath",pathname[:-4]+'Images/'+scannum+'/')
	time.sleep(1)
	epics.caput("QMPX3:TIFF1:FileName",'scan_'+scannum+'_img')
	time.sleep(1)
	epics.caput("dp_pilatusASD:cam1:FilePath",'/home/det'+pathname[5:-4]+'Images/'+scannum+'/')
	time.sleep(1)
	epics.caput("dp_pilatusASD:cam1:FileName",'scan_'+scannum+'_img_Pilatus')
	time.sleep(1)
	#epics.caput("S18_pilatus:cam1:FilePath",'/mnt/Sector_26'+pathname[5:-4]+'Images/'+scannum+'/')
	#time.sleep(1)
	#epics.caput("S18_pilatus:cam1:FileName",'scan_'+scannum+'_img_Pilatus')
	#time.sleep(1)
	#epics.caput("dp_pilatus4:cam1:FilePath",'/home/det/Sector_26_new/2019R2/20190806/'+'Images/'+scannum+'/')
	#time.sleep(1)
	#epics.caput("dp_pilatus4:cam1:FileName",'scan_'+scannum+'_img_Pilatus')
	#time.sleep(1)
	epics.caput("26idc:filter:Fi1:Set",0)
	time.sleep(1)


	curframe = inspect.currentframe()   #REK 20191206
	callframe = inspect.getouterframes(curframe, 2) #REK 20191206
	scanFunction = callframe[1][3]  #name of function 1 levels above prescan - should be the scan function that called this REK 20191206
	logger.updateLog(scanFunction = scanFunction, scanArgs = scanArgs)  #write to verbose logbook - REK 20191206

	return 0

def postscan():
	pathname = epics.caget(scanrecord+':saveData_fullPathName',as_string=True)
	epics.caput("QMPX3:TIFF1:FilePath",pathname[:-4]+'Images/')
	time.sleep(1)
	epics.caput("QMPX3:TIFF1:FileName",'image')
	time.sleep(1)
	epics.caput("dp_pilatusASD:cam1:FilePath",'/home/det'+pathname[5:-4]+'Images/')
	time.sleep(1)
	epics.caput("dp_pilatusASD:cam1:FileName",'image_Pilatus')
	time.sleep(1)
	#epics.caput("S18_pilatus:cam1:FilePath",'/mnt/Sector_26'+pathname[5:-4]+'Images/')
	#time.sleep(1)
	#epics.caput("S18_pilatus:cam1:FileName",'image_Pilatus')
	#time.sleep(1)
	#epics.caput("dp_pilatus4:cam1:FilePath",'/home/det/Sector_26_new/2019R2/20190806/'+'Images/')
	#time.sleep(1)
	#epics.caput("dp_pilatus4:cam1:FileName",'image_Pilatus')
	#time.sleep(1)
	epics.caput("26idc:filter:Fi1:Set",1)
	time.sleep(1)

# Define scanning functions

def scan(is2d=0):
	stopnow = prescan(scanArgs = locals())
	if (stopnow):
		return
	if (is2d==1):
		sc2.execute=1
	else:
		sc1.execute=1
	print("Scanning...")
	time.sleep(1)
	while(sc1.BUSY or sc2.BUSY):
		time.sleep(1)
	postscan()

def scan1d(motor,startpos,endpos,numpts,dettime, absolute=False):
	if motor in [fomx, fomy, samy]:
		epics.caput('26idcnpi:m34.STOP',1)
		epics.caput('26idcnpi:m35.STOP',1)
	if ( (motor in [hybridx, hybridy]) and ( (abs(hybridx.RBV-hybridx.VAL)>100) or (abs(hybridy.RBV-hybridy.VAL)>100) ) ):
		print("Please use lock_hybrid() to lock piezos at current position first...")
		return
	sc1.P1PV = motor.NAME+'.VAL'
	if absolute:
		sc1.P1AR=0
	else:
		sc1.P1AR=1
	sc1.P1SP = startpos
	sc1.P1EP = endpos
	sc1.NPTS = numpts
	count_time(dettime)
	fp = open(logbook,"a")
	fp.write(' ----- \n')
	fp.write('SCAN #: '+epics.caget(scanrecord+':saveData_scanNumber',as_string=True)+' ---- '+str(datetime.datetime.now())+'\n')
	if absolute:
		fp.write('Scanning '+motor.DESC+' from '+str(startpos)+' ---> '+str(endpos)+' in '+str(numpts)+' points at '+str(dettime)+' seconds acquisition\n')
	else:
		fp.write('Scanning '+motor.DESC+' from '+str(startpos+motor.VAL)+' ---> '+str(endpos+motor.VAL))
		fp.write(' in '+str(numpts)+' points at '+str(dettime)+' seconds acquisition\n')
	fp.write(' ----- \n')
	fp.close
	time.sleep(1)
	stopnow = prescan(scanArgs = locals());
	if (stopnow):
		return
	sc1.execute=1
	print("Scanning...")
	time.sleep(1)
	while(sc1.BUSY == 1):
		time.sleep(1)
	postscan()

def scan2d(motor1,startpos1,endpos1,numpts1,motor2,startpos2,endpos2,numpts2,dettime, absolute=False):
	if (motor1 in [fomx, fomy, samy]) or (motor2 in [fomx, fomy, samy]):
		epics.caput('26idcnpi:m34.STOP',1)
		epics.caput('26idcnpi:m35.STOP',1)
	if ( ( (motor1 in [hybridx, hybridy]) or (motor2 in [hybridx,hybridy]) ) and ( (abs(hybridx.RBV-hybridx.VAL)>100) or (abs(hybridy.RBV-hybridy.VAL)>100) ) ):
		print("Please use lock_hybrid() to lock piezos at current position first...")
		return
	sc2.P1PV = motor1.NAME+'.VAL'
	sc1.P1PV = motor2.NAME+'.VAL'
	if absolute:
		sc1.P1AR=0
		sc2.P1AR=0
	else:
		sc1.P1AR=1
		sc2.P1AR=1
	sc2.P1SP = startpos1
	sc1.P1SP = startpos2
	sc2.P1EP = endpos1
	sc1.P1EP = endpos2
	sc2.NPTS = numpts1
	sc1.NPTS = numpts2
	count_time(dettime)
	fp = open(logbook,"a")
	fp.write(' ----- \n')
	fp.write('SCAN #: '+epics.caget(scanrecord+':saveData_scanNumber',as_string=True)+' ---- '+str(datetime.datetime.now())+'\n')
	if absolute:
		fp.write('2D Scan:\n')
		fp.write('Inner loop: '+motor2.DESC+' from '+str(startpos2)+' ---> '+str(endpos2))
		fp.write(' in '+str(numpts2)+' points at '+str(dettime)+' seconds acquisition\n')
		fp.write('Outer loop: '+motor1.DESC+' from '+str(startpos1)+' ---> '+str(endpos1))
		fp.write(' in '+str(numpts1)+' points at '+str(dettime)+' seconds acquisition\n')   
	else:
		fp.write('2D Scan:\n')
		fp.write('Outer loop: '+motor1.DESC+' from '+str(startpos1+motor1.VAL)+' ---> '+str(endpos1+motor1.VAL))
		fp.write(' in '+str(numpts1)+' points at '+str(dettime)+' seconds acquisition\n')
		fp.write('Inner loop: '+motor2.DESC+' from '+str(startpos2+motor2.VAL)+' ---> '+str(endpos2+motor2.VAL))
		fp.write(' in '+str(numpts2)+' points at '+str(dettime)+' seconds acquisition\n')
	fp.write(' ----- \n')
	fp.close
	time.sleep(1)
	stopnow = prescan(scanArgs = locals());
	if (stopnow):
		return
	sc2.execute=1
	print("Scanning...")
	time.sleep(1)
	while(sc2.BUSY == 1):
		time.sleep(1)
	postscan()

def focalseries(z_range,numptsz,y_range,numptsy,dettime,motor1=fomz,motor2=hybridy):
	sc1.P1PV = motor2.NAME+'.VAL'
	sc2.P1PV = motor1.NAME+'.VAL'
	sc1.P1SP = -y_range/2.0
	sc2.P1SP = -z_range/2.0
	sc1.P1EP = y_range/2.0
	sc2.P1EP = z_range/2.0
	sc1.NPTS = numptsy
	sc2.NPTS = numptsz
	sc1.P1AR = 1
	sc2.P1AR = 1
	sc2.P2AR = 1
	sc2.P3AR = 1
	sc2.P2PV = hybridy.NAME+'.VAL'
	sc2.P2SP = 1.177*z_range/400   #change y offset here
	sc2.P2EP = -1.177*z_range/400
	sc2.P3PV = hybridx.NAME+'.VAL'
	sc2.P3SP = 0.3125*z_range/400   #change x offset here
	sc2.P3EP = -0.3125*z_range/400
	count_time(dettime)
	time.sleep(1)
	if ( (abs(hybridx.RBV-hybridx.VAL)>50) or (abs(hybridy.RBV-hybridy.VAL)>50) ):
		print("Please use lock_hybrid() to lock piezos at current position first...")
		sc2.P2PV = ''
		sc2.P3PV = ''
		return
	stopnow = prescan(scanArgs = locals());
	if (stopnow):
		return
	sc2.execute=1
	print("Scanning...")
	time.sleep(1)
	while(sc2.BUSY == 1):
		time.sleep(1)
	postscan()
	time.sleep(2)
	sc2.P2PV = ''
	sc2.P3PV = ''

def defocus(z_move):
	movr(fomz,z_move)
	movr(hybridy,-1.177*2*z_move/400) #SOH: added factor of 2 that is needed to get correct runout
	movr(hybridx,-0.31*2*z_move/400)

def timeseries(numpts,dettime=1.0):
	tempsettle1 = sc1.PDLY
	tempsettle2 = sc1.DDLY
	tempdrive = sc1.P1PV
	tempstart = sc1.P1SP
	tempend = sc1.P1EP
	sc1.PDLY = 0.0
	sc1.DDLY = 0.0
	sc1.P1PV = "26idcNES:sft01:ph01:ao03.VAL"
	sc1.P1AR = 1
	sc1.P1SP = 0.0
	sc1.P1EP = numpts*dettime
	sc1.NPTS = numpts+1
	count_time(dettime)
	fp = open(logbook,"a")
	fp.write(' ----- \n')
	fp.write('SCAN #: '+epics.caget(scanrecord+':saveData_scanNumber',as_string=True)+' ---- '+str(datetime.datetime.now())+'\n')
	fp.write('Timeseries: '+str(numpts)+' points at '+str(dettime)+' seconds acquisition\n')
	fp.write(' ----- \n')
	fp.close
	time.sleep(1)
	stopnow = prescan(scanArgs = locals());
	if (stopnow):
		return
	sc1.execute=1
	print("Scanning...")
	time.sleep(2)
	while(sc1.BUSY == 1):
		time.sleep(1)
	postscan()
	sc1.PDLY = tempsettle1
	sc1.DDLY = tempsettle2
	sc1.P1PV = tempdrive
	sc1.P1SP = tempstart
	sc1.P1EP = tempend

def spiral(stepsize,numpts,dettime):
	epics.caput('26idpvc:userCalc5.OUTN','')
	epics.caput('26idpvc:userCalc6.OUTN','')
	epics.caput('26idpvc:userCalc5.CALC$','B+'+str(stepsize)+'*SQRT(A)*COS(4*SQRT(A))')
	epics.caput('26idpvc:userCalc6.CALC$','B+'+str(stepsize)+'*SQRT(A)*SIN(4*SQRT(A))')
	epics.caput('26idpvc:sft01:ph01:ao04.VAL',hybridx.RBV)
	epics.caput('26idpvc:sft01:ph01:ao05.VAL',hybridy.RBV)
	epics.caput('26idpvc:sft01:ph01:ao01.VAL',0.0)
	time.sleep(1)
	# epics.caput('26idpvc:userCalc5.OUTN','26idcDEV:X_HYBRID_SP.VAL')
	# epics.caput('26idpvc:userCalc6.OUTN','26idcDEV:Y_HYBRID_SP.VAL')
	epics.caput('26idpvc:userCalc5.OUTN','26idcnpi:X_HYBRID_SP.VAL')
	epics.caput('26idpvc:userCalc6.OUTN','26idcnpi:Y_HYBRID_SP.VAL')
	temppos = sc1.P1PV
	tempcen = sc1.P1CP
	tempwidth = sc1.P1WD
	time.sleep(1)
	sc1.P1PV = '26idpvc:sft01:ph01:ao01.VAL'
	sc1.P1AR=1
	sc1.P1SP = 0.0
	sc1.P1EP = numpts-1
	sc1.NPTS = numpts
	count_time(dettime)
	fp = open(logbook,"a")
	fp.write(' ----- \n')
	fp.write('SCAN #: '+epics.caget(scanrecord+':saveData_scanNumber',as_string=True)+' ---- '+str(datetime.datetime.now())+'\n')
	fp.write('Scanning spiral from x:'+str(hybridx.VAL)+' y:'+str(hybridy.VAL)+' with step size of: '+str(stepsize)+' nm')
	fp.write(' in '+str(numpts)+' points at '+str(dettime)+' seconds acquisition\n')
	fp.write(' ----- \n')
	fp.close
	time.sleep(1)
	stopnow = prescan(scanArgs = locals());
	if (stopnow):
		return
	sc1.execute=1
	print("Scanning...")
	time.sleep(1)
	while(sc1.BUSY == 1):
		time.sleep(1)
	postscan()
	time.sleep(1)
	sc1.P1PV = temppos
	sc1.P1CP = tempcen
	sc1.P1WD = tempwidth

#def thetascan(thetapts,detchan,xscan,motor1,startpos1,endpos1,numpts1,motor2,startpos2,endpos2,numpts2,dettime):
def thetascan(thetapts,motor1,startpos1,endpos1,numpts1,motor2,startpos2,endpos2,numpts2,dettime):
	numthpts = np.size(thetapts)
	mov(samth,(thetapts[0]-0.5))
	time.sleep(1)
	#epics.caput("26idbSOFT:scan1.REFD",detchan); #Set horizontal reference XRF detector channel XRF 21 XRD 5
	for ii in range(numthpts):
		mov(samth,thetapts[ii])
		print ("theta",thetapts[ii])
		#mov(twotheta,thetapts[ii]*2-0.7)
		#print ("2theta",thetapts[ii]*2-0.7)
		#epics.caput("26idbSOFT:scan1.REFD",4); #Set horizontal reference XRF detector channel
		#sc1.PASM = 3  #Sets lineup scan to post-scan move to peak
		#sc1.PASM = 4  #Sets lineup scan to post-scan move to valley
		# sc1.PASM = 6  #Sets lineup scan to post-scan move to the -edge position (maybe we need +? Edge is in the "+" direction from the position)
		#time.sleep(1)
		sc1.PASM = 7  #Sets lineup scan to post-scan move to center of mass (7) or peak (3)
		print("scanning hybridx to recenter")
		_hx0 = epics.caget("26idcnpi:X_HYBRID_SP.VAL")
		#movr(hybridx,-10)
		xscan=10;
		scan1d(hybridx,-xscan/2,xscan/2, 21, 0.2)  # ADJUST X LINE UP SCAN HERE
		time.sleep(1)
		movr(hybridx,-.8)  #ADJUST RELATIVE MOVE HERE
		print("hybridx peak moved from ", _hx0, " to ", epics.caget("26idcnpi:X_HYBRID_SP.VAL"))
		#_hy0 = epics.caget("26idcnpi:Y_HYBRID_SP.VAL")
		epics.caput("26idbSOFT:scan1.REFD",4); #Set horizontal reference XRF detector channel
		# time.sleep(1)
		# print("scanning hybridy to recenter")
		# scan1d(hybridy,-15,15, 51, 1.5)  # ADJUST Y LINE UP SCAN HERE
		# time.sleep(1)
		# print("hybridy peak moved from ", _hy0, " to ", epics.caget("26idcnpi:Y_HYBRID_SP.VAL"))
		sc1.PASM = 2  #Sets post-scan move back to prior position'''
		print("Main scan theta: ",thetapts[ii]," Scan number:",epics.caget("26idbSOFT:saveData_scanNumber"))
		time.sleep(3)
		# print("too late man")
		scan2d(motor1,startpos1,endpos1,numpts1,motor2,startpos2,endpos2,numpts2,dettime)  #MESH SCAN
		# mov(samth,(thetapts[2]-0.1))
		# mov(samth,(thetapts[2]))

def rocking(theta_start,theta_end,npts,xstart,xend,nxpts,dettime):
	
	dstep=(theta_start-theta_end)/float(npts)
	currth=epics.caget("atto2:PIC867:1:m1.RBV")
   
	jj=theta_start 
	for ii in range(npts+1):
		mov(samth,currth+jj)
		sc1.PASM = 7  #Sets lineup scan to post-scan move to peak - REFERENCE DETECTOR MUST BE CORRECTLY SET
		time.sleep(1)
		scan1d(hybridx,xstart,xend,nxpts, 0.5)  # ADJUST X LINE UP SCAN HERE
		#scan1d(hybridy, -3,3,51, 0.3) #scan the y positoin
		sc1.PASM = 2  #Sets post-scan move back to prior position
		time.sleep(1)
		scan1d(hybridx,xstart,xend,nxpts+1,dettime)  #Smaller 1D scan with long exposures
		jj+=dstep

#Change in eV
def change_energy(E0,Z0,delta_E):
	stDCM = epics.caget("26idbDCM:sm8.RBV")
	stUDS = epics.caget("26idbDCM:sm2.RBV")
	stUUS = epics.caget("26idbDCM:sm3.RBV")
	defocus(delta_E*Z0/E0)
	epics.caput("26idbDCM:sm8.VAL",stDCM+delta_E)
	time.sleep(1)
	epics.caput("26idbDCM:sm2.VAL",stUDS+delta_E/1000.0)
	time.sleep(1)
	epics.caput("26idbDCM:sm3.VAL",stUUS+delta_E/1000.0)
	time.sleep(1)



#def energyscan(E0,Z0,EVrange,numpts,motor1,startpos1,endpos1,numpts1,motor2,startpos2,endpos2,numpts2,dettime):
#E0, Estep in eV
# Z0 is focal length at E0 from ZP calculator (in MICRONS)
def energyscan1d(E0,Z0,EVrange,numpts,motor1,startpos1,endpos1,numpts1,dettime): #input Z0 in MICRONS
	stDCM = epics.caget("26idbDCM:sm8.RBV")
	stUDS = epics.caget("26idbDCM:sm2.RBV")
	stUUS = epics.caget("26idbDCM:sm3.RBV")
	stHybY = epics.caget("26idcnpi:Y_HYBRID_SP.VAL")
	Estep = np.float(EVrange)/(numpts-1)
	HybYstep=0.4/30.0*0 #About 13 nm per eV around 9 keV. Toggle 0 as req 
	defocus((EVrange/2.0)*Z0/E0)
	epics.caput("26idbDCM:sm8.VAL",stDCM-EVrange/2.0)
	time.sleep(1)
	epics.caput("26idbDCM:sm2.VAL",stUDS-EVrange/2000.0)
	time.sleep(1)
	epics.caput("26idbDCM:sm3.VAL",stUUS-EVrange/2000.0)
	time.sleep(1)
	epics.caput("26idcnpi:Y_HYBRID_SP.VAL", stHybY-EVrange/2.0*HybYstep)
	time.sleep(1)
	print("Scanning at:%.1f eV" %(epics.caget("26idbDCM:sm8.RBV")))
	scan1d(motor1,startpos1,endpos1,numpts1,dettime)  #LINE SCAN
	for ii in range(1,numpts):
		 defocus(-Estep*Z0/E0)
		 time.sleep(1)
		 epics.caput("26idbDCM:sm8.VAL",stDCM-EVrange/2.0+Estep*ii)
		 time.sleep(1)
		 epics.caput("26idbDCM:sm2.VAL",stUDS-EVrange/2000.0+Estep*ii/1000)
		 time.sleep(1)
		 epics.caput("26idbDCM:sm3.VAL",stUUS-EVrange/2000.0+Estep*ii/1000)
		 time.sleep(1)
		 epics.caput("26idcnpi:Y_HYBRID_SP.VAL", stHybY-EVrange/2.0*HybYstep+ii*HybYstep*Estep)
		 time.sleep(1) 
		 scan1d(motor1,startpos1,endpos1,numpts1,dettime)  #LINE SCAN
	epics.caput("26idbDCM:sm8.VAL",stDCM)
	time.sleep(1)
	epics.caput("26idbDCM:sm2.VAL",stUDS)
	time.sleep(1)
	epics.caput("26idbDCM:sm3.VAL",stUUS)
	time.sleep(1)
	defocus((EVrange/2.0)*Z0/E0)
	time.sleep(1)
	epics.caput("26idcnpi:Y_HYBRID_SP.VAL",stHybY)

#Z0 in MICRONS        
#E0, Estep in eV
# Z0 is focal length at E0 from ZP calculator (in MICRONS)
def energyscan2d(E0,Z0,EVrange,numpts,motor1,startpos1,endpos1,numpts1,motor2,startpos2,endpos2,numpts2,dettime):
	stDCM = epics.caget("26idbDCM:sm8.RBV")
	stUDS = epics.caget("26idbDCM:sm2.RBV")
	stUUS = epics.caget("26idbDCM:sm3.RBV")
	stHybY = epics.caget("26idcnpi:Y_HYBRID_SP.VAL")
	Estep = np.float(EVrange)/(numpts-1)
	HybYstep=0.4/30.0*0 #About 13 nm per eV around 9 keV. Toggle 0 as req 
	defocus((EVrange/2.0)*Z0/E0)
	epics.caput("26idbDCM:sm8.VAL",stDCM-EVrange/2.0)
	time.sleep(1)
	epics.caput("26idbDCM:sm2.VAL",stUDS-EVrange/2000.0)
	time.sleep(1)
	epics.caput("26idbDCM:sm3.VAL",stUUS-EVrange/2000.0)
	time.sleep(1)
	epics.caput("26idcnpi:Y_HYBRID_SP.VAL", stHybY-EVrange/2.0*HybYstep)
	time.sleep(1)
	print("Scanning at:%.1f eV" %(epics.caget("26idbDCM:sm8.RBV")))
	scan2d(motor1,startpos1,endpos1,numpts1,motor2,startpos2,endpos2,numpts2,dettime)  #MESH SCAN
	for ii in range(1,numpts):
		 defocus(-Estep*Z0/E0)
		 time.sleep(1)
		 epics.caput("26idbDCM:sm8.VAL",stDCM-EVrange/2.0+Estep*ii)
		 time.sleep(1)
		 epics.caput("26idbDCM:sm2.VAL",stUDS-EVrange/2000.0+Estep*ii/1000)
		 time.sleep(1)
		 epics.caput("26idbDCM:sm3.VAL",stUUS-EVrange/2000.0+Estep*ii/1000)
		 time.sleep(1)
		 epics.caput("26idcnpi:Y_HYBRID_SP.VAL", stHybY-EVrange/2.0*HybYstep+ii*HybYstep*Estep)
		 time.sleep(1) 
		 print("Scanning at:%.1f eV" %(epics.caget("26idbDCM:sm8.RBV")))
		 scan2d(motor1,startpos1,endpos1,numpts1,motor2,startpos2,endpos2,numpts2,dettime)  #MESH SCAN
	epics.caput("26idbDCM:sm8.VAL",stDCM)
	time.sleep(1)
	epics.caput("26idbDCM:sm2.VAL",stUDS)
	time.sleep(1)
	epics.caput("26idbDCM:sm3.VAL",stUUS)
	time.sleep(1)
	defocus((EVrange/2.0)*Z0/E0)
	time.sleep(1)
	epics.caput("26idcnpi:Y_HYBRID_SP.VAL",stHybY)

	   

def theta2thetascan(thstartpos,thendpos,numpts,dettime):
	sc1.P2PV = '26idcDET:base:Theta.VAL'
	sc1.P2SP = 2*thstartpos
	sc1.P2EP = 2*thendpos
	sc1.P2AR=1
	time.sleep(1)
	scan1d(samth,thstartpos,thendpos,numpts,dettime)
	sc1.P2PV = ''
	


def panelscan(ypositions,xpositions,motor1,startpos1,endpos1,numpts1,motor2,startpos2,endpos2,numpts2,dettime):
	numypts = np.size(ypositions)
	for ii in range(numypts):
		mov(hybridy,ypositions[ii])
		mov(hybridx,xpositions[ii])
		time.sleep(1)
		#----Uncomment below for lineup scan ----
		#sc1.PASM = 3  #Sets lineup scan to post-scan move to peak - REFERENCE DETECTOR MUST BE CORRECTLY SET
		#time.sleep(1)
		#scan1d(hybridy,-1,1,51,3)  #LINE UP SCAN
		#sc1.PASM = 2  #Sets post-scan move back to prior position
		#time.sleep(1)
		#-----Uncomment above for lineup scan ----
		scan2d(motor1,startpos1,endpos1,numpts1,motor2,startpos2,endpos2,numpts2,dettime)  #MESH SCAN

def tempscan(temppts,motor1,startpos1,endpos1,numpts1,motor2,startpos2,endpos2,numpts2,dettime):
	numtemppts = np.size(temppts)
	for ii in range(numtemppts):
		print('Changing temperature to '+str(temppts[ii])+'...')
		epics.caput('26idcSOFT:LS336:tc1:OUT1:SP',temppts[ii])
		if ii>1:
			movr(samy,-0.1666*(temppts[ii]-temppts[ii-1]))
		time.sleep(900)
		scan2d(motor1,startpos1,endpos1,numpts1,motor2,startpos2,endpos2,numpts2,dettime)  #MESH SCAN

def pvscan(pvname,pvpts,motor1,startpos1,endpos1,numpts1,motor2,startpos2,endpos2,numpts2,dettime):
	numpvpts = np.size(pvpts);
	for ii in range(numpvpts):
		print('Changing '+pvname+' to '+str(pvpts[ii])+'...')
		time.sleep(1)
		epics.caput(pvname,pvpts[ii])
		time.sleep(1)
		#if ii>1:
		#    movr(hybridy,-0.1666*(temppts[ii]-temppts[ii-1]))
		#time.sleep(900)
		scan2d(motor1,startpos1,endpos1,numpts1,motor2,startpos2,endpos2,numpts2,dettime)
	epics.caput(pvname,0.0)
	time.sleep(1)
"""
def set_beam_energy(EkeV):
	und_us = [8.0299, 9.0497, 10.1098, 11.1395, 12.089]
	und_ds = [8.0701, 9.1298, 10.1398, 11.2202, 12.080]
	chi2 = [-0.1533, -0.2046, -0.2701, -0.3362, -0.4017]
	th2 = [5.27, 5.57, 5.67, 5.77, 5.77]
	if (EkeV<8.0) or (EkeV>12.0):
		print("Routine is only good between 8 and 12 keV- exiting")
		return
	i3 = float(EkeV-8)
	i1 = int(math.ceil(i3))
	i2 = int(math.floor(i3))
	if i1==i2:
		und_us_interp = und_us[i1]
		und_ds_interp = und_ds[i1]
		chi2_interp = chi2[i1]
		th2_interp = th2[i1]
	else:
		und_us_interp = (i1 - i3)*und_us[i2] + (i3 - i2)*und_us[i1]
		und_ds_interp = (i1 - i3)*und_ds[i2] + (i3 - i2)*und_ds[i1]
		chi2_interp = (i1 - i3)*chi2[i2] + (i3 - i2)*chi2[i1]
		th2_interp = (i1 - i3)*th2[i2] + (i3 - i2)*th2[i1]
	dcm_interp = EkeV*1000
	print('Moving to-  und_US:'+str(und_us_interp)+' und_DS:'+str(und_ds_interp)+' Chi2:'+str(chi2_interp)+' Th2:'+str(th2_interp)+' DCM:'+str(dcm_interp))
	epics.caput("26idbDCM:sm2.VAL",und_ds_interp)
	time.sleep(1)
	epics.caput("26idbDCM:sm3.VAL",und_us_interp)
	time.sleep(1)
	epics.caput("26idbDCM:sm5.VAL",chi2_interp)
	time.sleep(1)
	epics.caput("26idb:DAC1_1.VAL",th2_interp)
	time.sleep(1)
	epics.caput("26idbDCM:sm8.VAL",dcm_interp)
	time.sleep(1)
"""

def set_phase(phase):
	ph1 = [0,7,15,19,28,34,40,48,55,60,67,74,81,87,99,117,127,133,145,159,168,182,198,210,220,227,237,245,255,272,283,295,308,320,330,340,358]
	amp1 = [1.1,1.15,1.27,1.33,1.37,1.3,1.28,1.29,1.29,1.32,1.32,1.30,1.2,1.11,.98,.96,.93,.87,.82,.81,.84,.85,.88,.88,.88,.85,.86,.91,.91,.87,.85,.83,.83,.9,.93,.96,1.02]
	i1 = int(math.ceil(float(phase)/10))
	i2 = int(math.floor(float(phase)/10))
	if i1==i2:
		ph2=ph1[i1]
		amp2=amp1[i1]
	else:
		ph2 = (i1-float(phase)/10)*(ph1[i2])+(float(phase)/10-i2)*(ph1[i1])
		amp2 = (i1-float(phase)/10)*(amp1[i2])+(float(phase)/10-i2)*(amp1[i1])
	epics.caput('26idbDCM:sft01:ph01:ao05.VAL',amp2)
	epics.caput('26idbDCM:sft01:ph01:ao06.VAL',ph2)
	epics.caput('26idbDCM:sft01:ph01:ao08.VAL',float(phase))

def ramp_phase(phase):
	ph1=epics.caget('26idbDCM:sft01:ph01:ao08.VAL')
	while(ph1<>phase):
		step1 = (phase>ph1)*1.0-(phase<ph1)*1.0
		set_phase(float(ph1)+step1)
		#print([ph1,phase,step1,float(ph1)+step1])
		time.sleep(0.5)
		ph1=epics.caget('26idbDCM:sft01:ph01:ao08.VAL')
	print('Done')

def watch_phase():
	ph1 = np.round(epics.caget('26idbDCM:sft01:ph01:ao07.VAL'))
	while(1):
		time.sleep(1)
		ph2 = np.round(epics.caget('26idbDCM:sft01:ph01:ao07.VAL'))
		if(ph1<>ph2):
			ramp_phase(ph2)
			ph1=ph2

#def scan3d(outermotor,outerstart,outerstop,outernumpts,motor1,startpos1,endpos1,numpts1,motor2,startpos2,endpos2,numpts2,exp_time):
#    movr(outermotor,outerstart)
#    time.sleep(1)
#    for ii in range(outernumpts):
#        movr(outermotor,ii*((outerstop-outerstart)/outernumpts))
#        time.sleep(1)
#        scan2d(motor1,startpos1,endpos1,numpts1,motor2,startpos2,endpos2,numpts2,exp_time)
#        time.sleep(1)
#    mov(outermotor,(outerstart+outerstop)/2)

def scanfocusmaps(defocus_list, motor1, startpos1, endpos1, numpts1, motor2, startpos2, endopos2, numpts2, dettime):

	for ii in range(np.size(defocus_list)):
		defocus(defocus_list[ii]);
		scan2d( motor1, startpos1, endpos1, numpts1, motor2, startpos2, endopos2, numpts2, dettime )

def geniescan1d(motor, start, end, numpts1, dettime):

	epics.caput("S26idcGen1:cam1:ImageMode", 0)
	epics.caput("S26idcGen1:cam1:AcquireTime", dettime)
	_m_pos0 = motor.RBV
	_m_pos = np.linspace(start, end, numpts1) + _m_pos0
	_tmp_data = np.zeros(numpts1)
	for i in range(numpts1):
		motor.VAL = _m_pos[i]
		time.sleep(.5)
		epics.caput("S26idcGen1:cam1:Acquire", 1)
		time.sleep(dettime+.5)
		_tmp_data[i] = epics.caget("S26idcGen1:image1:ArrayData", as_numpy=1, count=1360*1024).sum()
	motor.VAL = _m_pos0
	pyplot.plot(_m_pos, _tmp_data)
	pyplot.show()
	print("peak at {0}".format(_m_pos[_tmp_data.argmax()]))

### Logging functions

class Logger():
	"""
	Object to handle logging of motor positions, filter statuses, and XRF ROI assignments per scan. 

	REK 20191206
	"""
	def __init__(self, sample = '', note = ''):
		self.rootDir = epics.caget(scanrecord+':saveData_fileSystem',as_string=True)
		self.logDir = os.path.join(self.rootDir, 'Logging')
		if not os.path.isdir(self.logDir):
			os.mkdir(self.logDir)

		self.logFilepath = os.path.join(self.logDir, 'verboselog.json')
		if not os.path.exist(self.logFilepath):
			with open(self.logFilepath, 'w') as f:
				json.dump({}, f)    #intialize as empty dictionary

		self.sample = sample    #current sample being measured
		self.note = note

		self.motorDict = {  #array of motor labels + epics addresses
					"fomx": '26idcnpi:m10.VAL',
					"fomy": '26idcnpi:m11.VAL',
					"fomz": '26idcnpi:m12.VAL',
					# "samx": '26idcnpi:m16.VAL',
					"samy": '26idcnpi:m17.VAL',
					# "samz": '26idcnpi:m18.VAL',
					"samth": 'atto2:PIC867:1:m1.VAL',
					"osax": '26idcnpi:m13.VAL',
					"osay": '26idcnpi:m14.VAL',
					"osaz": '26idcnpi:m15.VAL',
					"condx": '26idcnpi:m5.VAL',
					"attox": 'atto2:m3.VAL',
					"attoz": '26idcNES:sm27.VAL',
					"samchi": 'atto2:m1.VAL',
					"samphi": 'atto2:m2.VAL',
					"objx": '26idcnpi:m1.VAL',
					"xrfx": '26idcDET:m7.VAL',
					# "piezox": '26idcSOFT:sm1.VAL',
					# "piezoy": '26idcSOFT:sm2.VAL',
					"hybridx": '26idcnpi:X_HYBRID_SP.VAL',
					"hybridy": '26idcnpi:Y_HYBRID_SP.VAL',
					"twotheta": '26idcSOFT:sm3.VAL',
					"gamma":    !UPDATE!,
					"filter1": "26idc:filter:Fi1:Set",  #!UPDATE! Check that the filters are being read properly
					"filter2": "26idc:filter:Fi2:Set",
					"filter3": "26idc:filter:Fi3:Set",
					"filter4": "26idc:filter:Fi4:Set",
					"energy": "26idbDCM:sm8.RBV",	#!UPDATE! Check that the energy is being read properly
				}

	def updateLog(self, scanFunction, scanArgs):
		self.scanNumber = epics.caget(scanrecord+':saveData_scanNumber',as_string=True)

		self.scanEntry = {
			'BeamEnergy': epics.caget(!UPDATE!),
			'ROIs': {},
			'Sample': self.sample,
			'Note': self.note,
			'Date': str(datetime.datetime.now().date()),
			'Time': str(datetime.datetime.now().time()),
			'ScanFunction': scanFunction
			'ScanArgs': scanArgs
		}
		
		for label, key in motordict.items():
			self.scanEntry[label] = epics.caget(key, as_string = True)

		### find saved ROIs, get the element + energy range per ROI
		# iterate through all saved data entries, find with "ROI"
		for each in !UPDATE!DATAENTRIES:
			if 'ROI' in each:
				roiNum = int(each.split('')[0])
				roiElement = epics.caget(!UPDATE!, as_string = True)
				roiMin = epics.caget(!UPDATE!)
				roiMax = epics.caget(!UPDATE!)
				self.scanEntry['ROIs'][roinum] = {
					'Element': roiElement,
					'BinRange': (roiMin, roiMax)
				}
		
		### Add entry to log file
		with open(self.logbook_filepath, 'r') as f:
			fullLogbook = json.load(f)
		fullLogbook[scanNumber] = self.scanEntry
		with open(self.logbook_filepath, 'w') as f:
			json.dump(fullLogbook, f)


		self.lastScan
