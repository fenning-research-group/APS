from apsreporter import tableofcontents, section, scanlist, comparison, build

t = tableofcontents(
	title = '1%Eu_noTreatment',
	description = '1%Eu-CsFAPbI3 with no enviornmental stressor ',
	outputfile = '/Volumes/GoogleDrive/My Drive/APS/APS2019c2_2idd/XRF/h5RawData/1%Eu_noTreat_6pt98keV/tableofcontents.json',
	datafolder = '/Volumes/GoogleDrive/My Drive/APS/APS2019c2_2idd/XRF/h5RawData/1%Eu_noTreat_6pt98keV'
	)

t.energy = '/Volumes/GoogleDrive/My Drive/APS/APS2019c2_2idd/XRF/h5RawData/scanenergies.csv'
t.dwell = '/Volumes/GoogleDrive/My Drive/APS/APS2019c2_2idd/XRF/h5RawData/dwelltimes.csv'

s1 = section(
	title = 'Section 1',
	description = 'S1 desc'
)

sl1 = scanlist(
	datafolder = '/Volumes/GoogleDrive/My Drive/APS/APS2019c2_2idd/XRF/h5RawData/1%Eu_noTreat_6pt98keV',
	scans = {
		291: 'scan 291 Overview map of the sample area',
		292: 'scan 292 Electronic (XBIC) map taken with shorter dwell time (50ms)',
		293: 'scan 293 Chemistry map, a small region within the XBIC map'
	}
)

sl1 = scanlist(
	datafolder = 'G:\My Drive\FRG\Projects\APS\\2IDD_2019\Sample Data - 150C HEP\\data',
	scans = {
		63: 'scan 63 description',
		64: 'scan 64 description',
		65: 'scan 65 description'
	}
)

print('scan list 1')
sl1.setchannels()

c1 = comparison(
	title = 'Comparison 1 Title',
	description = 'Comparison 1 Description',
	datafolder = 'G:\My Drive\FRG\Projects\APS\\2IDD_2019\Sample Data - 150C HEP\\data',
	scans = {
		63: 'scan 63 description',
		64: 'scan 64 description',
		65: 'scan 65 description'
	}
)

c1.setchannels()



s1.contents = ('scans1', sl1)
s1.contents = ('comparison1', c1)

s2 = section(
	title = 'Section 2',
	description = 'S2 desc'
)
s2.contents = ('scans1', sl1)

t.contents = ('1%Eu_noTreatment', s1)
t.contents = ('1%Eu_noTreatment 2', s2)

t.show()
f = t.export()

b = build(
	tableofcontents = f,
	outputfolder = '/Volumes/GoogleDrive/My Drive/APS/APS2019c2_2idd/XRF/h5RawData/1%Eu_noTreat_6pt98keV/',
	title = 'sample.pdf'
	)

b.makepdf()