from apsreporter import tableofcontents, section, scanlist, build

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
print('Scans 1')
sl1.setchannels()

s1.contents = ('scans1', sl1)

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