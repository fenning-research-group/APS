from apsreporter import tableofcontents, section, scanlist, comparison, build

t = tableofcontents(
	title = '150C HEP',
	description = 'HEP samples annealed at 150 C',
	outputfile = 'G:\My Drive\FRG\Projects\APS\\2IDD_2019\Sample Data - 150C HEP\\tableofcontents.json',
	datafolder = 'G:\My Drive\FRG\Projects\APS\\2IDD_2019\Sample Data - 150C HEP\\data'
	)

t.energy = 'G:\My Drive\FRG\Projects\APS\\2IDD_2019\Sample Data - 150C HEP\\scanenergies.csv'
t.dwell = 'G:\My Drive\FRG\Projects\APS\\2IDD_2019\Sample Data - 150C HEP\\dwelltimes.csv'

s1 = section(
	title = 'Section 1',
	description = 'S1 desc'
)

sl1 = scanlist(
	datafolder = 'G:\My Drive\FRG\Projects\APS\\2IDD_2019\Sample Data - 150C HEP\\data',
	scans = {
		63: 'scan 63 description',
		64: 'scan 64 description',
		65: 'scan 65 description'
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

t.contents = ('150C HEP Scanlist', s1)
t.contents = ('150C HEP 2', s2)

t.show()
f = t.export()

b = build(
	tableofcontents = f,
	outputfolder = 'G:\My Drive\FRG\Projects\APS\\2IDD_2019\Sample Data - 150C HEP\\output',
	title = 'sample.pdf'
	)

b.makepdf()