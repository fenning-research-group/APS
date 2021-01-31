from distutils.core import setup

setup(
	name='apstools',
	version='0.1dev',
	packages=['apstools'],
	package_dir={'apstools': 'apstools'},
	package_data={'apstools': ['include/*']},
	install_requires = [
		'h5py',
		'scipy',
		'matplotlib',
		'numpy',
		'opencv-python',
		'tqdm',
		'skimage',
		'multiprocessing',
		'cmocean',
		'pandas'
		],
	license='Creative Commons Attribution-Noncommercial-Share Alike license',
	long_description=open('./apstools/README.md').read(),
)