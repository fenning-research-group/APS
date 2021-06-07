from distutils.core import setup

setup(
	name='apstools',
	version='0.2dev',
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
		'scikit-image',
		'cmocean',
		'pandas',
		'frgtools @ git+https://github.com/fenning-research-group/Python-Utilities.git#egg=frgtools&subdirectory=FrgTools',
		],
	license='Creative Commons Attribution-Noncommercial-Share Alike license',
	long_description=open('./apstools/README.md').read(),
)