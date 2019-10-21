from distutils.core import setup

setup(
	name='Henke',
	version='0.1dev',
	packages=['henke',],
	package_dir={'henke': 'henke'},
	package_data={'henke': ['include/*']},
	license='Creative Commons Attribution-Noncommercial-Share Alike license',
	long_description=open('README.txt').read(),
)
