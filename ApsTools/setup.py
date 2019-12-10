from distutils.core import setup

setup(
    name='apstools',
    version='0.1dev',
    packages=['apstools'],
	package_dir={'apstools': 'apstools'},
	package_data={'apstools': ['include/*']},
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('./apstools/README.md').read(),
)