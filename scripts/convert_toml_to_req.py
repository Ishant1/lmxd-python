import os
import re


parent_dir = os.path.dirname(os.path.dirname(__name__))
toml_file = os.path.join(parent_dir, "pyproject.toml")

with open(toml_file, "r") as f:
    alllines = f.readlines()

start_index=0
for i, lines in enumerate(alllines):
    if lines.__contains__("poetry.dependencies"):
        start_index = i

    if start_index>0:
        if lines=="\n":
            end_index = i
            break

all_packages = [re.sub(r" |\^|\"|\n","", line).split('=') for line in alllines[start_index+1:end_index]]

requirements_list = []
for package in all_packages:
    if len(package)==2:
        package_name, version = package
        if package_name != "python":
            requirements_list.append(f"{package_name}>={version}\n")


with open(os.path.join(parent_dir, "requirements.txt"), "w+") as f:
    f.writelines(requirements_list)




