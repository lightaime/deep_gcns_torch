import os
import shutil
import csv


def save_best_result(list_of_dict, file_name, dir_path='best_result'):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print("Directory ", dir_path, " is created.")
    csv_file_name = '{}/{}.csv'.format(dir_path, file_name)
    with open(csv_file_name, 'a+') as csv_file:
        csv_writer = csv.writer(csv_file)
        for _ in range(len(list_of_dict)):
            csv_writer.writerow(list_of_dict[_].values())


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
