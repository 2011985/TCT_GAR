import os

dataset_path = '/opt/DIN_GAR/data/NBA/'
label_path = '/opt/DIN_GAR/data/NBA/label'

categories = ['2p-succ.','2p-fail.-off.', '2p-fail.-def.', '2p-layup-succ.', '2p-layup-fail.-off.', '2p-layup-fail.-def.',
              '3p-succ.', '3p-fail.-off.', '3p-fail.-def.']


if __name__ == '__main__':
    assert len(set(categories)) == 9
    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i

    print(dict_categories)

    # files_input = ['kinetics_val.csv', 'kinetics_train.csv']
    files_output = ['val_videofolder.txt', 'train_videofolder.txt']
    for filename_output in files_output:
        stage = filename_output.split('_')[0]
        stage_path = os.path.join(dataset_path, stage)
        folders = []
        idx_categories = []
        categories_list = []
        count_cat = {k: 0 for k in dict_categories.keys()}
        for cat in os.listdir(stage_path):
            for video in os.listdir(os.path.join(stage_path, cat)):
                folders.append(os.path.join(stage_path, cat, video))
                categories_list.append(cat)
                idx_categories.append(dict_categories[cat])
                count_cat[cat] += 1

        print(folders)
        print(max(count_cat.values()))

        assert len(idx_categories) == len(folders)
        missing_folders = []
        output = []
        for i in range(len(folders)):
            img_dir = folders[i]
            curFolder = img_dir.split('/')[-1]
            curIDX = idx_categories[i]
            # counting the number of frames in each video folders
            if not os.path.exists(img_dir):
                missing_folders.append(img_dir)
                # print(missing_folders)
            else:
                dir_files = os.listdir(img_dir)
                output.append('%s %d %d' % (os.path.join(stage, categories_list[i], curFolder), len(dir_files), curIDX))
            print('%d/%d, missing %d' % (i, len(folders), len(missing_folders)))
        with open(os.path.join(label_path, filename_output), 'w') as f:
            f.write('\n'.join(output))
        with open(os.path.join(label_path, 'missing_' + filename_output), 'w') as f:
            f.write('\n'.join(missing_folders))
