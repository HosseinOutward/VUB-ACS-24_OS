import torch
import pandas as pd
import numpy as np


def get_class_id_array(array):
    temp = array == array
    if temp.sum() == 0: return np.zeros(len(array))
    unique_classes = np.unique(array[temp])
    class_id = {unique_classes[i]: i + 1 for i in range(len(unique_classes))}
    return np.array([class_id[i] if i in class_id else 0 for i in array])


class LogDataset(torch.utils.data.Dataset):
    def __init__(self, window_length=500, give_what='normal', shared_memory=None):
        self.shared_memory = shared_memory

        self.file_loc = r"Dataset/out/parsed_HDFS_.log_structured.csv"
        self.window_length = window_length
        self.give_what = give_what
        self.lens = pd.read_csv(self.file_loc, usecols=["Pid"], low_memory=False).shape[0] - self.window_length - 1
        self.class_col = [
            'Event_id',
            'Pid',
            'Parameter_0_ip',
            # 'Parameter_0_port',
            'Parameter_1_ip',
            'Parameter_1_port',
            'Parameter_2_ip',
            'Parameter_2_port',
        ]
        self.num_col = [
            'completeTime',
            # 'Parameter_0_packet_size',
            'Parameter_1_packet_size',
            # 'Parameter_2_packet_size',
        ]
        self.final_col = ['labels', 'blk_z',] + self.num_col + self.class_col
        self._file_col = pd.read_csv(self.file_loc, nrows=1).columns

    def __len__(self):
        return self.lens

    def read_file(self, idx):
        if self.shared_memory is not None:
            return pd.DataFrame(self.shared_memory[idx:idx+self.window_length*2], columns=self._file_col)
        return pd.read_csv(self.file_loc, nrows=self.window_length*2, skiprows=idx + 1, names=self._file_col)


    def __getitem__(self, idx):
        data = self.read_file(idx)
        if self.give_what == 'normal':
            data = data[data['labels'] == 0]
        data = data.iloc[:self.window_length]

        return self.prepare_data(data[self.final_col])

    def prepare_data(self, data):
        for col in self.num_col:
            # min_max_perc = np.percentile(data[col].dropna().values.astype(float), [2,98])

            temp = data[col].values.astype(float)
            if col == 'completeTime':
                min_max_perc=[temp.min(), temp.min()+0.1595*len(temp)/1000]
            elif col == 'Parameter_1_packet_size':
                min_max_perc=[28408864, 67108864]
            else: raise

            temp = (temp - min_max_perc[0]) / (min_max_perc[1] - min_max_perc[0])
            temp = np.clip(temp, 0, 1)
            # print(col, min_max_perc, data[col].mean())
            data[col] = temp

        data.fillna({'Parameter_1_packet_size': 0}, inplace=True)

        for col in self.class_col:
            if col == 'Event_id': continue
            data[col] = get_class_id_array(data[col].values)

        labels = data['labels'].values
        names = data['blk_z'].values
        data = [data[self.class_col].values.astype(int),data[self.num_col].values.astype(float)]

        temp = np.eye(int(self.window_length))#/10*8))
        loss_data = (np.array([temp[data[0].T[i].astype(int)]
                               for i in range(len(self.class_col))]).astype(int).transpose(1,0,2),
                     np.array([data[1].T[i][:, np.newaxis]
                               for i in range(len(self.num_col))]).transpose(1,0,2))

        return labels, names, data, loss_data

    def collate_fn(self, batch):
        labels, names, data, loss_data = list(zip(*batch))
        labels = np.array(labels).astype(int)
        names = np.array(names)
        data = list(zip(*data))
        data = (torch.tensor(np.array(data[0])),torch.tensor(np.array(data[1])))
        loss_data = list(zip(*loss_data))
        loss_data = (torch.tensor(np.array(loss_data[0])), torch.tensor(np.array(loss_data[1])))
        return labels, names, data, loss_data


def parse_df(df, template_id, df_label):
    df["Time"] = df['Time'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
    df["Time"] = df['Time'].apply(lambda x: int(x[:2]) * 60 + int(x[2:4]) + int(x[4:]) / 60)
    df["Date"] = df['Date'].apply(lambda x: (x - 81109) * 24 * 60)
    df["completeTime"] = df["Date"] + df["Time"] + df["LineId"] / 1e8 - 1235
    df.drop(columns=["Date", "Time", "LineId"], inplace=True)

    df['ParameterList'] = df['ParameterList'].apply(lambda x: eval(x))
    df['ParameterList'] = df['ParameterList'].apply(
        lambda x: [[a.split(":")[0], ":" + str(a.split(":")[1])] if ':' in a else [a] for a in x])
    df['ParameterList'] = df['ParameterList'].apply(lambda x: [a for b in x for a in b])
    df['blk_z'] = df['ParameterList'].apply(lambda x: x[np.array(['blk_' in a for a in x]).argmax()])
    df['blk_z'] = df['blk_z'].apply(lambda x: 'blk_' + x.split('blk_')[1].split(' ')[0])

    l = 3
    for i in range(l):
        # check if parameter is ip with port
        temp = df['ParameterList'].apply(lambda x: x[i] if len(x) > i else np.nan)
        temp = temp.apply(
            lambda x:
            (np.nan, np.nan) if type(x) is float and np.isnan(x) else
            (x, 'dir') if x.startswith('/') else
            (x, 'ip') if x.count('.') == 3 and x.replace('.', '').isdigit() else
            (x[1:], 'port') if ':' in x else
            (x, 'packet_size') if x.isdigit() and len(x) > 3 else
            (x, 'single_dig') if x.isdigit() and len(x) == 1 else
            (x, 'blk_z') if 'blk_' in x else
            (np.nan, np.nan)
        )
        selected = ['ip', 'port', 'packet_size']
        for s in selected:
            df['Parameter_%s_%s' % (i, s)] = temp.apply(lambda x: x[0] if x[1] == s else np.nan)

    df['Event_id'] = df['EventId'].apply(lambda x: template_id[template_id['EventId'] == x]['aaaa'].values[0])
    df['labels'] = df['blk_z'].apply(lambda x: int(df_label[x] == 'Anomaly'))  # if x in df_label else np.nan)

    df = df[['labels', 'blk_z', "completeTime", "Event_id", "Pid",
             *['Parameter_%s_%s' % (i, s) for i in range(l) for s in selected]]]

    return df


def replace_with_unique_classification_id(array):
    unique_classes = np.unique(array[array == array])
    class_id = {unique_classes[i]: i + 1 for i in range(len(unique_classes))}
    return np.array([class_id[i] if i in class_id else 0 for i in array])


if __name__ == '__main__':
    # next(iter(torch.utils.data.DataLoader(LogDataset(400), batch_size=150, collate_fn=LogDataset(2).collate_fn, shuffle=True)))
    # temp = LogDataset(500)[0]

    pass
