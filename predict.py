import csv
import numpy as np
from fpgnn.tool import set_predict_argument, get_scaler, load_args, load_data, load_model
from fpgnn.train import predict
from fpgnn.data import MoleDataSet

def predicting(args):
    print('Load args.')
    scaler = get_scaler(args.model_path)
    print('scaler',scaler)
    train_args = load_args(args.model_path)
    
    for key,value in vars(train_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    print('Load data.')
    test_data = load_data(args.predict_path,args)
    fir_data_len = len(test_data)
    all_data = test_data
    if fir_data_len == 0:
        raise ValueError('Data is empty.')
    
    smi_exist = []
    for i in range(fir_data_len):
        if test_data[i].mol is not None:
            smi_exist.append(i)
    test_data = MoleDataSet([test_data[i] for i in smi_exist])
    now_data_len = len(test_data)
    print('There are ',now_data_len,' smiles in total.')
    if fir_data_len - now_data_len > 0:
        print('There are ',fir_data_len - now_data_len, ' smiles invalid.')
    
    print('Load model')
    model = load_model(args.model_path,args.cuda)
    test_pred = predict(model,test_data,args.batch_size,scaler)
    assert len(test_data) == len(test_pred)
    test_pred = np.array(test_pred)
    test_pred = test_pred.tolist()
    
    print('Write result.')
    all_pred = [None] * len(all_data)
    for i,j in enumerate(smi_exist):
        all_pred[j] = test_pred[i]
    
    write_smile = all_data.smile()
    with open(args.result_path, 'w',newline = '') as file:
        writer = csv.writer(file)
        
        line = ['Smiles']
        line.extend(args.task_names)
        writer.writerow(line)
        
        for i in range(fir_data_len):
            line = []
            line.append(write_smile[i])
            if all_pred[i] is not None:
                line.extend(all_pred[i])
            else:
                row.extend(['']*args.task_num)
            writer.writerow(line)


if __name__=='__main__':
    args = set_predict_argument()
    predicting(args)
    