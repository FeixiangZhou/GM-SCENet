
config = dict()
config['lr'] = 1e-4
config['momentum'] = 0.9
config['weight_decay'] = 1e-4
config['epoch_num'] = 150
config['batch_size'] = 4
config['batch_size_valid'] = 1
config['nstack'] = 3
config['sample_num'] = 0
config['train_num'] = 0
config['valid_num'] = 0
config['test_num'] = 0
config['lowrmse'] = 100000
config['highPCK'] = 0
config['ncascaded'] = 1
config['sigma'] = 1.0
config['debug_vis'] = False
config['root_path'] = ''

#-------trainfile--------------------
config['trainsample'] = 'labeled-data/MAQ00261/CollectedData_fx23.csv'
# config['trainsample'] = 'labeled-data/mouse/CollectedData_Pranav_train.csv' #

#zebra
# config['trainsample'] = 'labeled-data/zebraimage/data_train.csv'
# config['trainsample'] = 'labeled-data/flyimage/data_train.csv'


#--------validfile-------------------
config['validsample'] = 'labeled-data-valid/MAQ00261/CollectedData_fx.csv' #valid 3000
# config['validsample'] = 'labeled-data/mouse/CollectedData_Pranav_test.csv'

# config['validsample'] = 'labeled-data/zebraimage/data_test.csv'
# config['validsample'] = 'labeled-data/flyimage/data_test.csv'




#--------testfile-------------------
# config['testsample'] = 'labeled-data-test/MAQ00261/CollectedData_fx2000.csv' #2000
# config['testsample'] = 'labeled-data/mouse/CollectedData_Pranav_test.csv'
# config['testsample'] = 'labeled-data/zebraimage/data_test.csv'
# config['testsample'] = 'labeled-data/flyimage/data_test.csv'


