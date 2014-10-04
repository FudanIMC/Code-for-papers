rcnn_exp_cache_features('train','2007');   % chunk1
rcnn_exp_cache_features('val','2007');     % chunk2
rcnn_exp_cache_features('test_1','2007');  % chunk3
rcnn_exp_cache_features('test_2','2007');  % chunk4
test_results = rcnn_exp_train_and_test('2007');
